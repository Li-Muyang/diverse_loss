from typing import Optional

import torch
import torch.nn.functional as F


def _plain_ce(logits, labels, num_items_in_batch, ignore_index):
    logits = logits.float()
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    return loss


def _reduce(loss_per_token, valid, num_items_in_batch):
    """Reduce per-token loss, ignoring invalid positions, matching HF's grad-accum convention."""
    loss_per_token = torch.where(valid, loss_per_token, torch.zeros_like(loss_per_token))
    if num_items_in_batch is not None:
        return loss_per_token.sum() / num_items_in_batch
    n_valid = valid.sum().clamp(min=1)
    return loss_per_token.sum() / n_valid


def _topk_ce(logits, labels, valid, safe_labels, k, num_items_in_batch, ignore_index):
    """Top-k CE computed from gathered (N, k) slices — never materializes (N, V) masks."""
    # (N, k) values and indices, already sorted descending by value.
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
    topk_vals = topk_vals.float()
    label_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).float()  # (N, 1)

    label_in_set = (topk_idx == safe_labels.unsqueeze(-1)).any(dim=-1, keepdim=True)  # (N, 1)
    # Append label logit to the normalizer, but -inf it out when label is already in top-k
    # to avoid double-counting.
    extra = torch.where(label_in_set, torch.full_like(label_logit, float("-inf")), label_logit)
    lse = torch.logsumexp(torch.cat([topk_vals, extra], dim=-1), dim=-1)  # (N,)

    loss_per_token = lse - label_logit.squeeze(-1)
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    n_valid = valid.sum()
    label_in_subset = (
        ((label_in_set.squeeze(-1) & valid).sum().float() / n_valid.clamp(min=1)).item()
        if n_valid > 0 else 1.0
    )
    return loss, label_in_subset


def _topp_ce(logits, labels, valid, safe_labels, p, num_items_in_batch, ignore_index):
    """Top-p CE. Still requires a full sort of logits (inherent to nucleus selection), but
    avoids the extra scatter-back and masked-fill on the original (N, V) tensor.
    """
    logits = logits.float()
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)  # (N, V)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    keep = (cumprobs - probs) < p
    keep[..., 0] = True
    # LSE directly in sorted order — no need to scatter back.
    lse_kept = torch.logsumexp(sorted_logits.masked_fill(~keep, float("-inf")), dim=-1)  # (N,)

    label_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (N,)
    label_in_set = ((sorted_idx == safe_labels.unsqueeze(-1)) & keep).any(dim=-1)  # (N,)
    extra = torch.where(label_in_set, torch.full_like(label_logit, float("-inf")), label_logit)
    lse = torch.logaddexp(lse_kept, extra)

    loss_per_token = lse - label_logit
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    n_valid = valid.sum()
    label_in_subset = (
        ((label_in_set & valid).sum().float() / n_valid.clamp(min=1)).item()
        if n_valid > 0 else 1.0
    )
    return loss, label_in_subset


def _topk_topp_ce(logits, labels, valid, safe_labels, k, p, num_items_in_batch, ignore_index):
    """Combined: apply top-p within the top-k set. All work on (N, k) tensors."""
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)  # (N, k) sorted desc
    topk_vals = topk_vals.float()
    # nucleus over the top-k set only
    probs = F.softmax(topk_vals, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    keep = (cumprobs - probs) < p
    keep[..., 0] = True
    lse_kept = torch.logsumexp(topk_vals.masked_fill(~keep, float("-inf")), dim=-1)  # (N,)

    label_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).float().squeeze(-1)  # (N,)
    label_in_set = ((topk_idx == safe_labels.unsqueeze(-1)) & keep).any(dim=-1)  # (N,)
    extra = torch.where(label_in_set, torch.full_like(label_logit, float("-inf")), label_logit)
    lse = torch.logaddexp(lse_kept, extra)

    loss_per_token = lse - label_logit
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    n_valid = valid.sum()
    label_in_subset = (
        ((label_in_set & valid).sum().float() / n_valid.clamp(min=1)).item()
        if n_valid > 0 else 1.0
    )
    return loss, label_in_subset


def _mask_topk_ce(logits, labels, valid, safe_labels, k, num_items_in_batch, ignore_index):
    """CE with the top-k logits *excluded* from the normalizer (label kept if it's in top-k).

    Uses the identity `logsumexp(S_kept) = lse_full + log1p(-Z_excluded / Z_full)`, so we
    can reuse F.cross_entropy's fused log_softmax kernel for the full-vocab part and only
    do one extra (N, k) reduction.

    Trick: F.cross_entropy(reduction='none') returns `lse_full - label_logit` per position,
    so we get both quantities in one pass without materializing `logits.float()` explicitly.
    """
    ce_full = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction="none")  # (N,)
    label_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).float().squeeze(-1)  # (N,)
    lse_full = ce_full + label_logit  # (N,)

    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
    topk_vals = topk_vals.float()

    # When the label is in top-k we must keep it in the normalizer; force its topk slot to
    # -inf before summing so it's excluded from Z_excluded.
    label_in_topk = topk_idx == safe_labels.unsqueeze(-1)  # (N, k)
    topk_vals_excl = topk_vals.masked_fill(label_in_topk, float("-inf"))

    # Z_excluded / Z_full, shifted for stability.
    ratio = (topk_vals_excl - lse_full.unsqueeze(-1)).exp().sum(dim=-1)
    # Clamp away from 1 so log1p(-1) = -inf doesn't blow up. Happens only on extremely
    # peaked distributions where essentially all mass sits in top-k.
    ratio = ratio.clamp(max=1.0 - 1e-6)
    lse_kept = lse_full + torch.log1p(-ratio)  # (N,)

    loss_per_token = lse_kept - label_logit
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    n_valid = valid.sum()
    # Diagnostic: fraction of valid labels that fell inside the masked-out top-k set.
    # High values mean the loss gives ~0 gradient on those positions (which is the point,
    # but also means aggressive masking may stall training).
    label_in_masked = label_in_topk.any(dim=-1) & valid
    label_in_mask_frac = (
        (label_in_masked.sum().float() / n_valid.clamp(min=1)).item() if n_valid > 0 else 0.0
    )
    return loss, label_in_mask_frac


def _mask_topp_ce(logits, labels, valid, safe_labels, p, num_items_in_batch, ignore_index):
    """CE with the minimal top-p nucleus *excluded* from the normalizer (label kept).

    Requires a full sort (inherent to nucleus selection). We compute logsumexp over the
    un-masked sorted logits directly — no scatter-back needed.
    """
    logits_f = logits.float()
    sorted_logits, sorted_idx = torch.sort(logits_f, dim=-1, descending=True)  # (N, V)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    # Minimal nucleus set in sorted order.
    nucleus = (cumprobs - probs) < p
    nucleus[..., 0] = True

    # Label's position in the sorted array — if it lies in the nucleus we must keep it.
    label_sorted_pos = (sorted_idx == safe_labels.unsqueeze(-1))  # (N, V)
    label_in_nucleus = (label_sorted_pos & nucleus).any(dim=-1)  # (N,)

    # Mask = nucleus AND NOT label position. Everything else is kept.
    drop = nucleus & ~label_sorted_pos
    lse_kept = torch.logsumexp(sorted_logits.masked_fill(drop, float("-inf")), dim=-1)  # (N,)

    label_logit = logits_f.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (N,)
    loss_per_token = lse_kept - label_logit
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    n_valid = valid.sum()
    label_in_mask_frac = (
        ((label_in_nucleus & valid).sum().float() / n_valid.clamp(min=1)).item()
        if n_valid > 0 else 0.0
    )
    return loss, label_in_mask_frac


def mask_topk_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    k: Optional[int] = None,
    p: Optional[float] = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float]:
    """Cross-entropy with the top-k and/or top-p logits *excluded* from the softmax
    normalizer (the inverse of `topk_ce_loss`).

    Intuition: the model's confident predictions are preserved — we don't push probability
    mass off the top-k tokens, which avoids disrupting the existing linguistic structure
    the pretrained model has learned. Gradient signal comes from pulling the label up
    against the low-confidence tail.

    Semantics:
      - If the label is in the masked set, it is *still kept* in the normalizer (otherwise
        the loss would be ill-defined). In practice this means the position gets very
        little gradient when the model is already confident on the label — desired.
      - When both `k` and `p` are set, the union of the top-k and top-p sets is excluded.

    Expects already-shifted `logits` of shape (N, V) and `labels` of shape (N,).

    Returns `(loss, label_in_masked_fraction)` — the fraction of valid positions where
    the label landed inside the masked-out set. High values mean most positions contribute
    little gradient (confident + correct); low values mean masking is doing most of the work.
    """
    V = logits.size(-1)
    use_topk = k is not None and 0 < k < V
    use_topp = p is not None and 0.0 < p < 1.0

    if not (use_topk or use_topp):
        return _plain_ce(logits, labels, num_items_in_batch, ignore_index), 0.0

    valid = labels != ignore_index
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    if use_topk and not use_topp:
        return _mask_topk_ce(logits, labels, valid, safe_labels, k, num_items_in_batch, ignore_index)
    if use_topp and not use_topk:
        return _mask_topp_ce(logits, labels, valid, safe_labels, p, num_items_in_batch, ignore_index)

    # Combined: fall back to general masking. Union of top-k and top-p sets dropped.
    logits_f = logits.float()
    _, topk_idx = torch.topk(logits_f, k, dim=-1)
    drop = torch.zeros_like(logits_f, dtype=torch.bool)
    drop.scatter_(-1, topk_idx, True)

    sorted_logits, sorted_idx = torch.sort(logits_f, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    nucleus = (cumprobs - probs) < p
    nucleus[..., 0] = True
    nucleus_unsorted = torch.zeros_like(drop)
    nucleus_unsorted.scatter_(-1, sorted_idx, nucleus)
    drop |= nucleus_unsorted

    # Protect the label position.
    drop.scatter_(-1, safe_labels.unsqueeze(-1), False)

    lse_kept = torch.logsumexp(logits_f.masked_fill(drop, float("-inf")), dim=-1)
    label_logit = logits_f.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    loss_per_token = lse_kept - label_logit
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    # Diagnostic: was the label inside the combined masked set (before we un-masked it).
    n_valid = valid.sum()
    label_col = safe_labels.unsqueeze(-1)
    # Recompute without the label-protection to know whether it was in the masked set.
    in_topk = (topk_idx == label_col).any(dim=-1)
    in_nucleus = (sorted_idx == label_col) & nucleus
    in_nucleus = in_nucleus.any(dim=-1)
    label_in_masked = (in_topk | in_nucleus) & valid
    label_in_mask_frac = (
        (label_in_masked.sum().float() / n_valid.clamp(min=1)).item() if n_valid > 0 else 0.0
    )
    return loss, label_in_mask_frac


def random_k_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: int,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float]:
    """CE with the softmax normalizer restricted to `k` *uniformly random* vocab indices
    per position (the "sampled softmax" / negative-sampling variant of `topk_ce_loss`).

    Motivation: `topk_ce_loss` always penalizes the model's own confident predictions,
    which can create a feedback loop where a small set of tokens are repeatedly pushed
    down. Random sampling gives every vocab token an equal chance of being a negative
    across training, removing that bias. In expectation over samples, this is an unbiased
    estimator of full-vocab CE up to a constant (see Mnih & Teh 2012, NCE; Mikolov et al.
    2013, word2vec skip-gram).

    Sampling: independent `torch.randint(0, V, (N, k))` per forward pass. With replacement
    for speed — for k=512 / V=150k, expected duplicates per row ≈ 1, so effective k ≈ 511.
    The label's logit is always appended to the normalizer if not already in the random
    set (no double-counting), so CE stays finite.

    Returns `(loss, label_in_sample_fraction)`. The diagnostic should be approximately
    k/V under the random-sampling hypothesis — a useful sanity check that sampling is
    uniform.
    """
    V = logits.size(-1)
    if k <= 0 or k >= V:
        return _plain_ce(logits, labels, num_items_in_batch, ignore_index), 1.0

    valid = labels != ignore_index
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    N = logits.size(0)
    rand_idx = torch.randint(0, V, (N, k), device=logits.device)  # (N, k)
    rand_vals = logits.gather(-1, rand_idx).float()  # (N, k)

    label_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).float()  # (N, 1)
    label_in_sample = (rand_idx == safe_labels.unsqueeze(-1)).any(dim=-1, keepdim=True)  # (N, 1)
    # If the label is already in the random set, its logit is already counted — mask the
    # append-slot to -inf to avoid double-counting.
    extra = torch.where(label_in_sample, torch.full_like(label_logit, float("-inf")), label_logit)
    lse = torch.logsumexp(torch.cat([rand_vals, extra], dim=-1), dim=-1)  # (N,)

    loss_per_token = lse - label_logit.squeeze(-1)
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    n_valid = valid.sum()
    frac = (
        ((label_in_sample.squeeze(-1) & valid).sum().float() / n_valid.clamp(min=1)).item()
        if n_valid > 0 else 0.0
    )
    return loss, frac


def gem_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    beta: float = 0.7,
    h: str = "linear",
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float]:
    """GEM loss (ICLR 2025, Li et al. "Preserving Diversity in Supervised Fine-tuning
    of Large Language Models"). Port of the reference implementation at
    https://github.com/pr-gigantic/GEM/blob/main/sft_trainer_v2.py#L37-L74.

    Per valid position j:
        p       = softmax(logits)                                  # model distribution (log-space)
        q       = softmax(logits / beta).detach()                  # sharpened target (no grad)
        weights = 1                                         if h == "linear"
                  sigmoid(0.01 * (logits - logit[label]))   if h == "logsigmoid"
        L       = -sum_j q_j * weights_j * (log p[label] - log p_j)

    Intuition: pull the label up in log-space, but weight the contribution of each
    competitor by how plausible `q` considers it (controlled by beta). This preserves the
    model's entropy over already-plausible alternatives and avoids the mode-seeking
    collapse of CE.

    Args:
        logits: (N, V) flat logits.
        labels: (N,) flat labels; positions equal to `ignore_index` are dropped before
            the computation (matches the reference code's `shift_logits[mask]` filtering).
        beta: in (0, 1]. Closer to 1 ≈ CE; closer to 0 preserves more diversity.
        h: `"linear"` (default, matches the paper's TrainingArguments default) or
            `"logsigmoid"`.
        num_items_in_batch: HF grad-accum convention — if provided, reduction is
            `sum / num_items_in_batch`; otherwise `mean` over valid positions.

    Returns:
        `(loss, q_on_label)` where `q_on_label` is the mean of `q[label]` across valid
        positions (diagnostic: high → q is CE-like; low → q spread over the top tokens).
    """
    valid = labels != ignore_index
    if not valid.any():
        # No valid tokens — return 0 with grad-flow preserved so the backward pass is a noop.
        return logits.sum() * 0.0, 0.0

    logits = logits[valid]
    labels = labels[valid]

    with torch.no_grad():
        logits_on_labels = torch.gather(
            logits, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        logits_diff = logits - logits_on_labels.unsqueeze(-1)
        if h == "linear":
            weights = torch.ones_like(logits_diff)
        elif h == "logsigmoid":
            weights = F.sigmoid(0.01 * logits_diff)
        else:
            raise ValueError(f"Unknown GEM h={h!r}; must be 'linear' or 'logsigmoid'")

    gene_log_probs = F.log_softmax(logits, dim=-1)
    q_probs = torch.exp(F.log_softmax(logits / beta, dim=-1)).detach()

    real_log_probs = torch.gather(gene_log_probs, dim=-1, index=labels.unsqueeze(-1))

    per_token = -torch.sum(q_probs * weights * (real_log_probs - gene_log_probs), dim=-1)
    if num_items_in_batch is not None:
        loss = per_token.sum() / num_items_in_batch
    else:
        loss = per_token.mean()

    with torch.no_grad():
        q_on_label = q_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean().item()

    return loss, q_on_label


def margin_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float]:
    """Pairwise-softmax margin loss — binary CE between the label logit and its best
    competitor's logit. Equivalent to `softplus(-gap) = -log(sigmoid(gap))` where
    `gap = logit[label] - max_{j != label} logit[j]`.

    This is the smooth version of hinge: gradient is non-zero everywhere and pushes the
    label-vs-competitor gap wider indefinitely, with the push fading exponentially as the
    gap grows. No margin hyperparameter — it's just binary classification over the
    label and its nearest rival.

    Implemented with a single `torch.topk(logits, 2)` — no `(N, V)` masks.

    Returns `(loss, top1_accuracy)` — fraction of valid positions where the label is the
    argmax (a useful diagnostic, though unlike hinge the loss keeps training past top1=1.0).
    """
    valid = labels != ignore_index
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    label_logit = logits.gather(-1, safe_labels.unsqueeze(-1)).float().squeeze(-1)  # (N,)

    top2_vals, top2_idx = torch.topk(logits, 2, dim=-1)  # (N, 2), sorted desc
    top2_vals = top2_vals.float()
    label_is_top1 = top2_idx[:, 0] == safe_labels  # (N,)
    best_other = torch.where(label_is_top1, top2_vals[:, 1], top2_vals[:, 0])  # (N,)

    gap = label_logit - best_other
    loss_per_token = F.softplus(-gap)
    loss = _reduce(loss_per_token, valid, num_items_in_batch)

    n_valid = valid.sum()
    top1_acc = (
        ((label_is_top1 & valid).sum().float() / n_valid.clamp(min=1)).item()
        if n_valid > 0 else 0.0
    )
    return loss, top1_acc


def topk_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    k: Optional[int] = None,
    p: Optional[float] = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float]:
    """Cross-entropy with the softmax normalizer restricted to the top-k and/or top-p logits.

    Expects already-shifted `logits` of shape (N, V) and `labels` of shape (N,). The ground-
    truth token is always included in the normalizer so the loss is finite.

    Efficiency notes:
      - top-k path never materializes a (N, V) mask; all work on (N, k+1) tensors.
      - top-p path still needs an (N, V) sort (inherent to nucleus selection) but avoids
        the extra scatter-back + masked_fill round-trip and the full-vocab log_softmax.
      - combined path runs top-p within the top-k set, so everything is (N, k).

    Follows HF's gradient-accumulation convention: `num_items_in_batch` provided → reduction
    is `sum / num_items_in_batch`; otherwise `mean` over valid positions.

    Returns `(loss, label_in_subset_fraction)` — second element is the fraction of valid
    labels already inside the kept set (diagnostic; low values mean the restriction is
    too aggressive).
    """
    V = logits.size(-1)
    use_topk = k is not None and 0 < k < V
    use_topp = p is not None and 0.0 < p < 1.0

    if not (use_topk or use_topp):
        return _plain_ce(logits, labels, num_items_in_batch, ignore_index), 1.0

    valid = labels != ignore_index
    safe_labels = torch.where(valid, labels, torch.zeros_like(labels))

    if use_topk and use_topp:
        return _topk_topp_ce(logits, labels, valid, safe_labels, k, p, num_items_in_batch, ignore_index)
    if use_topk:
        return _topk_ce(logits, labels, valid, safe_labels, k, num_items_in_batch, ignore_index)
    return _topp_ce(logits, labels, valid, safe_labels, p, num_items_in_batch, ignore_index)
