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
