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
