from typing import Optional

import torch
import torch.nn.functional as F


def _topk_keep_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    _, topk_idx = torch.topk(logits, k, dim=-1)
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(-1, topk_idx, True)
    return keep


def _topp_keep_mask(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Minimal nucleus set: smallest set of highest-prob tokens whose cumulative probability
    is >= p. The top-1 is always kept.
    """
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)
    keep_sorted = (cumprobs - probs) < p
    keep_sorted[..., 0] = True
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep.scatter_(-1, sorted_idx, keep_sorted)
    return keep


def topk_ce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_items_in_batch: Optional[int] = None,
    k: Optional[int] = None,
    p: Optional[float] = None,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, float]:
    """Cross-entropy with the softmax normalizer restricted to the top-k and/or top-p logits.

    Expects already-shifted `logits` of shape (N, V) and `labels` of shape (N,). The caller
    is responsible for the next-token shift.

    When both `k` and `p` are set, top-k is applied first and top-p within that set.
    The ground-truth logit is always kept in the subset, so CE stays finite when the label
    falls outside the top-k/p band.

    Follows HF's gradient-accumulation convention: if `num_items_in_batch` is provided, the
    loss is summed and divided by `num_items_in_batch`; otherwise reduction is "mean".

    Returns `(loss, label_in_subset_fraction)` where the second element is the fraction of
    valid-label positions whose ground-truth token was inside the top-k/p set (a useful
    diagnostic — low values mean the restriction is too aggressive).
    """
    V = logits.size(-1)
    use_topk = k is not None and 0 < k < V
    use_topp = p is not None and 0.0 < p < 1.0

    logits = logits.float()  # matches HF ForCausalLMLoss precision handling

    if use_topk or use_topp:
        keep = torch.ones_like(logits, dtype=torch.bool)
        if use_topk:
            keep &= _topk_keep_mask(logits, k)
        if use_topp:
            keep &= _topp_keep_mask(logits, p)

        valid = labels != ignore_index
        safe_idx = torch.where(valid, labels, torch.zeros_like(labels)).unsqueeze(-1)

        # Diagnostic: fraction of valid labels already inside the kept set.
        in_subset = keep.gather(-1, safe_idx).squeeze(-1) & valid
        n_valid = valid.sum()
        label_in_subset = (in_subset.sum().float() / n_valid.clamp(min=1)).item() if n_valid > 0 else 1.0

        # Force the label index to be kept so CE is finite.
        keep.scatter_(-1, safe_idx, True)
        logits = logits.masked_fill(~keep, float("-inf"))
    else:
        label_in_subset = 1.0

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    return loss, label_in_subset
