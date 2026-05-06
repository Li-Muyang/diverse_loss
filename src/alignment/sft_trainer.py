from typing import Any, Optional

import torch
import trl

from .losses import (
    focal_loss,
    gem_loss,
    margin_loss,
    mask_topk_ce_loss,
    random_k_ce_loss,
    topk_ce_loss,
)


class SFTTrainer(trl.SFTTrainer):
    """`trl.SFTTrainer` with optional alternative training losses.

    Seven modes, controlled by `SFTConfig` fields (all mutually exclusive):
      - `topk_ce_k` / `topk_ce_p`: *restrict* the CE normalizer to the top-k/p logits
        (penalize only confident competitors; label is always kept).
      - `mask_topk_ce_k` / `mask_topk_ce_p`: *exclude* the top-k/p logits from the CE
        normalizer (preserve the model's confident predictions; label is always kept).
      - `margin_loss`: pairwise softmax CE between label and best competitor.
      - `gem_loss` + `gem_beta` + `gem_h`: GEM loss (Li et al. ICLR 2025), a diversity-
        preserving alternative to CE.
      - `random_k_ce_k`: sampled-softmax CE — normalizer over `k` uniformly random vocab
        indices (unbiased negative sampling).
      - `focal_loss` + `focal_gamma`: focal loss — down-weights easy positions via
        `(1 - p_t)^gamma`.
      - None set: identical to the upstream trainer.

    The token-accuracy metric is always computed against the unrestricted logits.
    """

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        keep_k = getattr(self.args, "topk_ce_k", None)
        keep_p = getattr(self.args, "topk_ce_p", None)
        mask_k = getattr(self.args, "mask_topk_ce_k", None)
        mask_p = getattr(self.args, "mask_topk_ce_p", None)
        use_keep = (keep_k is not None and keep_k > 0) or (keep_p is not None and 0.0 < keep_p < 1.0)
        use_mask = (mask_k is not None and mask_k > 0) or (mask_p is not None and 0.0 < mask_p < 1.0)
        use_margin = bool(getattr(self.args, "margin_loss", False))
        use_gem = bool(getattr(self.args, "gem_loss", False))
        random_k = getattr(self.args, "random_k_ce_k", None)
        use_random = random_k is not None and random_k > 0
        use_focal = bool(getattr(self.args, "focal_loss", False))

        if sum([use_keep, use_mask, use_margin, use_gem, use_random, use_focal]) > 1:
            raise ValueError(
                "`topk_ce_*`, `mask_topk_ce_*`, `margin_loss`, `gem_loss`, "
                "`random_k_ce_k`, and `focal_loss` are mutually exclusive. "
                "Set at most one."
            )

        if not (use_keep or use_mask or use_margin or use_gem or use_random or use_focal):
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        if self.args.use_liger_kernel:
            raise ValueError(
                "Alternative loss options are incompatible with `use_liger_kernel=True` "
                "because Liger fuses logits with the CE kernel and does not expose logits."
            )

        mode = "train" if model.training else "eval"
        labels = inputs.pop("labels")

        # Model accepts loss kwargs — forwarding `num_items_in_batch` would trigger HF's
        # built-in loss and cause a double loss computation. Ours replaces it.
        outputs = model(**inputs)
        logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
        V = shift_logits.size(-1)
        flat_logits = shift_logits.view(-1, V)
        flat_labels = shift_labels.view(-1)

        if use_keep:
            loss, diag_frac = topk_ce_loss(
                flat_logits,
                flat_labels,
                num_items_in_batch=num_items_in_batch,
                k=keep_k,
                p=keep_p,
            )
            diag_key = "label_in_topk_fraction"
        elif use_mask:
            loss, diag_frac = mask_topk_ce_loss(
                flat_logits,
                flat_labels,
                num_items_in_batch=num_items_in_batch,
                k=mask_k,
                p=mask_p,
            )
            diag_key = "label_in_masked_fraction"
        elif use_margin:
            loss, diag_frac = margin_loss(
                flat_logits,
                flat_labels,
                num_items_in_batch=num_items_in_batch,
            )
            diag_key = "margin_top1_accuracy"
        elif use_gem:
            loss, diag_frac = gem_loss(
                flat_logits,
                flat_labels,
                num_items_in_batch=num_items_in_batch,
                beta=self.args.gem_beta,
                h=self.args.gem_h,
            )
            diag_key = "gem_q_on_label"
        elif use_random:
            loss, diag_frac = random_k_ce_loss(
                flat_logits,
                flat_labels,
                k=random_k,
                num_items_in_batch=num_items_in_batch,
            )
            diag_key = "label_in_random_fraction"
        else:  # focal
            loss, diag_frac = focal_loss(
                flat_logits,
                flat_labels,
                gamma=self.args.focal_gamma,
                num_items_in_batch=num_items_in_batch,
            )
            diag_key = "focal_mean_p_t"

        # Put labels back so downstream logging (e.g. accuracy) can still find them.
        inputs["labels"] = labels

        # Token-accuracy + token-count bookkeeping, mirroring upstream SFTTrainer.
        if mode == "train":
            if "attention_mask" in inputs:
                num_tokens_in_batch = (
                    self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
                )
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(
                    inputs["position_ids"].size(1), device=inputs["position_ids"].device
                )
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        with torch.no_grad():
            mask = shift_labels != -100
            predictions = shift_logits.argmax(dim=-1)
            correct = ((predictions == shift_labels) & mask).sum()
            total = mask.sum()
            correct = self.accelerator.gather_for_metrics(correct)
            total = self.accelerator.gather_for_metrics(total)
            total_sum = total.sum()
            accuracy = (correct.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)
            self._metrics[mode][diag_key].append(diag_frac)

        return (loss, outputs) if return_outputs else loss
