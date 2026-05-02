from typing import Any, Optional

import torch
import trl

from .losses import topk_ce_loss


class SFTTrainer(trl.SFTTrainer):
    """`trl.SFTTrainer` with optional top-k / top-p restricted cross-entropy.

    When `args.topk_ce_k` or `args.topk_ce_p` is set, the softmax normalizer in the
    training loss is restricted to the top-k and/or top-p logits per position (the
    label's own logit is always kept). When both are None, behavior is identical to
    the upstream trainer.

    The token-accuracy metric is computed against the unrestricted logits, matching
    the upstream trainer's semantics.
    """

    def compute_loss(
        self,
        model,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        k = getattr(self.args, "topk_ce_k", None)
        p = getattr(self.args, "topk_ce_p", None)
        use_topk = k is not None and k > 0
        use_topp = p is not None and 0.0 < p < 1.0

        if not (use_topk or use_topp):
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        if self.args.use_liger_kernel:
            raise ValueError(
                "topk_ce_k / topk_ce_p are incompatible with `use_liger_kernel=True` because "
                "Liger fuses logits with the CE kernel and does not expose logits for masking."
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

        loss, label_in_subset = topk_ce_loss(
            flat_logits,
            flat_labels,
            num_items_in_batch=num_items_in_batch,
            k=k if use_topk else None,
            p=p if use_topp else None,
        )

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
            self._metrics[mode]["label_in_topk_fraction"].append(label_in_subset)

        return (loss, outputs) if return_outputs else loss
