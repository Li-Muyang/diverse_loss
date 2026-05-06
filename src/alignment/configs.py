# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

import trl


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(trl.ScriptArguments):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    Args:
        dataset_mixture (`dict[str, Any]` or `None`, *optional*, defaults to `None`):
            Configuration for creating dataset mixtures with advanced options.
            Format:
              dataset_mixture:
                datasets:
                  - id: dataset_id1
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                  - id: dataset_id2
                    config: config_name
                    columns:
                      - col1
                      - col2
                    weight: 0.5
                seed: 42
                test_split_size: 0.1
    """

    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={"help": "Configuration for creating dataset mixtures with advanced options like shuffling."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [set(dataset.columns) for dataset in datasets_list if dataset.columns is not None]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


@dataclass
class SFTConfig(trl.SFTConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    topk_ce_k: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, restrict the cross-entropy softmax normalizer to the top-k logits per "
                "position. The ground-truth token is always kept in the subset. Disabled when None."
            )
        },
    )
    topk_ce_p: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "If set in (0, 1), restrict the cross-entropy softmax normalizer to the minimal "
                "nucleus (top-p) set per position. Applied after top-k when both are set. "
                "Disabled when None."
            )
        },
    )
    mask_topk_ce_k: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Inverse of `topk_ce_k`: *exclude* the top-k logits from the cross-entropy "
                "softmax normalizer per position. The ground-truth logit is always kept. "
                "Intuition: avoid pushing mass off the model's confident predictions, "
                "preserving learned linguistic structure. Mutually exclusive with "
                "`topk_ce_k`/`topk_ce_p`. Disabled when None."
            )
        },
    )
    mask_topk_ce_p: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Inverse of `topk_ce_p`: *exclude* the minimal top-p nucleus from the "
                "cross-entropy softmax normalizer per position. Mutually exclusive with "
                "`topk_ce_k`/`topk_ce_p`. Disabled when None."
            )
        },
    )
    margin_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use a pairwise-softmax margin loss instead of CE: binary CE "
                "between the label logit and its best competitor's logit "
                "(`softplus(-(logit[label] - max_other_logit))`). Smooth version of hinge — "
                "keeps pushing the label/competitor gap wider, with exponentially fading "
                "gradient as the gap grows. Mutually exclusive with topk_ce_* options."
            )
        },
    )
    gem_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use the GEM loss (Li et al. ICLR 2025, 'Preserving Diversity in "
                "Supervised Fine-tuning of Large Language Models') instead of CE. "
                "Pulls the label up in log-space, weighted by a sharpened target "
                "distribution `q = softmax(logits / gem_beta)` so already-plausible "
                "non-label tokens are not aggressively squashed. Mutually exclusive with "
                "other alternative losses."
            )
        },
    )
    gem_beta: float = field(
        default=0.7,
        metadata={
            "help": (
                "GEM temperature in (0, 1]. Closer to 1 makes GEM behave more like CE; "
                "closer to 0 preserves more diversity."
            )
        },
    )
    gem_h: str = field(
        default="linear",
        metadata={
            "help": (
                "GEM `h` function: 'linear' (default, paper's default) or 'logsigmoid' "
                "(adaptive re-weighting by `sigmoid(0.01 * (logit - label_logit))`). "
                "The difference is usually negligible."
            )
        },
    )
    random_k_ce_k: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "If set, restrict the cross-entropy softmax normalizer to `k` uniformly "
                "random vocab indices per position (sampled softmax / negative sampling). "
                "Unbiased in expectation over samples; removes the self-reinforcing bias "
                "of `topk_ce_k`. The label's logit is always kept. Mutually exclusive with "
                "other alternative losses. Disabled when None."
            )
        },
    )
    focal_loss: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, use focal loss (Lin et al. 2017) instead of CE: "
                "`-(1 - p_t)^gamma * log(p_t)` where p_t is the model's probability on "
                "the label token. Down-weights easy (confident) positions so training "
                "focuses on hard ones. Mutually exclusive with other alternative losses."
            )
        },
    )
    focal_gamma: float = field(
        default=2.0,
        metadata={
            "help": (
                "Focal-loss focusing parameter. `gamma = 0` reduces to plain CE; higher "
                "gamma increases the down-weighting of easy positions. Paper default: 2.0."
            )
        },
    )


@dataclass
class DPOConfig(trl.DPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})


@dataclass
class ORPOConfig(trl.ORPOConfig):
    """
    args for callbacks, benchmarks etc
    """

    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
