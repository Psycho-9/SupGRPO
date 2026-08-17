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

import os
import re
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union, Sized

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, selective_log_softmax
# from trl import GRPOTrainer

from accelerate.utils import is_peft_model, set_seed
import PIL.Image

import copy
from torch.utils.data import RandomSampler, Sampler
import warnings

if is_peft_available():
    from peft import PeftConfig, PeftModel, get_peft_model

if is_wandb_available():
    import wandb

from ..vlm_modules.vlm_module import VLMBaseModule
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        mini_repeat_count (`int`):
            Number of times to repeat each index per batch.
        batch_size (`int`, *optional*, defaults to `1`):
            Number of unique indices per batch.
        repeat_count (`int`, *optional*, defaults to `1`):
            Number of times to repeat the full sampling process.
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()
        indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count


class VLMGRPOTrainer(Trainer):
    """
    SupGRPO Trainer
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen3-VL-8B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        vlm_module: VLMBaseModule = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        freeze_vision_modules: Optional[bool] = False,
        attn_implementation: str = "flash_attention_2",
        torch_dtype: str = "bfloat16",
        **kwargs,
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        
        self.vlm_module = vlm_module

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        # FIXME
        # Remember to modify it in the invernvl
        model_init_kwargs["attn_implementation"] = attn_implementation
        if model_init_kwargs.get("torch_dtype") is None:
            model_init_kwargs["torch_dtype"] = torch_dtype
        
        assert isinstance(model, str), "model must be a string in the current implementation"
        model_id = model
        torch_dtype = model_init_kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
            pass  # torch_dtype is already a torch.dtype or "auto" or None
        elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
            torch_dtype = getattr(torch, torch_dtype)
        else:
            raise ValueError(
                "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        model_init_kwargs["use_cache"] = (
            False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
        )
        model_cls = self.vlm_module.get_model_class(model_id, model_init_kwargs)
        # Some model classes (e.g. Qwen3-VL) don't accept `use_cache` as a from_pretrained
        # kwarg; pass it through the config instead.
        _use_cache = model_init_kwargs.pop("use_cache", None)
        try:
            model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        except TypeError as e:
            if "use_cache" in str(e):
                model = model_cls.from_pretrained(model_id, **model_init_kwargs)
            else:
                raise
        if _use_cache is not None and hasattr(model, "config"):
            model.config.use_cache = _use_cache

        # LoRA
        self.vision_modules_keywords = self.vlm_module.get_vision_modules_keywords()
        if peft_config is not None:
            init_adapter_path = os.environ.get("SUPGRPO_INIT_ADAPTER", "").strip()
            if init_adapter_path:
                print(f"Loading initial LoRA adapter from {init_adapter_path}...")
                model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
            else:
                print("Applying LoRA...")
            def find_all_linear_names(model, multimodal_keywords):
                cls = torch.nn.Linear
                lora_module_names = set()
                for name, module in model.named_modules():
                    # LoRA is not applied to the vision modules
                    if any(mm_keyword in name for mm_keyword in multimodal_keywords):
                        continue
                    if isinstance(module, cls):
                        lora_module_names.add(name)
                for m in lora_module_names:  # needed for 16-bit
                    if "embed_tokens" in m:
                        lora_module_names.remove(m)
                return list(lora_module_names)
            if not init_adapter_path:
                target_modules = find_all_linear_names(model, self.vision_modules_keywords)
                peft_config.target_modules = target_modules
                model = get_peft_model(model, peft_config)

        # Freeze vision modules
        if freeze_vision_modules:
            print("Freezing vision modules...")
            for n, p in model.named_parameters():
                if any(keyword in n for keyword in self.vision_modules_keywords):
                    p.requires_grad = False
        # Compute the number of trainable parameters and print the parameter that is trainable
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in trainable_params)
        # for n, p in model.named_parameters():
        #     if p.requires_grad:
        #         print(n, p.shape)
        print(f"Total trainable parameters: {total_params}")

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        # Reference model
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = model_cls.from_pretrained(model_id, **model_init_kwargs)
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)

        # Processing class
        if processing_class is None:
            processing_cls = self.vlm_module.get_processing_class()
            processing_class = processing_cls.from_pretrained(model_id, trust_remote_code=model_init_kwargs.get("trust_remote_code", None))
            for component, processing_keyword in self.vlm_module.get_custom_processing_keywords():
                if processing_keyword in kwargs:
                    # If we cannot find component in processing_class, return the processing_class itself
                    processing_component = getattr(processing_class, component, processing_class)
                    setattr(processing_component, processing_keyword, kwargs[processing_keyword])
            if getattr(processing_class, "tokenizer",  None) is not None:
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                assert isinstance(processing_class, PreTrainedTokenizerBase), "processing_class must be an instance of PreTrainedTokenizerBase if it has no tokenizer attribute"
                pad_token_id = processing_class.pad_token_id

        self.vlm_module.post_model_init(model, processing_class)
        self.vlm_module.post_model_init(self.ref_model, processing_class)

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_prompt_length = None
        if args.max_prompt_length is not None:
            warnings.warn("Setting max_prompt_length is currently not supported, it has been set to None")

        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1,
            pad_token_id=pad_token_id,
            use_cache=True,   # force KV-cache during generation even when the model was
                              # loaded with use_cache=False for gradient checkpointing.
                              # Qwen3-VL's get_rope_index needs cached decoding, else it
                              # treats each step as prefill -> attention_mask/input_ids
                              # shape mismatch.
        )
        self.beta = args.beta
        self.epsilon_low = args.epsilon
        self.epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon

        # SupGRPO: weight lambda for matching-based online SFT on coordinate tokens (Eq. 9).
        # Paper default 1e-4; overridable via env var for ablation.
        self.sft_coord_lambda = float(os.environ.get("SFT_COORD_LAMBDA", getattr(args, "sft_coord_lambda", 1e-4)))
        self.forward_batch_size = int(os.environ.get("SUPGRPO_FORWARD_BATCH_SIZE", "4"))
        if self.forward_batch_size < 1:
            raise ValueError("SUPGRPO_FORWARD_BATCH_SIZE must be positive")
        print(f"[SupGRPO] sft_coord_lambda = {self.sft_coord_lambda}")
        print(f"[SupGRPO] forward_batch_size = {self.forward_batch_size}")

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        # Tracks the number of iterations (forward + backward passes), including those within a gradient accumulation cycle
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates
        self._buffered_inputs = [None] * args.gradient_accumulation_steps

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # This fork generates `num_generations` completions for each prompt directly.
        # The original TRL divisibility check only applies to the old path where a
        # global batch is pre-expanded with repeated prompts before generation.

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            # if self.is_deepspeed_enabled:
            if is_deepspeed_zero3_enabled():
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()
        # Enable gradient checkpointing for non-PEFT models.
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep=None,
        sft_labels=None,
        **custom_multimodal_inputs,
    ):
        """Compute token log-probabilities in completion micro-batches.

        `forward_batch_size` only slices the model forward. The returned values and
        loss normalization still cover the complete per-device GRPO batch.
        """
        batch_size = input_ids.size(0)
        per_token_logps = []
        sft_nll = torch.zeros((), device=input_ids.device, dtype=torch.float32)
        for start in range(0, batch_size, self.forward_batch_size):
            end = min(start + self.forward_batch_size, batch_size)
            model_kwargs = self._slice_multimodal_inputs(
                custom_multimodal_inputs, start, end, batch_size
            )
            if logits_to_keep is not None:
                model_kwargs["logits_to_keep"] = logits_to_keep
            logits = model(
                input_ids=input_ids[start:end],
                attention_mask=attention_mask[start:end],
                **model_kwargs,
            ).logits[:, :-1, :]
            if logits_to_keep is None:
                targets = input_ids[start:end, 1:]
            else:
                targets = input_ids[start:end, -logits.size(1):]
            per_token_logps.append(selective_log_softmax(logits, targets))

            if sft_labels is not None:
                labels = sft_labels[start:end]
                mask = labels.reshape(-1) != -100
                if mask.any():
                    vocab_size = logits.size(-1)
                    selected_logits = logits.reshape(-1, vocab_size)[mask].float()
                    selected_targets = labels.reshape(-1)[mask]
                    sft_nll = sft_nll + torch.nn.functional.cross_entropy(
                        selected_logits, selected_targets, reduction="sum"
                    )
        result = torch.cat(per_token_logps, dim=0)
        return (result, sft_nll) if sft_labels is not None else result


    def _prepare_inputs(self, inputs):
        # Simple pass-through, just like original
        return inputs

    def _get_key_from_inputs(self, x, key):
        ele = x.get(key, None)
        assert ele is not None, f"The key {key} is not found in the input"
        if isinstance(ele, list):
            return [e for e in ele]
        else:
            return [ele]

    def _repeat_multimodal_inputs(self, multimodal_inputs, repeats: int):
        """Repeat multimodal tensors so B prompt inputs align with B*G completions.

        Qwen-VL flattens image patches into `pixel_values` while `image_grid_thw`
        keeps one row per image. For grouped GRPO we need prompt0's image repeated
        G times, then prompt1's image repeated G times, matching the completion
        order used below.
        """
        if repeats == 1:
            return multimodal_inputs
        repeated = {}
        image_grid = multimodal_inputs.get("image_grid_thw")
        batch_size = int(image_grid.shape[0]) if torch.is_tensor(image_grid) else None
        patch_counts = None
        if torch.is_tensor(image_grid):
            patch_counts = [
                int(x.item()) for x in (image_grid[:, 0] * image_grid[:, 1] * image_grid[:, 2])
            ]

        for key, value in multimodal_inputs.items():
            if value is None or not torch.is_tensor(value):
                repeated[key] = value
                continue
            if key == "image_grid_thw":
                repeated[key] = value.repeat_interleave(repeats, dim=0)
                continue
            if key == "pixel_values" and patch_counts and sum(patch_counts) == value.shape[0]:
                chunks = torch.split(value, patch_counts, dim=0)
                repeated[key] = torch.cat(
                    [chunk for chunk in chunks for _ in range(repeats)], dim=0
                )
                continue
            if batch_size is not None and value.shape[0] == batch_size:
                repeated[key] = value.repeat_interleave(repeats, dim=0)
            else:
                repeated[key] = value
        return repeated

    @staticmethod
    def _slice_multimodal_inputs(multimodal_inputs, start, end, batch_size):
        """Slice Qwen-VL image patches consistently with a sequence sub-batch."""
        sliced = {}
        image_grid = multimodal_inputs.get("image_grid_thw")
        patch_counts = None
        if torch.is_tensor(image_grid) and image_grid.shape[0] == batch_size:
            patch_counts = [
                int(value.item()) for value in (image_grid[:, 0] * image_grid[:, 1] * image_grid[:, 2])
            ]
        for key, value in multimodal_inputs.items():
            if value is None or not torch.is_tensor(value):
                sliced[key] = value
            elif key == "pixel_values" and patch_counts and sum(patch_counts) == value.shape[0]:
                patch_start = sum(patch_counts[:start])
                patch_end = sum(patch_counts[:end])
                sliced[key] = value[patch_start:patch_end]
            elif value.shape[0] == batch_size:
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        return sliced

    def _generate_and_score_completions(self, inputs: dict[str, Union[torch.Tensor, Any]], model) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        expanded_prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        prompts_text = self.vlm_module.prepare_prompt(self.processing_class, inputs)
        # Handle both pre-loaded images and image paths
        images = []
        for x in inputs:
            if "image" in x:
                imgs = self._get_key_from_inputs(x, "image")
            elif "image_path" in x and x["image_path"] is not None:
                imgs = [PIL.Image.open(p) for p in self._get_key_from_inputs(x, "image_path")]
            else:
                imgs = []

            for img in imgs:
                width, height = img.size
                if width < 28 or height < 28:
                    scale = max(28 / width, 28 / height)
                    new_size = (round(width * scale), round(height * scale))
                    img = img.resize(new_size, PIL.Image.Resampling.LANCZOS)
                images.append(img)
                

        prompt_inputs = self.vlm_module.prepare_model_inputs(
            self.processing_class,
            prompts_text,
            images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]


        # max_prompt_length is not supported yet
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_inputs["input_ids"] = prompt_ids
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]
        #     prompt_inputs["attention_mask"] = prompt_mask

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # Force KV-cache ON for generation even if the model was loaded with
            # use_cache=False (gradient checkpointing). Qwen3-VL's get_rope_index
            # requires cached decoding, else each decode step is treated as prefill
            # (single-token input_ids indexed by full-length attention_mask ->
            # IndexError in get_rope_index). Two things must be undone during generate:
            #   1. gradient checkpointing (transformers forces use_cache=False whenever
            #      GC is active, which prevents the KV cache from ever being built);
            #   2. use_cache=False on EVERY nested config (top-level, text_config, and
            #      the inner language_model.config) -- the inner config is what the
            #      decoder forward actually reads, so flipping only the top config
            #      (the old fix) had no effect.
            _restore = []  # list of (obj, attr, prev_value) to restore in finally

            def _set_use_cache(obj, value):
                cfg = getattr(obj, "config", None)
                if cfg is not None and hasattr(cfg, "use_cache"):
                    _restore.append((cfg, "use_cache", cfg.use_cache))
                    cfg.use_cache = value
                # Qwen3-VL nests text_config inside the top config
                if cfg is not None and getattr(cfg, "text_config", None) is not None \
                        and hasattr(cfg.text_config, "use_cache"):
                    _restore.append((cfg.text_config, "use_cache", cfg.text_config.use_cache))
                    cfg.text_config.use_cache = value

            # Toggle use_cache on the wrapper + any inner sub-modules that carry a config
            base = unwrapped_model
            if is_peft_model(base):
                base = base.base_model.model if hasattr(base.base_model, "model") else base.base_model
            for obj in (unwrapped_model, base,
                        getattr(base, "model", None),
                        getattr(base, "language_model", None),
                        getattr(getattr(base, "model", None), "language_model", None)):
                if obj is not None:
                    _set_use_cache(obj, True)

            # Disable gradient checkpointing during generation (forces use_cache off)
            _gc_was_enabled = bool(getattr(base, "is_gradient_checkpointing", False))
            _was_training = unwrapped_model.training
            try:
                if _gc_was_enabled and hasattr(base, "gradient_checkpointing_disable"):
                    base.gradient_checkpointing_disable()
                unwrapped_model.eval()
                generation_config = copy.deepcopy(self.generation_config)
                generation_config.num_return_sequences = self.num_generations
                generate_returned_result = unwrapped_model.generate(
                    **{k: v for k, v in prompt_inputs.items() if k not in self.vlm_module.get_non_generate_params()},
                    generation_config=generation_config
                    )
            finally:
                if _was_training:
                    unwrapped_model.train()
                if _gc_was_enabled and hasattr(base, "gradient_checkpointing_enable"):
                    base.gradient_checkpointing_enable()
                for obj, attr, prev in reversed(_restore):
                    setattr(obj, attr, prev)
            prompt_length = prompt_ids.size(1)
            if not self.vlm_module.is_embeds_input():
                prompt_completion_ids = generate_returned_result
                prompt_ids = prompt_completion_ids[:, :prompt_length]
                completion_ids = prompt_completion_ids[:, prompt_length:]
            else:
                # In this case, the input of the LLM backbone is the embedding of the combination of the image and text prompt
                # So the returned result of the `generate` method only contains the completion ids
                completion_ids = generate_returned_result
                prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        # Get the multimodal inputs
        multimodal_keywords = self.vlm_module.get_custom_multimodal_keywords()
        multimodal_inputs = {k: prompt_inputs[k] if k in prompt_inputs else None for k in multimodal_keywords}
        multimodal_inputs = self._repeat_multimodal_inputs(multimodal_inputs, self.num_generations)
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep=completion_ids.size(1) + 1,
                    **multimodal_inputs,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep=completion_ids.size(1) + 1,
                    **multimodal_inputs,
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep=completion_ids.size(1) + 1,
                        **multimodal_inputs,
                    )

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        # No need to duplicate prompts as we're not generating multiple completions per prompt

        rewards_per_func = torch.zeros(len(expanded_prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(expanded_prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(expanded_prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                # print(f"reward_kwargs: {reward_kwargs}")
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum local rewards from all reward functions. Each rank has
        # [prompt0_gen0..G-1, prompt1_gen0..G-1, ...], so advantages can be
        # computed locally without depending on cross-rank batch divisibility.
        local_rewards_per_func = rewards_per_func
        rewards = local_rewards_per_func.sum(dim=1)
        
        # Compute grouped-wise rewards
        # Each group consists of num_generations completions for the same prompt
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        gathered_rewards_per_func = self.accelerator.gather_for_metrics(local_rewards_per_func)
        reward_per_func = gathered_rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "multimodal_inputs": multimodal_inputs,
            "solutions": reward_kwargs["solution"]
        }

    # ------------------------------------------------------------------
    # SupGRPO: matching-based online SFT on coordinate tokens (Eq. 8-9)
    # ------------------------------------------------------------------
    @staticmethod
    def _iou_xyxy(box1, box2):
        inter_x1 = max(box1[0], box2[0]); inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2]); inter_y2 = min(box1[3], box2[3])
        iw = max(0.0, inter_x2 - inter_x1); ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union = a1 + a2 - inter
        return float(inter) / union if union > 0 else 0.0

    @staticmethod
    def _norm_text(s):
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        return s.strip()

    def _build_sft_coord_labels(self, completion_ids, solutions):
        """Build per-token CE targets (B, C) for matching-based online SFT on coordinate
        tokens. A predicted box is matched to a GT box iff (normalized text identical AND
        IoU>0). At the token positions where the model emitted each matched box's coordinate
        digits, the target is set to the corresponding GT coordinate-digit token id (aligned
        per coordinate number); all other positions are -100 (ignored). Returns a LongTensor
        on the same device as completion_ids and the number of matched instances."""
        tok = getattr(self.processing_class, "tokenizer", self.processing_class)
        device = completion_ids.device
        B, C = completion_ids.shape
        labels = torch.full((B, C), -100, dtype=torch.long, device=device)
        # {"bbox_2d": [x1, y1, x2, y2], "text_content": "..."}
        inst_pats = [
            (
                re.compile(
                    r'\{[^{}]*\x22bbox_2d\x22\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\][^{}]*'
                    r'\x22text_content\x22\s*:\s*\x22((?:[^\x22\\]|\\.)*)\x22[^{}]*\}'
                ),
                (1, 2, 3, 4),
                5,
            ),
            (
                re.compile(
                    r'\{[^{}]*\x22text_content\x22\s*:\s*\x22((?:[^\x22\\]|\\.)*)\x22[^{}]*'
                    r'\x22bbox_2d\x22\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\][^{}]*\}'
                ),
                (2, 3, 4, 5),
                1,
            ),
        ]
        n_matched_total = 0
        ids_list = completion_ids.tolist()
        for b in range(B):
            sol = solutions[b] if b < len(solutions) else []
            if not sol:
                continue
            ids = ids_list[b]
            # Per-token strings -> exact char offsets aligned with completion_ids positions.
            pieces = tok.batch_decode([[i] for i in ids], skip_special_tokens=False)
            starts, ends, pos, text = [], [], 0, []
            for p in pieces:
                starts.append(pos); pos += len(p); ends.append(pos); text.append(p)
            text = "".join(text)

            gt_boxes = [ [float(v) for v in item['bbox_2d']] for item in sol ]
            gt_texts = [ self._norm_text(str(item['text_content'])) for item in sol ]
            gt_matched = [False] * len(sol)

            matches = []
            used_spans = set()
            for inst_pat, coord_groups, text_group in inst_pats:
                for m in inst_pat.finditer(text):
                    span = m.span()
                    if span in used_spans:
                        continue
                    used_spans.add(span)
                    matches.append((m.start(), m, coord_groups, text_group))
            matches.sort(key=lambda item: item[0])
            for _, m, coord_groups, text_group in matches:
                pred_box = [float(m.group(k)) for k in coord_groups]
                pred_text = self._norm_text(re.sub(r'\\(.)', r'\1', m.group(text_group)))
                # find a GT box: identical text AND IoU>0
                match_j = -1
                for j in range(len(sol)):
                    if not gt_matched[j] and gt_texts[j] == pred_text and self._iou_xyxy(pred_box, gt_boxes[j]) > 0:
                        match_j = j; break
                if match_j < 0:
                    continue
                gt_matched[match_j] = True
                n_matched_total += 1
                # For each of the 4 coordinate numbers, set targets at the pred number's token positions
                for ci in range(4):
                    cs, ce = m.span(coord_groups[ci])             # char span of predicted number ci
                    tok_positions = [k for k in range(len(ids)) if starts[k] < ce and ends[k] > cs]
                    gt_num_str = str(int(round(gt_boxes[match_j][ci])))
                    gt_num_ids = tok.encode(gt_num_str, add_special_tokens=False)
                    for jj in range(min(len(tok_positions), len(gt_num_ids))):
                        labels[b, tok_positions[jj]] = gt_num_ids[jj]
        return labels, n_matched_total

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        # Check if we need to generate new completions or use buffered ones
        if self.state.global_step % self.num_iterations == 0:
            inputs = self._generate_and_score_completions(inputs, model)
            self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
        else:
            inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
        self._step += 1

        # Get the prepared inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        multimodal_inputs = inputs["multimodal_inputs"]
        
        # Concatenate for full sequence
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Build matching labels before the forward so coordinate NLL can be reduced
        # inside each memory-bounded completion sub-batch.
        sft_labels, n_matched = self._build_sft_coord_labels(completion_ids, inputs["solutions"])
        per_token_logps, sft_nll = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            logits_to_keep=completion_ids.size(1) + 1,
            sft_labels=sft_labels,
            **multimodal_inputs,
        )

        # Get the advantages from inputs
        advantages = inputs["advantages"]

        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip its computation
        # and use per_token_logps.detach() instead
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()

        # Compute the policy ratio and clipped version
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Add KL penalty if beta > 0
        if self.beta > 0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            per_token_loss = per_token_loss + self.beta * per_token_kl

            # Log KL divergence
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # Compute final loss
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log clip ratio
        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())

        # ---------------- SupGRPO: matching-based online SFT on coordinate tokens ----------------
        # L_SupGRPO = -J_GRPO + lambda * L_SFT-coord   (Eq. 9, lambda default 1e-4)
        # Eq. 8 sums coordinate NLLs; Eq. 9 averages that sum over sampled sequences.
        sft_coord_loss = sft_nll / max(completion_ids.size(0), 1)

        self._metrics["sft_coord_loss"].append(float(sft_coord_loss.detach().item()))
        self._metrics["sft_matched_boxes"].append(float(n_matched))

        # Combine: GRPO policy loss + lambda * coordinate SFT loss
        combined_loss = loss + self.sft_coord_lambda * sft_coord_loss

        return combined_loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))

    def _get_train_sampler(self, dataset=None) -> Sampler:
        """Returns a sampler that ensures proper data sampling for GRPO training.
        (transformers>=4.50 passes the dataset positionally; accept and ignore it,
        we always sample from self.train_dataset.)"""
        dataset = dataset if dataset is not None else self.train_dataset
        if dataset is None or isinstance(dataset, IterableDataset):
            return None
        generator = torch.Generator()
        if self.args.seed is not None:
            generator.manual_seed(self.args.seed)
        return RandomSampler(dataset, generator=generator)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        """Returns a sampler for evaluation."""
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )
