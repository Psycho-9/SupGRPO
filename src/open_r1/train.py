"""Qwen3-VL text-spotting training entry point for SupGRPO."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image
from torch.utils.data import Dataset
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

from .trainer import GRPOConfig, VLMGRPOTrainer
from .vlm_modules import QwenVLModule


@dataclass
class SupGRPOScriptArguments(ScriptArguments):
    reward_funcs: list[str] = field(
        default_factory=lambda: ["precision", "recall", "content", "format"],
        metadata={"help": "Any of: precision recall content format"},
    )
    image_root: str = field(
        default="data/images",
        metadata={"help": "Root joined with each sample's relative image path"},
    )
    max_pixels: Optional[int] = 12_845_056
    min_pixels: Optional[int] = 3_136


@dataclass
class SupGRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = True


class TextSpottingDataset(Dataset):
    def __init__(self, config_path: str, image_root: str):
        config_file = Path(config_path).resolve()
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        self.image_root = Path(image_root).resolve()
        self.samples = []
        for source in config.get("datasets", []):
            json_path = Path(source["json_path"])
            if not json_path.is_absolute():
                json_path = config_file.parent / json_path
            records = json.loads(json_path.read_text(encoding="utf-8"))
            records = self._sample(records, source.get("sampling_strategy", "all"))
            print(f"Loaded {len(records)} samples from {json_path}")
            self.samples.extend(records)
        if not self.samples:
            raise ValueError(f"No training samples were loaded from {config_file}")

    @staticmethod
    def _sample(records, strategy):
        if strategy == "all":
            return records
        name, value = strategy.split(":", 1)
        count = math.ceil(float(value[:-1]) * len(records) / 100) if value.endswith("%") else int(value)
        if name == "first":
            return records[:count]
        if name == "end":
            return records[-count:]
        raise ValueError(f"Unsupported deterministic sampling strategy: {strategy}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image_path = self.image_root / sample["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Training image not found: {image_path}")
        solution = sample.get("normalized_solution")
        if solution is None:
            raise KeyError(f"Sample {index} has no normalized_solution")
        question = QwenVLModule.get_question_template().format(question=sample["problem"])
        return {
            "image": Image.open(image_path).convert("RGB"),
            "problem": sample["problem"],
            "solution": solution,
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                }
            ],
        }


def main(script_args, training_args, model_args):
    module = QwenVLModule()
    registry = {
        "precision": module.precision_reward,
        "recall": module.recall_reward,
        "content": module.content_reward,
        "format": module.format_reward_spotting,
    }
    unknown = set(script_args.reward_funcs) - registry.keys()
    if unknown:
        raise ValueError(f"Unknown reward functions: {sorted(unknown)}")
    dataset = TextSpottingDataset(script_args.dataset_name, script_args.image_root)
    trainer = VLMGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=[registry[name] for name in script_args.reward_funcs],
        args=training_args,
        vlm_module=module,
        train_dataset=dataset,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        torch_dtype=model_args.torch_dtype,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((SupGRPOScriptArguments, GRPOConfig, SupGRPOModelConfig))
    main(*parser.parse_args_and_config())
