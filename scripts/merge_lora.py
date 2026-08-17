import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description="Merge a SupGRPO LoRA adapter into Qwen3-VL-8B.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    AutoProcessor.from_pretrained(args.model_path).save_pretrained(output_dir)


if __name__ == "__main__":
    main()

