"""Multi-GPU Qwen3-VL inference for text spotting."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


QUESTION_TEMPLATE = (
    "{question} First output the thinking process in <think> </think> tags "
    "and then output the final answer in <answer> </answer> tags. "
    "Output the final answer in JSON format."
)


def default_question(dataset_name: str) -> str:
    if dataset_name.lower() == "ctw1500":
        return (
            "Spot all the text in the image with line-level, and output in JSON format. "
            "Provide the position of each extracted line while ensuring the content is accurate and complete."
        )
    return (
        "Spot all the text in the image with word-level, and output in JSON format. "
        "Provide the position of each extracted word while ensuring the content is accurate and complete."
    )


def build_messages(item, image):
    question = item.get("problem") or default_question(item.get("dataset_name", ""))
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QUESTION_TEMPLATE.format(question=question)},
            ],
        }
    ]


def worker(rank, items, args, progress_queue, lock):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": device},
    ).eval()
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).merge_and_unload()
    processor = AutoProcessor.from_pretrained(args.model_path, fix_mistral_regex=True)
    processor.image_processor.max_pixels = args.max_pixels
    processor.image_processor.min_pixels = args.min_pixels

    for item in items:
        image_path = Path(args.image_root) / item["image"]
        try:
            image = Image.open(image_path).convert("RGB")
            messages = build_messages(item, image)
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[prompt], images=image_inputs, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
            completion_ids = generated[:, inputs.input_ids.shape[1] :]
            output = processor.batch_decode(
                completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            record = {
                "image": item["image"],
                "dataset_name": item.get("dataset_name"),
                "output": output,
            }
            with lock:
                with open(args.output, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            raise RuntimeError(f"Inference failed for {image_path}") from exc
        finally:
            progress_queue.put(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--data-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-pixels", type=int, default=12_845_056)
    parser.add_argument("--min-pixels", type=int, default=3_136)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_gpus < 1:
        raise ValueError("At least one CUDA device is required")
    data = json.loads(Path(args.data_json).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    chunks = [data[index:: args.num_gpus] for index in range(args.num_gpus)]
    with mp.Manager() as manager:
        progress_queue = manager.Queue()
        lock = manager.Lock()
        process_args = [(rank, chunks[rank], args, progress_queue, lock) for rank in range(args.num_gpus)]
        with mp.Pool(args.num_gpus) as pool:
            result = pool.starmap_async(worker, process_args)
            with tqdm(total=len(data)) as progress:
                while not result.ready():
                    try:
                        progress.update(progress_queue.get(timeout=0.2))
                    except queue.Empty:
                        continue
                result.get()
                while not progress_queue.empty():
                    progress.update(progress_queue.get())


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

