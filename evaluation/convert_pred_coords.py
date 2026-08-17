import argparse
import json
import os
import re
from copy import deepcopy
from PIL import Image


BBOX_KEYS = ("bbox_2d", "bbox", "position", "coordinates")


def extract_json_payload(output):
    if not output or not output.strip():
        return None
    answer_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)
    content = answer_match.group(1) if answer_match else output
    content = re.sub(r"```(?:json)?\s*(.*?)```", r"\1", content.strip(), flags=re.DOTALL | re.IGNORECASE)

    candidates = [content]
    obj_start, obj_end = content.find("{"), content.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(content[obj_start:obj_end + 1])
    arr_start, arr_end = content.find("["), content.rfind("]")
    if arr_start != -1 and arr_end > arr_start:
        candidates.append(content[arr_start:arr_end + 1])

    seen = set()
    for json_str in candidates:
        json_str = re.sub(r",\s*([}\]])", r"\1", json_str.strip())
        if not json_str or json_str in seen:
            continue
        seen.add(json_str)
        try:
            return json.loads(json_str)
        except Exception:
            continue
    recovered = []
    bbox_key_pattern = r"\\?\"(?:bbox_2d|bbox|position|coordinates)\\?\"\s*:\s*\[([^\]]+)\]"
    text_key_pattern = r"\\?\"(?:text_content|text|word|content)\\?\"\s*:\s*\\?\"((?:\\.|[^\"\\])*)\\?\""
    for obj_text in re.findall(r"\{[^{}]*\}", content, flags=re.DOTALL):
        if not any(key in obj_text for key in BBOX_KEYS):
            continue
        obj_clean = re.sub(r",\s*([}\]])", r"\1", obj_text.strip())
        try:
            recovered.append(json.loads(obj_clean))
            continue
        except Exception:
            pass
        text_match = re.search(text_key_pattern, obj_text, flags=re.DOTALL)
        box_match = re.search(bbox_key_pattern, obj_text, flags=re.DOTALL)
        if not (text_match and box_match):
            continue
        nums = [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", box_match.group(1))]
        if len(nums) >= 4:
            recovered.append({"text_content": text_match.group(1), "bbox_2d": nums[:4]})
    if recovered:
        return recovered
    return None


def _scales(width, height, mode):
    if mode == "normalized_to_raw":
        return width / 1000.0, height / 1000.0
    if mode == "raw_to_normalized":
        return 1000.0 / width, 1000.0 / height
    raise ValueError(f"Unsupported mode: {mode}")


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _convert_number_list(values, width, height, mode):
    sx, sy = _scales(width, height, mode)
    out = []
    for idx, value in enumerate(values):
        scale = sx if idx % 2 == 0 else sy
        out.append(round(float(value) * scale, 2))
    return out


def convert_bbox(box, width, height, mode):
    if isinstance(box, dict):
        sx, sy = _scales(width, height, mode)
        x_keys = {"x", "x1", "x2", "left", "right", "width"}
        y_keys = {"y", "y1", "y2", "top", "bottom", "height"}
        out = {}
        for key, value in box.items():
            if key in x_keys and _is_number(value):
                out[key] = round(float(value) * sx, 2)
            elif key in y_keys and _is_number(value):
                out[key] = round(float(value) * sy, 2)
            else:
                out[key] = value
        return out
    if not isinstance(box, list):
        return box
    if len(box) == 1 and isinstance(box[0], list):
        return [convert_bbox(box[0], width, height, mode)]
    if box and all(isinstance(item, list) for item in box):
        return [convert_bbox(item, width, height, mode) for item in box]
    if len(box) >= 2 and all(_is_number(v) for v in box):
        return _convert_number_list(box, width, height, mode)
    return box


def convert_obj(obj, width, height, mode):
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            if key in BBOX_KEYS:
                try:
                    out[key] = convert_bbox(value, width, height, mode)
                except Exception:
                    out[key] = value
            else:
                out[key] = convert_obj(value, width, height, mode)
        return out
    if isinstance(obj, list):
        return [convert_obj(item, width, height, mode) for item in obj]
    return obj


def image_size(image_root, image_path):
    full_path = os.path.join(image_root, image_path)
    with Image.open(full_path) as image:
        return image.size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, help="Input prediction JSONL")
    parser.add_argument("--out", required=True, help="Output prediction JSONL")
    parser.add_argument("--image-root", default=".", help="Root joined with each prediction image path")
    parser.add_argument("--mode", default="normalized_to_raw",
                        choices=["normalized_to_raw", "raw_to_normalized"])
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cache = {}
    converted = kept = failed = 0
    with open(args.pred, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            new_row = deepcopy(row)
            image_path = row.get("image")
            payload = extract_json_payload(row.get("output", ""))
            if image_path and payload is not None:
                try:
                    if image_path not in cache:
                        cache[image_path] = image_size(args.image_root, image_path)
                    width, height = cache[image_path]
                    payload = convert_obj(payload, width, height, args.mode)
                    new_row["output"] = "<answer> " + json.dumps(payload, ensure_ascii=False) + " </answer>"
                    converted += 1
                except Exception:
                    failed += 1
            else:
                kept += 1
            fout.write(json.dumps(new_row, ensure_ascii=False) + "\n")
    print(f"converted={converted} kept={kept} failed={failed} -> {args.out}")


if __name__ == "__main__":
    main()
