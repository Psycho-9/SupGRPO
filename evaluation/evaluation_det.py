import json
import re
import os
from collections import defaultdict
from tqdm import tqdm

TARGET_DATASETS = ['ic15', 'totaltext', 'ctw1500', 'ats']
GT_FILE_PATH = os.environ.get("GT_FILE_PATH", "gt.json")
PRED_FILE_PATH = os.environ.get("PRED_FILE_PATH", "result.json")

import string
_punct_tbl = str.maketrans({p: " " for p in string.punctuation})

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = s.translate(_punct_tbl)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_json_from_output(output_str: str):
    if not output_str or not output_str.strip():
        return None
    answer_match = re.search(r"<answer>(.*?)</answer>", output_str, re.DOTALL | re.IGNORECASE)
    content = answer_match.group(1) if answer_match else output_str
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
        if not any(key in obj_text for key in ("bbox_2d", "bbox", "position", "coordinates")):
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

def _normalize_bbox(b):
    x1, y1, x2, y2 = b
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [x1, y1, x2, y2]

def _valid_bbox(b):
    x1, y1, x2, y2 = b
    return (x2 - x1) > 0 and (y2 - y1) > 0

def to_bbox(pos):
    if not pos:
        return None
    try:
        if isinstance(pos, tuple):
            pos = list(pos)
        if isinstance(pos, list):
            while len(pos) == 1 and isinstance(pos[0], (list, tuple)):
                pos = list(pos[0])
            if len(pos) >= 2 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in pos):
                xs = [float(p[0]) for p in pos]
                ys = [float(p[1]) for p in pos]
                bbox = _normalize_bbox([min(xs), min(ys), max(xs), max(ys)])
                return bbox if _valid_bbox(bbox) else None
            if len(pos) >= 4 and all(isinstance(p, (int, float, str)) for p in pos):
                vals = [float(p) for p in pos]
                if len(vals) >= 6 and len(vals) % 2 == 0:
                    xs = vals[0::2]
                    ys = vals[1::2]
                    bbox = _normalize_bbox([min(xs), min(ys), max(xs), max(ys)])
                else:
                    bbox = _normalize_bbox(vals[:4])
                return bbox if _valid_bbox(bbox) else None
        if isinstance(pos, dict):
            if {"x", "y", "width", "height"} <= pos.keys():
                x1, y1, w, h = float(pos["x"]), float(pos["y"]), float(pos["width"]), float(pos["height"])
                bbox = _normalize_bbox([x1, y1, x1 + w, y1 + h])
                return bbox if _valid_bbox(bbox) else None
            if {"x1", "y1", "x2", "y2"} <= pos.keys():
                bbox = _normalize_bbox([float(pos["x1"]), float(pos["y1"]), float(pos["x2"]), float(pos["y2"])])
                return bbox if _valid_bbox(bbox) else None
            if {"top", "left", "bottom", "right"} <= pos.keys():
                bbox = _normalize_bbox([float(pos["left"]), float(pos["top"]), float(pos["right"]), float(pos["bottom"])])
                return bbox if _valid_bbox(bbox) else None
    except (ValueError, TypeError):
        return None
    return None

def parse_predictions(output_str):
    results = []
    data = extract_json_from_output(output_str)
    if not data:
        return results
    seen = set()
    def recursive_extract(obj):
        if isinstance(obj, dict):
            text_keys = ["text_content", "text", "content", "word"]
            pos_keys = ["bbox_2d", "bbox", "position", "coordinates"]
            text_content = next((str(obj[k]) for k in text_keys if k in obj), None)
            pos_info = next((obj[k] for k in pos_keys if k in obj), None)
            bbox = to_bbox(pos_info)
            if bbox:
                normalized_text = normalize_text(text_content) if text_content else ""
                key = (tuple(bbox), normalized_text)
                if key not in seen:
                    seen.add(key)
                    results.append({"text": normalized_text, "bbox": bbox})
                return
            for v in obj.values():
                recursive_extract(v)
        elif isinstance(obj, list):
            for item in obj:
                recursive_extract(item)
    recursive_extract(data)
    results = [r for r in results if _valid_bbox(r["bbox"])]
    return results

def parse_gt_boxes(gt_item):
    gt_boxes = []
    for sol in gt_item.get("solution", []):
        text_content = sol.get("text_content")
        bbox = to_bbox(sol.get("bbox_2d") or sol.get("position"))
        if bbox:
            normalized_text = normalize_text(text_content) if text_content else ""
            gt_boxes.append({"text": normalized_text, "bbox": bbox})
    return gt_boxes

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1g, y1g, x2g, y2g = box2
    xi1, yi1 = max(x1, x1g), max(y1, y1g)
    xi2, yi2 = min(x2, x2g), min(y2, y2g)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2g - x1g) * (y2g - y1g)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_matches(gt_boxes, pred_boxes):
    iou_thresh = 0.5
    gt_matched = [False] * len(gt_boxes)
    pred_matched = [False] * len(pred_boxes)
    tp = 0
    potential_matches = []
    for p_idx, pb in enumerate(pred_boxes):
        best_iou, best_g_idx = -1.0, -1
        for g_idx, gb in enumerate(gt_boxes):
            current_iou = iou(pb['bbox'], gb['bbox'])
            if current_iou > best_iou:
                best_iou, best_g_idx = current_iou, g_idx
        if best_iou >= iou_thresh:
            potential_matches.append((best_iou, p_idx, best_g_idx))
    potential_matches.sort(key=lambda x: x[0], reverse=True)
    for _, p_idx, g_idx in potential_matches:
        if not gt_matched[g_idx] and not pred_matched[p_idx]:
            tp += 1
            gt_matched[g_idx] = True
            pred_matched[p_idx] = True
    fp = sum(1 for m in pred_matched if not m)
    fn = len(gt_boxes) - tp
    return tp, fp, fn

def evaluate_targeted_metrics(gt_file, pred_file, target_datasets):
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data_all = json.load(f)
    pred_data_all = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    pred_data_all.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    gt_map = {}
    basename2ds = defaultdict(set)
    for gt in gt_data_all:
        ds_name = gt.get('dataset_name')
        if ds_name in target_datasets:
            img_base = os.path.basename(gt['image'])
            gt_map[(ds_name, img_base)] = parse_gt_boxes(gt)
            basename2ds[img_base].add(ds_name)
    pred_map = defaultdict(list)
    for pred in pred_data_all:
        img_field = pred.get('image')
        if not img_field:
            continue
        img_base = os.path.basename(img_field)
        ds_in_pred = pred.get('dataset_name')
        assigned = False
        if ds_in_pred and (ds_in_pred, img_base) in gt_map:
            pred_map[(ds_in_pred, img_base)] = parse_predictions(pred.get('output', ''))
            assigned = True
        else:
            dsets = list(basename2ds.get(img_base, []))
            if len(dsets) == 1:
                ds_only = dsets[0]
                pred_map[(ds_only, img_base)] = parse_predictions(pred.get('output', ''))
                assigned = True
        if not assigned:
            continue
    final_results = {}
    for ds in tqdm(target_datasets, desc="Evaluating"):
        keys = [k for k in gt_map.keys() if k[0] == ds]
        if not keys:
            continue
        total_tp, total_fp, total_fn = 0, 0, 0
        for key in keys:
            gt_boxes = gt_map[key]
            pred_boxes = pred_map.get(key, [])
            if not gt_boxes and not pred_boxes:
                continue
            tp, fp, fn = calculate_matches(gt_boxes, pred_boxes)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        final_results[ds] = {
            "detection": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            }
        }
    return final_results

if __name__ == "__main__":
    if not os.path.exists(GT_FILE_PATH) or not os.path.exists(PRED_FILE_PATH):
        print(f"Error: Ensure '{GT_FILE_PATH}' and '{PRED_FILE_PATH}' exist.")
    else:
        results = evaluate_targeted_metrics(
            gt_file=GT_FILE_PATH,
            pred_file=PRED_FILE_PATH,
            target_datasets=TARGET_DATASETS,
        )
        print("\n" + "=" * 40)
        print(" " * 10 + "Evaluation Results")
        print("=" * 40)
        print("IoU Threshold: 0.5")
        print("-" * 40)
        for ds, res in results.items():
            d = res["detection"]
            print(f"Dataset: {ds.upper()}")
            print(f"  Precision: {d['precision']:.4f}")
            print(f"  Recall:    {d['recall']:.4f}")
            print(f"  F1-Score:  {d['f1']:.4f}")
            print(f"  (TP: {d['tp']} | FP: {d['fp']} | FN: {d['fn']})")
            print("-" * 40)
