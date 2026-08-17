"""Qwen-VL integration and the four SupGRPO text-spotting rewards."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from trl.data_utils import maybe_apply_chat_template

from .vlm_module import VLMBaseModule


_BBOX_PATTERN = re.compile(
    r'"?bbox_2d"?\s*:\s*\[\s*'
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)
_TEXT_PATTERN = re.compile(
    r'"?(?:text_content|text|word)"?\s*:\s*"((?:[^"\\]|\\.)*)"'
)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)


def _parse_instances(text: str) -> list[dict[str, Any]]:
    """Extract spotting instances while tolerating text around the JSON answer."""
    answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    payload = answer.group(1) if answer else text
    payload = re.sub(r"```(?:json)?|```", "", payload, flags=re.IGNORECASE).strip()

    candidates = [payload]
    left, right = payload.find("["), payload.rfind("]")
    if left >= 0 and right > left:
        candidates.append(payload[left : right + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

    recovered = []
    for block in re.findall(r"\{[^{}]*\}", payload, flags=re.DOTALL):
        box_match = _BBOX_PATTERN.search(block)
        text_match = _TEXT_PATTERN.search(block)
        if box_match and text_match:
            recovered.append(
                {
                    "bbox_2d": [float(box_match.group(i)) for i in range(1, 5)],
                    "text_content": re.sub(r"\\(.)", r"\1", text_match.group(1)),
                }
            )
    return recovered


def _normalize_text(text: str) -> str:
    return re.sub(r"[^\w\s]", "", str(text).lower()).strip()


def _box(item: dict[str, Any]) -> list[float] | None:
    value = item.get("bbox_2d")
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _iou(box1: list[float], box2: list[float]) -> float:
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2] - 1, box2[2] - 1)
    inter_y2 = min(box1[3] - 1, box2[3] - 1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        intersection = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        intersection = 0.0
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return float(intersection / union) if union > 0 else 0.0


def _detection_counts(predicted: list[list[float]], target: list[list[float]]) -> tuple[int, int, int]:
    pred_matched = [False] * len(predicted)
    target_matched = [False] * len(target)
    true_positive = 0
    for pred_index, pred_box in enumerate(predicted):
        for target_index, target_box in enumerate(target):
            if not target_matched[target_index] and _iou(pred_box, target_box) > 0.5:
                pred_matched[pred_index] = True
                target_matched[target_index] = True
                true_positive += 1
                break
    return true_positive, len(predicted) - sum(pred_matched), len(target) - sum(target_matched)


class QwenVLModule(VLMBaseModule):
    """Qwen3-VL adapter used by the SupGRPO trainer."""

    def get_vlm_key(self):
        return "qwen3_vl"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        del model_id, model_init_kwargs
        return Qwen3VLForConditionalGeneration

    def get_processing_class(self):
        return AutoProcessor

    def get_vision_modules_keywords(self):
        return ["visual"]

    def get_custom_multimodal_keywords(self):
        return ["pixel_values", "image_grid_thw"]

    def get_non_generate_params(self):
        return []

    def get_custom_processing_keywords(self):
        return [("image_processor", "max_pixels"), ("image_processor", "min_pixels")]

    def prepare_prompt(self, processing_class, inputs):
        return [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]

    def prepare_model_inputs(
        self,
        processing_class,
        prompts_text,
        images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ):
        return processing_class(
            text=prompts_text,
            images=images or None,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )

    @staticmethod
    def get_question_template() -> str:
        return (
            "{question} First output the thinking process in <think> </think> tags "
            "and then output the final answer in <answer> </answer> tags. "
            "Output the final answer in JSON format."
        )

    @staticmethod
    def format_reward_spotting(completions, solution, **kwargs):
        """Eq. 4: one iff the answer is a valid list of spotting objects."""
        del solution, kwargs
        rewards = []
        for completion in completions:
            text = _completion_text(completion)
            answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
            payload = answer.group(1) if answer else ""
            payload = re.sub(r"```(?:json)?|```", "", payload, flags=re.IGNORECASE).strip()
            try:
                parsed = json.loads(payload)
                valid = isinstance(parsed, list) and all(
                    isinstance(item, dict)
                    and _box(item) is not None
                    and isinstance(item.get("text_content"), str)
                    for item in parsed
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                valid = False
            rewards.append(float(valid))
        return rewards

    @staticmethod
    def content_reward(completions, solution, **kwargs):
        """Eq. 5: multiset word F1, preserving duplicate transcripts."""
        del kwargs
        rewards = []
        for completion, targets in zip(completions, solution):
            predicted_words = Counter(
                _normalize_text(item.get("text_content", "")) for item in _parse_instances(_completion_text(completion))
            )
            target_words = Counter(_normalize_text(item.get("text_content", "")) for item in targets)
            predicted_words.pop("", None)
            target_words.pop("", None)
            true_positive = sum((predicted_words & target_words).values())
            precision = true_positive / sum(predicted_words.values()) if predicted_words else 0.0
            recall = true_positive / sum(target_words.values()) if target_words else 0.0
            rewards.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
        return rewards

    @staticmethod
    def precision_reward(completions, solution, **kwargs):
        """Eq. 6: detection precision at IoU > 0.5."""
        del kwargs
        rewards = []
        for completion, targets in zip(completions, solution):
            predicted = [box for item in _parse_instances(_completion_text(completion)) if (box := _box(item))]
            target = [box for item in targets if (box := _box(item))]
            true_positive, false_positive, _ = _detection_counts(predicted, target)
            denominator = true_positive + false_positive
            rewards.append(true_positive / denominator if denominator else 0.0)
        return rewards

    @staticmethod
    def recall_reward(completions, solution, **kwargs):
        """Eq. 7: detection recall at IoU > 0.5."""
        del kwargs
        rewards = []
        for completion, targets in zip(completions, solution):
            predicted = [box for item in _parse_instances(_completion_text(completion)) if (box := _box(item))]
            target = [box for item in targets if (box := _box(item))]
            true_positive, _, false_negative = _detection_counts(predicted, target)
            denominator = true_positive + false_negative
            rewards.append(true_positive / denominator if denominator else 0.0)
        return rewards

