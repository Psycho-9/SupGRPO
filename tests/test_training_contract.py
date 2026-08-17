import json
import re
from pathlib import Path

import torch
from torch import nn

from open_r1.trainer.grpo_trainer import VLMGRPOTrainer


ROOT = Path(__file__).resolve().parents[1]


def test_paper_hyperparameters_are_explicit():
    launcher = (ROOT / "scripts" / "train_qwen3_vl_8b.sh").read_text(encoding="utf-8")
    for fragment in (
        'NPROC="${NPROC:-4}"',
        "--per_device_train_batch_size 2",
        "--num_generations 8",
        "--max_completion_length 1024",
        "--num_train_epochs 1",
        "--learning_rate 1e-6",
        "--beta 0.04",
        'SFT_COORD_LAMBDA="${SFT_COORD_LAMBDA:-1e-4}"',
    ):
        assert fragment in launcher


def test_zero3_config_is_valid():
    config = json.loads((ROOT / "configs" / "zero3.json").read_text(encoding="utf-8"))
    assert config["zero_optimization"]["stage"] == 3


def test_qwen_patch_slicing_tracks_sequence_ranges():
    grids = torch.tensor([[1, 2, 2], [1, 1, 3], [1, 2, 1]])
    pixels = torch.arange(9 * 2).reshape(9, 2)
    sliced = VLMGRPOTrainer._slice_multimodal_inputs(
        {"image_grid_thw": grids, "pixel_values": pixels}, 1, 3, 3
    )
    assert torch.equal(sliced["image_grid_thw"], grids[1:3])
    assert torch.equal(sliced["pixel_values"], pixels[4:9])


def test_forward_slicing_matches_full_batch_logps_and_sft_nll():
    class Output:
        def __init__(self, logits):
            self.logits = logits

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(17, 8)
            self.head = nn.Linear(8, 17, bias=False)

        def forward(self, input_ids, attention_mask, logits_to_keep=None, **kwargs):
            del attention_mask, kwargs
            logits = self.head(self.embedding(input_ids))
            if logits_to_keep is not None:
                logits = logits[:, -logits_to_keep:]
            return Output(logits)

    torch.manual_seed(7)
    model = TinyModel()
    input_ids = torch.randint(0, 17, (6, 9))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.full((6, 4), -100)
    labels[0, 1] = 3
    labels[4, 2] = 9

    trainer = object.__new__(VLMGRPOTrainer)
    trainer.forward_batch_size = 2
    sliced_logps, sliced_nll = trainer._get_per_token_logps(
        model,
        input_ids,
        attention_mask,
        logits_to_keep=5,
        sft_labels=labels,
    )

    logits = model(input_ids, attention_mask, logits_to_keep=5).logits[:, :-1]
    targets = input_ids[:, -logits.size(1):]
    expected_logps = logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = labels.reshape(-1) != -100
    expected_nll = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1))[mask].float(),
        labels.reshape(-1)[mask],
        reduction="sum",
    )
    assert torch.allclose(sliced_logps, expected_logps, atol=1e-6)
    assert torch.allclose(sliced_nll, expected_nll, atol=1e-6)


def test_no_test_assisted_artifacts_or_absolute_paths():
    forbidden = ("benchmark_" + "memory", "metric-" + "target")
    absolute_machine_path = re.compile(r"(?<![.\w])/(?:home|data|root|mnt|modelarts)/")
    suffixes = {".py", ".sh", ".json", ".yaml", ".yml", ".md", ".toml"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix not in suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{token!r} found in {path.relative_to(ROOT)}"
        assert not absolute_machine_path.search(text), f"absolute machine path found in {path.relative_to(ROOT)}"


def test_peft_reference_does_not_allocate_a_second_model():
    source = (ROOT / "src" / "open_r1" / "trainer" / "grpo_trainer.py").read_text(encoding="utf-8")
    peft_branch = source.index("elif is_peft_model(model):")
    zero3_branch = source.index("elif is_deepspeed_zero3_enabled():", peft_branch)
    assert peft_branch < zero3_branch
