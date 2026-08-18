# SupGRPO: Enhancing GRPO with Matching-based Online SFT for Text Spotting

This is the official PyTorch implementation of **"SupGRPO: Enhancing GRPO
with Matching-based Online SFT for Text Spotting"**, accepted by **ECCV 2026**.

## Overview

Text spotting requires accurate recognition and spatial localization. SupGRPO
combines GRPO with matching-based online SFT applied only to coordinate tokens.
The implementation contains the four paper rewards (format, multiset word F1,
IoU precision, and IoU recall) and the coordinate-token SFT objective.

## News

- **2026/06:** SupGRPO was accepted by ECCV 2026 Poster.

## Installation

The reference environment uses Python 3.11 and CUDA 12.4:

```bash
git clone https://github.com/Psycho-9/SupGRPO.git
cd SupGRPO
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e '.[test,eval]'
```

## Preparation

Place the Qwen3-VL-8B-Instruct base model at:

```text
models/Qwen3-VL-8B-Instruct/
```

Prepare ATS, Total-Text, ICDAR 2015, CTW1500, and ReCTS following
[data/README.md](data/README.md). The default training manifest is
[configs/text_spotting.yaml](configs/text_spotting.yaml). All image paths in
the JSON files must be relative to `data/images/`.

Training coordinates must be scaled to the Qwen 0-1000 coordinate space. Each
sample must provide `image`, `problem`, and `normalized_solution`; the exact
schema is shown in the data guide.

## Training

Run the paper-disclosed Qwen3-VL-8B configuration on four GPUs:

```bash
bash scripts/train_qwen3_vl_8b.sh
```

## Evaluation

```bash
bash scripts/evaluate_qwen3_vl_8b.sh
```

The evaluation pipeline performs deterministic generation, converts normalized
coordinates back to pixels, and reports recognition F1, detection F1, and
end-to-end Hmean for ATS, Total-Text, ICDAR 2015, and CTW1500. Results are saved
under `evaluation/results/`.

## Tests

```bash
pytest -q
bash -n scripts/*.sh
```

The tests cover the four rewards, the paper hyperparameter contract, the
Qwen-VL multimodal slicing logic, and the repository release boundary.

## Repository Layout

```text
configs/                 Training data and DeepSpeed configurations
data/                    Dataset schema (datasets are not redistributed)
evaluation/              Inference and official metric pipeline
scripts/                 Train, evaluate, and LoRA merge entry points
src/open_r1/             SupGRPO trainer and Qwen3-VL rewards
tests/                   Unit and release-contract tests
```

## Acknowledgements

This implementation builds on Hugging Face Transformers, TRL, PEFT, DeepSpeed,
and the multimodal Open-R1 training codebase. Their licenses and contributions
are acknowledged in the source headers and project license.

## Citation

Please cite the ECCV 2026 paper if this repository is useful in your research:

```bibtex
```

The author field will be added to the BibTeX entry when the official conference
record is available.

## License

Released under the [Apache License 2.0](LICENSE).

