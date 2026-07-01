# SupGRPO: Enhancing GRPO with Matching-based Online SFT for Text Spotting

This is the official PyTorch implementation of the paper: 
**"SupGRPO: Enhancing GRPO with Matching-based Online SFT for Text Spotting"** (Accepted by **ECCV 2026**).

---

## 🌟 Overview
Text spotting requires both accurate text recognition and precise spatial localization. Current specialised spotters excel at predicting tight bounding boxes in natural scenes, but falter on complex or artistic text, whereas multimodal large language models (MLLMs) possess strong recognition capabilities yet remain weak at localisation. To equip the text spotter with general and powerful recognition capabilities and to maximize its localization ability, we explore two MLLM-based fine-tuning methods: Supervised Fine-Tuning (SFT) and reinforcement learning fine-tuning based on Group Relative Policy Optimisation (GRPO). An interesting finding is that SFT is less effective than GRPO at enhancing recognition, while GRPO is less effective than SFT at enhancing detection. To compensate for each other's shortcomings, we introduce a joint training strategy, SupGRPO, which simultaneously optimizes the model using both SFT and GRPO. SupGRPO employs the specially designed reward functions and develops a matching‑based online SFT applied solely to coordinate tokens. It both mitigates the reward sparsity problem of GRPO and avoids the instance order dependency problem of SFT. To evaluate particularly challenging cases, we curate ATS, a dataset for artistic text spotting. Experiments demonstrate that SupGRPO improves both text recognition and detection, and attains superior performance. We will release ATS and our code upon acceptance.

---

## 🚀 News
- **[2026/06]** SupGRPO is accepted by **ECCV 2026**! 🎉

---

## 🔧 Installation


