# SupGRPO

This is the official PyTorch implementation of the paper: 
**"SupGRPO: Enhancing GRPO with Matching-based Online SFT for Text Spotting"** (Accepted by **ECCV 2026**).

---

## 🌟 Overview
Recent advances in Multimodal Large Language Models (MLLMs) have demonstrated promising results in perception and reasoning. However, effectively optimizing these models for structured dense prediction tasks like Text Spotting remains challenging. In this work, we propose **SupGRPO**, which significantly enhances Group Relative Policy Optimization (GRPO) by integrating *Matching-based Online SFT*. This design effectively decouples perception and reasoning, mitigating training instability and achieving state-of-the-art performance.

---

## 🚀 News
- **[2026/07]** SupGRPO is accepted by **ECCV 2026**! 🎉
- **[2026/07]** Code and checkpoints are released.

---

## 🔧 Installation

```bash
# Clone the repository
git clone [https://github.com/yzLi/SupGRPO.git](https://github.com/yzLi/SupGRPO.git)
cd SupGRPO

# Create environment
conda create -n supgrpo python=3.10 -y
conda activate supgrpo

# Install dependencies
pip install -r requirements.txt
