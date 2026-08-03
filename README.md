# Pathology-Aware Cross-Modal Semantic Learning Prostate Cancer Grading

Official PyTorch implementation of our pathology-aware multimodal framework for weakly supervised prostate cancer grading from whole-slide images (WSIs). The proposed framework integrates multi-scale visual representations with large language model (LLM)-generated histopathology concepts through cross-attention and hierarchical semantic fusion for accurate Grade Group prediction.

---

# Overview

Whole-slide images contain gigapixel-resolution tissue with substantial morphological heterogeneity, making weakly supervised prostate cancer grading challenging.

This repository implements a pathology-aware multimodal framework consisting of:

- Multi-scale patch extraction
- Iterative Refinement Module (IRM)
- Frozen LLM-generated histopathology concepts
- Text-guided cross-attention
- Hierarchical multi-scale semantic fusion
- Semantic prototype learning
- Cosine similarity classification

The framework is designed for efficient weakly supervised learning without requiring pixel-level annotations.

---

# Repository Structure

```
ProstateSemanticMIL/

configs/
datasets/
preprocessing/
prompts/
models/
losses/
trainer/
utils/

train.py
test.py
infer.py

README.md
requirements.txt
LICENSE
```

---

# Installation

Create a new environment

```bash
conda create -n prostate python=3.7
conda activate prostate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

The repository expects precomputed multi-scale feature bags.

Example structure:

```
data/

metadata/
    tcga_prad_folds.csv

features/

    TCGA-XX-0001.pt
    TCGA-XX-0002.pt
    ...
```

Each feature file should contain

```python
{
    "5x": Tensor[N5,2048],
    "10x": Tensor[N10,2048],
    "20x": Tensor[N20,2048]
}
```

---

# Prompt Generation

Generate pathology descriptions using Gemini.

```bash
python -m prompts.generate_descriptions
```

Generate frozen text embeddings.

```bash
python -m prompts.build_prompt_embeddings
```

---

# Training

Train one fold

```bash
python train.py --config configs/default.yaml --fold 0
```

Train all folds

```bash
python run_five_folds.py
```

Resume training

```bash
python train.py \
--config configs/default.yaml \
--resume outputs/fold0/last.pt
```

---

# Evaluation

```bash
python test.py \
--config configs/default.yaml \
--checkpoint outputs/fold0/best.pt
```

---

# Inference

Single-slide prediction

```bash
python infer.py \
--config configs/default.yaml \
--checkpoint outputs/fold0/best.pt \
--features data/features/TCGA-XX-0001.pt
```

---

# Model Components

### Multi-scale Patch Extraction

Extracts tissue patches from

- 5×

- 10×

- 20×

magnifications.

---

### Iterative Refinement Module (IRM)

Progressively removes less informative patches and retains only Top-M patches for efficient global reasoning.

---

### Histopathology Semantic Concepts

Gemini 2.5 Pro generates pathology-aware descriptions for each Grade Group.

Descriptions are encoded using a frozen biomedical language model.

---

### Cross Attention

Visual tokens act as **queries**.

Text tokens act as **keys** and **values**.

This enables pathology-guided visual feature refinement.

---

### Hierarchical Semantic Fusion

Semantic-enhanced representations from all magnifications are aggregated using a transformer encoder with a learnable CLS token.

---

### Semantic Prototype Learning

Each Grade Group is represented by a semantic prototype.

Prediction is performed using cosine similarity.

---


# Acknowledgements

This work was conducted at the University of Connecticut.

---

# License

This repository is released under the MIT License.
