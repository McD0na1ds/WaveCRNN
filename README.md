# Wave Classification with Heterogeneous Knowledge Distillation

This project implements a novel approach to wave type classification using heterogeneous knowledge distillation. We distill knowledge from a large pre-trained DINOv2 model (teacher) into a lightweight ViT+LSTM model (student).

## Approach

Our solution addresses the challenge of knowledge distillation between heterogeneous models (different architectures and feature dimensions) through:

1. **Task Loss**: Standard cross-entropy loss for wave classification
2. **Feature Imitation Loss**: MSE loss between teacher and student features after mapping student features to teacher space using a linear adapter
3. **Relational Knowledge Loss**: MSE loss between normalized Gram matrices of teacher and student features, capturing relationships between features regardless of dimensionality

## Model Architecture

### Teacher Model
- Pre-trained DINOv2 (ViT-S/14 with 384-dim features)

### Student Model
- Lightweight ViT encoder (custom implementation with 14x14 patches to match DINOv2)
- LSTM module for temporal analysis of patch features
- Attention mechanism for feature focusing
- Classification head

## Dataset Structure

```
datasets/
├── train/
│   ├── Plunging/
│   ├── Spilling/
│   └── Surging/
└── val/
    ├── Plunging/
    ├── Spilling/
    └── Surging/
```

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. The DINOv2 model will be automatically downloaded from PyTorch Hub on first run.

## Training

To train the model:
```bash
python train.py
```

The training script will:
- Train for 50 epochs
- Validate every 5 epochs
- Save the best model based on validation accuracy
- Save the final model after training

## Key Features

- Mixed precision training for efficiency
- Cosine learning rate scheduling
- Comprehensive logging of loss components
- Best model checkpointing
- Feature dimension matching between teacher and student models

## Results

The student model achieves high accuracy while being significantly smaller and faster than the teacher model.