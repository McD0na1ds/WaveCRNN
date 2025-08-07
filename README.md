# Wave Classification with Heterogeneous Knowledge Distillation

This project implements a novel approach to wave type classification using heterogeneous knowledge distillation with **video sequence processing**. We distill knowledge from a large pre-trained DINOv2 model (teacher) into a lightweight ViT+LSTM model (student) that processes sequences of 60 images.

## Approach

Our solution addresses the challenge of knowledge distillation between heterogeneous models (different architectures and feature dimensions) through:

1. **Task Loss**: Standard cross-entropy loss for wave classification
2. **Feature Imitation Loss**: MSE loss between teacher and student features after mapping student features to teacher space using a linear adapter
3. **Relational Knowledge Loss**: MSE loss between normalized Gram matrices of teacher and student features, capturing relationships between features regardless of dimensionality

## Model Architecture

### Teacher Model
- Pre-trained DINOv2 (ViT-S/14 with 384-dim features)
- Processes each frame individually, features averaged across sequence

### Student Model
- Lightweight ViT encoder (custom implementation with 14x14 patches to match DINOv2)
- **Sequence Processing**: Each of 60 images processed through ViT individually
- LSTM module for temporal analysis of frame-level features
- Attention mechanism for feature focusing
- Classification head combining ViT and LSTM outputs

## Dataset Structure

**NEW: Video Sequence Format**
```
datasets/
├── train/
│   ├── Plunging/
│   │   ├── sequence_01/     # 60 images: frame_001.jpg, frame_002.jpg, ...
│   │   ├── sequence_02/     # 60 images: frame_001.jpg, frame_002.jpg, ...
│   │   └── ...
│   ├── Spilling/
│   └── Surging/
└── val/
    ├── Plunging/
    ├── Spilling/
    └── Surging/
```

Each folder should contain exactly **60 images** representing a video sequence.

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
- Train for 50 epochs with sequence processing
- Validate every 5 epochs
- Save the best model based on validation accuracy
- Use reduced batch size (4) due to sequence memory requirements
- Save the final model after training

## Key Features

- **Video Sequence Processing**: 60 images per sample for temporal understanding
- **Temporal Feature Learning**: LSTM processes frame-level ViT features
- Mixed precision training for efficiency
- Cosine learning rate scheduling
- Comprehensive logging of loss components
- Best model checkpointing
- Feature dimension matching between teacher and student models

## Testing

Run comprehensive tests to verify the implementation:

```bash
# Test sequence processing
python test_sequence_model.py

# Test full pipeline with distillation
python test_full_pipeline.py

# Test model functionality
python test_model.py
```

## Sequence Implementation Details

For detailed information about the sequence processing implementation, see [SEQUENCE_IMPLEMENTATION.md](SEQUENCE_IMPLEMENTATION.md).

Key changes from the original implementation:
- **Input**: Changed from single images to sequences of 60 images
- **Processing**: Each frame processed through ViT, then sequence through LSTM
- **Memory**: Reduced batch size to handle increased memory requirements
- **Distillation**: Teacher features averaged across sequence for compatibility

## Results

The student model achieves high accuracy while being significantly smaller and faster than the teacher model, with improved temporal understanding through sequence processing.