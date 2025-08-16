# Wave Classification with Heterogeneous Knowledge Distillation

This project implements a novel approach to wave type classification using heterogeneous knowledge distillation with **video sequence processing**. We distill knowledge from a large pre-trained DINOv2 model (teacher) into a lightweight ViT+LSTM model (student) that processes sequences of 60 images.

## Approach

Our solution addresses the challenge of knowledge distillation between heterogeneous models (different architectures and feature dimensions) through:

1. **Task Loss**: Standard cross-entropy loss for wave classification
2. **Feature Imitation Loss**: MSE loss between teacher and student features after mapping student features to teacher space using a linear adapter
3. **Relational Knowledge Loss**: MSE loss between normalized Gram matrices of teacher and student features, capturing relationships between features regardless of dimensionality

## Model Architecture

### Teacher Model
- Pre-trained DINOv2 Base (ViT-B/14 with 768-dim features) 
- Processes each frame individually, features averaged across sequence

### Student Model
- Lightweight ViT encoder (custom implementation with 14x14 patches, 768-dim to match DINOv2 Base)
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

### 5-Fold Cross Validation Training (Recommended)
For robust model evaluation with early stopping and comprehensive metrics tracking:
```bash
python train_kfold.py
```

This will:
- Train on 5 different data splits for robust evaluation
- Use DINOv2 Base (768-dim) teacher model
- Apply early stopping with patience mechanism
- Save per-epoch metrics (accuracy/loss) to .npy files
- Save best model weights for each fold
- Generate comprehensive 5-fold CV results

### Single Training Run (Legacy)
To train with a single train/val split:
```bash
python train.py
```

For detailed information about the 5-fold training system, see [KFOLD_TRAINING_GUIDE.md](KFOLD_TRAINING_GUIDE.md).

## Key Features

- **Video Sequence Processing**: 60 images per sample for temporal understanding
- **Temporal Feature Learning**: LSTM processes frame-level ViT features
- **5-Fold Cross Validation**: Robust evaluation with `train_kfold.py`
- **Early Stopping**: Automatic training termination with patience mechanism
- **Comprehensive Metrics**: Per-epoch accuracy/loss saved to .npy files
- **DINOv2 Base Teacher**: Upgraded to 768-dim features for better knowledge distillation
- Mixed precision training for efficiency
- Cosine learning rate scheduling
- Comprehensive logging of loss components
- Best model checkpointing per fold
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