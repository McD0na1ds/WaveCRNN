# 5-Fold Cross Validation Training Guide

This document explains how to use the new 5-fold cross validation training system with DINOv2 Base teacher model.

## Key Improvements

### 1. Teacher Model Upgrade
- **Changed from:** DINOv2 ViT-Small/14 (384-dim features)
- **Changed to:** DINOv2 ViT-Base/14 (768-dim features)
- **Benefits:** Better feature representations from larger pre-trained model

### 2. Student Model Enhancements
- **Updated dimensions:** All components now handle 768-dim features
- **Improved architecture:** 12 attention heads (up from 6) for better capacity
- **Maintained efficiency:** Still lightweight compared to teacher

### 3. 5-Fold Cross Validation Training
- **Dataset structure:** `datasets_5fold/fold_0/1/2/3/4/`
- **Comprehensive evaluation:** Better estimate of model performance
- **Fold-specific results:** Individual metrics and models for each fold

### 4. Early Stopping & Metrics Tracking
- **Early stopping:** Configurable patience (default: 10 epochs)
- **Metrics saving:** Per-epoch accuracy and loss saved to `.npy` files
- **Best model saving:** Automatic best weight preservation per fold

## Dataset Structure

```
datasets_5fold/
├── fold_0/
│   ├── train/
│   │   ├── Plunging/
│   │   │   ├── sequence_01/     # 60 images: frame_001.jpg, frame_002.jpg, ...
│   │   │   ├── sequence_02/     # 60 images: frame_001.jpg, frame_002.jpg, ...
│   │   │   └── ...
│   │   ├── Spilling/
│   │   └── Surging/
│   └── val/
│       ├── Plunging/
│       ├── Spilling/
│       └── Surging/
├── fold_1/
│   ├── train/
│   └── val/
├── fold_2/
│   ├── train/
│   └── val/
├── fold_3/
│   ├── train/
│   └── val/
└── fold_4/
    ├── train/
    └── val/
```

## Usage

### 5-Fold Cross Validation Training
```bash
python train_kfold.py
```

### Single Fold Training (Original)
```bash
python train.py
```

## Configuration

### Hyperparameters (in `train_kfold.py`)
```python
hyperparams = {
    'batch_size': 4,        # Reduced due to sequence processing
    'num_epochs': 50,       # Maximum epochs per fold
    'learning_rate': 1e-4,  # AdamW learning rate
    'validate_every': 5,    # Validation frequency
    'sequence_length': 60,  # Images per sequence
    'patience': 10,         # Early stopping patience
}
```

### Early Stopping
- Monitors validation accuracy
- Stops training after `patience` epochs without improvement
- Automatically saves best model when validation improves

## Output Files

### Per-Fold Results
For each fold `i`, the following files are created:

**Model Checkpoints:**
- `best_student_model_fold_i.pth` - Best performing model
- `final_student_model_fold_i.pth` - Final model after training

**Metrics (NumPy arrays):**
- `fold_i_results/train_losses.npy` - Training losses per epoch
- `fold_i_results/train_accs.npy` - Training accuracies per epoch  
- `fold_i_results/val_losses.npy` - Validation losses per epoch
- `fold_i_results/val_accs.npy` - Validation accuracies per epoch

### Overall Results
- `kfold_results.npy` - Complete 5-fold CV summary with statistics

## Model Architecture Changes

### Student Model (LightweightViT)
```python
# Old configuration (384-dim)
LightweightViT(embed_dim=384, num_heads=6)

# New configuration (768-dim) 
LightweightViT(embed_dim=768, num_heads=12)
```

### Feature Adapter
```python
# Old: 384 → 384 (identity mapping)
FeatureAdapter(384, 384)

# New: 768 → 768 (identity mapping)
FeatureAdapter(768, 768)
```

## Testing

### Test Model Functionality
```bash
python test_model.py
```

### Test Training Pipeline
```bash
python test_training_pipeline.py
```

## Memory Considerations

- **Batch size reduced** from 16 to 4 due to larger model capacity
- **Sequence processing** requires ~60x memory per sample
- **GPU memory:** Monitor usage and reduce batch size if needed
- **Early stopping** helps prevent overlong training

## Results Analysis

### Loading Fold Results
```python
import numpy as np

# Load metrics for fold 0
train_losses = np.load('fold_0_results/train_losses.npy')
train_accs = np.load('fold_0_results/train_accs.npy')
val_accs = np.load('fold_0_results/val_accs.npy')

# Load overall k-fold results
results = np.load('kfold_results.npy', allow_pickle=True).item()
print(f"Mean CV Accuracy: {results['mean_acc']:.4f} ± {results['std_acc']:.4f}")
```

### Loading Best Model
```python
import torch
from model import StudentModel, FeatureAdapter

# Load best model for fold 0
checkpoint = torch.load('best_student_model_fold_0.pth')
model = StudentModel(num_classes=3, embed_dim=768)
model.load_state_dict(checkpoint['model_state_dict'])

adapter = FeatureAdapter(768, 768)
adapter.load_state_dict(checkpoint['adapter_state_dict'])
```

## Migration from Previous Version

### Code Changes Required
1. **Dataset structure:** Reorganize into 5 folds
2. **Model instantiation:** Add `embed_dim=768` parameter
3. **Feature adapter:** Update dimensions to 768
4. **Teacher model:** Will automatically use DINOv2 Base

### Backward Compatibility
- **Original `train.py`** still works with updated models
- **Test scripts** updated to handle new dimensions
- **Model files** remain compatible (just with different dimensions)

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size` in hyperparameters
   - Reduce `sequence_length` for testing

2. **Missing Datasets**
   - Script will skip missing folds gracefully
   - Check dataset structure matches expected format

3. **Teacher Model Loading**
   - Requires internet for DINOv2 download
   - Falls back to mock teacher if unavailable

4. **Early Stopping Too Aggressive**
   - Increase `patience` in hyperparameters
   - Reduce `validate_every` for more frequent checks

### Performance Tips
1. Use GPU for training (set `CUDA_VISIBLE_DEVICES`)
2. Monitor memory usage and adjust batch size
3. Use smaller sequence lengths during development
4. Save intermediate results frequently

## Expected Training Time

- **Per fold:** 2-4 hours (depending on dataset size and hardware)
- **Full 5-fold CV:** 10-20 hours total
- **Early stopping:** May significantly reduce training time

## Performance Improvements Expected

- **Better features:** DINOv2 Base provides richer representations
- **More robust evaluation:** 5-fold CV gives better performance estimates  
- **Efficient training:** Early stopping prevents overfitting
- **Better tracking:** Comprehensive metrics for analysis