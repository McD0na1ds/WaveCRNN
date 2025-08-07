# Sequence-Based Wave Classification

This document explains the updated implementation that processes video sequences (60 images per folder) instead of individual images.

## Key Changes

### 1. Dataset Structure
The dataset now expects each folder to contain **60 images** representing a video sequence:

```
datasets/
├── train/
│   ├── Plunging/
│   │   ├── sequence_01/        # 60 images: frame_001.jpg, frame_002.jpg, ...
│   │   ├── sequence_02/        # 60 images: frame_001.jpg, frame_002.jpg, ...
│   │   └── ...
│   ├── Spilling/
│   │   ├── sequence_01/        # 60 images
│   │   └── ...
│   └── Surging/
│       ├── sequence_01/        # 60 images
│       └── ...
└── val/
    ├── Plunging/
    ├── Spilling/
    └── Surging/
```

### 2. Model Architecture

The updated architecture processes sequences as follows:

```
Input: (batch_size, 60, 3, 224, 224)
  ↓
ViT processes each frame individually
  ↓ 
Frame features: (batch_size, 60, num_patches, feature_dim)
  ↓
Average patch features per frame: (batch_size, 60, feature_dim)
  ↓
LSTM processes frame sequence
  ↓
Classification output: (batch_size, num_classes)
```

### 3. Knowledge Distillation

The distillation process now:
1. Processes all 60 frames through DINOv2 teacher model
2. Averages teacher features across the sequence
3. Matches with student features averaged across the sequence
4. Maintains the same loss components (task, feature, relational)

## Usage

### Training
```python
python train.py
```

The training script now:
- Uses reduced batch size (4 instead of 16) due to memory requirements
- Processes sequences of 60 images per sample
- Maintains the same distillation framework

### Custom Sequence Length
You can modify the sequence length by changing the `sequence_length` parameter:

```python
# In train.py
sequence_length = 60  # Change this value

# In dataset loading
train_loader, val_loader, class_to_idx = get_wave_dataloaders(
    train_dir, val_dir, batch_size=batch_size, sequence_length=sequence_length
)

# In model creation
student_model = StudentModel(num_classes=len(class_to_idx), sequence_length=sequence_length)
```

### Testing
Run the comprehensive tests:

```bash
# Test sequence dataset loading
python test_sequence_model.py

# Test full pipeline with teacher-student distillation
python test_full_pipeline.py

# Test updated model functionality
python test_model.py

# Test dataset loading (requires actual dataset)
python test_dataset.py
```

## Technical Details

### Memory Considerations
- **Batch size reduced** from 16 to 4 due to sequence processing
- Each sample now requires 60x more memory than single images
- Consider further reducing batch size if you encounter OOM errors

### Sequence Handling
- **Padding**: If a folder has fewer than 60 images, the last image is repeated
- **Ordering**: Images are sorted by filename to ensure consistent temporal ordering
- **Processing**: All frames in a sequence are processed through ViT simultaneously for efficiency

### Feature Matching
- Student features are averaged across the sequence for distillation
- Teacher features are also averaged across the sequence
- This maintains compatibility with the original distillation framework

## Compatibility

### Backward Compatibility
The original single-image functionality is **not preserved**. The model now expects sequence inputs only.

### Dataset Migration
To migrate from single-image to sequence-based dataset:
1. Group your 60 images per sample into subdirectories
2. Ensure consistent naming (e.g., frame_001.jpg, frame_002.jpg, ...)
3. Maintain the class structure (Plunging/Spilling/Surging)

## Performance

### Expected Changes
- **Training time**: Significantly longer due to sequence processing
- **Memory usage**: 60x higher per sample (mitigated by reduced batch size)
- **Model capacity**: Better temporal understanding through LSTM processing

### Optimization Tips
1. Use smaller sequence lengths for faster experimentation
2. Consider gradient accumulation if batch size needs to be further reduced
3. Use mixed precision training (already enabled)
4. Monitor GPU memory usage and adjust batch size accordingly

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Solution: Reduce batch size further (try batch_size=2 or 1)
   - Alternative: Reduce sequence_length for testing

2. **Dataset Loading Errors**
   - Check that each folder contains exactly 60 images (or adjust sequence_length)
   - Ensure images are named consistently and sorted correctly

3. **Teacher Model Loading**
   - The DINOv2 model requires internet access
   - In offline environments, a mock teacher will be used automatically

4. **Slow Training**
   - This is expected due to sequence processing
   - Consider using fewer sequences for initial testing
   - Use smaller sequence lengths during development

### Validation
Always run the test scripts before training to ensure your setup is correct:

```bash
python test_sequence_model.py  # Should show all tests passing
python test_full_pipeline.py   # Should show successful loss computation
```