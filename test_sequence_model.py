#!/usr/bin/env python3
"""
Test script for sequence-based model implementation
"""

import torch
import torch.nn.functional as F
from model import StudentModel, get_dinov2_model, FeatureAdapter
from dataset import WaveDataset, get_wave_dataloaders
from pathlib import Path
import tempfile
import os
from PIL import Image
import numpy as np

def create_dummy_dataset(temp_dir, num_sequences=2, sequence_length=60):
    """Create a dummy dataset for testing"""
    classes = ['Plunging', 'Spilling', 'Surging']
    
    for class_name in classes:
        class_dir = Path(temp_dir) / 'train' / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for seq_idx in range(num_sequences):
            seq_dir = class_dir / f'sequence_{seq_idx}'
            seq_dir.mkdir(exist_ok=True)
            
            for img_idx in range(sequence_length):
                # Create a dummy image (random noise)
                img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                img.save(seq_dir / f'frame_{img_idx:03d}.jpg')
    
    # Create validation set (smaller)
    for class_name in classes:
        class_dir = Path(temp_dir) / 'val' / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        seq_dir = class_dir / 'sequence_0'
        seq_dir.mkdir(exist_ok=True)
        
        for img_idx in range(sequence_length):
            img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(seq_dir / f'frame_{img_idx:03d}.jpg')

def test_sequence_dataset():
    """Test sequence dataset loading"""
    print("Testing sequence dataset...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy dataset
        create_dummy_dataset(temp_dir, num_sequences=2, sequence_length=60)
        
        # Test dataset loading
        train_dir = Path(temp_dir) / 'train'
        val_dir = Path(temp_dir) / 'val'
        
        try:
            train_loader, val_loader, class_to_idx = get_wave_dataloaders(
                train_dir, val_dir, batch_size=2, sequence_length=60
            )
            
            print(f"✓ Dataset loaded successfully")
            print(f"  Classes: {class_to_idx}")
            print(f"  Training samples: {len(train_loader.dataset)}")
            print(f"  Validation samples: {len(val_loader.dataset)}")
            
            # Test batch loading
            for sequences, labels in train_loader:
                print(f"  Batch shape: sequences {sequences.shape}, labels {labels.shape}")
                expected_shape = (2, 60, 3, 224, 224)  # (batch, seq_len, channels, height, width)
                assert sequences.shape == expected_shape, f"Expected {expected_shape}, got {sequences.shape}"
                break
                
            print("✓ Sequence dataset test passed!")
            return True
            
        except Exception as e:
            print(f"✗ Dataset test failed: {e}")
            return False

def test_sequence_model():
    """Test sequence model forward pass"""
    print("\nTesting sequence model...")
    
    try:
        # Create model
        model = StudentModel(num_classes=3, sequence_length=60)
        model.eval()
        
        # Create dummy input (batch_size=2, seq_len=60, channels=3, height=224, width=224)
        dummy_input = torch.randn(2, 60, 3, 224, 224)
        
        # Forward pass
        with torch.no_grad():
            outputs, features = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {outputs.shape}")
        print(f"  Features shape: {features.shape}")
        
        # Check shapes
        assert outputs.shape == (2, 3), f"Expected output shape (2, 3), got {outputs.shape}"
        # Features should be (batch_size, num_patches, feature_dim)
        assert len(features.shape) == 3, f"Expected 3D features, got shape {features.shape}"
        
        print("✓ Sequence model test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test full integration with dataset and model"""
    print("\nTesting integration...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Create dummy dataset
            create_dummy_dataset(temp_dir, num_sequences=1, sequence_length=10)  # Smaller for faster testing
            
            # Create dataloader
            train_dir = Path(temp_dir) / 'train'
            val_dir = Path(temp_dir) / 'val'
            
            train_loader, val_loader, class_to_idx = get_wave_dataloaders(
                train_dir, val_dir, batch_size=1, sequence_length=10
            )
            
            # Create model
            model = StudentModel(num_classes=len(class_to_idx), sequence_length=10)
            model.eval()
            
            # Test with real batch
            for sequences, labels in train_loader:
                with torch.no_grad():
                    outputs, features = model(sequences)
                    
                print(f"✓ Integration test successful")
                print(f"  Input shape: {sequences.shape}")
                print(f"  Output shape: {outputs.shape}")
                print(f"  Labels shape: {labels.shape}")
                break
                
            print("✓ Integration test passed!")
            return True
            
        except Exception as e:
            print(f"✗ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Run all tests"""
    print("Running sequence model tests...\n")
    
    tests = [
        test_sequence_dataset,
        test_sequence_model,
        test_integration
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n{'='*50}")
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! The sequence implementation is working correctly.")
    else:
        print("❌ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()