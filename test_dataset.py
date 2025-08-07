import sys
sys.path.append('/Users/wuxi/PycharmProjects/waveClass')

from dataset import get_wave_dataloaders
from pathlib import Path

def test_dataset_loading():
    """Test dataset loading functionality for sequences"""
    train_dir = Path('./datasets/train')
    val_dir = Path('./datasets/val')
    
    try:
        train_loader, val_loader, class_to_idx = get_wave_dataloaders(
            train_dir, val_dir, batch_size=4, sequence_length=60  # Updated for sequences
        )
        print(f"Success! Loaded dataset:")
        print(f"  Classes: {class_to_idx}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Validation samples: {len(val_loader.dataset)}")
        
        # Try to load a batch
        for sequences, labels in train_loader:  # Updated variable name
            print(f"  Batch shape: sequences {sequences.shape}, labels {labels.shape}")
            print(f"  Expected sequence shape: (batch_size, 60, 3, 224, 224)")
            break
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Note: Make sure your dataset has folders with 60 images each")

if __name__ == "__main__":
    test_dataset_loading()