#!/usr/bin/env python3
"""
Quick test to verify the k-fold training pipeline works with minimal data
"""
import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_kfold import train_fold
import torch

def test_single_fold_training():
    """Test training on a single fold with minimal data"""
    device = torch.device('cpu')  # Use CPU for testing
    
    # Minimal hyperparameters for quick test
    hyperparams = {
        'batch_size': 1,
        'num_epochs': 2,  # Just 2 epochs for testing
        'learning_rate': 1e-4,
        'validate_every': 1,  # Validate every epoch
        'sequence_length': 5,  # Use only 5 frames instead of 60
        'patience': 2,
    }
    
    # Test paths
    train_dir = Path('./test_datasets_5fold/fold_0/train')
    val_dir = Path('./test_datasets_5fold/fold_0/val')
    
    print("Testing single fold training with minimal data...")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    print(f"Hyperparams: {hyperparams}")
    
    try:
        # Test the training function
        best_acc, train_losses, train_accs, val_losses, val_accs = train_fold(
            fold_num=0,
            train_dir=train_dir,
            val_dir=val_dir,
            device=device,
            hyperparams=hyperparams
        )
        
        print(f"\n✅ Test successful!")
        print(f"Best accuracy: {best_acc:.4f}")
        print(f"Training losses: {train_losses}")
        print(f"Training accuracies: {train_accs}")
        print(f"Validation accuracies: {val_accs}")
        
        # Check if files were created
        expected_files = [
            'best_student_model_fold_0.pth',
            'final_student_model_fold_0.pth',
            'fold_0_results/train_losses.npy',
            'fold_0_results/train_accs.npy',
            'fold_0_results/val_losses.npy',
            'fold_0_results/val_accs.npy'
        ]
        
        print("\nChecking created files:")
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✅ {file_path}")
            else:
                print(f"❌ {file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_fold_training()
    if success:
        print("\n🎉 Training pipeline test completed successfully!")
    else:
        print("\n💥 Training pipeline test failed!")
        sys.exit(1)