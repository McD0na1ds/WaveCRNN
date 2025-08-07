import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

class WaveDataset(Dataset):
    """Dataset class for wave classification with video sequences"""
    
    def __init__(self, root_dir, transform=None, class_to_idx=None, sequence_length=60):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from class names to indices.
            sequence_length (int): Number of images per sequence (default: 60).
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.sequence_length = sequence_length
        self.classes = ['Plunging', 'Spilling', 'Surging']
        
        # Create class to index mapping if not provided
        if class_to_idx is None:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            
        # Collect all sequence directories and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                # Each subdirectory represents one video sequence
                for subdir in class_dir.iterdir():
                    if subdir.is_dir():
                        # Count jpg files in the subdirectory
                        jpg_files = list(subdir.glob('*.jpg'))
                        if len(jpg_files) == self.sequence_length:
                            # Sort files to ensure consistent ordering
                            jpg_files.sort()
                            self.samples.append((jpg_files, self.class_to_idx[class_name]))
                        elif len(jpg_files) > 0:
                            # If not exactly sequence_length images, take first sequence_length or pad
                            jpg_files.sort()
                            self.samples.append((jpg_files[:self.sequence_length], self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_paths, label = self.samples[idx]
        
        # Load sequence of images
        sequence = []
        for img_path in img_paths:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            sequence.append(image)
        
        # Pad sequence if needed (in case some folders have fewer than sequence_length images)
        while len(sequence) < self.sequence_length:
            # Repeat the last image if not enough images
            if sequence:
                sequence.append(sequence[-1].clone() if hasattr(sequence[-1], 'clone') else sequence[-1])
            else:
                # Create a zero tensor if no images found (shouldn't happen but for safety)
                dummy_image = torch.zeros(3, 224, 224)
                sequence.append(dummy_image)
        
        # Stack into a tensor of shape (sequence_length, channels, height, width)
        sequence_tensor = torch.stack(sequence)
        
        return sequence_tensor, label

def get_wave_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4, sequence_length=60):
    """Create train and validation dataloaders for wave dataset"""
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WaveDataset(train_dir, transform=train_transform, sequence_length=sequence_length)
    val_dataset = WaveDataset(val_dir, transform=val_transform, 
                             class_to_idx=train_dataset.class_to_idx, sequence_length=sequence_length)
    
    # Check if datasets are empty
    if len(train_dataset) == 0:
        raise ValueError(f"No training samples found in {train_dir}")
    if len(val_dataset) == 0:
        raise ValueError(f"No validation samples found in {val_dir}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx