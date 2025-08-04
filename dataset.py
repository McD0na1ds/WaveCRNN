import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path

class WaveDataset(Dataset):
    """Dataset class for wave classification"""
    
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            class_to_idx (dict, optional): Mapping from class names to indices.
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = ['Plunging', 'Spilling', 'Surging']
        
        # Create class to index mapping if not provided
        if class_to_idx is None:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            
        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                # Look for images in subdirectories
                for subdir in class_dir.iterdir():
                    if subdir.is_dir():
                        # Look for jpg files in the subdirectory
                        for img_path in subdir.glob('*.jpg'):
                            self.samples.append((str(img_path), self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_wave_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4):
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
    train_dataset = WaveDataset(train_dir, transform=train_transform)
    val_dataset = WaveDataset(val_dir, transform=val_transform, 
                             class_to_idx=train_dataset.class_to_idx)
    
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