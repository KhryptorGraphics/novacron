"""Data loading and preprocessing for CIFAR-10 image classification."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from typing import Tuple, Dict
import os

class CIFAR10DataLoader:
    """CIFAR-10 data loader with comprehensive preprocessing pipeline."""
    
    def __init__(self, data_dir: str = './data', batch_size: int = 32, num_workers: int = 2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        # Define data transformations for training and validation
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and split CIFAR-10 dataset into train/val/test loaders."""
        
        # Download and load training data
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.train_transform
        )
        
        # Split training data into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Update validation dataset transform
        val_dataset.dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, download=False, transform=self.val_transform
        )
        
        # Load test data
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.test_transform
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
        
        return train_loader, val_loader, test_loader
    
    def get_data_stats(self) -> Dict[str, int]:
        """Get dataset statistics."""
        train_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=False)
        test_dataset = torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=False)
        
        return {
            'train_size': int(0.8 * len(train_dataset)),
            'val_size': int(0.2 * len(train_dataset)),
            'test_size': len(test_dataset),
            'num_classes': 10,
            'input_shape': (3, 32, 32)
        }
    
    def visualize_samples(self, num_samples: int = 8) -> None:
        """Visualize sample images from the dataset."""
        import matplotlib.pyplot as plt
        
        # Load a batch of training data
        train_loader, _, _ = self.load_data()
        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        
        # Denormalize images for visualization
        def denormalize(tensor):
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            return tensor * std + mean
        
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(num_samples):
            row = i // 4
            col = i % 4
            
            img = denormalize(images[i]).permute(1, 2, 0)
            img = torch.clamp(img, 0, 1)
            
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'Class: {self.classes[labels[i]]}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('data/sample_images.png')
        plt.show()

# MLE-Star Stage 1: Situation Analysis
def analyze_data_situation() -> Dict[str, any]:
    """Analyze the data situation for CIFAR-10 classification task."""
    
    data_loader = CIFAR10DataLoader()
    stats = data_loader.get_data_stats()
    
    situation_analysis = {
        'problem_type': 'multi-class image classification',
        'dataset': 'CIFAR-10',
        'data_characteristics': stats,
        'challenges': [
            'Small image resolution (32x32)',
            'High inter-class similarity',
            'Limited training data per class',
            'Potential for overfitting'
        ],
        'recommended_approaches': [
            'Data augmentation for training robustness',
            'Regularization techniques (dropout, batch normalization)',
            'Transfer learning from pre-trained models',
            'Ensemble methods for improved accuracy'
        ]
    }
    
    return situation_analysis

if __name__ == '__main__':
    # Test data loading
    loader = CIFAR10DataLoader()
    train_loader, val_loader, test_loader = loader.load_data()
    
    print("Data loading successful!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Analyze situation
    analysis = analyze_data_situation()
    print("\nSituation Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")
