"""CNN model architecture for CIFAR-10 classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from typing import Dict, List, Optional
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block for improved gradient flow."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out

class CIFAR10CNN(nn.Module):
    """Advanced CNN model for CIFAR-10 classification with residual connections."""
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super(CIFAR10CNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global average pooling and classification
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'CIFAR10CNN',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'architecture': 'ResNet-inspired CNN',
            'input_shape': (3, 32, 32),
            'output_shape': (self.num_classes,),
            'dropout_rate': self.dropout_rate
        }

class ModelFactory:
    """Factory class for creating different model configurations."""
    
    @staticmethod
    def create_model(model_type: str = 'cnn', **kwargs) -> nn.Module:
        """Create model instance based on type."""
        
        if model_type == 'cnn':
            return CIFAR10CNN(**kwargs)
        elif model_type == 'simple_cnn':
            return SimpleCNN(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_optimizer(model: nn.Module, optimizer_type: str = 'adam', 
                        learning_rate: float = 0.001, **kwargs):
        """Create optimizer for model training."""
        
        if optimizer_type == 'adam':
            return Adam(model.parameters(), lr=learning_rate, **kwargs)
        elif optimizer_type == 'sgd':
            return SGD(model.parameters(), lr=learning_rate, momentum=0.9, **kwargs)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    @staticmethod
    def create_scheduler(optimizer, scheduler_type: str = 'step', **kwargs):
        """Create learning rate scheduler."""
        
        if scheduler_type == 'step':
            return StepLR(optimizer, step_size=30, gamma=0.1, **kwargs)
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(optimizer, T_max=100, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class SimpleCNN(nn.Module):
    """Simple CNN baseline model."""
    
    def __init__(self, num_classes: int = 10):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second convolutional block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third convolutional block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# MLE-Star Stage 2: Task Definition
def define_model_task() -> Dict[str, any]:
    """Define the specific ML task and model requirements."""
    
    task_definition = {
        'task_type': 'multi-class image classification',
        'input_specification': {
            'format': '32x32 RGB images',
            'channels': 3,
            'normalization': 'ImageNet statistics',
            'augmentation': 'rotation, flip, color jitter'
        },
        'output_specification': {
            'classes': 10,
            'class_names': ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
            'output_format': 'logits/probabilities'
        },
        'success_metrics': {
            'primary': 'accuracy',
            'secondary': ['precision', 'recall', 'f1-score'],
            'target_accuracy': 0.85
        },
        'model_constraints': {
            'max_parameters': '10M',
            'inference_time': '<50ms',
            'memory_usage': '<2GB'
        },
        'training_strategy': {
            'epochs': 100,
            'batch_size': 128,
            'learning_rate': 0.001,
            'optimization': 'Adam with cosine annealing'
        }
    }
    
    return task_definition

if __name__ == '__main__':
    # Test model creation
    model = ModelFactory.create_model('cnn', num_classes=10, dropout_rate=0.5)
    print("Model created successfully!")
    print(model.get_model_info())
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Define task
    task = define_model_task()
    print("\nTask Definition:")
    for key, value in task.items():
        print(f"{key}: {value}")
