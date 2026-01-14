#!/usr/bin/env python3
"""
PyTorch Model Implementation for ml-experiment
MLE-Star Framework - Deep Learning Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

class MLPClassifier(nn.Module):
    """
    Multi-Layer Perceptron Classifier with configurable architecture
    """
    
    def __init__(self, input_size, hidden_layers, output_size, dropout_rate=0.2, 
                 activation='relu', batch_norm=False):
        super(MLPClassifier, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        self.batch_norm = batch_norm
        
        layers = []
        prev_size = input_size
        
        # Hidden layers
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)
    
    def get_embeddings(self, x, layer_idx=-2):
        """Get embeddings from a specific layer"""
        for i, layer in enumerate(self.network):
            x = layer(x)
            if i == layer_idx:
                return x
        return x

class CNNClassifier(nn.Module):
    """
    Convolutional Neural Network for image classification
    """
    
    def __init__(self, input_channels, num_classes, input_size=32):
        super(CNNClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
    
    def forward(self, x):
        # Convolutional layers
        x = self.pool(F.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(F.relu(self.batch_norm2(self.conv2(x))))
        x = self.adaptive_pool(F.relu(self.batch_norm3(self.conv3(x))))
        
        # Flatten
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for sequence data
    """
    
    def __init__(self, input_size, hidden_size, num_layers, num_classes, 
                 dropout=0.2, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        # Apply dropout and final linear layer
        out = self.dropout(hidden)
        out = self.fc(out)
        
        return out

class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier
    """
    
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes,
                 max_seq_length=1000, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_length, d_model)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        seq_len = x.size(1)
        
        # Project input to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x += self.positional_encoding[:seq_len, :]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class ModelFactory:
    """Factory class for creating different model architectures"""
    
    @staticmethod
    def create_model(model_type, config):
        """
        Create model based on type and configuration
        
        Args:
            model_type: Type of model ('mlp', 'cnn', 'lstm', 'transformer')
            config: Model configuration dictionary
        
        Returns:
            PyTorch model instance
        """
        model_config = config.get('model', {})
        
        if model_type == 'mlp':
            return MLPClassifier(
                input_size=model_config.get('input_size'),
                hidden_layers=model_config.get('hidden_layers', [128, 64]),
                output_size=model_config.get('output_size'),
                dropout_rate=model_config.get('dropout_rate', 0.2),
                activation=model_config.get('activation', 'relu'),
                batch_norm=model_config.get('batch_norm', False)
            )
        
        elif model_type == 'cnn':
            return CNNClassifier(
                input_channels=model_config.get('input_channels', 3),
                num_classes=model_config.get('output_size'),
                input_size=model_config.get('input_size', 32)
            )
        
        elif model_type == 'lstm':
            return LSTMClassifier(
                input_size=model_config.get('input_size'),
                hidden_size=model_config.get('hidden_size', 128),
                num_layers=model_config.get('num_layers', 2),
                num_classes=model_config.get('output_size'),
                dropout=model_config.get('dropout_rate', 0.2),
                bidirectional=model_config.get('bidirectional', False)
            )
        
        elif model_type == 'transformer':
            return TransformerClassifier(
                input_size=model_config.get('input_size'),
                d_model=model_config.get('d_model', 512),
                nhead=model_config.get('nhead', 8),
                num_layers=model_config.get('num_layers', 6),
                num_classes=model_config.get('output_size'),
                max_seq_length=model_config.get('max_seq_length', 1000),
                dropout=model_config.get('dropout_rate', 0.1)
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

class ModelTrainer:
    """
    PyTorch model trainer with advanced features
    """
    
    def __init__(self, model, config, device=None):
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Training configuration
        training_config = config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 0.001)
        self.batch_size = training_config.get('batch_size', 32)
        self.epochs = training_config.get('epochs', 100)
        self.weight_decay = training_config.get('weight_decay', 1e-5)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def _setup_optimizer(self):
        """Setup optimizer based on configuration"""
        optimizer_name = self.config.get('training', {}).get('optimizer', 'adam')
        
        if optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        scheduler_config = self.config.get('training', {}).get('lr_scheduler', {})
        scheduler_type = scheduler_config.get('type', 'step')
        
        if scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs
            )
        else:
            return None
    
    def _setup_loss_function(self):
        """Setup loss function based on task"""
        loss_name = self.config.get('training', {}).get('loss_function', 'crossentropy')
        
        if loss_name == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader=None, save_path=None):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            save_path: Path to save best model
        
        Returns:
            Training history
        """
        best_val_acc = 0
        save_path = save_path or Path("./outputs/models/best_model.pth")
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        for epoch in range(self.epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate_epoch(val_loader)
            else:
                val_loss, val_acc = train_loss, train_acc
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record history
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"Best model saved with validation accuracy: {val_acc:.2f}%")
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f'Epoch {epoch+1}/{self.epochs}: '
                           f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                           f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                           f'LR: {current_lr:.6f}')
        
        return self.history
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.numpy())
        
        return np.array(predictions), np.array(true_labels)
    
    def save_model(self, path):
        """Save complete model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load complete model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', {})
        logger.info(f"Model loaded from {path}")
        return checkpoint

# Utility functions
def create_data_loaders(X_train, y_train, X_val=None, y_val=None, 
                       X_test=None, y_test=None, batch_size=32):
    """Create PyTorch data loaders"""
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if X_val is not None and y_val is not None:
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_loader = None
    if X_test is not None and y_test is not None:
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Example usage
    config = {
        'model': {
            'input_size': 10,
            'hidden_layers': [128, 64, 32],
            'output_size': 2,
            'dropout_rate': 0.2,
            'activation': 'relu'
        },
        'training': {
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'adam',
            'lr_scheduler': {'type': 'step', 'step_size': 30, 'gamma': 0.1}
        }
    }
    
    # Create model
    model = ModelFactory.create_model('mlp', config)
    print(f"Model created with {count_parameters(model)} parameters")
    
    # Create trainer
    trainer = ModelTrainer(model, config)
    print("Trainer initialized successfully")