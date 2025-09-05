"""Training pipeline for CIFAR-10 image classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from pathlib import Path
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import CIFAR10DataLoader
from model import ModelFactory, CIFAR10CNN
from evaluator import ModelEvaluator

class CIFAR10Trainer:
    """Comprehensive trainer for CIFAR-10 classification with MLE-Star methodology."""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.data_loader = CIFAR10DataLoader(
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 2)
        )
        
        self.model = ModelFactory.create_model(
            model_type=config['model_type'],
            num_classes=10,
            dropout_rate=config.get('dropout_rate', 0.5)
        ).to(self.device)
        
        self.optimizer = ModelFactory.create_optimizer(
            self.model,
            optimizer_type=config['optimizer_type'],
            learning_rate=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        self.scheduler = ModelFactory.create_scheduler(
            self.optimizer,
            scheduler_type=config['scheduler_type'],
            T_max=config['epochs']
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.evaluator = ModelEvaluator()
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup tensorboard logging and model checkpoints."""
        self.log_dir = Path('models/experiments') / f"run_{int(time.time())}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def train_epoch(self, train_loader, epoch: int) -> Tuple[float, float]:
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                batch_acc = 100. * correct / total
                self.writer.add_scalar('Batch/Train_Loss', loss.item(), 
                                     epoch * len(train_loader) + batch_idx)
                self.writer.add_scalar('Batch/Train_Accuracy', batch_acc, 
                                     epoch * len(train_loader) + batch_idx)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader, epoch: int) -> Tuple[float, float]:
        """Validate the model for one epoch."""
        self.model.eval()
        total_loss = 0.0
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
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_acc = checkpoint['best_val_acc']
        self.training_history = checkpoint['training_history']
    
    def train(self) -> Dict[str, any]:
        """Execute full training pipeline - MLE-Star Stage 3: Action."""
        print("Starting training...")
        start_time = time.time()
        
        # Load data
        train_loader, val_loader, test_loader = self.data_loader.load_data()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['epochs']):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc = self.validate_epoch(val_loader, epoch)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)
            
            # Log metrics
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            
            # Save best model
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            self.save_checkpoint(epoch, is_best)
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f'Epoch {epoch+1}/{self.config["epochs"]} ({epoch_time:.2f}s) - '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, LR: {current_lr:.6f}')
            
            # Early stopping
            if self.config.get('early_stopping', False):
                if epoch > 20 and val_acc < max(self.training_history['val_acc'][-10:]):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        # Final evaluation on test set
        test_results = self.evaluate_final(test_loader)
        
        # Save training results
        results = {
            'training_time': total_time,
            'best_val_accuracy': self.best_val_acc,
            'final_test_results': test_results,
            'training_history': self.training_history,
            'model_info': self.model.get_model_info(),
            'config': self.config
        }
        
        with open(self.log_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def evaluate_final(self, test_loader) -> Dict[str, any]:
        """Final evaluation on test set."""
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / 'best.pth')
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Evaluate
        test_loss, test_acc = self.validate_epoch(test_loader, 0)
        
        # Detailed evaluation
        all_preds, all_targets = self.get_predictions(test_loader)
        
        # Classification report
        class_names = self.data_loader.classes
        report = classification_report(all_targets, all_preds, 
                                     target_names=class_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.log_dir / 'confusion_matrix.png')
        plt.close()
        
        return {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def get_predictions(self, data_loader) -> Tuple[List[int], List[int]]:
        """Get all predictions and targets from data loader."""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        return all_preds, all_targets
    
    def plot_training_history(self):
        """Plot training curves."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy curves
        ax2.plot(epochs, self.training_history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(epochs, self.training_history['learning_rate'], 'g-')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Validation accuracy with best marker
        ax4.plot(epochs, self.training_history['val_acc'], 'r-', label='Validation Accuracy')
        best_epoch = np.argmax(self.training_history['val_acc']) + 1
        best_acc = max(self.training_history['val_acc'])
        ax4.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax4.scatter([best_epoch], [best_acc], color='gold', s=100, zorder=5)
        ax4.set_title('Validation Accuracy with Best Model')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Accuracy (%)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

# MLE-Star Stage 3: Action Implementation
def execute_training_action(config: Dict[str, any]) -> Dict[str, any]:
    """Execute the training action with comprehensive monitoring."""
    
    # Initialize trainer
    trainer = CIFAR10Trainer(config)
    
    # Execute training
    print("=== MLE-Star Stage 3: Action - Training Execution ===")
    results = trainer.train()
    
    # Plot training curves
    trainer.plot_training_history()
    
    # Close tensorboard writer
    trainer.writer.close()
    
    return results

if __name__ == '__main__':
    # Training configuration
    config = {
        'model_type': 'cnn',
        'batch_size': 128,
        'epochs': 100,
        'learning_rate': 0.001,
        'optimizer_type': 'adam',
        'scheduler_type': 'cosine',
        'dropout_rate': 0.5,
        'weight_decay': 1e-4,
        'num_workers': 4,
        'early_stopping': True
    }
    
    # Execute training
    results = execute_training_action(config)
    
    print("\n=== Training Results ===")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
    print(f"Final Test Accuracy: {results['final_test_results']['test_accuracy']:.2f}%")
    print(f"Training Time: {results['training_time']:.2f}s")
