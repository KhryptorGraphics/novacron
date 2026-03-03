"""Training pipeline for BERT-based sentiment analysis."""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import os
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import logging

from data_processor import SentimentDataLoader, TextPreprocessor
from bert_model import ModelFactory, SentimentBERT

class SentimentTrainer:
    """Comprehensive trainer for sentiment analysis with BERT."""
    
    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize data processor
        preprocessor_config = config.get('preprocessor', {})
        self.data_loader = SentimentDataLoader(
            model_name=config['model_name'],
            max_length=config['max_length'],
            batch_size=config['batch_size'],
            preprocessor_config=preprocessor_config
        )
        
        # Initialize model
        self.model = ModelFactory.create_model(
            model_type=config['model_type'],
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            dropout_rate=config.get('dropout_rate', 0.3)
        ).to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        # Setup logging directory
        self.setup_logging()
    
    def setup_logging(self):
        """Setup tensorboard logging and model checkpoints."""
        self.log_dir = Path('models/nlp_experiments') / f"run_{int(time.time())}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        self.checkpoint_dir = self.log_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def prepare_data(self, data_path: Optional[str] = None):
        """Prepare training, validation, and test data."""
        # Load data
        df, data_stats = self.data_loader.load_imdb_data(data_path)
        self.logger.info(f"Loaded data: {data_stats}")
        
        # Prepare data loaders
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.prepare_data(
            df, test_size=0.2, val_size=0.1
        )
        
        # Calculate training steps for scheduler
        self.total_steps = len(self.train_loader) * self.config['epochs']
        
        # Initialize optimizer and scheduler
        self.optimizer = ModelFactory.create_optimizer(
            self.model,
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        self.scheduler = ModelFactory.create_scheduler(
            self.optimizer,
            num_training_steps=self.total_steps,
            num_warmup_steps=self.config.get('warmup_steps', self.total_steps // 10)
        )
        
        self.logger.info(f"Training setup complete. Total steps: {self.total_steps}")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            logits = outputs['logits']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_predictions:.3f}',
                'LR': f'{current_lr:.2e}'
            })
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Batch/Train_Loss', loss.item(), step)
                self.writer.add_scalar('Batch/Learning_Rate', current_lr, step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float, Dict[str, float]]:
        """Validate model for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]} [Val]')
            
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # ROC AUC for binary classification
        if self.config['num_classes'] == 2:
            probabilities_pos = np.array(all_probabilities)[:, 1]
            roc_auc = roc_auc_score(all_labels, probabilities_pos)
        else:
            roc_auc = 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        return avg_loss, accuracy, metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_score': self.best_val_score,
            'training_history': self.training_history,
            'config': self.config,
            'metrics': metrics
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            self.logger.info(f"New best model saved with score: {self.best_val_score:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_score = checkpoint['best_val_score']
        self.training_history = checkpoint['training_history']
    
    def train(self, data_path: Optional[str] = None) -> Dict[str, any]:
        """Execute full training pipeline - MLE-Star Stage 3: Action."""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        # Prepare data
        self.prepare_data(data_path)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config['epochs']):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc, val_metrics = self.validate_epoch(epoch)
            
            # Update training history
            current_lr = self.scheduler.get_last_lr()[0]
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rate'].append(current_lr)
            
            # Log metrics to tensorboard
            self.writer.add_scalar('Epoch/Train_Loss', train_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Accuracy', train_acc, epoch)
            self.writer.add_scalar('Epoch/Val_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Val_Accuracy', val_acc, epoch)
            self.writer.add_scalar('Epoch/Val_F1', val_metrics['f1_score'], epoch)
            self.writer.add_scalar('Epoch/Val_ROC_AUC', val_metrics['roc_auc'], epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', current_lr, epoch)
            
            # Save best model
            is_best = val_acc > self.best_val_score
            if is_best:
                self.best_val_score = val_acc
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            self.logger.info(
                f'Epoch {epoch+1}/{self.config["epochs"]} ({epoch_time:.2f}s) - '
                f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, '
                f'Val F1: {val_metrics["f1_score"]:.4f}'
            )
            
            # Early stopping
            if self.config.get('early_stopping', False):
                if epoch > 2 and val_acc < max(self.training_history['val_acc'][-3:]):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        # Final evaluation on test set
        test_results = self.evaluate_final()
        
        # Close tensorboard writer
        self.writer.close()
        
        # Prepare results
        results = {
            'training_time': total_time,
            'best_val_accuracy': self.best_val_score,
            'final_test_results': test_results,
            'training_history': self.training_history,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {},
            'config': self.config
        }
        
        # Save results
        with open(self.log_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def evaluate_final(self) -> Dict[str, any]:
        """Final evaluation on test set."""
        self.logger.info("Running final evaluation on test set...")
        
        # Load best model
        best_checkpoint = torch.load(self.checkpoint_dir / 'best.pth')
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        # Evaluate on test set
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc='Final Evaluation'):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive metrics
        test_loss = total_loss / len(self.test_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # ROC AUC
        if self.config['num_classes'] == 2:
            probabilities_pos = np.array(all_probabilities)[:, 1]
            roc_auc = roc_auc_score(all_labels, probabilities_pos)
        else:
            roc_auc = 0.0
        
        # Class-wise metrics
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, zero_division=0
        )
        
        return {
            'test_loss': test_loss,
            'test_accuracy': accuracy,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1_score': f1,
            'test_roc_auc': roc_auc,
            'per_class_precision': precision_per_class.tolist(),
            'per_class_recall': recall_per_class.tolist(),
            'per_class_f1': f1_per_class.tolist(),
            'predictions': all_predictions,
            'true_labels': all_labels,
            'probabilities': all_probabilities
        }

# MLE-Star Stage 3: Action Implementation for NLP
def execute_nlp_training_action(config: Dict[str, any], 
                               data_path: Optional[str] = None) -> Dict[str, any]:
    """Execute NLP training action with comprehensive monitoring."""
    
    print("=== MLE-Star Stage 3: Action - NLP Training Execution ===")
    
    # Initialize trainer
    trainer = SentimentTrainer(config)
    
    # Execute training
    results = trainer.train(data_path)
    
    return results

if __name__ == '__main__':
    # Training configuration
    config = {
        'model_type': 'bert',
        'model_name': 'bert-base-uncased',
        'num_classes': 2,
        'max_length': 256,
        'batch_size': 16,
        'epochs': 5,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'dropout_rate': 0.3,
        'warmup_steps': None,  # Will be calculated as 10% of total steps
        'early_stopping': True,
        'preprocessor': {
            'lowercase': True,
            'remove_stopwords': False,  # Keep stopwords for BERT
            'lemmatize': False,  # BERT handles word variations
            'remove_punctuation': False  # Keep punctuation for context
        }
    }
    
    # Execute training
    results = execute_nlp_training_action(config)
    
    print("\n=== NLP Training Results ===")
    print(f"Best Validation Accuracy: {results['best_val_accuracy']:.4f}")
    print(f"Final Test Accuracy: {results['final_test_results']['test_accuracy']:.4f}")
    print(f"Final Test F1-Score: {results['final_test_results']['test_f1_score']:.4f}")
    print(f"Training Time: {results['training_time']:.2f}s")
