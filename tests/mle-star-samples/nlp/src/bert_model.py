"""BERT-based model for sentiment analysis."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoModel, AutoConfig, 
    get_linear_schedule_with_warmup,
    BertModel, BertConfig
)
import numpy as np
from typing import Dict, Optional, Tuple
import logging

class SentimentBERT(nn.Module):
    """BERT-based sentiment classification model."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', 
                 num_classes: int = 2, dropout_rate: float = 0.3,
                 freeze_bert: bool = False):
        super(SentimentBERT, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load BERT model and config
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize classifier weights
        self._init_weights(self.classifier)
    
    def _init_weights(self, module):
        """Initialize weights using normal distribution."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model."""
        
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classify
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state,
            'pooled_output': pooled_output
        }
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': f'SentimentBERT-{self.model_name}',
            'base_model': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'num_classes': self.num_classes,
            'dropout_rate': self.dropout_rate,
            'hidden_size': self.config.hidden_size,
            'max_position_embeddings': self.config.max_position_embeddings
        }

class SentimentRoBERTa(nn.Module):
    """RoBERTa-based sentiment classification model."""
    
    def __init__(self, model_name: str = 'roberta-base', 
                 num_classes: int = 2, dropout_rate: float = 0.3):
        super(SentimentRoBERTa, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load RoBERTa model
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through RoBERTa model."""
        
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use mean pooling of last hidden state
        last_hidden_state = outputs.last_hidden_state
        mean_pooling = torch.mean(last_hidden_state, dim=1)
        
        # Classification
        pooled_output = self.dropout(mean_pooling)
        logits = self.classifier(pooled_output)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': last_hidden_state,
            'pooled_output': pooled_output
        }

class AdvancedSentimentModel(nn.Module):
    """Advanced sentiment model with multiple pooling strategies."""
    
    def __init__(self, model_name: str = 'bert-base-uncased',
                 num_classes: int = 2, dropout_rate: float = 0.3,
                 pooling_strategy: str = 'cls'):
        super(AdvancedSentimentModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pooling_strategy = pooling_strategy
        
        # Load transformer model
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Pooling layers
        hidden_size = self.config.hidden_size
        
        if pooling_strategy == 'multi':
            # Multiple pooling strategies combined
            self.pooling_layers = nn.ModuleDict({
                'cls': nn.Identity(),
                'mean': nn.Identity(),
                'max': nn.Identity()
            })
            classifier_input_size = hidden_size * 3
        else:
            classifier_input_size = hidden_size
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(classifier_input_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # Initialize weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def _apply_pooling(self, hidden_states, attention_mask):
        """Apply specified pooling strategy."""
        
        if self.pooling_strategy == 'cls':
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pooling_strategy == 'mean':
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling_strategy == 'max':
            # Max pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            hidden_states = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.pooling_strategy == 'multi':
            # Combine multiple pooling strategies
            cls_pooling = hidden_states[:, 0, :]
            
            # Mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            mean_pooling = sum_embeddings / sum_mask
            
            # Max pooling
            masked_hidden = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            max_pooling = torch.max(masked_hidden, dim=1)[0]
            
            return torch.cat([cls_pooling, mean_pooling, max_pooling], dim=1)
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling_strategy}")
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with advanced pooling."""
        
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Apply pooling
        pooled_output = self._apply_pooling(outputs.last_hidden_state, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state,
            'pooled_output': pooled_output
        }

class ModelFactory:
    """Factory class for creating sentiment analysis models."""
    
    @staticmethod
    def create_model(model_type: str = 'bert', **kwargs) -> nn.Module:
        """Create model instance based on type."""
        
        if model_type == 'bert':
            return SentimentBERT(**kwargs)
        elif model_type == 'roberta':
            return SentimentRoBERTa(**kwargs)
        elif model_type == 'advanced':
            return AdvancedSentimentModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_optimizer(model: nn.Module, learning_rate: float = 2e-5,
                        weight_decay: float = 0.01, eps: float = 1e-8) -> AdamW:
        """Create AdamW optimizer with appropriate parameters for transformers."""
        
        # Different learning rates for different parts
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': weight_decay
            },
            {
                'params': [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=eps)
    
    @staticmethod
    def create_scheduler(optimizer, num_training_steps: int,
                        num_warmup_steps: Optional[int] = None):
        """Create learning rate scheduler with warmup."""
        
        if num_warmup_steps is None:
            num_warmup_steps = num_training_steps // 10  # 10% warmup
        
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

# MLE-Star Stage 2: Task Definition for NLP
def define_nlp_task() -> Dict[str, any]:
    """Define the specific NLP task and model requirements."""
    
    task_definition = {
        'task_type': 'binary sentiment classification',
        'input_specification': {
            'format': 'variable-length text sequences',
            'tokenizer': 'bert-base-uncased',
            'max_sequence_length': 512,
            'preprocessing': 'optional text cleaning and normalization'
        },
        'output_specification': {
            'classes': 2,
            'class_names': ['negative', 'positive'],
            'output_format': 'logits/probabilities'
        },
        'success_metrics': {
            'primary': 'accuracy',
            'secondary': ['precision', 'recall', 'f1-score', 'auc-roc'],
            'target_accuracy': 0.90
        },
        'model_constraints': {
            'max_parameters': '110M (BERT-base)',
            'inference_time': '<200ms per sample',
            'memory_usage': '<8GB during training'
        },
        'training_strategy': {
            'epochs': 5,
            'batch_size': 16,
            'learning_rate': 2e-5,
            'warmup_steps': '10% of total steps',
            'weight_decay': 0.01,
            'optimization': 'AdamW with linear schedule'
        },
        'architecture_details': {
            'base_model': 'BERT (Bidirectional Encoder Representations from Transformers)',
            'fine_tuning': 'Full model fine-tuning',
            'pooling_strategy': 'CLS token or mean pooling',
            'classification_head': 'Linear layer with dropout'
        }
    }
    
    return task_definition

if __name__ == '__main__':
    # Test model creation
    print("Testing NLP model creation...")
    
    # Create different model types
    models = {
        'BERT': ModelFactory.create_model('bert', num_classes=2, dropout_rate=0.3),
        'Advanced': ModelFactory.create_model('advanced', pooling_strategy='multi', num_classes=2)
    }
    
    for name, model in models.items():
        print(f"\n{name} Model Info:")
        if hasattr(model, 'get_model_info'):
            info = model.get_model_info()
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    # Test forward pass
    model = models['BERT']
    batch_size, seq_len = 2, 128
    
    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    print(f"\nForward pass test:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    # Test optimizer creation
    optimizer = ModelFactory.create_optimizer(model, learning_rate=2e-5)
    print(f"\nOptimizer created: {type(optimizer).__name__}")
    
    # Task definition
    task = define_nlp_task()
    print("\n=== NLP Task Definition ===")
    for key, value in task.items():
        print(f"{key}: {value}")
