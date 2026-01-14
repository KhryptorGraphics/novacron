"""Model evaluation and analysis for CIFAR-10 classification."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import json
from pathlib import Path
import time

class ModelEvaluator:
    """Comprehensive model evaluation with MLE-Star methodology."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    def evaluate_model(self, model: nn.Module, data_loader, 
                      criterion: nn.Module) -> Dict[str, Any]:
        """Comprehensive model evaluation."""
        
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # Measure inference time
                start_time = time.time()
                output = model(data)
                inference_time = time.time() - start_time
                inference_times.append(inference_time / data.size(0))  # per sample
                
                # Calculate loss
                loss = criterion(output, target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        avg_inference_time = np.mean(inference_times)
        
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        return {
            'loss': avg_loss,
            'inference_time_per_sample': avg_inference_time,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            **metrics
        }
    
    def _calculate_metrics(self, targets: List[int], predictions: List[int], 
                          probabilities: List[List[float]]) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        
        # Basic metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, average='weighted', zero_division=0)
        recall = recall_score(targets, predictions, average='weighted', zero_division=0)
        f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_precision = precision_score(targets, predictions, average=None, zero_division=0)
        per_class_recall = recall_score(targets, predictions, average=None, zero_division=0)
        per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
        
        # Classification report
        class_report = classification_report(targets, predictions, 
                                           target_names=self.class_names, 
                                           output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        
        # ROC metrics (for multi-class)
        targets_binary = label_binarize(targets, classes=range(10))
        probabilities_array = np.array(probabilities)
        
        roc_metrics = self._calculate_roc_metrics(targets_binary, probabilities_array)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'roc_metrics': roc_metrics
        }
    
    def _calculate_roc_metrics(self, targets_binary: np.ndarray, 
                              probabilities: np.ndarray) -> Dict[str, Any]:
        """Calculate ROC-AUC metrics for multi-class classification."""
        
        roc_auc_scores = []
        
        for i in range(10):
            if np.sum(targets_binary[:, i]) > 0:  # Check if class exists in targets
                fpr, tpr, _ = roc_curve(targets_binary[:, i], probabilities[:, i])
                roc_auc = auc(fpr, tpr)
                roc_auc_scores.append(roc_auc)
            else:
                roc_auc_scores.append(0.0)
        
        # Macro and weighted averages
        macro_roc_auc = np.mean(roc_auc_scores)
        
        return {
            'per_class_roc_auc': roc_auc_scores,
            'macro_roc_auc': macro_roc_auc
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix heatmap."""
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_per_class_metrics(self, metrics: Dict[str, Any], 
                              save_path: Optional[str] = None) -> None:
        """Plot per-class performance metrics."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Per-class metrics bar plot
        x = np.arange(len(self.class_names))
        width = 0.25
        
        ax1.bar(x - width, metrics['per_class_precision'], width, 
               label='Precision', alpha=0.8, color='skyblue')
        ax1.bar(x, metrics['per_class_recall'], width, 
               label='Recall', alpha=0.8, color='lightgreen')
        ax1.bar(x + width, metrics['per_class_f1'], width, 
               label='F1-Score', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Classes', fontsize=12)
        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(self.class_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # ROC AUC scores
        ax2.bar(range(len(self.class_names)), metrics['roc_metrics']['per_class_roc_auc'],
               color='orange', alpha=0.8)
        ax2.set_xlabel('Classes', fontsize=12)
        ax2.set_ylabel('ROC AUC Score', fontsize=12)
        ax2.set_title('Per-Class ROC AUC Scores', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassifications(self, targets: List[int], predictions: List[int]) -> Dict[str, Any]:
        """Analyze misclassification patterns."""
        
        misclassified_indices = [i for i, (t, p) in enumerate(zip(targets, predictions)) if t != p]
        
        if not misclassified_indices:
            return {'misclassification_rate': 0.0, 'patterns': {}}
        
        misclassification_rate = len(misclassified_indices) / len(targets)
        
        # Analyze confusion patterns
        confusion_patterns = {}
        for i in misclassified_indices:
            true_class = self.class_names[targets[i]]
            pred_class = self.class_names[predictions[i]]
            pattern = f"{true_class} -> {pred_class}"
            
            if pattern not in confusion_patterns:
                confusion_patterns[pattern] = 0
            confusion_patterns[pattern] += 1
        
        # Sort patterns by frequency
        sorted_patterns = dict(sorted(confusion_patterns.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        return {
            'misclassification_rate': misclassification_rate,
            'total_misclassified': len(misclassified_indices),
            'patterns': sorted_patterns,
            'top_confusion_pairs': list(sorted_patterns.keys())[:5]
        }
    
    def generate_evaluation_report(self, model: nn.Module, test_loader, 
                                 criterion: nn.Module, save_dir: str) -> Dict[str, Any]:
        """Generate comprehensive evaluation report - MLE-Star Stage 4: Results."""
        
        print("=== MLE-Star Stage 4: Results - Model Evaluation ===")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Evaluate model
        results = self.evaluate_model(model, test_loader, criterion)
        
        # Analyze misclassifications
        misclass_analysis = self.analyze_misclassifications(
            results['targets'], results['predictions']
        )
        results['misclassification_analysis'] = misclass_analysis
        
        # Plot visualizations
        cm = np.array(results['confusion_matrix'])
        self.plot_confusion_matrix(cm, save_path / 'confusion_matrix.png')
        self.plot_per_class_metrics(results, save_path / 'per_class_metrics.png')
        
        # Generate detailed report
        report = self._generate_detailed_report(results)
        
        # Save results
        results_to_save = {k: v for k, v in results.items() 
                          if k not in ['predictions', 'targets', 'probabilities']}
        
        with open(save_path / 'evaluation_results.json', 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        with open(save_path / 'evaluation_report.md', 'w') as f:
            f.write(report)
        
        print(f"Evaluation results saved to: {save_path}")
        return results
    
    def _generate_detailed_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed evaluation report in markdown format."""
        
        report = f"""# CIFAR-10 Model Evaluation Report

## Overall Performance

- **Accuracy**: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)
- **Precision**: {results['precision']:.4f}
- **Recall**: {results['recall']:.4f}
- **F1-Score**: {results['f1_score']:.4f}
- **Loss**: {results['loss']:.4f}
- **Inference Time**: {results['inference_time_per_sample']*1000:.2f}ms per sample
- **Macro ROC AUC**: {results['roc_metrics']['macro_roc_auc']:.4f}

## Per-Class Performance

| Class | Precision | Recall | F1-Score | ROC AUC |
|-------|-----------|--------|----------|----------|
"""
        
        for i, class_name in enumerate(self.class_names):
            precision = results['per_class_precision'][i]
            recall = results['per_class_recall'][i]
            f1 = results['per_class_f1'][i]
            roc_auc = results['roc_metrics']['per_class_roc_auc'][i]
            
            report += f"| {class_name} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {roc_auc:.3f} |\n"
        
        report += f"""

## Misclassification Analysis

- **Misclassification Rate**: {results['misclassification_analysis']['misclassification_rate']*100:.2f}%
- **Total Misclassified**: {results['misclassification_analysis']['total_misclassified']}

### Top Confusion Patterns

"""
        
        for i, pattern in enumerate(results['misclassification_analysis']['top_confusion_pairs'][:5], 1):
            count = results['misclassification_analysis']['patterns'][pattern]
            report += f"{i}. {pattern}: {count} cases\n"
        
        report += f"""

## Model Architecture Summary

- Model has been evaluated on CIFAR-10 test set
- Input shape: (3, 32, 32)
- Output classes: 10
- Evaluation completed successfully

## Recommendations

"""
        
        # Add recommendations based on performance
        if results['accuracy'] < 0.8:
            report += "- Consider increasing model complexity or training time\n"
            report += "- Implement additional data augmentation techniques\n"
        
        if results['misclassification_analysis']['misclassification_rate'] > 0.2:
            report += "- Analyze top confusion patterns for targeted improvements\n"
            report += "- Consider class-specific data augmentation\n"
        
        if results['inference_time_per_sample'] > 0.1:
            report += "- Consider model optimization for faster inference\n"
            report += "- Implement model quantization or pruning\n"
        
        return report

class AblationStudy:
    """Conduct ablation studies to understand component contributions."""
    
    def __init__(self, base_model_factory):
        self.base_model_factory = base_model_factory
        self.evaluator = ModelEvaluator()
    
    def run_ablation_study(self, test_loader, criterion, 
                          ablation_configs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run comprehensive ablation study."""
        
        print("=== Running Ablation Study ===")
        
        results = {}
        
        for config in ablation_configs:
            config_name = config.get('name', 'unnamed')
            print(f"Testing configuration: {config_name}")
            
            # Create model with configuration
            model = self.base_model_factory(config)
            
            # Evaluate model
            eval_results = self.evaluator.evaluate_model(model, test_loader, criterion)
            
            results[config_name] = {
                'config': config,
                'accuracy': eval_results['accuracy'],
                'loss': eval_results['loss'],
                'inference_time': eval_results['inference_time_per_sample']
            }
        
        # Analyze results
        self._analyze_ablation_results(results)
        
        return results
    
    def _analyze_ablation_results(self, results: Dict[str, Any]) -> None:
        """Analyze and visualize ablation study results."""
        
        configs = list(results.keys())
        accuracies = [results[config]['accuracy'] for config in configs]
        losses = [results[config]['loss'] for config in configs]
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(configs, accuracies, color='skyblue', alpha=0.8)
        ax1.set_title('Accuracy Comparison Across Configurations')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Configuration')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Loss comparison
        bars2 = ax2.bar(configs, losses, color='lightcoral', alpha=0.8)
        ax2.set_title('Loss Comparison Across Configurations')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Configuration')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, loss in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Example usage
    evaluator = ModelEvaluator()
    
    print("ModelEvaluator initialized successfully!")
    print(f"Device: {evaluator.device}")
    print(f"Classes: {evaluator.class_names}")
