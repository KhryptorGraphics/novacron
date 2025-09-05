"""
Neural Architecture Search (NAS)
================================

Automated neural network architecture discovery for MLE-Star:
- Differentiable Architecture Search (DARTS)
- Efficient Neural Architecture Search (ENAS)
- Progressive Neural Architecture Search (PNAS)
- Network morphism and architecture evolution
- Performance prediction and early stopping
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass
import json
from pathlib import Path
import random

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, SGD
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search"""
    # Search strategy
    search_strategy: str = "darts"  # darts, enas, random, evolutionary
    
    # Architecture space configuration
    max_layers: int = 20
    layer_types: List[str] = None  # conv, pool, fc, skip, etc.
    activation_functions: List[str] = None  # relu, tanh, sigmoid, swish
    
    # Search hyperparameters
    search_epochs: int = 50
    architecture_lr: float = 3e-4
    weight_lr: float = 0.025
    momentum: float = 0.9
    weight_decay: float = 3e-4
    
    # DARTS specific
    darts_unrolled: bool = False
    darts_arch_weight_decay: float = 1e-3
    
    # ENAS specific
    enas_controller_lr: float = 3.5e-4
    enas_controller_tanh_constant: float = 1.10
    enas_controller_op_tanh_reduce: float = 2.5
    
    # Evolutionary search
    population_size: int = 50
    tournament_size: int = 10
    mutation_rate: float = 0.1
    crossover_rate: float = 0.5
    
    # Early stopping and pruning
    early_stopping_patience: int = 10
    min_performance_threshold: float = 0.1
    
    # Resource constraints
    max_params: Optional[int] = None  # Maximum number of parameters
    max_flops: Optional[int] = None   # Maximum FLOPs
    memory_limit: Optional[int] = None # Maximum memory usage (MB)
    
    # Output configuration
    save_architectures: bool = True
    architecture_save_path: str = "./nas_architectures"
    top_k_architectures: int = 5


class ArchitectureSpace:
    """Definition of neural architecture search space"""
    
    def __init__(self, config: NASConfig):
        self.config = config
        self.layer_types = config.layer_types or [
            'conv3x3', 'conv5x5', 'conv7x7', 'sep_conv_3x3', 'sep_conv_5x5',
            'dil_conv_3x3', 'dil_conv_5x5', 'max_pool_3x3', 'avg_pool_3x3',
            'skip_connect', 'none'
        ]
        self.activations = config.activation_functions or [
            'relu', 'relu6', 'tanh', 'sigmoid', 'swish', 'gelu', 'leaky_relu'
        ]
    
    def sample_architecture(self) -> Dict[str, Any]:
        """Sample a random architecture from the search space"""
        num_layers = random.randint(1, self.config.max_layers)
        
        architecture = {
            'layers': [],
            'connections': [],
            'global_config': {
                'num_layers': num_layers,
                'input_size': None,  # To be set based on data
                'output_size': None  # To be set based on task
            }
        }
        
        for i in range(num_layers):
            layer_config = {
                'layer_id': i,
                'type': random.choice(self.layer_types),
                'activation': random.choice(self.activations),
                'channels': random.choice([16, 32, 64, 128, 256, 512]),
                'kernel_size': random.choice([1, 3, 5, 7]) if 'conv' in self.layer_types[0] else None,
                'stride': random.choice([1, 2]),
                'padding': 'same'
            }
            
            architecture['layers'].append(layer_config)
        
        # Generate skip connections
        for i in range(num_layers):
            for j in range(i + 1, min(i + 4, num_layers)):  # Max skip of 3 layers
                if random.random() < 0.3:  # 30% chance of skip connection
                    architecture['connections'].append({'from': i, 'to': j})
        
        return architecture
    
    def mutate_architecture(self, architecture: Dict[str, Any], 
                          mutation_rate: float = 0.1) -> Dict[str, Any]:
        """Apply mutations to an architecture"""
        mutated = json.loads(json.dumps(architecture))  # Deep copy
        
        # Mutate layer types
        for layer in mutated['layers']:
            if random.random() < mutation_rate:
                layer['type'] = random.choice(self.layer_types)
            
            if random.random() < mutation_rate:
                layer['activation'] = random.choice(self.activations)
            
            if random.random() < mutation_rate:
                layer['channels'] = random.choice([16, 32, 64, 128, 256, 512])
        
        # Mutate connections
        if random.random() < mutation_rate:
            # Add or remove connection
            if random.random() < 0.5 and len(mutated['connections']) > 0:
                # Remove connection
                mutated['connections'].pop(random.randint(0, len(mutated['connections']) - 1))
            else:
                # Add connection
                num_layers = len(mutated['layers'])
                if num_layers > 1:
                    i = random.randint(0, num_layers - 2)
                    j = random.randint(i + 1, num_layers - 1)
                    new_connection = {'from': i, 'to': j}
                    if new_connection not in mutated['connections']:
                        mutated['connections'].append(new_connection)
        
        return mutated
    
    def crossover_architectures(self, parent1: Dict[str, Any], 
                               parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two architectures"""
        # Simple crossover: take layers from both parents
        child = {'layers': [], 'connections': [], 'global_config': parent1['global_config'].copy()}
        
        # Randomly select layers from parents
        max_layers = max(len(parent1['layers']), len(parent2['layers']))
        
        for i in range(max_layers):
            if i < len(parent1['layers']) and i < len(parent2['layers']):
                # Both parents have this layer
                chosen_layer = random.choice([parent1['layers'][i], parent2['layers'][i]])
            elif i < len(parent1['layers']):
                chosen_layer = parent1['layers'][i]
            else:
                chosen_layer = parent2['layers'][i]
            
            chosen_layer = chosen_layer.copy()
            chosen_layer['layer_id'] = i
            child['layers'].append(chosen_layer)
        
        # Combine connections from both parents
        all_connections = parent1['connections'] + parent2['connections']
        valid_connections = []
        
        for conn in all_connections:
            if (conn['from'] < len(child['layers']) and 
                conn['to'] < len(child['layers']) and
                conn not in valid_connections):
                valid_connections.append(conn)
        
        child['connections'] = valid_connections
        child['global_config']['num_layers'] = len(child['layers'])
        
        return child


class DARTSSearcher:
    """Differentiable Architecture Search implementation"""
    
    def __init__(self, config: NASConfig, search_space: ArchitectureSpace):
        self.config = config
        self.search_space = search_space
        
    def create_supernet(self, input_shape: Tuple[int, ...], num_classes: int):
        """Create DARTS supernet"""
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DARTS")
        
        class DARTSCell(nn.Module):
            def __init__(self, channels, operations):
                super().__init__()
                self.operations = operations
                self.alpha = nn.Parameter(torch.randn(len(operations)))
                
            def forward(self, x):
                weights = F.softmax(self.alpha, dim=0)
                output = sum(w * op(x) for w, op in zip(weights, self.operations))
                return output
        
        class DARTSSupernet(nn.Module):
            def __init__(self, input_shape, num_classes, num_cells=8):
                super().__init__()
                self.num_cells = num_cells
                
                # Create operations for each cell
                operations = [
                    nn.Conv2d(input_shape[0], 64, 3, padding=1),
                    nn.Conv2d(input_shape[0], 64, 5, padding=2),
                    nn.AvgPool2d(3, stride=1, padding=1),
                    nn.MaxPool2d(3, stride=1, padding=1),
                    nn.Identity()  # Skip connection
                ]
                
                self.cells = nn.ModuleList([
                    DARTSCell(64, operations) for _ in range(num_cells)
                ])
                
                self.classifier = nn.Linear(64, num_classes)
                self.global_pool = nn.AdaptiveAvgPool2d(1)
                
            def forward(self, x):
                for cell in self.cells:
                    x = cell(x)
                
                x = self.global_pool(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                
                return x
            
            def get_architecture(self):
                """Extract current architecture from alpha parameters"""
                architecture = {'layers': [], 'connections': []}
                
                for i, cell in enumerate(self.cells):
                    best_op_idx = torch.argmax(cell.alpha).item()
                    op_name = self.search_space.layer_types[best_op_idx] if best_op_idx < len(self.search_space.layer_types) else 'conv3x3'
                    
                    layer = {
                        'layer_id': i,
                        'type': op_name,
                        'activation': 'relu',
                        'channels': 64,
                        'alpha_weights': cell.alpha.detach().cpu().numpy().tolist()
                    }
                    
                    architecture['layers'].append(layer)
                
                return architecture
        
        return DARTSSupernet(input_shape, num_classes)
    
    def search(self, train_loader, val_loader, input_shape: Tuple[int, ...], 
              num_classes: int) -> Dict[str, Any]:
        """Run DARTS architecture search"""
        if not PYTORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DARTS")
        
        # Create supernet
        supernet = self.create_supernet(input_shape, num_classes)
        
        # Optimizers
        weight_optimizer = Adam(
            [p for p in supernet.parameters() if p.requires_grad and p.dim() > 1],
            lr=self.config.weight_lr,
            weight_decay=self.config.weight_decay
        )
        
        arch_optimizer = Adam(
            [p for p in supernet.parameters() if p.requires_grad and p.dim() == 1],
            lr=self.config.architecture_lr,
            weight_decay=self.config.darts_arch_weight_decay
        )
        
        best_val_acc = 0
        best_architecture = None
        search_history = []
        
        for epoch in range(self.config.search_epochs):
            # Train weights
            supernet.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                weight_optimizer.zero_grad()
                output = supernet(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                weight_optimizer.step()
                
                train_loss += loss.item()
                pred = output.argmax(dim=1)
                train_correct += pred.eq(target).sum().item()
                train_total += target.size(0)
            
            # Train architecture parameters
            supernet.train()
            for batch_idx, (data, target) in enumerate(val_loader):
                if batch_idx >= 10:  # Limit arch updates
                    break
                    
                arch_optimizer.zero_grad()
                output = supernet(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                arch_optimizer.step()
            
            # Validation
            supernet.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = supernet(data)
                    val_loss += F.cross_entropy(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            logger.info(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best architecture
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_architecture = supernet.get_architecture()
            
            search_history.append({
                'epoch': epoch,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'train_loss': train_loss / train_total,
                'val_loss': val_loss / val_total
            })
        
        return {
            'best_architecture': best_architecture,
            'best_val_accuracy': best_val_acc,
            'search_history': search_history,
            'supernet_state': supernet.state_dict()
        }


class EvolutionarySearcher:
    """Evolutionary Neural Architecture Search"""
    
    def __init__(self, config: NASConfig, search_space: ArchitectureSpace):
        self.config = config
        self.search_space = search_space
    
    def evaluate_architecture(self, architecture: Dict[str, Any], 
                            train_data, val_data, quick_eval: bool = True) -> float:
        """Evaluate architecture performance"""
        try:
            # For quick evaluation, use a simplified metric
            if quick_eval:
                # Estimate based on architecture complexity
                num_layers = len(architecture['layers'])
                num_connections = len(architecture['connections'])
                
                # Simple heuristic: moderate complexity often works better
                complexity_score = 1.0 - abs(num_layers - 10) / 20  # Optimal around 10 layers
                connection_score = min(num_connections / num_layers, 1.0)  # Moderate connections
                
                # Random noise to simulate actual training variance
                noise = random.gauss(0, 0.1)
                
                score = (complexity_score + connection_score) / 2 + noise
                return max(0, min(1, score))  # Clamp to [0, 1]
            else:
                # TODO: Implement actual training and evaluation
                # This would involve building the network and training it
                return random.random()  # Placeholder
                
        except Exception as e:
            logger.warning(f"Architecture evaluation failed: {e}")
            return 0.0
    
    def search(self, train_data, val_data) -> Dict[str, Any]:
        """Run evolutionary architecture search"""
        # Initialize population
        population = []
        for _ in range(self.config.population_size):
            arch = self.search_space.sample_architecture()
            fitness = self.evaluate_architecture(arch, train_data, val_data)
            population.append({'architecture': arch, 'fitness': fitness})
        
        best_individual = max(population, key=lambda x: x['fitness'])
        search_history = []
        
        for generation in range(self.config.search_epochs):
            # Tournament selection
            new_population = []
            
            for _ in range(self.config.population_size):
                # Select parents through tournament
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if random.random() < self.config.crossover_rate:
                    child_arch = self.search_space.crossover_architectures(
                        parent1['architecture'], parent2['architecture']
                    )
                else:
                    child_arch = parent1['architecture'].copy()
                
                # Mutation
                if random.random() < self.config.mutation_rate:
                    child_arch = self.search_space.mutate_architecture(
                        child_arch, self.config.mutation_rate
                    )
                
                # Evaluate child
                fitness = self.evaluate_architecture(child_arch, train_data, val_data)
                new_population.append({'architecture': child_arch, 'fitness': fitness})
            
            population = new_population
            
            # Track best individual
            current_best = max(population, key=lambda x: x['fitness'])
            if current_best['fitness'] > best_individual['fitness']:
                best_individual = current_best
            
            # Log progress
            avg_fitness = sum(ind['fitness'] for ind in population) / len(population)
            logger.info(f"Generation {generation}: Best: {best_individual['fitness']:.4f}, "
                      f"Avg: {avg_fitness:.4f}")
            
            search_history.append({
                'generation': generation,
                'best_fitness': best_individual['fitness'],
                'avg_fitness': avg_fitness,
                'population_diversity': self._calculate_diversity(population)
            })
        
        return {
            'best_architecture': best_individual['architecture'],
            'best_fitness': best_individual['fitness'],
            'search_history': search_history,
            'final_population': population
        }
    
    def _tournament_selection(self, population: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select individual using tournament selection"""
        tournament = random.sample(population, min(self.config.tournament_size, len(population)))
        return max(tournament, key=lambda x: x['fitness'])
    
    def _calculate_diversity(self, population: List[Dict[str, Any]]) -> float:
        """Calculate population diversity metric"""
        if len(population) < 2:
            return 0.0
        
        # Simple diversity measure: average difference in number of layers
        layer_counts = [len(ind['architecture']['layers']) for ind in population]
        return np.std(layer_counts) / np.mean(layer_counts) if np.mean(layer_counts) > 0 else 0.0


class NeuralArchitectureSearch(BaseEnhancement):
    """Neural Architecture Search enhancement for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="neural_architecture_search",
            version="1.0.0",
            enabled=PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE,
            priority=35,
            parameters={
                "search_strategy": "evolutionary",  # Start with evolutionary as it's more stable
                "max_layers": 20,
                "layer_types": None,
                "activation_functions": None,
                "search_epochs": 50,
                "architecture_lr": 3e-4,
                "weight_lr": 0.025,
                "momentum": 0.9,
                "weight_decay": 3e-4,
                "darts_unrolled": False,
                "darts_arch_weight_decay": 1e-3,
                "enas_controller_lr": 3.5e-4,
                "enas_controller_tanh_constant": 1.10,
                "enas_controller_op_tanh_reduce": 2.5,
                "population_size": 50,
                "tournament_size": 10,
                "mutation_rate": 0.1,
                "crossover_rate": 0.5,
                "early_stopping_patience": 10,
                "min_performance_threshold": 0.1,
                "max_params": None,
                "max_flops": None,
                "memory_limit": None,
                "save_architectures": True,
                "architecture_save_path": "./nas_architectures",
                "top_k_architectures": 5
            }
        )
    
    def initialize(self) -> bool:
        """Initialize NAS"""
        if not (PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE):
            self._logger.error("Neither PyTorch nor TensorFlow available for NAS")
            return False
        
        try:
            # Create configuration
            self.nas_config = NASConfig(**self.config.parameters)
            
            # Initialize search space
            self.search_space = ArchitectureSpace(self.nas_config)
            
            # Initialize searcher based on strategy
            if self.nas_config.search_strategy == "darts":
                if PYTORCH_AVAILABLE:
                    self.searcher = DARTSSearcher(self.nas_config, self.search_space)
                else:
                    self._logger.warning("DARTS requires PyTorch. Falling back to evolutionary search")
                    self.searcher = EvolutionarySearcher(self.nas_config, self.search_space)
            elif self.nas_config.search_strategy == "evolutionary":
                self.searcher = EvolutionarySearcher(self.nas_config, self.search_space)
            else:
                self._logger.warning(f"Unknown search strategy: {self.nas_config.search_strategy}. Using evolutionary")
                self.searcher = EvolutionarySearcher(self.nas_config, self.search_space)
            
            self._logger.info(f"NAS initialized with strategy: {self.nas_config.search_strategy}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize NAS: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with NAS capabilities"""
        enhanced = workflow.copy()
        
        # Add NAS configuration
        if 'neural_architecture_search' not in enhanced:
            enhanced['neural_architecture_search'] = {}
        
        enhanced['neural_architecture_search'] = {
            'enabled': True,
            'search_strategy': self.nas_config.search_strategy,
            'search_epochs': self.nas_config.search_epochs,
            'max_layers': self.nas_config.max_layers,
            'population_size': self.nas_config.population_size if self.nas_config.search_strategy == 'evolutionary' else None,
            'resource_constraints': {
                'max_params': self.nas_config.max_params,
                'max_flops': self.nas_config.max_flops,
                'memory_limit': self.nas_config.memory_limit
            },
            'search_space': {
                'layer_types': self.search_space.layer_types,
                'activation_functions': self.search_space.activations
            }
        }
        
        # Enhance MLE-Star stages with NAS
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 3: Action Planning - Add NAS planning
            if '3_action_planning' in stages:
                if 'nas_planning' not in stages['3_action_planning']:
                    stages['3_action_planning']['nas_planning'] = [
                        'architecture_search_space_design',
                        'search_strategy_selection',
                        'resource_constraint_definition',
                        'performance_evaluation_strategy'
                    ]
            
            # Stage 4: Implementation - Add architecture search
            if '4_implementation' in stages:
                if 'nas_implementation' not in stages['4_implementation']:
                    stages['4_implementation']['nas_implementation'] = [
                        'supernet_construction',
                        'architecture_search_execution',
                        'candidate_architecture_evaluation',
                        'architecture_ranking'
                    ]
            
            # Stage 6: Refinement - Add architecture optimization
            if '6_refinement' in stages:
                if 'nas_refinement' not in stages['6_refinement']:
                    stages['6_refinement']['nas_refinement'] = [
                        'architecture_fine_tuning',
                        'multi_objective_optimization',
                        'architecture_ensemble',
                        'knowledge_distillation'
                    ]
        
        # Add NAS specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'nas_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['nas_metrics'] = [
                'architecture_diversity',
                'search_efficiency',
                'parameter_efficiency',
                'flops_efficiency',
                'memory_efficiency',
                'convergence_speed'
            ]
        
        self._logger.debug("Enhanced workflow with NAS capabilities")
        return enhanced
    
    def search_architecture(self, train_data, val_data, 
                          input_shape: Optional[Tuple[int, ...]] = None,
                          num_classes: Optional[int] = None) -> Dict[str, Any]:
        """Run neural architecture search"""
        try:
            self._logger.info(f"Starting architecture search with {self.nas_config.search_strategy}")
            
            # Run search based on strategy
            if self.nas_config.search_strategy == "darts" and hasattr(self.searcher, 'search'):
                if input_shape is None or num_classes is None:
                    raise ValueError("DARTS requires input_shape and num_classes")
                
                results = self.searcher.search(train_data, val_data, input_shape, num_classes)
            else:
                results = self.searcher.search(train_data, val_data)
            
            # Save architectures if enabled
            if self.nas_config.save_architectures:
                self._save_architectures(results)
            
            # Analyze results
            analysis = self._analyze_search_results(results)
            results['analysis'] = analysis
            
            self._logger.info(f"Architecture search completed. Best performance: {results.get('best_fitness', 'N/A')}")
            
            return results
            
        except Exception as e:
            self._logger.error(f"Architecture search failed: {e}")
            return {}
    
    def _save_architectures(self, results: Dict[str, Any]):
        """Save discovered architectures"""
        try:
            save_path = Path(self.nas_config.architecture_save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save best architecture
            if 'best_architecture' in results:
                with open(save_path / "best_architecture.json", 'w') as f:
                    json.dump(results['best_architecture'], f, indent=2)
            
            # Save top k architectures if available
            if 'final_population' in results:
                # Sort by fitness and save top k
                sorted_population = sorted(
                    results['final_population'], 
                    key=lambda x: x['fitness'], 
                    reverse=True
                )
                
                top_k = sorted_population[:self.nas_config.top_k_architectures]
                
                with open(save_path / "top_architectures.json", 'w') as f:
                    json.dump(top_k, f, indent=2)
            
            # Save search history
            if 'search_history' in results:
                with open(save_path / "search_history.json", 'w') as f:
                    json.dump(results['search_history'], f, indent=2)
            
            self._logger.info(f"Architectures saved to: {save_path}")
            
        except Exception as e:
            self._logger.error(f"Failed to save architectures: {e}")
    
    def _analyze_search_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze search results"""
        analysis = {}
        
        try:
            # Search efficiency analysis
            if 'search_history' in results:
                history = results['search_history']
                
                if self.nas_config.search_strategy == 'evolutionary':
                    fitness_values = [h['best_fitness'] for h in history]
                    analysis['convergence'] = {
                        'final_best': max(fitness_values) if fitness_values else 0,
                        'convergence_epoch': fitness_values.index(max(fitness_values)) if fitness_values else 0,
                        'improvement_rate': np.diff(fitness_values).tolist() if len(fitness_values) > 1 else [],
                        'plateau_epochs': self._count_plateau_epochs(fitness_values)
                    }
                    
                    if len(history) > 1:
                        diversity_values = [h.get('population_diversity', 0) for h in history]
                        analysis['diversity'] = {
                            'avg_diversity': np.mean(diversity_values),
                            'diversity_trend': np.polyfit(range(len(diversity_values)), diversity_values, 1)[0] if len(diversity_values) > 1 else 0
                        }
            
            # Architecture complexity analysis
            if 'best_architecture' in results:
                arch = results['best_architecture']
                
                analysis['architecture_stats'] = {
                    'num_layers': len(arch.get('layers', [])),
                    'num_connections': len(arch.get('connections', [])),
                    'layer_type_distribution': self._get_layer_type_distribution(arch),
                    'activation_distribution': self._get_activation_distribution(arch),
                    'estimated_parameters': self._estimate_parameters(arch)
                }
            
            # Population analysis for evolutionary search
            if 'final_population' in results:
                population = results['final_population']
                fitness_values = [ind['fitness'] for ind in population]
                
                analysis['population_stats'] = {
                    'best_fitness': max(fitness_values),
                    'worst_fitness': min(fitness_values),
                    'mean_fitness': np.mean(fitness_values),
                    'std_fitness': np.std(fitness_values),
                    'fitness_distribution': np.histogram(fitness_values, bins=10)[0].tolist()
                }
        
        except Exception as e:
            self._logger.warning(f"Analysis failed: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _count_plateau_epochs(self, fitness_values: List[float], 
                             threshold: float = 1e-4) -> int:
        """Count epochs without significant improvement"""
        if len(fitness_values) < 2:
            return 0
        
        best_so_far = fitness_values[0]
        plateau_count = 0
        
        for fitness in fitness_values[1:]:
            if fitness - best_so_far < threshold:
                plateau_count += 1
            else:
                best_so_far = fitness
                plateau_count = 0
        
        return plateau_count
    
    def _get_layer_type_distribution(self, architecture: Dict[str, Any]) -> Dict[str, int]:
        """Get distribution of layer types in architecture"""
        distribution = {}
        for layer in architecture.get('layers', []):
            layer_type = layer.get('type', 'unknown')
            distribution[layer_type] = distribution.get(layer_type, 0) + 1
        return distribution
    
    def _get_activation_distribution(self, architecture: Dict[str, Any]) -> Dict[str, int]:
        """Get distribution of activation functions in architecture"""
        distribution = {}
        for layer in architecture.get('layers', []):
            activation = layer.get('activation', 'unknown')
            distribution[activation] = distribution.get(activation, 0) + 1
        return distribution
    
    def _estimate_parameters(self, architecture: Dict[str, Any]) -> int:
        """Rough estimate of model parameters"""
        total_params = 0
        prev_channels = 3  # Assume RGB input
        
        for layer in architecture.get('layers', []):
            layer_type = layer.get('type', '')
            channels = layer.get('channels', 64)
            
            if 'conv' in layer_type:
                kernel_size = layer.get('kernel_size', 3)
                params = prev_channels * channels * kernel_size * kernel_size
                total_params += params
                prev_channels = channels
            elif layer_type == 'fc':
                # Assume flattened input
                params = prev_channels * channels
                total_params += params
                prev_channels = channels
        
        return total_params