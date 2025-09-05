"""
JAX/Flax Integration
===================

Advanced neural network capabilities using JAX/Flax for:
- High-performance automatic differentiation
- Just-in-time compilation with XLA
- Functional programming paradigms
- Advanced optimization techniques
- Distributed training support
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import flax
    import flax.linen as nn
    import optax
    from flax.training import train_state, checkpoints
    from flax.core import freeze, unfreeze
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from ..core import BaseEnhancement, EnhancementConfig

logger = logging.getLogger(__name__)


@dataclass
class JAXFlaxConfig:
    """Configuration for JAX/Flax integration"""
    # JAX configuration
    enable_x64: bool = False
    jit_compilation: bool = True
    platform: str = "gpu"  # gpu, cpu, tpu
    
    # Distributed training
    distributed: bool = False
    num_devices: Optional[int] = None
    
    # Memory optimization
    memory_fraction: Optional[float] = None
    preallocate_memory: bool = False
    
    # Optimization
    gradient_accumulation_steps: int = 1
    gradient_clipping: Optional[float] = None
    
    # Training state
    checkpoint_dir: str = "checkpoints"
    save_every_steps: int = 1000
    keep_checkpoints: int = 5


class JAXFlaxModel(nn.Module):
    """Base Flax model with MLE-Star integration"""
    features: List[int]
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        x = nn.Dense(self.features[-1])(x)
        return x


class JAXFlaxTrainer:
    """JAX/Flax training utilities for MLE-Star"""
    
    def __init__(self, config: JAXFlaxConfig):
        self.config = config
        self._setup_jax()
    
    def _setup_jax(self):
        """Configure JAX environment"""
        if self.config.enable_x64:
            jax.config.update("jax_enable_x64", True)
        
        if self.config.memory_fraction:
            jax.config.update("jax_gpu_memory_fraction", self.config.memory_fraction)
        
        if self.config.preallocate_memory:
            jax.config.update("jax_gpu_preallocate", True)
        
        logger.info(f"JAX devices: {jax.devices()}")
        logger.info(f"JAX local devices: {jax.local_devices()}")
    
    def create_train_state(self, model: nn.Module, learning_rate: float,
                          input_shape: Tuple[int, ...], key: Optional[jax.Array] = None) -> train_state.TrainState:
        """Create training state with model, optimizer, and parameters"""
        if key is None:
            key = jax.random.PRNGKey(42)
        
        # Initialize parameters
        dummy_input = jnp.ones(input_shape)
        params = model.init(key, dummy_input, training=False)
        
        # Create optimizer
        optimizer = optax.adam(learning_rate)
        
        if self.config.gradient_clipping:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clipping),
                optimizer
            )
        
        # Create training state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=optimizer
        )
        
        return state
    
    @staticmethod
    @jax.jit
    def train_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray],
                   dropout_key: jax.Array) -> Tuple[train_state.TrainState, Dict[str, float]]:
        """Single training step with gradient computation"""
        
        def loss_fn(params):
            logits = state.apply_fn(
                params, batch['x'], 
                training=True, 
                rngs={'dropout': dropout_key}
            )
            
            # Cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, batch['y']
            ).mean()
            
            # Compute accuracy
            predictions = jnp.argmax(logits, axis=-1)
            accuracy = jnp.mean(predictions == batch['y'])
            
            return loss, {'accuracy': accuracy}
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, metrics), grads = grad_fn(state.params)
        
        # Update state
        state = state.apply_gradients(grads=grads)
        
        metrics['loss'] = loss
        return state, metrics
    
    @staticmethod
    @jax.jit
    def eval_step(state: train_state.TrainState, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Single evaluation step"""
        logits = state.apply_fn(state.params, batch['x'], training=False)
        
        # Compute metrics
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['y']
        ).mean()
        
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == batch['y'])
        
        return {'loss': loss, 'accuracy': accuracy}
    
    def save_checkpoint(self, state: train_state.TrainState, step: int):
        """Save model checkpoint"""
        checkpoints.save_checkpoint(
            ckpt_dir=self.config.checkpoint_dir,
            target=state,
            step=step,
            keep=self.config.keep_checkpoints
        )
    
    def load_checkpoint(self, state: train_state.TrainState, step: Optional[int] = None) -> train_state.TrainState:
        """Load model checkpoint"""
        return checkpoints.restore_checkpoint(
            ckpt_dir=self.config.checkpoint_dir,
            target=state,
            step=step
        )


class JAXFlaxIntegration(BaseEnhancement):
    """JAX/Flax integration for MLE-Star"""
    
    def _default_config(self) -> EnhancementConfig:
        return EnhancementConfig(
            name="jax_flax_integration",
            version="1.0.0",
            enabled=JAX_AVAILABLE,
            priority=50,
            parameters={
                "enable_x64": False,
                "jit_compilation": True,
                "platform": "gpu",
                "distributed": False,
                "memory_fraction": None,
                "preallocate_memory": False,
                "gradient_accumulation_steps": 1,
                "gradient_clipping": None,
                "checkpoint_dir": "checkpoints",
                "save_every_steps": 1000,
                "keep_checkpoints": 5
            }
        )
    
    def initialize(self) -> bool:
        """Initialize JAX/Flax integration"""
        if not JAX_AVAILABLE:
            self._logger.error("JAX/Flax not available. Install with: pip install jax[gpu] flax optax")
            return False
        
        try:
            # Create JAX/Flax configuration
            self.jax_config = JAXFlaxConfig(**self.config.parameters)
            
            # Initialize trainer
            self.trainer = JAXFlaxTrainer(self.jax_config)
            
            self._logger.info("JAX/Flax integration initialized successfully")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to initialize JAX/Flax: {e}")
            return False
    
    def enhance_workflow(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance workflow with JAX/Flax capabilities"""
        enhanced = workflow.copy()
        
        # Add JAX/Flax specific configuration
        if 'model' not in enhanced:
            enhanced['model'] = {}
        
        # Add JAX framework option
        if 'framework_options' not in enhanced['model']:
            enhanced['model']['framework_options'] = {}
        
        enhanced['model']['framework_options']['jax_flax'] = {
            'enabled': True,
            'model_class': 'JAXFlaxModel',
            'trainer_class': 'JAXFlaxTrainer',
            'config': self.jax_config.__dict__
        }
        
        # Enhance training configuration
        if 'training' not in enhanced:
            enhanced['training'] = {}
        
        enhanced['training']['jax_flax_options'] = {
            'jit_compilation': self.jax_config.jit_compilation,
            'distributed': self.jax_config.distributed,
            'gradient_accumulation_steps': self.jax_config.gradient_accumulation_steps,
            'gradient_clipping': self.jax_config.gradient_clipping
        }
        
        # Add JAX-specific evaluation metrics
        if 'evaluation' not in enhanced:
            enhanced['evaluation'] = {}
        
        if 'jax_metrics' not in enhanced['evaluation']:
            enhanced['evaluation']['jax_metrics'] = [
                'compilation_time',
                'memory_usage',
                'device_utilization',
                'gradient_norm'
            ]
        
        # Add checkpoint configuration
        if 'checkpointing' not in enhanced:
            enhanced['checkpointing'] = {}
        
        enhanced['checkpointing']['jax_flax'] = {
            'checkpoint_dir': self.jax_config.checkpoint_dir,
            'save_every_steps': self.jax_config.save_every_steps,
            'keep_checkpoints': self.jax_config.keep_checkpoints,
            'format': 'flax_msgpack'
        }
        
        # Add MLE-Star stage enhancements
        if 'mle_star_workflow' in enhanced:
            stages = enhanced['mle_star_workflow'].get('stages', {})
            
            # Stage 3: Action Planning - Add JAX considerations
            if '3_action_planning' in stages:
                stages['3_action_planning']['jax_considerations'] = [
                    'functional_programming_paradigm',
                    'jit_compilation_requirements',
                    'memory_layout_optimization',
                    'device_placement_strategy'
                ]
            
            # Stage 4: Implementation - Add JAX-specific outputs
            if '4_implementation' in stages:
                if 'outputs' not in stages['4_implementation']:
                    stages['4_implementation']['outputs'] = []
                stages['4_implementation']['outputs'].extend([
                    'jax_model_definition',
                    'training_state_management',
                    'jit_compiled_functions'
                ])
            
            # Stage 6: Refinement - Add JAX optimization techniques
            if '6_refinement' in stages:
                if 'jax_optimizations' not in stages['6_refinement']:
                    stages['6_refinement']['jax_optimizations'] = [
                        'xla_optimization',
                        'memory_efficient_training',
                        'gradient_accumulation',
                        'distributed_training'
                    ]
        
        self._logger.debug("Enhanced workflow with JAX/Flax capabilities")
        return enhanced
    
    def create_model_template(self, task_type: str, **kwargs) -> str:
        """Generate JAX/Flax model template code"""
        if task_type == "classification":
            return self._classification_template(**kwargs)
        elif task_type == "regression":
            return self._regression_template(**kwargs)
        else:
            return self._generic_template(**kwargs)
    
    def _classification_template(self, num_classes: int = 10, hidden_dims: List[int] = None) -> str:
        """Generate classification model template"""
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        features = hidden_dims + [num_classes]
        
        return f"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class ClassificationModel(nn.Module):
    features: Sequence[int] = {features}
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat, name=f'dense_{{i}}')(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not training
                )
        
        # Output layer
        x = nn.Dense(self.features[-1], name='output')(x)
        return x

# Usage example:
model = ClassificationModel()
key = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 784))  # Adjust input dimension
params = model.init(key, dummy_input, training=False)
"""
    
    def _regression_template(self, output_dim: int = 1, hidden_dims: List[int] = None) -> str:
        """Generate regression model template"""
        if hidden_dims is None:
            hidden_dims = [128, 64]
        
        features = hidden_dims + [output_dim]
        
        return f"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class RegressionModel(nn.Module):
    features: Sequence[int] = {features}
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat, name=f'dense_{{i}}')(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not training
                )
        
        # Output layer (no activation for regression)
        x = nn.Dense(self.features[-1], name='output')(x)
        return x

# Usage example:
model = RegressionModel()
key = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 10))  # Adjust input dimension
params = model.init(key, dummy_input, training=False)
"""
    
    def _generic_template(self, **kwargs) -> str:
        """Generate generic model template"""
        return """
import jax
import jax.numpy as jnp
import flax.linen as nn

class GenericModel(nn.Module):
    features: list
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = False):
        # Implement your model architecture here
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
            if training:
                x = nn.Dropout(rate=self.dropout_rate)(
                    x, deterministic=not training
                )
        
        return x

# Customize this template for your specific use case
"""