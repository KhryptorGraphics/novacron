#!/usr/bin/env python3
"""
{{experimentName}} - Main Entry Point
Generated with MLE-Star methodology
Framework: {{mlFramework}}
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Import project modules
sys.path.append(str(Path(__file__).parent))

try:
    from data.data_pipeline import DataPipeline
    from models.model import MLModel
    from models.train_optimize import ModelTrainer
    from models.evaluation import ModelEvaluator
except ImportError as e:
    logging.warning(f"Some modules not yet implemented: {e}")
    logging.info("Run MLE-Star stages to generate missing components")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration: {e}")
        return None

def setup_environment(config):
    """Setup environment based on configuration"""
    import os
    import random
    import numpy as np
    
    # Set random seeds for reproducibility
    if config.get('environment', {}).get('set_random_seed', True):
        seed = config.get('data', {}).get('random_seed', 42)
        random.seed(seed)
        np.random.seed(seed)
        
        # Framework-specific random seeds
        if '{{mlFramework}}' == 'pytorch':
            try:
                import torch
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            except ImportError:
                pass
        elif '{{mlFramework}}' == 'tensorflow':
            try:
                import tensorflow as tf
                tf.random.set_seed(seed)
            except ImportError:
                pass
    
    # Create output directories
    output_paths = [
        config.get('paths', {}).get('output_root', './outputs'),
        config.get('model', {}).get('model_save_path', './outputs/models'),
        config.get('logging', {}).get('log_dir', './logs'),
        config.get('visualization', {}).get('plot_dir', './outputs/figures')
    ]
    
    for path in output_paths:
        Path(path).mkdir(parents=True, exist_ok=True)

def run_training(config):
    """Run model training pipeline"""
    logger.info("Starting training pipeline...")
    
    try:
        # Initialize components
        data_pipeline = DataPipeline(config)
        model = MLModel(config)
        trainer = ModelTrainer(config)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        train_data, val_data, test_data = data_pipeline.prepare_data()
        
        # Train model
        logger.info("Training model...")
        trained_model = trainer.train(model, train_data, val_data)
        
        # Save model
        model_path = Path(config['model']['model_save_path']) / 'trained_model'
        trainer.save_model(trained_model, model_path)
        logger.info(f"Model saved to: {model_path}")
        
        return trained_model
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def run_evaluation(config, model_path=None):
    """Run model evaluation"""
    logger.info("Starting model evaluation...")
    
    try:
        # Initialize components
        data_pipeline = DataPipeline(config)
        evaluator = ModelEvaluator(config)
        
        # Load test data
        logger.info("Loading test data...")
        _, _, test_data = data_pipeline.prepare_data()
        
        # Load model
        if model_path:
            model = evaluator.load_model(model_path)
        else:
            model_path = Path(config['model']['model_save_path']) / 'trained_model'
            model = evaluator.load_model(model_path)
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = evaluator.evaluate(model, test_data)
        
        # Save results
        results_path = Path(config['paths']['output_root']) / 'evaluation_results.yaml'
        with open(results_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        logger.info(f"Evaluation results saved to: {results_path}")
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def run_inference(config, model_path, data_path, output_path):
    """Run inference on new data"""
    logger.info("Starting inference...")
    
    try:
        # Initialize components
        evaluator = ModelEvaluator(config)
        
        # Load model
        model = evaluator.load_model(model_path)
        
        # Load inference data
        # Implementation depends on data format
        logger.info(f"Loading inference data from: {data_path}")
        
        # Run inference
        logger.info("Running inference...")
        # predictions = model.predict(inference_data)
        
        # Save predictions
        logger.info(f"Saving predictions to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='{{experimentName}} - MLE-Star ML Project'
    )
    parser.add_argument(
        'command',
        choices=['train', 'evaluate', 'inference', 'validate'],
        help='Command to execute'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to trained model (for evaluation/inference)'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        help='Path to inference data'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        help='Path for output results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        sys.exit(1)
    
    # Setup environment
    setup_environment(config)
    
    try:
        if args.command == 'train':
            logger.info("Executing training command...")
            model = run_training(config)
            logger.info("Training completed successfully!")
            
        elif args.command == 'evaluate':
            logger.info("Executing evaluation command...")
            results = run_evaluation(config, args.model_path)
            logger.info("Evaluation completed successfully!")
            logger.info(f"Results: {results}")
            
        elif args.command == 'inference':
            if not args.model_path or not args.data_path or not args.output_path:
                logger.error("inference requires --model-path, --data-path, and --output-path")
                sys.exit(1)
            
            logger.info("Executing inference command...")
            success = run_inference(config, args.model_path, args.data_path, args.output_path)
            if success:
                logger.info("Inference completed successfully!")
                
        elif args.command == 'validate':
            logger.info("Validating project setup...")
            
            # Check required directories
            required_dirs = ['data', 'models', 'outputs', 'configs']
            for dir_name in required_dirs:
                if not Path(dir_name).exists():
                    logger.error(f"Required directory missing: {dir_name}")
                else:
                    logger.info(f"✓ Directory exists: {dir_name}")
            
            # Check configuration
            logger.info(f"✓ Configuration loaded: {args.config}")
            
            # Check framework availability
            framework = config.get('model', {}).get('framework', '{{mlFramework}}')
            try:
                if framework == 'pytorch':
                    import torch
                    logger.info(f"✓ PyTorch available: {torch.__version__}")
                elif framework == 'tensorflow':
                    import tensorflow as tf
                    logger.info(f"✓ TensorFlow available: {tf.__version__}")
                elif framework == 'scikit-learn':
                    import sklearn
                    logger.info(f"✓ Scikit-learn available: {sklearn.__version__}")
            except ImportError:
                logger.error(f"✗ {framework} not available")
            
            logger.info("Validation completed!")
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()