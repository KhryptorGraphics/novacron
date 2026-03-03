#!/usr/bin/env python3
"""Test script to validate MLE-Star sample projects setup."""

import os
import sys
import importlib
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

def test_project_structure() -> Tuple[bool, List[str]]:
    """Test that all required directories and files exist."""
    
    base_dir = Path(__file__).parent
    errors = []
    
    # Test main structure
    required_files = [
        'README.md',
        'requirements.txt',
        'mle_star_workflow_runner.py'
    ]
    
    for file_path in required_files:
        if not (base_dir / file_path).exists():
            errors.append(f"Missing main file: {file_path}")
    
    # Test project structures
    projects = ['computer-vision', 'nlp', 'tabular-data']
    
    for project in projects:
        project_dir = base_dir / project
        
        if not project_dir.exists():
            errors.append(f"Missing project directory: {project}")
            continue
            
        # Check required subdirectories
        required_dirs = ['src', 'config', 'data', 'models', 'notebooks', 'tests', 'docs']
        for dir_name in required_dirs:
            if not (project_dir / dir_name).exists():
                errors.append(f"Missing directory: {project}/{dir_name}")
        
        # Check config file
        config_file = project_dir / 'config' / 'mle_star_config.yaml'
        if not config_file.exists():
            errors.append(f"Missing config file: {project}/config/mle_star_config.yaml")
        
        # Check key source files
        if project == 'computer-vision':
            src_files = ['data_loader.py', 'model.py', 'trainer.py', 'evaluator.py']
        elif project == 'nlp':
            src_files = ['data_processor.py', 'bert_model.py', 'trainer.py']
        elif project == 'tabular-data':
            src_files = ['data_preprocessor.py', 'model_ensemble.py']
        
        for src_file in src_files:
            if not (project_dir / 'src' / src_file).exists():
                errors.append(f"Missing source file: {project}/src/{src_file}")
    
    return len(errors) == 0, errors

def test_configuration_files() -> Tuple[bool, List[str]]:
    """Test that configuration files are valid YAML and contain required sections."""
    
    base_dir = Path(__file__).parent
    errors = []
    
    projects = ['computer-vision', 'nlp', 'tabular-data']
    required_sections = [
        'project',
        'mle_star_workflow',
        'data',
        'model',
        'training',
        'evaluation'
    ]
    
    for project in projects:
        config_file = base_dir / project / 'config' / 'mle_star_config.yaml'
        
        if not config_file.exists():
            errors.append(f"Config file not found: {project}")
            continue
            
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            for section in required_sections:
                if section not in config:
                    errors.append(f"Missing config section '{section}' in {project}")
            
            # Check MLE-Star workflow stages
            if 'mle_star_workflow' in config and 'stages' in config['mle_star_workflow']:
                workflow_stages = config['mle_star_workflow']['stages']
                expected_stages = [
                    '1_situation_analysis',
                    '2_task_definition', 
                    '3_action_planning',
                    '4_implementation',
                    '5_results_evaluation',
                    '6_refinement',
                    '7_deployment_prep'
                ]
                
                for stage in expected_stages:
                    if stage not in workflow_stages:
                        errors.append(f"Missing workflow stage '{stage}' in {project}")
            
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML in {project} config: {e}")
        except Exception as e:
            errors.append(f"Error reading {project} config: {e}")
    
    return len(errors) == 0, errors

def test_python_imports() -> Tuple[bool, List[str]]:
    """Test that Python modules can be imported without errors."""
    
    base_dir = Path(__file__).parent
    errors = []
    
    # Add project source directories to path
    sys.path.append(str(base_dir / 'computer-vision' / 'src'))
    sys.path.append(str(base_dir / 'nlp' / 'src'))
    sys.path.append(str(base_dir / 'tabular-data' / 'src'))
    
    # Test imports
    test_imports = [
        # Computer Vision
        ('data_loader', 'CIFAR10DataLoader'),
        ('model', 'CIFAR10CNN'),
        ('trainer', 'CIFAR10Trainer'),
        ('evaluator', 'ModelEvaluator'),
        
        # NLP
        ('data_processor', 'SentimentDataLoader'),
        ('bert_model', 'SentimentBERT'),
        ('trainer', 'SentimentTrainer'),
        
        # Tabular Data
        ('data_preprocessor', 'TabularDataPreprocessor'),
        ('model_ensemble', 'TabularEnsemble'),
    ]
    
    for module_name, class_name in test_imports:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                # Try to instantiate (basic test)
                if class_name == 'CIFAR10DataLoader':
                    obj = getattr(module, class_name)()
                elif class_name == 'CIFAR10CNN':
                    obj = getattr(module, class_name)(num_classes=10)
                elif class_name == 'SentimentDataLoader':
                    obj = getattr(module, class_name)()
                elif class_name == 'SentimentBERT':
                    # Skip BERT instantiation as it requires downloading model
                    pass
                elif class_name == 'TabularDataPreprocessor':
                    obj = getattr(module, class_name)('target', 'classification')
                elif class_name == 'TabularEnsemble':
                    obj = getattr(module, class_name)('classification')
                else:
                    obj = getattr(module, class_name)()
                    
            else:
                errors.append(f"Class {class_name} not found in module {module_name}")
                
        except ImportError as e:
            errors.append(f"Cannot import {module_name}: {e}")
        except Exception as e:
            errors.append(f"Error testing {module_name}.{class_name}: {e}")
    
    return len(errors) == 0, errors

def test_workflow_runner() -> Tuple[bool, List[str]]:
    """Test the main workflow runner."""
    
    base_dir = Path(__file__).parent
    errors = []
    
    try:
        # Import workflow runner
        sys.path.append(str(base_dir))
        from mle_star_workflow_runner import MLEStarWorkflowRunner
        
        # Test initialization
        runner = MLEStarWorkflowRunner(base_dir)
        
        # Test configuration loading
        if len(runner.configs) != 3:
            errors.append(f"Expected 3 project configs, got {len(runner.configs)}")
        
        # Test project validation
        for project in ['computer-vision', 'nlp', 'tabular-data']:
            if not runner.validate_project_setup(project):
                errors.append(f"Project setup validation failed for {project}")
        
        # Test stage execution (situation analysis only)
        try:
            stage_config = {'description': 'Test stage'}
            result = runner.execute_stage('computer-vision', '1_situation_analysis', stage_config)
            
            if not isinstance(result, dict):
                errors.append("Stage execution should return dictionary")
            elif 'status' not in result:
                errors.append("Stage result missing 'status' field")
                
        except Exception as e:
            errors.append(f"Stage execution test failed: {e}")
            
    except ImportError as e:
        errors.append(f"Cannot import workflow runner: {e}")
    except Exception as e:
        errors.append(f"Workflow runner test failed: {e}")
    
    return len(errors) == 0, errors

def test_sample_data_generation() -> Tuple[bool, List[str]]:
    """Test synthetic data generation for each project."""
    
    errors = []
    
    # Test Computer Vision data loading
    try:
        from data_loader import CIFAR10DataLoader, analyze_data_situation
        
        loader = CIFAR10DataLoader()
        stats = loader.get_data_stats()
        
        if stats['num_classes'] != 10:
            errors.append(f"CIFAR-10 should have 10 classes, got {stats['num_classes']}")
            
        # Test situation analysis
        analysis = analyze_data_situation()
        if 'problem_type' not in analysis:
            errors.append("CV situation analysis missing problem_type")
            
    except Exception as e:
        errors.append(f"CV data generation test failed: {e}")
    
    # Test NLP data processing
    try:
        from data_processor import SentimentDataLoader, analyze_nlp_situation
        
        loader = SentimentDataLoader()
        df, stats = loader.load_imdb_data()
        
        if len(df) == 0:
            errors.append("NLP synthetic dataset is empty")
            
        # Test situation analysis
        analysis = analyze_nlp_situation()
        if 'problem_type' not in analysis:
            errors.append("NLP situation analysis missing problem_type")
            
    except Exception as e:
        errors.append(f"NLP data generation test failed: {e}")
    
    # Test Tabular data preprocessing
    try:
        from data_preprocessor import TabularDataPreprocessor, analyze_tabular_situation
        
        preprocessor = TabularDataPreprocessor('target', 'classification')
        df = preprocessor.create_synthetic_dataset(n_samples=100)
        
        if len(df) != 100:
            errors.append(f"Tabular dataset should have 100 samples, got {len(df)}")
            
        # Test situation analysis
        analysis = analyze_tabular_situation(df, 'classification')
        if 'problem_type' not in analysis:
            errors.append("Tabular situation analysis missing problem_type")
            
    except Exception as e:
        errors.append(f"Tabular data generation test failed: {e}")
    
    return len(errors) == 0, errors

def run_all_tests() -> Dict[str, Any]:
    """Run all validation tests and return results."""
    
    print("\n=== MLE-Star Sample Projects Validation ===")
    print("Running comprehensive setup tests...\n")
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Configuration Files", test_configuration_files),
        ("Python Imports", test_python_imports),
        ("Workflow Runner", test_workflow_runner),
        ("Sample Data Generation", test_sample_data_generation)
    ]
    
    results = {
        'tests_run': len(tests),
        'tests_passed': 0,
        'tests_failed': 0,
        'details': {},
        'overall_success': False
    }
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        
        try:
            success, errors = test_func()
            
            results['details'][test_name] = {
                'success': success,
                'errors': errors
            }
            
            if success:
                results['tests_passed'] += 1
                print(f"  ✅ {test_name}: PASSED")
            else:
                results['tests_failed'] += 1
                print(f"  ❌ {test_name}: FAILED")
                for error in errors:
                    print(f"     - {error}")
                    
        except Exception as e:
            results['tests_failed'] += 1
            results['details'][test_name] = {
                'success': False,
                'errors': [f"Test execution failed: {e}"]
            }
            print(f"  ❌ {test_name}: ERROR - {e}")
        
        print()
    
    results['overall_success'] = results['tests_failed'] == 0
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Tests Passed: {results['tests_passed']}")
    print(f"Tests Failed: {results['tests_failed']}")
    print(f"Success Rate: {(results['tests_passed']/results['tests_run']*100):.1f}%")
    
    if results['overall_success']:
        print("\n🎉 All tests passed! MLE-Star sample projects are ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
    
    return results

if __name__ == '__main__':
    results = run_all_tests()
    
    # Save results
    results_file = Path(__file__).parent / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)
