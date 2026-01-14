#!/usr/bin/env python3
"""
Neural Pattern Analyzer - Part 2: Testing, Deployment and Main Analyzer
"""

from pattern_analyzer import *

class TestingPatternDetector(PatternDetector):
    """Detects testing patterns for 100% coverage"""
    
    def __init__(self):
        self.patterns_db = {
            'unit_tests': r'(test_|describe\(|it\(|@Test)',
            'integration_tests': r'(integration[_-]?test|e2e[_-]?test)',
            'mocking': r'(mock|stub|spy|fake)',
            'assertions': r'(assert|expect|should)',
            'test_fixtures': r'(fixture|setup|teardown|before|after)',
            'coverage': r'(coverage|cover|gcov|lcov)',
            'tdd': r'(test[_-]?driven|red[_-]?green[_-]?refactor)',
            'bdd': r'(given|when|then|feature|scenario)',
            'property_testing': r'(property|hypothesis|quickcheck)',
            'load_testing': r'(load[_-]?test|stress[_-]?test|performance[_-]?test)',
        }
        
    def detect(self, code: str, file_path: str) -> List[Pattern]:
        """Detect testing patterns"""
        patterns = []
        
        for pattern_name, regex in self.patterns_db.items():
            matches = re.finditer(regex, code, re.IGNORECASE)
            for match in matches:
                pattern = Pattern(
                    category='TESTING',
                    name=f"test_{pattern_name}",
                    description=f"Testing pattern: {pattern_name}",
                    file_path=file_path,
                    line_number=code[:match.start()].count('\n') + 1,
                    code_snippet=match.group(),
                    confidence=0.87,
                    performance_impact=0.0,
                    security_score=0.6,
                    maintainability_score=0.95
                )
                
                # Check for test coverage
                if 'test' in file_path.lower() or 'spec' in file_path.lower():
                    pattern.metadata['is_test_file'] = True
                    pattern.confidence = 0.95
                    
                patterns.append(pattern)
                
        return patterns
    
    def analyze(self, pattern: Pattern) -> Dict[str, Any]:
        """Analyze testing pattern"""
        analysis = {
            'coverage_impact': 0.0,
            'test_quality': 0.0,
            'recommendations': [],
            'metrics': {
                'estimated_coverage': 0.0,
                'test_types': []
            }
        }
        
        if 'unit_tests' in pattern.name:
            analysis['coverage_impact'] = 0.6
            analysis['test_quality'] = 0.85
            analysis['metrics']['test_types'].append('unit')
        elif 'integration_tests' in pattern.name:
            analysis['coverage_impact'] = 0.3
            analysis['test_quality'] = 0.9
            analysis['metrics']['test_types'].append('integration')
        elif 'mocking' in pattern.name:
            analysis['test_quality'] = 0.8
            analysis['recommendations'].append('Ensure mocks are properly maintained')
            
        # Estimate coverage based on patterns found
        if pattern.metadata.get('is_test_file'):
            analysis['metrics']['estimated_coverage'] = 85.0
        else:
            analysis['metrics']['estimated_coverage'] = 40.0
            analysis['recommendations'].append('Increase test coverage to reach 100% target')
            
        return analysis

class DeploymentPatternDetector(PatternDetector):
    """Detects deployment patterns for blue-green strategy"""
    
    def __init__(self):
        self.patterns_db = {
            'blue_green': r'(blue[_-]?green|canary|rolling[_-]?update)',
            'ci_cd': r'(ci[_-]?cd|continuous[_-]?integration|continuous[_-]?deployment)',
            'containerization': r'(docker|kubernetes|k8s|container)',
            'infrastructure_as_code': r'(terraform|cloudformation|ansible|puppet)',
            'monitoring': r'(prometheus|grafana|datadog|newrelic)',
            'logging': r'(elasticsearch|logstash|kibana|elk)',
            'secrets_management': r'(vault|secret[_-]?manager|kms)',
            'load_balancing': r'(load[_-]?balanc|nginx|haproxy|alb|elb)',
            'auto_scaling': r'(auto[_-]?scal|horizontal[_-]?pod|hpa)',
            'health_checks': r'(health[_-]?check|liveness|readiness)',
        }
        
    def detect(self, code: str, file_path: str) -> List[Pattern]:
        """Detect deployment patterns"""
        patterns = []
        
        for pattern_name, regex in self.patterns_db.items():
            matches = re.finditer(regex, code, re.IGNORECASE)
            for match in matches:
                pattern = Pattern(
                    category='DEPLOYMENT',
                    name=f"deploy_{pattern_name}",
                    description=f"Deployment pattern: {pattern_name}",
                    file_path=file_path,
                    line_number=code[:match.start()].count('\n') + 1,
                    code_snippet=match.group(),
                    confidence=0.86,
                    performance_impact=0.4,
                    security_score=0.8,
                    maintainability_score=0.85
                )
                
                # Special handling for blue-green deployment
                if 'blue_green' in pattern.name:
                    pattern.confidence = 0.95
                    pattern.metadata['deployment_strategy'] = 'blue-green'
                    
                patterns.append(pattern)
                
        return patterns
    
    def analyze(self, pattern: Pattern) -> Dict[str, Any]:
        """Analyze deployment pattern"""
        analysis = {
            'deployment_readiness': 0.0,
            'automation_level': 0.0,
            'recommendations': [],
            'best_practices': []
        }
        
        if 'blue_green' in pattern.name:
            analysis['deployment_readiness'] = 0.95
            analysis['automation_level'] = 0.9
            analysis['best_practices'].append('Blue-green deployment detected - excellent strategy')
        elif 'ci_cd' in pattern.name:
            analysis['deployment_readiness'] = 0.85
            analysis['automation_level'] = 0.85
            analysis['best_practices'].append('CI/CD pipeline in place')
        elif 'containerization' in pattern.name:
            analysis['deployment_readiness'] = 0.8
            analysis['best_practices'].append('Containerized deployment')
            
        if pattern.metadata.get('deployment_strategy') != 'blue-green':
            analysis['recommendations'].append('Consider implementing blue-green deployment')
            
        return analysis

class NeuralPatternAnalyzer:
    """Main neural pattern analyzer that coordinates all detectors"""
    
    def __init__(self):
        self.detectors = [
            APIPatternDetector(),
            ErrorHandlingPatternDetector(),
            PerformancePatternDetector(),
            SecurityPatternDetector(),
            TestingPatternDetector(),
            DeploymentPatternDetector()
        ]
        
        self.patterns_db: Dict[str, List[Pattern]] = defaultdict(list)
        self.neural_models: Dict[str, NeuralModel] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {
            'total_patterns_detected': 0,
            'anti_patterns_found': 0,
            'files_analyzed': 0,
            'quality_score': 0.0,
            'coverage_estimate': 0.0,
            'security_score': 0.0,
            'performance_score': 0.0,
        }
        
        # Initialize neural models for each category
        for category in PATTERN_CATEGORIES:
            self.neural_models[category] = NeuralModel(
                name=f"model_{category.lower()}",
                hyperparameters={
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100,
                    'hidden_layers': [128, 64, 32]
                }
            )
        
        self.logger = logging.getLogger(__name__)
        
    def analyze_file(self, file_path: str) -> List[Pattern]:
        """Analyze a single file for patterns"""
        patterns = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
                
            for detector in self.detectors:
                detected_patterns = detector.detect(code, file_path)
                patterns.extend(detected_patterns)
                
                # Analyze each pattern
                for pattern in detected_patterns:
                    analysis = detector.analyze(pattern)
                    pattern.metadata['analysis'] = analysis
                    
                    # Store in database
                    self.patterns_db[pattern.category].append(pattern)
                    
                    # Update metrics
                    if pattern.is_anti_pattern:
                        self.metrics['anti_patterns_found'] += 1
                        
            self.metrics['files_analyzed'] += 1
            self.metrics['total_patterns_detected'] += len(patterns)
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            
        return patterns
    
    def analyze_directory(self, directory: str, extensions: List[str] = None) -> Dict[str, Any]:
        """Analyze all files in a directory"""
        if extensions is None:
            extensions = ['.py', '.js', '.ts', '.go', '.java', '.cpp', '.c']
            
        all_patterns = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    patterns = self.analyze_file(file_path)
                    all_patterns.extend(patterns)
                    
        # Generate summary report
        report = self.generate_report()
        
        return {
            'patterns': all_patterns,
            'report': report,
            'metrics': self.metrics
        }
    
    def train_models(self):
        """Train neural models on detected patterns"""
        for category, patterns in self.patterns_db.items():
            if patterns and category in self.neural_models:
                # Prepare training data
                training_data = [
                    {
                        'pattern': pattern.to_dict(),
                        'quality_score': pattern.metadata.get('analysis', {}).get('quality_score', 0.5),
                        'is_anti_pattern': pattern.is_anti_pattern
                    }
                    for pattern in patterns
                ]
                
                # Train the model
                model = self.neural_models[category]
                model.train(training_data)
                
                # Store learned patterns
                model.patterns = patterns[:100]  # Keep top patterns
                
                self.logger.info(f"Trained model for {category} with {len(training_data)} samples")
    
    def learn_from_patterns(self) -> Dict[str, List[str]]:
        """Learn from detected patterns and generate recommendations"""
        learnings = defaultdict(list)
        
        for category, patterns in self.patterns_db.items():
            # Find most common patterns
            pattern_counts = Counter(p.name for p in patterns)
            common_patterns = pattern_counts.most_common(5)
            
            # Find anti-patterns
            anti_patterns = [p for p in patterns if p.is_anti_pattern]
            
            # Generate learnings
            if common_patterns:
                learnings[category].append(f"Most common patterns: {', '.join([p[0] for p in common_patterns])}")
                
            if anti_patterns:
                learnings[category].append(f"Found {len(anti_patterns)} anti-patterns that need fixing")
                
            # Category-specific learnings
            if category == 'SECURITY':
                vuln_count = sum(1 for p in patterns if 'vuln_' in p.name)
                if vuln_count > 0:
                    learnings[category].append(f"Critical: {vuln_count} security vulnerabilities detected")
                    
            elif category == 'PERFORMANCE':
                perf_patterns = [p for p in patterns if not p.is_anti_pattern]
                if perf_patterns:
                    avg_impact = np.mean([p.performance_impact for p in perf_patterns])
                    learnings[category].append(f"Average performance impact: {avg_impact:.2%}")
                    
            elif category == 'TESTING':
                test_files = sum(1 for p in patterns if p.metadata.get('is_test_file'))
                learnings[category].append(f"Test files analyzed: {test_files}")
                
            elif category == 'DEPLOYMENT':
                blue_green = any('blue_green' in p.name for p in patterns)
                if not blue_green:
                    learnings[category].append("Recommendation: Implement blue-green deployment")
                    
        return dict(learnings)
    
    def suggest_improvements(self) -> List[Dict[str, Any]]:
        """Suggest improvements based on learned patterns"""
        suggestions = []
        
        # Analyze anti-patterns
        for category, patterns in self.patterns_db.items():
            anti_patterns = [p for p in patterns if p.is_anti_pattern]
            
            for anti_pattern in anti_patterns:
                suggestion = {
                    'priority': 'high' if anti_pattern.security_score < 0.3 else 'medium',
                    'category': category,
                    'pattern': anti_pattern.name,
                    'file': anti_pattern.file_path,
                    'line': anti_pattern.line_number,
                    'description': anti_pattern.description,
                    'recommendations': anti_pattern.suggestions,
                    'impact': {
                        'performance': anti_pattern.performance_impact,
                        'security': anti_pattern.security_score,
                        'maintainability': anti_pattern.maintainability_score
                    }
                }
                suggestions.append(suggestion)
                
        # Sort by priority and impact
        suggestions.sort(key=lambda x: (
            0 if x['priority'] == 'high' else 1,
            -x['impact']['security']
        ))
        
        return suggestions[:50]  # Return top 50 suggestions
    
    def propagate_patterns(self) -> Dict[str, Any]:
        """Propagate successful patterns across the codebase"""
        propagation_strategy = {
            'patterns_to_propagate': [],
            'files_to_update': [],
            'estimated_impact': {}
        }
        
        # Find high-quality patterns
        for category, patterns in self.patterns_db.items():
            quality_patterns = [
                p for p in patterns 
                if not p.is_anti_pattern 
                and p.confidence > 0.85
                and p.security_score > 0.8
            ]
            
            if quality_patterns:
                # Select best pattern from each category
                best_pattern = max(quality_patterns, key=lambda p: (
                    p.confidence * p.security_score * p.maintainability_score
                ))
                
                propagation_strategy['patterns_to_propagate'].append({
                    'category': category,
                    'pattern': best_pattern.name,
                    'description': best_pattern.description,
                    'example': best_pattern.code_snippet,
                    'benefits': {
                        'security': best_pattern.security_score,
                        'performance': best_pattern.performance_impact,
                        'maintainability': best_pattern.maintainability_score
                    }
                })
                
                # Find files that could benefit from this pattern
                files_without_pattern = self._find_files_without_pattern(best_pattern)
                propagation_strategy['files_to_update'].extend(files_without_pattern)
                
        # Estimate impact
        propagation_strategy['estimated_impact'] = {
            'files_affected': len(set(propagation_strategy['files_to_update'])),
            'quality_improvement': 0.25,  # Conservative estimate
            'security_improvement': 0.3,
            'performance_improvement': 0.2
        }
        
        return propagation_strategy
    
    def _find_files_without_pattern(self, pattern: Pattern) -> List[str]:
        """Find files that could benefit from a pattern"""
        # This is a simplified version - in production, would do more sophisticated matching
        files = []
        
        # Look for files in the same category that don't have this pattern
        for p in self.patterns_db[pattern.category]:
            if p.file_path != pattern.file_path and p.name != pattern.name:
                if p.file_path not in files:
                    files.append(p.file_path)
                    
        return files[:10]  # Limit to 10 files for now
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive pattern recognition report"""
        
        # Calculate overall scores
        all_patterns = [p for patterns in self.patterns_db.values() for p in patterns]
        
        if all_patterns:
            self.metrics['quality_score'] = np.mean([
                p.maintainability_score for p in all_patterns if not p.is_anti_pattern
            ])
            self.metrics['security_score'] = np.mean([
                p.security_score for p in all_patterns
            ])
            self.metrics['performance_score'] = np.mean([
                p.performance_impact for p in all_patterns if p.performance_impact > 0
            ])
            
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'files_analyzed': self.metrics['files_analyzed'],
                'total_patterns': self.metrics['total_patterns_detected'],
                'anti_patterns': self.metrics['anti_patterns_found'],
                'quality_score': f"{self.metrics['quality_score']:.2%}",
                'security_score': f"{self.metrics['security_score']:.2%}",
                'performance_score': f"{self.metrics['performance_score']:.2%}",
            },
            'categories': {},
            'learnings': self.learn_from_patterns(),
            'improvements': self.suggest_improvements()[:10],  # Top 10 improvements
            'propagation_strategy': self.propagate_patterns(),
            'model_accuracy': {
                category: model.accuracy 
                for category, model in self.neural_models.items()
            }
        }
        
        # Category breakdown
        for category, patterns in self.patterns_db.items():
            report['categories'][category] = {
                'total': len(patterns),
                'anti_patterns': sum(1 for p in patterns if p.is_anti_pattern),
                'unique_patterns': len(set(p.name for p in patterns)),
                'top_patterns': Counter(p.name for p in patterns).most_common(5)
            }
            
        return report
    
    def save_knowledge_base(self, filepath: str):
        """Save the knowledge base to file"""
        knowledge = {
            'patterns': {cat: [p.to_dict() for p in patterns] 
                        for cat, patterns in self.patterns_db.items()},
            'models': {name: {'accuracy': model.accuracy, 
                            'last_trained': model.last_trained.isoformat() if model.last_trained else None}
                      for name, model in self.neural_models.items()},
            'metrics': self.metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge, f, indent=2)
            
        self.logger.info(f"Knowledge base saved to {filepath}")
    
    def load_knowledge_base(self, filepath: str):
        """Load knowledge base from file"""
        try:
            with open(filepath, 'r') as f:
                knowledge = json.load(f)
                
            self.metrics = knowledge.get('metrics', {})
            # Would also load patterns and models in production
            
            self.logger.info(f"Knowledge base loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading knowledge base: {e}")


def main():
    """Main entry point for neural pattern analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Pattern Analyzer for NovaCron')
    parser.add_argument('directory', help='Directory to analyze')
    parser.add_argument('--output', '-o', default='pattern_report.json', help='Output report file')
    parser.add_argument('--knowledge-base', '-k', default='knowledge_base.json', help='Knowledge base file')
    parser.add_argument('--train', action='store_true', help='Train neural models')
    parser.add_argument('--load-knowledge', action='store_true', help='Load existing knowledge base')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = NeuralPatternAnalyzer()
    
    # Load existing knowledge if requested
    if args.load_knowledge:
        analyzer.load_knowledge_base(args.knowledge_base)
    
    # Analyze directory
    print(f"Analyzing directory: {args.directory}")
    results = analyzer.analyze_directory(args.directory)
    
    # Train models if requested
    if args.train:
        print("Training neural models...")
        analyzer.train_models()
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results['report'], f, indent=2)
    
    # Save knowledge base
    analyzer.save_knowledge_base(args.knowledge_base)
    
    # Print summary
    print("\n=== Pattern Recognition Report ===")
    print(f"Files Analyzed: {results['metrics']['files_analyzed']}")
    print(f"Total Patterns: {results['metrics']['total_patterns_detected']}")
    print(f"Anti-patterns: {results['metrics']['anti_patterns_found']}")
    print(f"Quality Score: {results['metrics']['quality_score']:.2%}")
    print(f"Security Score: {results['metrics']['security_score']:.2%}")
    print(f"Performance Score: {results['metrics']['performance_score']:.2%}")
    
    # Print top improvements
    print("\n=== Top Improvements Needed ===")
    for i, improvement in enumerate(results['report']['improvements'][:5], 1):
        print(f"{i}. [{improvement['priority'].upper()}] {improvement['description']}")
        print(f"   File: {improvement['file']}:{improvement['line']}")
        if improvement['recommendations']:
            print(f"   Fix: {improvement['recommendations'][0]}")
    
    print(f"\nFull report saved to: {args.output}")
    print(f"Knowledge base saved to: {args.knowledge_base}")


if __name__ == "__main__":
    main()