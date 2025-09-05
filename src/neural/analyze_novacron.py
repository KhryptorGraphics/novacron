#!/usr/bin/env python3
"""
Run Neural Pattern Analysis on NovaCron Codebase
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pattern_analyzer_part2 import NeuralPatternAnalyzer
import json
from datetime import datetime

def analyze_novacron():
    """Analyze the NovaCron codebase for patterns"""
    
    print("=" * 80)
    print("NovaCron v10 Neural Pattern Analysis")
    print("=" * 80)
    
    # Initialize analyzer
    analyzer = NeuralPatternAnalyzer()
    
    # Define directories to analyze
    directories = [
        "/home/kp/novacron/backend",
        "/home/kp/novacron/frontend/src",
        "/home/kp/novacron/src"
    ]
    
    all_results = []
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"\nAnalyzing: {directory}")
            print("-" * 40)
            
            # Analyze directory
            results = analyzer.analyze_directory(directory, extensions=['.go', '.js', '.ts', '.py'])
            
            # Print summary for this directory
            print(f"Files analyzed: {results['metrics']['files_analyzed']}")
            print(f"Patterns found: {results['metrics']['total_patterns_detected']}")
            print(f"Anti-patterns: {results['metrics']['anti_patterns_found']}")
            
            all_results.append(results)
    
    # Train neural models
    print("\n" + "=" * 80)
    print("Training Neural Models")
    print("=" * 80)
    analyzer.train_models()
    
    # Generate comprehensive report
    final_report = analyzer.generate_report()
    
    # Save reports
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"/home/kp/novacron/docs/neural_analysis_report_{timestamp}.json"
    knowledge_file = f"/home/kp/novacron/src/neural/knowledge_base_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2)
    
    analyzer.save_knowledge_base(knowledge_file)
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("NEURAL PATTERN ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Overall Metrics:")
    print(f"   Total Files Analyzed: {final_report['summary']['files_analyzed']}")
    print(f"   Total Patterns Detected: {final_report['summary']['total_patterns']}")
    print(f"   Anti-patterns Found: {final_report['summary']['anti_patterns']}")
    
    print(f"\n🎯 Quality Scores:")
    print(f"   Code Quality: {final_report['summary']['quality_score']}")
    print(f"   Security: {final_report['summary']['security_score']}")
    print(f"   Performance: {final_report['summary']['performance_score']}")
    
    print(f"\n📈 Pattern Categories:")
    for category, data in final_report['categories'].items():
        print(f"   {category}:")
        print(f"      Total: {data['total']}")
        print(f"      Anti-patterns: {data['anti_patterns']}")
        print(f"      Unique: {data['unique_patterns']}")
        if data['top_patterns']:
            print(f"      Top Pattern: {data['top_patterns'][0][0]} ({data['top_patterns'][0][1]} occurrences)")
    
    print(f"\n🧠 Neural Model Accuracy:")
    for model, accuracy in final_report['model_accuracy'].items():
        print(f"   {model}: {accuracy:.2%}")
    
    print(f"\n💡 Key Learnings:")
    for category, learnings in final_report['learnings'].items():
        if learnings:
            print(f"   {category}:")
            for learning in learnings[:2]:  # Show top 2 learnings per category
                print(f"      - {learning}")
    
    print(f"\n⚠️  Top Improvements Needed:")
    for i, improvement in enumerate(final_report['improvements'][:5], 1):
        print(f"   {i}. [{improvement['priority'].upper()}] {improvement['description']}")
        print(f"      File: {improvement['file']}:{improvement['line']}")
        if improvement['recommendations']:
            print(f"      Fix: {improvement['recommendations'][0]}")
    
    print(f"\n🚀 Pattern Propagation Strategy:")
    propagation = final_report['propagation_strategy']
    print(f"   Patterns to propagate: {len(propagation['patterns_to_propagate'])}")
    print(f"   Files that could benefit: {propagation['estimated_impact']['files_affected']}")
    print(f"   Estimated quality improvement: {propagation['estimated_impact']['quality_improvement']:.0%}")
    print(f"   Estimated security improvement: {propagation['estimated_impact']['security_improvement']:.0%}")
    print(f"   Estimated performance improvement: {propagation['estimated_impact']['performance_improvement']:.0%}")
    
    if propagation['patterns_to_propagate']:
        print(f"\n   Best Patterns to Propagate:")
        for pattern_info in propagation['patterns_to_propagate'][:3]:
            print(f"      - {pattern_info['category']}: {pattern_info['pattern']}")
            print(f"        Security: {pattern_info['benefits']['security']:.2f}, "
                  f"Performance: {pattern_info['benefits']['performance']:.2f}, "
                  f"Maintainability: {pattern_info['benefits']['maintainability']:.2f}")
    
    print(f"\n📁 Reports saved:")
    print(f"   Full Report: {report_file}")
    print(f"   Knowledge Base: {knowledge_file}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    
    return final_report

if __name__ == "__main__":
    analyze_novacron()