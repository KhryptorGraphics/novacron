#!/usr/bin/env python3
"""
DWCP v3 Benchmark Results Extraction Script
Parses Go benchmark output and generates structured summary
"""

import re
import json
import sys
from collections import defaultdict
from typing import Dict, List, Any

def parse_benchmark_line(line: str) -> Dict[str, Any]:
    """Parse a single benchmark result line"""
    # Pattern: BenchmarkName-96    iterations    ns/op    custom_metrics    B/op    allocs/op
    pattern = r'(Benchmark[\w/]+)-\d+\s+(\d+)\s+([\d.]+)\s+(\w+/op)(.*)$'
    match = re.match(pattern, line)

    if not match:
        return None

    name = match.group(1)
    iterations = int(match.group(2))
    time_value = float(match.group(3))
    time_unit = match.group(4).replace('/op', '')
    custom_metrics = match.group(5).strip()

    result = {
        'name': name,
        'iterations': iterations,
        'time': time_value,
        'time_unit': time_unit,
        'custom_metrics': {}
    }

    # Parse custom metrics (e.g., "226.1 GB/s    1.000 streams    0 B/op")
    metric_pattern = r'([\d.]+)\s+(\S+)'
    for match in re.finditer(metric_pattern, custom_metrics):
        value = float(match.group(1))
        unit = match.group(2)
        result['custom_metrics'][unit] = value

    return result

def categorize_benchmarks(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Categorize benchmarks by component"""
    categories = defaultdict(list)

    for result in results:
        name = result['name']
        if 'AMST' in name:
            categories['AMST'].append(result)
        elif 'HDE' in name:
            categories['HDE'].append(result)
        elif 'PBA' in name:
            categories['PBA'].append(result)
        elif 'ASS' in name or 'ACP' in name:
            categories['ASS_ACP'].append(result)
        elif 'ITP' in name:
            categories['ITP'].append(result)
        elif 'Migration' in name or 'VMMigration' in name:
            categories['Migration'].append(result)
        elif 'Scalability' in name:
            categories['Scalability'].append(result)
        elif 'Competitor' in name or 'DWCP' in name or 'VMware' in name or 'HyperV' in name or 'KVM' in name:
            categories['Competitor'].append(result)
        elif 'Stress' in name:
            categories['Stress'].append(result)
        else:
            categories['Other'].append(result)

    return dict(categories)

def generate_summary(categories: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Generate summary statistics"""
    summary = {}

    for category, results in categories.items():
        summary[category] = {
            'total_benchmarks': len(results),
            'key_results': []
        }

        # Extract key metrics for each category
        for result in results[:10]:  # Top 10 per category
            summary[category]['key_results'].append({
                'name': result['name'].replace('Benchmark', ''),
                'time': f"{result['time']} {result['time_unit']}",
                'custom_metrics': result['custom_metrics']
            })

    return summary

def main():
    if len(sys.argv) < 2:
        print("Usage: extract-benchmark-summary.py <benchmark_results_file>")
        sys.exit(1)

    results_file = sys.argv[1]

    all_results = []

    with open(results_file, 'r') as f:
        for line in f:
            if line.startswith('Benchmark'):
                result = parse_benchmark_line(line)
                if result:
                    all_results.append(result)

    # Categorize and summarize
    categories = categorize_benchmarks(all_results)
    summary = generate_summary(categories)

    # Print JSON summary
    print(json.dumps(summary, indent=2))

    # Print human-readable summary
    print("\n" + "="*80)
    print("DWCP v3 Benchmark Summary")
    print("="*80)

    for category, data in summary.items():
        print(f"\n{category}: {data['total_benchmarks']} benchmarks")
        print("-" * 40)
        for result in data['key_results'][:5]:
            print(f"  {result['name'][:60]}")
            print(f"    Time: {result['time']}")
            if result['custom_metrics']:
                for metric, value in result['custom_metrics'].items():
                    print(f"    {metric}: {value}")

if __name__ == '__main__':
    main()
