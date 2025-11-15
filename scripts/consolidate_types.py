#!/usr/bin/env python3
"""
Phase 1: Consolidate Type Redeclarations
Fixes all duplicate type definitions across packages
"""

import os
import re
from pathlib import Path

# Type redeclarations to fix
TYPE_CONSOLIDATIONS = {
    # backend/community/marketplace
    'backend/community/marketplace': {
        'remove_from': {
            'marketplace_scale_v2.go': [
                ('AppCategory', 138),
                ('Permission', 198),
                ('APIEndpoint', 207),
                ('UserProfile', 358),
                ('PricingModel', 414),
                ('PriceTier', 424),
                ('PayoutEngine', 462),
                ('EnterpriseMarketplace', 583),
                ('EnterpriseApp', 600),
            ],
            'app_store.go': [
                ('PricingModel', 46),
            ],
        },
        'keep_in': 'app_engine_v2.go',  # Primary source
    },

    # backend/community/opensource
    'backend/community/opensource': {
        'remove_from': {
            'opensource_leadership.go': [
                ('Contributor', 181),
                ('ContributionType', 228),
                ('Reward', 446),
                ('Badge', 455),
                ('CommunityGovernance', 467),
                ('Proposal', 516),
                ('Vote', 535),
                ('Vulnerability', 715),
            ],
        },
        'keep_in': 'contribution_platform.go',  # Primary source
    },

    # backend/community/certification
    'backend/community/certification': {
        'remove_from': {
            'advanced_cert.go': [
                ('DeveloperProfile', 46),
                ('ResourceSpec', 106),
            ],
        },
        'keep_in': 'acceleration.go',  # Primary source
    },
}

def find_type_definition(filepath, typename, approx_line):
    """Find the exact range of a type definition"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Search around the approximate line
    search_start = max(0, approx_line - 5)
    search_end = min(len(lines), approx_line + 100)

    type_start = None
    brace_count = 0
    type_end = None

    for i in range(search_start, search_end):
        line = lines[i]

        # Find type declaration
        if type_start is None and f'type {typename}' in line:
            type_start = i
            if '{' in line:
                brace_count = 1
            continue

        if type_start is not None:
            brace_count += line.count('{') - line.count('}')

            if brace_count == 0:
                type_end = i + 1
                break

    return type_start, type_end

def remove_type_definitions(filepath, types_to_remove):
    """Remove type definitions from a file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Collect all ranges to remove
    ranges_to_remove = []
    for typename, approx_line in types_to_remove:
        start, end = find_type_definition(filepath, typename, approx_line)
        if start is not None and end is not None:
            ranges_to_remove.append((start, end, typename))
            print(f"  Found {typename} at lines {start+1}-{end+1}")

    # Sort ranges in reverse order to remove from bottom up
    ranges_to_remove.sort(reverse=True)

    # Remove the ranges
    for start, end, typename in ranges_to_remove:
        print(f"  Removing {typename} (lines {start+1}-{end+1})")
        del lines[start:end]

    # Write back
    with open(filepath, 'w') as f:
        f.writelines(lines)

    return len(ranges_to_remove)

def consolidate_package(package_path, config):
    """Consolidate types in a package"""
    base_dir = Path('/home/kp/repos/novacron')
    package_dir = base_dir / package_path

    print(f"\n=== Processing {package_path} ===")
    print(f"Primary source: {config['keep_in']}")

    total_removed = 0
    for filename, types_list in config['remove_from'].items():
        filepath = package_dir / filename
        if not filepath.exists():
            print(f"  Warning: {filepath} not found")
            continue

        print(f"\n  Cleaning {filename}...")
        removed = remove_type_definitions(filepath, types_list)
        total_removed += removed

    return total_removed

def main():
    print("=" * 70)
    print("PHASE 1: TYPE REDECLARATION CONSOLIDATION")
    print("=" * 70)

    total_fixed = 0

    for package_path, config in TYPE_CONSOLIDATIONS.items():
        fixed = consolidate_package(package_path, config)
        total_fixed += fixed

    print("\n" + "=" * 70)
    print(f"COMPLETED: Removed {total_fixed} duplicate type definitions")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    exit(main())
