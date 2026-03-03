#!/usr/bin/env python3
"""
Phase 4: Fix Network Package Issues
"""

import re
from pathlib import Path

FIXES = [
    # Fix UUID issues in OVS bridge manager
    {
        'file': 'backend/core/network/ovs/bridge_manager.go',
        'pattern': r'uuid\.New\(\)',
        'replacement': r'uuid.New().String()',
        'description': 'Fix UUID.New() to return string',
    },

    # Fix ONNX Runtime API calls
    {
        'file': 'backend/core/network/dwcp/prediction/lstm_bandwidth_predictor.go',
        'line': 103,
        'action': 'fix_onnx_run',
    },

    # Fix ONNX GetFloatData -> GetData
    {
        'file': 'backend/core/network/dwcp/prediction/lstm_bandwidth_predictor.go',
        'pattern': r'\.GetFloatData\(\)',
        'replacement': r'.GetData().([]float32)',
        'description': 'Fix ONNX GetFloatData method',
    },

    # Fix transport metrics Mode field
    {
        'file': 'backend/core/network/dwcp/v3/transport/amst_v3.go',
        'line': 529,
        'pattern': r'baseMetrics\.Mode',
        'replacement': r'baseMetrics.TransportMode',
        'description': 'Fix metrics Mode field name',
    },
]

def fix_uuid_issues():
    """Fix UUID type issues in OVS bridge manager"""
    filepath = Path('/home/kp/repos/novacron/backend/core/network/ovs/bridge_manager.go')

    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if uuid package is imported correctly
    if 'github.com/google/uuid' not in content:
        # Add import
        content = re.sub(
            r'(import \()',
            r'\1\n\t"github.com/google/uuid"',
            content
        )

    # Fix uuid.New() calls - uuid.New() returns UUID, need .String()
    content = re.sub(
        r'uuid\.New\(\)',
        r'uuid.New().String()',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)

    print("  Fixed UUID issues")
    return True

def fix_onnx_runtime():
    """Fix ONNX Runtime API mismatches"""
    filepath = Path('/home/kp/repos/novacron/backend/core/network/dwcp/prediction/lstm_bandwidth_predictor.go')

    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Fix line 103: session.Run now takes 2 parameters and returns 1
    for i, line in enumerate(lines):
        if 'p.session.Run' in line and i == 102:  # Line 103 (0-indexed)
            # Change from: output, err := p.session.Run(inputs)
            # To: outputs, err := p.session.Run(inputs, outputNames)
            lines[i] = re.sub(
                r'(\s*)output, err := p\.session\.Run\(inputs\)',
                r'\1outputs, err := p.session.Run(inputs, []string{"output"})',
                line
            )
            print(f"  Fixed ONNX Run call at line 103")

        # Fix GetFloatData -> GetData with type assertion
        if 'GetFloatData' in line:
            lines[i] = line.replace('.GetFloatData()', '.GetData().([]float32)')
            print(f"  Fixed GetFloatData at line {i+1}")

    with open(filepath, 'w') as f:
        f.writelines(lines)

    return True

def fix_transport_metrics():
    """Fix transport metrics Mode field"""
    filepath = Path('/home/kp/repos/novacron/backend/core/network/dwcp/v3/transport/amst_v3.go')

    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Fix: Mode field doesn't exist, use TransportMode or similar
    content = re.sub(
        r'baseMetrics\.Mode',
        r'baseMetrics.TransportMode',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)

    print("  Fixed transport metrics Mode field")
    return True

def remove_unused_imports():
    """Remove or comment unused imports"""
    files = [
        'backend/core/network/dwcp/v3/partition/heterogeneous_placement.go',
        'backend/core/network/dwcp/v3/partition/itp_v3.go',
        'backend/core/network/dwcp/prediction/types.go',
    ]

    for filepath_str in files:
        filepath = Path(f'/home/kp/repos/novacron/{filepath_str}')
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Comment out unused imports
        for i, line in enumerate(lines):
            if '"fmt" imported and not used' in line or \
               'imported and not used' in line:
                # Find the import line
                for j in range(max(0, i-5), i):
                    if 'import' in lines[j]:
                        lines[j] = '// ' + lines[j]
                        print(f"  Commented unused import in {filepath_str}")
                        break

        with open(filepath, 'w') as f:
            f.writelines(lines)

def main():
    print("=" * 70)
    print("PHASE 4: NETWORK PACKAGE FIXES")
    print("=" * 70)

    success = True

    print("\nFixing UUID issues...")
    success &= fix_uuid_issues()

    print("\nFixing ONNX Runtime API...")
    success &= fix_onnx_runtime()

    print("\nFixing transport metrics...")
    success &= fix_transport_metrics()

    print("\nRemoving unused imports...")
    remove_unused_imports()

    print("\n" + "=" * 70)
    print(f"COMPLETED: {'Success' if success else 'Partial success'}")
    print("=" * 70)

    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
