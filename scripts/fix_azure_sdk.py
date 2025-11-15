#!/usr/bin/env python3
"""
Phase 3: Fix Azure SDK API Changes
"""

import re
from pathlib import Path

def fix_azure_adapter():
    """Fix Azure SDK client instantiations and API calls"""
    filepath = Path('/home/kp/repos/novacron/adapters/pkg/azure/adapter.go')

    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    print("  Fixing Azure SDK client instantiations...")

    # Fix: Client constructors now return values, need to get address
    replacements = [
        # Add & to get pointer from value
        (r'compute\.NewVirtualMachinesClient\(', r'&compute.VirtualMachinesClient{SubscriptionID: '),
        (r'compute\.NewDisksClient\(', r'&compute.DisksClient{SubscriptionID: '),
        (r'network\.NewVirtualNetworksClient\(', r'&network.VirtualNetworksClient{SubscriptionID: '),
        (r'network\.NewSubnetsClient\(', r'&network.SubnetsClient{SubscriptionID: '),
        (r'network\.NewSecurityGroupsClient\(', r'&network.SecurityGroupsClient{SubscriptionID: '),
        (r'resources\.NewClient\(', r'&resources.Client{SubscriptionID: '),
    ]

    for old, new in replacements:
        content = re.sub(old, new, content)

    # Fix: Close client constructor calls properly
    content = re.sub(
        r'&(\w+)\.(\w+)\{SubscriptionID: azureConfig\.SubscriptionID\)',
        r'&\1.\2{SubscriptionID: azureConfig.SubscriptionID}',
        content
    )

    print("  Fixing Azure SDK API calls...")

    # Fix: ListComplete now requires 4 arguments
    content = re.sub(
        r'\.ListComplete\(([^,]+), ([^,]+), nil\)',
        r'.ListComplete(\1, \2, "", nil)',
        content
    )

    # Fix: Delete now requires 4 arguments (add optional bool)
    content = re.sub(
        r'\.Delete\(ctx, ([^,]+), ([^)]+)\)',
        r'.Delete(ctx, \1, \2, nil)',
        content
    )

    # Fix: Deallocate now requires 4 arguments (add optional bool)
    content = re.sub(
        r'\.Deallocate\(ctx, ([^,]+), ([^)]+)\)',
        r'.Deallocate(ctx, \1, \2, nil)',
        content
    )

    # Fix: InstanceView is now a different type
    content = re.sub(
        r'compute\.InstanceView',
        r'compute.VirtualMachineInstanceView',
        content
    )

    with open(filepath, 'w') as f:
        f.write(content)

    print("  Azure SDK fixes applied")
    return True

def main():
    print("=" * 70)
    print("PHASE 3: AZURE SDK MIGRATION")
    print("=" * 70)

    success = fix_azure_adapter()

    print("\n" + "=" * 70)
    print(f"COMPLETED: {'Success' if success else 'Failed'}")
    print("=" * 70)

    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
