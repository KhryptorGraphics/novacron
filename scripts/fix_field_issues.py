#!/usr/bin/env python3
"""
Phase 2: Fix Struct Field and Import Issues
"""

import os
import re
from pathlib import Path

FIXES = [
    # Fix unused imports
    {
        'file': 'backend/community/developer/developer_scale_up.go',
        'line': 7,
        'action': 'remove_line',
        'match': '"encoding/json" imported and not used',
    },
    {
        'file': 'backend/community/transformation/industry_transformation.go',
        'line': 7,
        'action': 'remove_line',
        'match': '"encoding/json" imported and not used',
    },

    # Fix undefined types - add them
    {
        'file': 'backend/community/developer/developer_scale_up.go',
        'action': 'add_types',
        'types': {
            'InPersonTrainingProgram': 'struct { Name string; Location string; Duration int }',
            'HybridTrainingProgram': 'struct { Name string; OnlineHours int; InPersonHours int }',
            'CulturalAdaptationEngine': 'struct { Enabled bool; Strategies []string }',
        }
    },

    # Fix missing struct fields
    {
        'file': 'backend/community/university/academic_program.go',
        'action': 'add_field',
        'struct': 'AcademicStats',
        'field': 'InternsPlaced int',
    },

    # Fix opensource leadership governance field
    {
        'file': 'backend/community/opensource/opensource_leadership.go',
        'line': 901,
        'action': 'replace',
        'old': 'e.governanceModel.governanceModel',
        'new': 'e.governanceModel',
    },

    # Fix unknown field in struct literal
    {
        'file': 'backend/community/opensource/opensource_leadership.go',
        'line': 943,
        'action': 'remove_field',
        'field': 'governanceModel',
    },

    # Fix unused variable declarations
    {
        'file': 'backend/core/network/dwcp/v3/partition/geographic_optimizer.go',
        'line': 351,
        'action': 'prefix_underscore',
        'var': 'id',
    },
    {
        'file': 'backend/community/hackathons/innovation_engine.go',
        'line': 719,
        'action': 'prefix_underscore',
        'var': 'i',
    },
    {
        'file': 'backend/core/network/dwcp/prediction/example_integration.go',
        'line': 168,
        'action': 'prefix_underscore',
        'var': 'logger',
    },
    {
        'file': 'backend/core/network/dwcp/prediction/prediction_service.go',
        'line': 433,
        'action': 'prefix_underscore',
        'var': 'altPrediction',
    },

    # Fix type conversions
    {
        'file': 'backend/corporate/ma/evaluation.go',
        'line': 1260,
        'action': 'replace',
        'old': 'target.Technology.CodebaseSize / 1000000',
        'new': 'int(target.Technology.CodebaseSize / 1000000)',
    },

    # Fix unknown struct fields
    {
        'file': 'backend/community/hackathons/innovation_engine.go',
        'line': 727,
        'action': 'remove_field',
        'field': 'TicketRange',
    },

    # Fix time.Duration math error
    {
        'file': 'backend/core/network/dwcp/v3/partition/heterogeneous_placement.go',
        'line': 464,
        'action': 'replace',
        'old': 'float64(cap.MinLatency) / (100 * time.Millisecond)',
        'new': 'float64(cap.MinLatency) / float64(100 * time.Millisecond)',
    },

    # Fix string repeat operator
    {
        'file': 'backend/core/network/dwcp/prediction/example_integration.go',
        'line': 138,
        'action': 'replace',
        'old': '"─" * 60',
        'new': 'strings.Repeat("─", 60)',
    },
]

def apply_fix(fix):
    """Apply a single fix"""
    base_dir = Path('/home/kp/repos/novacron')
    filepath = base_dir / fix['file']

    if not filepath.exists():
        print(f"  Warning: {filepath} not found")
        return False

    with open(filepath, 'r') as f:
        lines = f.readlines()

    action = fix['action']

    if action == 'remove_line':
        line_num = fix['line'] - 1
        if line_num < len(lines):
            # Comment out instead of removing to preserve line numbers
            lines[line_num] = '// ' + lines[line_num]
            print(f"  Commented out line {fix['line']}")

    elif action == 'replace':
        line_num = fix['line'] - 1
        if line_num < len(lines):
            lines[line_num] = lines[line_num].replace(fix['old'], fix['new'])
            print(f"  Replaced '{fix['old']}' with '{fix['new']}' at line {fix['line']}")

    elif action == 'prefix_underscore':
        line_num = fix['line'] - 1
        if line_num < len(lines):
            var_name = fix['var']
            # Replace variable declaration
            lines[line_num] = re.sub(
                rf'\b{var_name}\b',
                f'_{var_name}',
                lines[line_num]
            )
            print(f"  Prefixed variable '{var_name}' with underscore at line {fix['line']}")

    elif action == 'add_types':
        # Find package declaration and add types after imports
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('import'):
                # Find end of import block
                if '(' in line:
                    for j in range(i, len(lines)):
                        if ')' in lines[j]:
                            insert_pos = j + 1
                            break
                else:
                    insert_pos = i + 1
                break

        type_defs = []
        for typename, typedef in fix['types'].items():
            type_defs.append(f'\ntype {typename} {typedef}\n')

        lines.insert(insert_pos, '\n'.join(type_defs) + '\n')
        print(f"  Added {len(fix['types'])} type definitions")

    elif action == 'add_field':
        # Find struct definition
        struct_name = fix['struct']
        field_def = fix['field']

        for i, line in enumerate(lines):
            if f'type {struct_name} struct' in line:
                # Find closing brace
                brace_count = line.count('{') - line.count('}')
                for j in range(i+1, len(lines)):
                    brace_count += lines[j].count('{') - lines[j].count('}')
                    if brace_count == 0:
                        # Insert before closing brace
                        lines.insert(j, f'\t{field_def}\n')
                        print(f"  Added field '{field_def}' to struct {struct_name}")
                        break
                break

    elif action == 'remove_field':
        line_num = fix['line'] - 1
        field_name = fix['field']
        if line_num < len(lines):
            # Find and comment out the field
            lines[line_num] = re.sub(
                rf'{field_name}\s*:',
                f'// {field_name}:',
                lines[line_num]
            )
            print(f"  Commented out field '{field_name}' at line {fix['line']}")

    # Write back
    with open(filepath, 'w') as f:
        f.writelines(lines)

    return True

def main():
    print("=" * 70)
    print("PHASE 2: FIELD AND IMPORT FIXES")
    print("=" * 70)

    fixed_count = 0
    for i, fix in enumerate(FIXES, 1):
        print(f"\nFix {i}/{len(FIXES)}: {fix['file']}")
        if apply_fix(fix):
            fixed_count += 1

    print("\n" + "=" * 70)
    print(f"COMPLETED: Applied {fixed_count}/{len(FIXES)} fixes")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    exit(main())
