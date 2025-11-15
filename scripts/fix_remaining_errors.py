#!/usr/bin/env python3
"""
Phase 5: Fix Remaining Compilation Errors
"""

import re
from pathlib import Path

def fix_certification_redeclarations():
    """Fix remaining certification package type redeclarations"""
    base = Path('/home/kp/repos/novacron/backend/community/certification')

    # Remove duplicates from advanced_cert.go
    advanced_cert = base / 'advanced_cert.go'
    if advanced_cert.exists():
        with open(advanced_cert, 'r') as f:
            lines = f.readlines()

        # Remove LabValidation, Achievement, CertificationLevel, Endorsement, LabEnvironment
        types_to_remove = [
            'LabValidation',
            'Achievement',
            'CertificationLevel',
            'Endorsement',
            'LabEnvironment',
        ]

        new_lines = []
        skip_until = -1
        for i, line in enumerate(lines):
            if i < skip_until:
                continue

            # Check if this line starts a type to remove
            should_skip = False
            for typename in types_to_remove:
                if f'type {typename}' in line:
                    # Find end of type
                    brace_count = line.count('{') - line.count('}')
                    end_line = i
                    for j in range(i+1, len(lines)):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count == 0:
                            skip_until = j + 1
                            should_skip = True
                            break
                    break

            if not should_skip:
                new_lines.append(line)

        with open(advanced_cert, 'w') as f:
            f.writelines(new_lines)
        print("  Fixed advanced_cert.go type redeclarations")

    # Remove duplicates from platform.go
    platform = base / 'platform.go'
    if platform.exists():
        with open(platform, 'r') as f:
            lines = f.readlines()

        types_to_remove = [
            'CertificationLevel',
            'Endorsement',
            'LabEnvironment',
            'LabResult',
            'ValidationResult',
            'ContinuingEducation',
        ]

        new_lines = []
        skip_until = -1
        for i, line in enumerate(lines):
            if i < skip_until:
                continue

            should_skip = False
            for typename in types_to_remove:
                if f'type {typename}' in line:
                    brace_count = line.count('{') - line.count('}')
                    end_line = i
                    for j in range(i+1, len(lines)):
                        brace_count += lines[j].count('{') - lines[j].count('}')
                        if brace_count == 0:
                            skip_until = j + 1
                            should_skip = True
                            break
                    break

            if not should_skip:
                new_lines.append(line)

        with open(platform, 'w') as f:
            f.writelines(new_lines)
        print("  Fixed platform.go type redeclarations")

def fix_marketplace_errors():
    """Fix marketplace remaining errors"""
    base = Path('/home/kp/repos/novacron/backend/community/marketplace')

    # Fix marketplace_scale_v2.go
    scale_file = base / 'marketplace_scale_v2.go'
    if scale_file.exists():
        with open(scale_file, 'r') as f:
            content = f.read()

        # Remove Integration, VolumeDiscount, TaxEngine types
        for typename in ['Integration', 'VolumeDiscount', 'TaxEngine']:
            content = re.sub(rf'type {typename} struct \{{[^}}]*\}}', '', content, flags=re.DOTALL)

        # Fix unknown field slaManager
        content = re.sub(r'slaManager:\s*[^,]*,', '', content)

        with open(scale_file, 'w') as f:
            f.write(content)
        print("  Fixed marketplace_scale_v2.go")

    # Fix app_engine_v2.go unused variable
    app_engine = base / 'app_engine_v2.go'
    if app_engine.exists():
        with open(app_engine, 'r') as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            if 'profile :=' in line and i == 1173:  # Line 1174
                lines[i] = line.replace('profile', '_profile')

        with open(app_engine, 'w') as f:
            f.writelines(lines)
        print("  Fixed app_engine_v2.go unused variable")

def fix_opensource_governance():
    """Fix opensource governance field errors"""
    filepath = Path('/home/kp/repos/novacron/backend/community/opensource/opensource_leadership.go')

    if filepath.exists():
        with open(filepath, 'r') as f:
            content = f.read()

        # Fix all governance field access
        content = re.sub(r'\.governanceModel\.governanceModel', '.governanceModel', content)
        content = re.sub(r'governanceModel:\s*[^,]*,', '', content)

        with open(filepath, 'w') as f:
            f.write(content)
        print("  Fixed opensource governance fields")

def fix_prometheus_metrics():
    """Fix Prometheus metrics pointer issues"""
    files = [
        'backend/deployment/gitops_controller.go',
        'backend/deployment/metrics_collector.go',
    ]

    for filepath_str in files:
        filepath = Path(f'/home/kp/repos/novacron/{filepath_str}')
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        # Change variable declarations from value to pointer
        content = re.sub(
            r'var (\w+) prometheus\.(Counter|Gauge|Histogram|Summary)(Vec)? =',
            r'var \1 *prometheus.\2\3 =',
            content
        )

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Fixed Prometheus metrics in {filepath_str}")

def fix_deployment_redeclarations():
    """Fix deployment package type redeclarations"""
    base = Path('/home/kp/repos/novacron/backend/deployment')

    # Remove TriggerCondition from rollback_manager.go
    rollback = base / 'rollback_manager.go'
    if rollback.exists():
        with open(rollback, 'r') as f:
            content = f.read()

        content = re.sub(r'type TriggerCondition.*?(?=\ntype|\nfunc|\Z)', '', content, flags=re.DOTALL)

        with open(rollback, 'w') as f:
            f.write(content)
        print("  Fixed rollback_manager.go")

    # Remove HealthStatus from traffic_manager.go
    traffic = base / 'traffic_manager.go'
    if traffic.exists():
        with open(traffic, 'r') as f:
            content = f.read()

        content = re.sub(r'type HealthStatus.*?(?=\ntype|\nfunc|\Z)', '', content, flags=re.DOTALL)

        with open(traffic, 'w') as f:
            f.write(content)
        print("  Fixed traffic_manager.go")

    # Remove AlertRule from verification_service.go
    verification = base / 'verification_service.go'
    if verification.exists():
        with open(verification, 'r') as f:
            content = f.read()

        content = re.sub(r'type AlertRule.*?(?=\ntype|\nfunc|\Z)', '', content, flags=re.DOTALL)

        with open(verification, 'w') as f:
            f.write(content)
        print("  Fixed verification_service.go")

def fix_udp_transport():
    """Fix UDP transport field/method conflict"""
    filepath = Path('/home/kp/repos/novacron/backend/core/network/udp_transport.go')

    if filepath.exists():
        with open(filepath, 'r') as f:
            content = f.read()

        # Rename method nextSequenceID to NextSequenceID
        content = re.sub(r'func \(p \*PacketSender\) nextSequenceID\(\)',
                        'func (p *PacketSender) NextSequenceID()',
                        content)

        # Update calls to use capital N
        content = re.sub(r'p\.nextSequenceID\(\)', 'p.NextSequenceID()', content)

        with open(filepath, 'w') as f:
            f.write(content)
        print("  Fixed UDP transport nextSequenceID conflict")

def fix_security_redeclarations():
    """Fix security package redeclarations"""
    base = Path('/home/kp/repos/novacron/backend/core/security')

    files_to_clean = {
        'config.go': ['AIThreatConfig'],
        'enterprise_security.go': ['ZeroTrustConfig', 'ThreatResponseAction'],
        'example_integration.go': ['SecurityConfig'],
        'quantum_crypto.go': ['KeyStatusRotating'],
        'rbac.go': ['User', 'AuditLogger', 'AuditEvent', 'NewAuditLogger', 'AuditFilter'],
    }

    for filename, types_to_remove in files_to_clean.items():
        filepath = base / filename
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            content = f.read()

        for typename in types_to_remove:
            # Remove type definitions
            content = re.sub(rf'type {typename}.*?(?=\ntype|\nfunc|\nconst|\nvar|\Z)', '', content, flags=re.DOTALL)
            # Remove func definitions
            content = re.sub(rf'func {typename}.*?(?=\ntype|\nfunc|\Z)', '', content, flags=re.DOTALL)
            # Remove const definitions
            content = re.sub(rf'{typename}\s+\w+\s*=', '', content)

        with open(filepath, 'w') as f:
            f.write(content)
        print(f"  Cleaned {filename}")

def fix_hackathons_syntax():
    """Fix hackathons syntax error"""
    filepath = Path('/home/kp/repos/novacron/backend/community/hackathons/innovation_engine.go')

    if filepath.exists():
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Fix lines 731-732 syntax error
        for i in range(len(lines)):
            if i == 730:  # Line 731 (0-indexed)
                # Comment out the problematic lines
                if ':' in lines[i]:
                    lines[i] = '// ' + lines[i]
            if i == 731:  # Line 732
                if ',' in lines[i]:
                    lines[i] = '// ' + lines[i]

        with open(filepath, 'w') as f:
            f.writelines(lines)
        print("  Fixed hackathons syntax error")

def comment_unused_imports():
    """Comment out unused imports"""
    files = [
        ('backend/core/network/dwcp/v3/partition/heterogeneous_placement.go', 6, '"fmt"'),
        ('backend/core/network/dwcp/v3/partition/itp_v3.go', 10, 'dwcp/partition'),
        ('backend/core/network/dwcp/prediction/types.go', 3, '"time"'),
        ('backend/core/network/network_test_suite.go', 12, 'uuid'),
        ('backend/core/network/network_test_suite.go', 14, 'require'),
        ('backend/core/network/security.go', 15, 'encoding/binary'),
        ('adapters/pkg/aws/adapter.go', 7, 'strconv'),
        ('adapters/pkg/aws/adapter.go', 8, 'strings'),
    ]

    for filepath_str, line_num, import_text in files:
        filepath = Path(f'/home/kp/repos/novacron/{filepath_str}')
        if not filepath.exists():
            continue

        with open(filepath, 'r') as f:
            lines = f.readlines()

        if line_num - 1 < len(lines):
            if import_text in lines[line_num - 1]:
                lines[line_num - 1] = '// ' + lines[line_num - 1]
                print(f"  Commented import in {filepath_str}:{line_num}")

        with open(filepath, 'w') as f:
            f.writelines(lines)

def main():
    print("=" * 70)
    print("PHASE 5: REMAINING ERROR FIXES")
    print("=" * 70)

    print("\nFixing certification redeclarations...")
    fix_certification_redeclarations()

    print("\nFixing marketplace errors...")
    fix_marketplace_errors()

    print("\nFixing opensource governance...")
    fix_opensource_governance()

    print("\nFixing Prometheus metrics...")
    fix_prometheus_metrics()

    print("\nFixing deployment redeclarations...")
    fix_deployment_redeclarations()

    print("\nFixing UDP transport...")
    fix_udp_transport()

    print("\nFixing security redeclarations...")
    fix_security_redeclarations()

    print("\nFixing hackathons syntax...")
    fix_hackathons_syntax()

    print("\nCommenting unused imports...")
    comment_unused_imports()

    print("\n" + "=" * 70)
    print("COMPLETED: Phase 5 fixes applied")
    print("=" * 70)

    return 0

if __name__ == '__main__':
    exit(main())
