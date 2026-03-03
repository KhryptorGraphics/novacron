#!/bin/bash
set -e

echo "========================================================================"
echo "COMPILATION ERROR RESOLUTION - MASTER SCRIPT"
echo "========================================================================"
echo "Agent: 29 - Compilation Error Resolution Expert"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Create logs directory
mkdir -p logs

echo "========================================================================"
echo "PHASE 1: TYPE REDECLARATION CONSOLIDATION"
echo "========================================================================"
python3 scripts/consolidate_types.py 2>&1 | tee logs/phase1_types.log
echo "Phase 1 Complete"
echo ""

echo "========================================================================"
echo "PHASE 2: FIELD AND IMPORT FIXES"
echo "========================================================================"
python3 scripts/fix_field_issues.py 2>&1 | tee logs/phase2_fields.log
echo "Phase 2 Complete"
echo ""

echo "========================================================================"
echo "PHASE 3: AZURE SDK MIGRATION"
echo "========================================================================"
python3 scripts/fix_azure_sdk.py 2>&1 | tee logs/phase3_azure.log
echo "Phase 3 Complete"
echo ""

echo "========================================================================"
echo "PHASE 4: NETWORK PACKAGE FIXES"
echo "========================================================================"
python3 scripts/fix_network_issues.py 2>&1 | tee logs/phase4_network.log
echo "Phase 4 Complete"
echo ""

echo "========================================================================"
echo "VERIFICATION: REBUILD ALL PACKAGES"
echo "========================================================================"
echo "Running: go build ./..."
go build ./... 2>&1 | tee logs/final_build.log

# Count errors
ERROR_COUNT=$(grep -c "^#.*error" logs/final_build.log || echo "0")
echo ""
echo "========================================================================"
echo "BUILD RESULTS"
echo "========================================================================"
echo "Total errors remaining: $ERROR_COUNT"
echo ""

if [ "$ERROR_COUNT" -eq 0 ]; then
    echo "✅ SUCCESS: All compilation errors fixed!"
    echo "✅ Full codebase builds successfully"
else
    echo "⚠️  PARTIAL SUCCESS: $ERROR_COUNT packages still have errors"
    echo "See logs/final_build.log for details"
fi

echo ""
echo "========================================================================"
echo "SUMMARY"
echo "========================================================================"
echo "End Time: $(date)"
echo "Logs saved to: logs/"
echo "  - phase1_types.log"
echo "  - phase2_fields.log"
echo "  - phase3_azure.log"
echo "  - phase4_network.log"
echo "  - final_build.log"
echo "========================================================================"
