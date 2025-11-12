# DWCP v1 → v3 Upgrade Test Suite

Comprehensive test suite for DWCP v1.0 → v3.0 upgrade with **90%+ coverage target**.

## Test Categories

### 1. Backward Compatibility Tests (`backward_compat_test.go`)
**MOST CRITICAL** - Ensures v1 still works after v3 upgrade.

Tests:
- ✅ `TestV1StillWorks` - All v1 functionality after upgrade
- ✅ `TestDualModeOperation` - v1 and v3 running simultaneously
- ✅ `TestFeatureFlagRollout` - 0% → 10% → 50% → 100% rollout
- ✅ `TestInstantRollback` - Rollback from v3 to v1 in <5 seconds
- ✅ `TestZeroDowntimeUpgrade` - No service interruption during upgrade
- ✅ `TestBackwardCompatibilityAfterRevert` - v1 works after v3 → v1 revert

**Coverage Target:** 95%+

### 2. AMST v3 Tests (`amst_v3_test.go`)
Tests AMST hybrid multi-stream transport.

Tests:
- ✅ `TestAMSTv3HybridTransport` - Datacenter/Internet/Hybrid modes
- ✅ `TestAMSTv3AdaptiveStreams` - 4-16 streams (internet), 16-256 (datacenter)
- ✅ `TestAMSTv3CongestionControl` - WAN congestion handling
- ✅ `TestAMSTv3PerformanceTargets` - 10-100 Gbps (DC), 100-900 Mbps (Internet)
- ✅ `TestAMSTv3Reliability` - Failure recovery and bandwidth limits
- ✅ `TestAMSTv3Concurrent` - Concurrent operations stress test

**Coverage Target:** 85%+

### 3. HDE v3 Tests (`hde_v3_test.go`)
Tests HDE ML-based compression and CRDT integration.

Tests:
- ✅ `TestHDEv3MLCompression` - ML-based compression selection
- ✅ `TestHDEv3CRDTIntegration` - Conflict-free state sync
- ✅ `TestHDEv3DeltaEncoding` - Enhanced delta encoding
- ✅ `TestHDEv3CompressionTargets` - 70-85% compression target
- ✅ `TestHDEv3DictionaryTraining` - Dictionary-based compression
- ✅ `TestHDEv3MemoryManagement` - Baseline cleanup
- ✅ `TestHDEv3Concurrent` - Concurrent compression

**Coverage Target:** 85%+

### 4. Mode Switching Tests (`mode_switching_test.go`)
Tests datacenter ↔ internet ↔ hybrid mode switching.

Tests:
- ✅ `TestModeSwitching` - Mode transitions <2 seconds
- ✅ `TestModeDetection` - Automatic mode detection
- ✅ `TestModeThresholds` - Latency/bandwidth thresholds
- ✅ `TestModeConsistency` - Mode persistence during operations
- ✅ `TestModeSwitchingPerformance` - Zero data loss during switch

**Coverage Target:** 90%+

### 5. Integration Tests (`integration_test.go`)
Tests v3 integration with existing NovaCron systems.

Tests:
- ✅ `TestV3WithMigration` - Integration with VM migration
- ✅ `TestV3WithFederation` - Multi-cloud federation
- ✅ `TestV3WithMultiCloud` - Hybrid cloud deployment
- ✅ `TestV3EndToEndWorkflow` - Complete migration workflow
- ✅ `TestV3PerformanceUnderLoad` - High throughput scenarios
- ✅ `TestV3ErrorHandling` - Error recovery
- ✅ `TestV3SecurityIntegration` - Secure transport and encryption

**Coverage Target:** 80%+

### 6. Performance Benchmarks (`benchmark_test.go`)
Comparative benchmarks: v1 vs v3 performance.

Benchmarks:
- ✅ `BenchmarkAMSTv1VsV3` - Transport performance comparison
- ✅ `BenchmarkHDECompression` - Compression performance (1KB → 10MB)
- ✅ `BenchmarkHDECompressionLevels` - Level 0/3/9 comparison
- ✅ `BenchmarkHDEDeltaEncoding` - Delta encoding performance
- ✅ `BenchmarkModeDetection` - Mode detection latency
- ✅ `BenchmarkModeSwitching` - Mode switching overhead
- ✅ `BenchmarkFeatureFlagEvaluation` - Feature flag performance
- ✅ `BenchmarkConcurrentCompression` - Parallel compression
- ✅ `BenchmarkAMSTMetricsCollection` - Metrics overhead
- ✅ `BenchmarkCompressionRatio` - Compression ratios by data type

## Running Tests

### Quick Start
```bash
cd /home/kp/novacron/backend/core/network/dwcp/v3/tests
./run_tests.sh
```

### Individual Test Suites
```bash
# Backward compatibility (CRITICAL)
go test -v -race backward_compat_test.go

# AMST v3 component tests
go test -v -race amst_v3_test.go

# HDE v3 component tests
go test -v -race hde_v3_test.go

# Mode switching tests
go test -v -race mode_switching_test.go

# Integration tests
go test -v -race integration_test.go

# Benchmarks
go test -v -bench=. -benchmem benchmark_test.go
```

### Coverage Analysis
```bash
# Generate coverage for specific test
go test -coverprofile=coverage.out backward_compat_test.go

# View coverage in HTML
go tool cover -html=coverage.out -o coverage.html

# View coverage percentages
go tool cover -func=coverage.out
```

### Comprehensive Coverage Report
```bash
# Run all tests with combined coverage
go test -v -race -coverprofile=coverage_v3.out ./...
go tool cover -html=coverage_v3.out -o coverage_v3.html

# Check coverage percentage
go tool cover -func=coverage_v3.out | grep total
```

## Coverage Targets

| Test Suite | Target | Critical |
|-----------|--------|----------|
| Backward Compatibility | 95%+ | ⚠️ YES |
| AMST v3 | 85%+ | - |
| HDE v3 | 85%+ | - |
| Mode Switching | 90%+ | - |
| Integration | 80%+ | - |
| **Overall** | **90%+** | ⚠️ YES |

## Success Criteria

### Functional Requirements
- ✅ All v1 functionality works after v3 upgrade
- ✅ v1 and v3 can run simultaneously (dual-mode)
- ✅ Gradual rollout: 0% → 10% → 50% → 100%
- ✅ Instant rollback: <5 seconds from v3 → v1
- ✅ Zero downtime during upgrade
- ✅ Mode switching: <2 seconds

### Performance Requirements
- ✅ Datacenter mode: 10-100 Gbps, <10ms latency
- ✅ Internet mode: 100-900 Mbps, 50-500ms latency
- ✅ Compression: 70-85% bandwidth savings
- ✅ HDE delta hit rate: >50%

### Reliability Requirements
- ✅ No data loss during mode switching
- ✅ Graceful degradation under network stress
- ✅ Concurrent operations without race conditions
- ✅ Error recovery and retry logic

## Test Execution Timeline

### Phase 1: Unit Tests (15 minutes)
- Backward compatibility tests
- Component tests (AMST, HDE)
- Mode switching tests

### Phase 2: Integration Tests (10 minutes)
- Migration integration
- Federation integration
- Multi-cloud scenarios

### Phase 3: Benchmarks (20 minutes)
- Performance comparison (v1 vs v3)
- Compression benchmarks
- Mode switching overhead

### Phase 4: Coverage Analysis (5 minutes)
- Generate combined coverage report
- Verify 90%+ coverage target
- Identify coverage gaps

**Total Execution Time:** ~50 minutes

## CI/CD Integration

### GitHub Actions (Recommended)
```yaml
name: DWCP v3 Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-go@v4
        with:
          go-version: '1.21'
      - name: Run DWCP v3 Tests
        run: |
          cd backend/core/network/dwcp/v3/tests
          ./run_tests.sh
      - name: Upload Coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./backend/core/network/dwcp/v3/coverage/coverage_combined.out
```

### Local Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

cd backend/core/network/dwcp/v3/tests
./run_tests.sh

if [ $? -ne 0 ]; then
    echo "DWCP v3 tests failed. Commit aborted."
    exit 1
fi
```

## Debugging Failed Tests

### View Detailed Logs
```bash
# Check individual test logs
cat /home/kp/novacron/backend/core/network/dwcp/v3/coverage/*.log

# Run single test with verbose output
go test -v -run TestV1StillWorks backward_compat_test.go
```

### Enable Race Detection
```bash
go test -v -race backward_compat_test.go
```

### Increase Timeout
```bash
go test -v -timeout 10m integration_test.go
```

### Debug Specific Test
```bash
go test -v -run "TestV1StillWorks/v1_amst_transfer" backward_compat_test.go
```

## Test Maintenance

### Adding New Tests
1. Create test function with descriptive name
2. Use `t.Run()` for subtests
3. Add assertions with clear messages
4. Update coverage targets
5. Run full test suite

### Updating Existing Tests
1. Preserve backward compatibility tests
2. Update performance benchmarks for new features
3. Verify coverage doesn't decrease
4. Document changes in git commit

## Contact

For test failures or questions:
- File an issue in the NovaCron repository
- Tag: `dwcp-v3-upgrade`, `testing`
- Include: Test output, coverage report, system info

## References

- [DWCP v3 Upgrade Plan](../UPGRADE_PLAN_V1_TO_V3.md)
- [Migration Strategy](../MIGRATION_STRATEGY_V1_TO_V3.md)
- [Implementation Status](../DWCP_V1_TO_V3_IMPLEMENTATION_STATUS.md)
