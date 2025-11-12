# Test Infrastructure Recommendations - Priority Action Plan

**Date**: November 10, 2025
**Agent**: QA Test Engineer
**Status**: Ready for Implementation

---

## Critical Discovery: DWCP v3 Tests Found ✅

**Previous Assessment**: Tests missing ❌
**Actual Status**: 20+ test files exist ✅

This changes the Phase 0 blocker from "create tests" to "run benchmarks"!

---

## Immediate Actions - This Sprint (P0)

### 1. Run DWCP v3 Performance Benchmarks 🔴 CRITICAL

**Priority**: P0
**Effort**: 2-3 days
**Blocking**: novacron-38p (Phase 0 Go/No-Go decision)

#### Commands to Execute

```bash
# Navigate to DWCP v3 tests
cd /home/kp/novacron/backend/core/network/dwcp/v3/tests

# Run comprehensive benchmarks
go test -bench=. -benchmem -timeout 30m -run=^$ 2>&1 | tee benchmark-results.txt

# Run specific component benchmarks
go test -bench=BenchmarkAMST -benchmem -timeout 10m
go test -bench=BenchmarkHDE -benchmem -timeout 10m
go test -bench=BenchmarkPBA -benchmem -timeout 10m

# Generate CPU profile
go test -bench=. -cpuprofile=cpu.prof -memprofile=mem.prof

# Analyze profiles
go tool pprof cpu.prof
go tool pprof mem.prof
```

#### Expected Metrics to Validate

| Metric | Target | Test File |
|--------|--------|-----------|
| Bandwidth Utilization | >70% | `benchmark_test.go` |
| Compression Ratio | >5x | `hde_v3_test.go` |
| Migration Time | Reduced vs v1/v2 | `performance_comparison_test.go` |
| CPU Overhead | <30% | `benchmark_test.go` |
| Memory Efficiency | <20% overhead | `pba_v3_test.go` |

#### Documentation

Create `/home/kp/novacron/docs/DWCP-PHASE0-BENCHMARK-RESULTS.md`:

```markdown
# DWCP v3 Phase 0 Benchmark Results

## Test Environment
- CPU: [specs]
- Memory: [specs]
- Network: [specs]
- Date: [date]

## Results

### AMST (Adaptive Multi-Stream Transport)
- Bandwidth Utilization: XX%
- TCP Streams: 16-256
- RDMA Performance: XX Gbps

### HDE (Hierarchical Delta Encoding)
- Compression Ratio: XXx
- CPU Overhead: XX%
- Compression Time: XX ms

### PBA (Predictive Block Allocator)
- Memory Efficiency: XX%
- Block Allocation Time: XX µs
- Zero-copy Performance: XX%

## Go/No-Go Decision
[PASS/FAIL based on metrics]
```

**Acceptance Criteria**:
- ✅ All benchmarks executed
- ✅ Metrics documented
- ✅ Go/No-Go decision made
- ✅ Results stored in Beads (novacron-38p)

---

### 2. Update Test Documentation 🟠 HIGH

**Priority**: P0
**Effort**: 1 day
**Blocking**: None (but critical for team awareness)

#### Tasks

1. Update `docs/TEST-INFRASTRUCTURE-ASSESSMENT.md`:
   - Change "DWCP v3 tests MISSING" to "DWCP v3 tests FOUND"
   - Update coverage metrics from ~75% to ~80%
   - Update blocker status for novacron-38p

2. Update `docs/TEST-STRATEGY-SUMMARY.md`:
   - Add DWCP v3 test inventory
   - Update Phase 0 status
   - Update timeline to production

3. Create `docs/TEST-QUICK-START.md`:
   - How to run all test suites
   - How to run DWCP v3 benchmarks
   - How to analyze coverage

**Acceptance Criteria**:
- ✅ All documentation updated
- ✅ Team notified of discovery
- ✅ Beads issues updated

---

## Short-Term Actions - Next Sprint (P1)

### 3. Implement E2E Testing Framework ⚠️ HIGH

**Priority**: P1
**Effort**: 1 week
**Blocking**: novacron-aca (Phase 5 production)

#### Setup

```bash
# Install Playwright
cd /home/kp/novacron
npm install --save-dev @playwright/test

# Create E2E test structure
mkdir -p tests/e2e/{vm-operations,workload-distribution,disaster-recovery,security}

# Create Playwright configuration
cat > playwright.config.ts << 'EOF'
import { defineConfig } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60000,
  retries: 2,
  workers: 4,
  use: {
    baseURL: 'http://localhost:8080',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { browserName: 'chromium' },
    },
  ],
});
EOF
```

#### E2E Test Scenarios

**Priority 1 - VM Operations**:
```javascript
// tests/e2e/vm-operations/vm-migration.test.js
test('Complete VM migration workflow', async ({ page }) => {
  // 1. Login to dashboard
  await page.goto('/login');
  await page.fill('[name="username"]', 'admin');
  await page.fill('[name="password"]', 'password');
  await page.click('button[type="submit"]');

  // 2. Navigate to VM list
  await page.goto('/vms');
  await expect(page.locator('.vm-list')).toBeVisible();

  // 3. Select VM and initiate migration
  await page.click('[data-vm-id="vm-1"]');
  await page.click('button[data-action="migrate"]');

  // 4. Configure migration target
  await page.selectOption('[name="target-host"]', 'host-2');
  await page.click('button[data-action="start-migration"]');

  // 5. Wait for migration to complete
  await page.waitForSelector('.migration-complete', { timeout: 60000 });

  // 6. Verify VM is running on new host
  const vmStatus = await page.locator('[data-vm-id="vm-1"] .status').textContent();
  expect(vmStatus).toBe('Running');

  const vmHost = await page.locator('[data-vm-id="vm-1"] .host').textContent();
  expect(vmHost).toBe('host-2');
});
```

**Priority 2 - Workload Distribution**:
```javascript
// tests/e2e/workload-distribution/auto-scaling.test.js
test('Auto-scaling under load', async ({ page }) => {
  await page.goto('/dashboard');

  // Monitor VM count
  const initialVMs = await page.locator('.vm-count').textContent();

  // Trigger load increase
  await page.click('button[data-action="simulate-load"]');

  // Wait for auto-scaling to trigger
  await page.waitForFunction(
    (initial) => {
      const current = document.querySelector('.vm-count').textContent;
      return parseInt(current) > parseInt(initial);
    },
    initialVMs,
    { timeout: 120000 }
  );

  // Verify new VMs are running
  const newVMs = await page.locator('.vm-list .vm-card').count();
  expect(newVMs).toBeGreaterThan(parseInt(initialVMs));
});
```

**Priority 3 - Disaster Recovery**:
```javascript
// tests/e2e/disaster-recovery/node-failure.test.js
test('System recovery after node failure', async ({ page }) => {
  await page.goto('/cluster');

  // Record initial cluster state
  const initialNodes = await page.locator('.node-list .node-card').count();

  // Simulate node failure
  await page.click('[data-node-id="node-1"] button[data-action="simulate-failure"]');

  // Wait for failure detection
  await page.waitForSelector('[data-node-id="node-1"].node-failed', { timeout: 30000 });

  // Verify VM migration to healthy nodes
  await page.waitForFunction(
    () => {
      const failedNode = document.querySelector('[data-node-id="node-1"]');
      const vmCount = failedNode.querySelector('.vm-count').textContent;
      return parseInt(vmCount) === 0;
    },
    { timeout: 120000 }
  );

  // Verify cluster is healthy
  const clusterStatus = await page.locator('.cluster-status').textContent();
  expect(clusterStatus).toContain('Degraded but Operational');
});
```

**Acceptance Criteria**:
- ✅ Playwright configured
- ✅ 3+ critical path E2E tests implemented
- ✅ Tests integrated with CI/CD
- ✅ Documentation created

---

### 4. Implement Load Testing Framework 🔴 CRITICAL

**Priority**: P1
**Effort**: 1-2 weeks
**Blocking**: novacron-aca (Phase 5 production)

#### Setup

```bash
# Install k6
brew install k6  # macOS
# OR
sudo apt-get install k6  # Ubuntu

# Create load test structure
mkdir -p tests/load/{concurrent-ops,sustained-load,stress-tests,spike-tests}
```

#### Load Test Scenarios

**Scenario 1: 1000 Concurrent VM Operations**

```javascript
// tests/load/concurrent-ops/1000-concurrent-vms.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '5m', target: 100 },   // Ramp up to 100
    { duration: '10m', target: 500 },  // Ramp up to 500
    { duration: '10m', target: 1000 }, // Ramp up to 1000
    { duration: '30m', target: 1000 }, // Stay at 1000
    { duration: '5m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.05'],
  },
};

export default function () {
  // Create VM
  const createRes = http.post(
    'http://localhost:8080/api/vms',
    JSON.stringify({
      name: `vm-${__VU}-${__ITER}`,
      cpu: 2,
      memory: 4096,
      disk: 20480,
    }),
    { headers: { 'Content-Type': 'application/json' } }
  );

  const createSuccess = check(createRes, {
    'VM created': (r) => r.status === 201,
    'VM ID returned': (r) => r.json('id') !== undefined,
  });
  errorRate.add(!createSuccess);

  if (!createSuccess) {
    console.error(`Failed to create VM: ${createRes.status}`);
    return;
  }

  const vmId = createRes.json('id');
  sleep(1);

  // Start VM
  const startRes = http.post(`http://localhost:8080/api/vms/${vmId}/start`);
  const startSuccess = check(startRes, {
    'VM started': (r) => r.status === 200,
  });
  errorRate.add(!startSuccess);

  sleep(2);

  // Check VM status
  const statusRes = http.get(`http://localhost:8080/api/vms/${vmId}`);
  check(statusRes, {
    'VM status retrieved': (r) => r.status === 200,
    'VM is running': (r) => r.json('status') === 'running',
  });

  sleep(5);

  // Stop VM
  http.post(`http://localhost:8080/api/vms/${vmId}/stop`);
  sleep(1);

  // Delete VM
  http.del(`http://localhost:8080/api/vms/${vmId}`);
}
```

**Scenario 2: Sustained Load (24 hours)**

```javascript
// tests/load/sustained-load/24hour-sustained.js
export const options = {
  stages: [
    { duration: '10m', target: 500 },   // Ramp up
    { duration: '23h40m', target: 500 }, // Sustained
    { duration: '10m', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
  },
};
```

**Scenario 3: Stress Testing**

```javascript
// tests/load/stress-tests/breaking-point.js
export const options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 500 },
    { duration: '5m', target: 1000 },
    { duration: '5m', target: 2000 },
    { duration: '5m', target: 3000 },  // Find breaking point
    { duration: '10m', target: 0 },
  ],
};
```

#### Run Load Tests

```bash
# Run 1000 concurrent VMs test
k6 run tests/load/concurrent-ops/1000-concurrent-vms.js

# Run with InfluxDB for metrics
k6 run --out influxdb=http://localhost:8086/k6 tests/load/concurrent-ops/1000-concurrent-vms.js

# Run sustained load test
k6 run tests/load/sustained-load/24hour-sustained.js
```

**Acceptance Criteria**:
- ✅ k6 configured
- ✅ 1000+ concurrent operations test passing
- ✅ System maintains <500ms p95 latency
- ✅ Error rate <1%
- ✅ Resource monitoring integrated

---

## Medium-Term Actions - This Quarter (P2)

### 5. Enhance Chaos Engineering ⚠️ MEDIUM

**Priority**: P2
**Effort**: 1-2 weeks

#### Setup Chaos Mesh

```bash
# Install Chaos Mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm install chaos-mesh chaos-mesh/chaos-mesh -n chaos-testing --create-namespace

# Verify installation
kubectl get pods -n chaos-testing
```

#### Chaos Scenarios

**Network Partition**:
```yaml
# tests/chaos/network/partition.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: network-partition
spec:
  action: partition
  mode: all
  selector:
    namespaces:
      - novacron
  direction: both
  duration: '10m'
```

**Node Failure**:
```yaml
# tests/chaos/failure/node-failure.yaml
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: node-failure
spec:
  action: pod-kill
  mode: one
  selector:
    namespaces:
      - novacron
    labelSelectors:
      app: novacron-node
  scheduler:
    cron: '@every 30m'
```

---

### 6. Security Audit & Penetration Testing ⚠️ MEDIUM

**Priority**: P2
**Effort**: 2 weeks

#### OWASP ZAP Scan

```bash
# Run baseline scan
docker run -t owasp/zap2docker-stable zap-baseline.py \
  -t http://localhost:8080 \
  -r zap-report.html

# Run full scan
docker run -t owasp/zap2docker-stable zap-full-scan.py \
  -t http://localhost:8080 \
  -r zap-full-report.html
```

#### Penetration Testing Checklist

- [ ] SQL injection testing
- [ ] XSS attack prevention
- [ ] CSRF token validation
- [ ] Authentication bypass attempts
- [ ] Authorization escalation testing
- [ ] Encryption strength validation
- [ ] Secrets management audit
- [ ] API rate limiting
- [ ] Input validation
- [ ] Session management

---

## Success Criteria

### Phase 0 (novacron-38p) - Immediate

- ✅ DWCP v3 benchmarks executed
- ✅ Bandwidth >70% validated
- ✅ Compression >5x validated
- ✅ CPU overhead <30% validated
- ✅ Go/No-Go decision documented
- ✅ Results stored in Beads

**Timeline**: 2-3 days

---

### Phase 5 (novacron-aca) - Production Readiness

- ✅ E2E tests for all critical paths
- ✅ Load tests validate 1000+ concurrent ops
- ✅ Chaos tests demonstrate resilience
- ✅ Security audit passed
- ✅ Code coverage >90%
- ✅ Test execution <5 minutes
- ✅ CI/CD pipeline green

**Timeline**: 6-8 weeks

---

## Quick Command Reference

### Run Tests

```bash
# JavaScript tests
npm test
npm run test:unit
npm run test:integration
npm test -- --coverage

# Go tests
cd backend/core
go test ./...
go test -v ./network/dwcp/v3/...
go test -bench=. ./network/dwcp/v3/tests/
go test -cover ./...

# E2E tests (after setup)
npx playwright test
npx playwright test --headed

# Load tests (after setup)
k6 run tests/load/concurrent-ops/1000-concurrent-vms.js
```

### View Coverage

```bash
# JavaScript coverage
npm test -- --coverage
open coverage/lcov-report/index.html

# Go coverage
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out
```

---

## Coordination

### Memory Keys

```json
{
  "swarm/testing/priority-actions": [
    "Run DWCP v3 benchmarks (P0, 2-3 days)",
    "Update test documentation (P0, 1 day)",
    "Implement E2E framework (P1, 1 week)",
    "Implement load testing (P1, 1-2 weeks)"
  ],
  "swarm/testing/blockers": {
    "novacron-38p": "Need to run benchmarks (2-3 days)",
    "novacron-aca": "Need E2E and load tests (6-8 weeks)"
  }
}
```

### Coordination Hooks

```bash
# Start task
npx claude-flow@alpha hooks pre-task --description "Run DWCP v3 benchmarks"

# During work
npx claude-flow@alpha hooks notify --message "Benchmark execution in progress"

# Complete task
npx claude-flow@alpha hooks post-task --task-id "dwcp-benchmarks"
```

---

**Document Created**: November 10, 2025
**Next Review**: November 17, 2025 (after benchmarks)
**Status**: Ready for Implementation
