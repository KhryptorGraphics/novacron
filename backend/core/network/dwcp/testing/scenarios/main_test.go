package scenarios

import (
	"math/rand"
	"testing"
)

// TestMain seeds the legacy global RNG so this package's simulations are
// REPRODUCIBLE.
//
// The scenarios here drive a simulator whose packet-loss decision is
// `rand.Float64() < profile.Rate` and whose latency model draws from the
// same global source (network_simulator.go:254-285,350; chaos_engineering.go:
// 157). The tests then assert a statistical contract on the outcome - e.g.
// TestMultiRegionFailover requires a >= 95% success rate and
// TestPacketLossResilience a >= 98% one.
//
// Without a fixed seed that is a dice roll, not an assertion: a draw that
// drops 2 of 20 operations is a legitimate outcome of the configured loss
// rate, and it failed the arm64 CI leg on 2026-10-29 with exactly that
// message ("Success rate too low: 90.00% (expected >= 95%)") while passing
// 20/20 locally. A test that fails on a valid draw is a flaky test, and it
// makes the gate untrustworthy.
//
// This seeds the LEGACY math/rand global on purpose. The simulator reads the
// legacy global generator directly (math/rand/v2 has no global seed and no
// process-wide source to seed), so v2 cannot be used here without threading
// an explicit *rand.Rand through the simulator, the chaos engine and the
// workload generator - a much larger change than the fix this test needs.
// The seed is fixed, not derived from the clock: the point is that the same
// input yields the same result on every machine, so a failure reproduces.
func TestMain(m *testing.M) {
	rand.Seed(20261029)
	m.Run()
}
