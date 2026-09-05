package dwcp

import "time"

// SetMetricsIntervalForTest shortens the metrics tick so lifecycle tests can
// provoke Stop/tick interleavings deterministically. Call before Start.
func SetMetricsIntervalForTest(m *Manager, d time.Duration) { m.metricsInterval = d }
