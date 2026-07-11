package upgrade

import (
	"context"
	"testing"
	"time"
)

// fakeMetricsCollector drives DetectMode to a deterministic mode by returning
// fixed latency/bandwidth. It satisfies the anonymous interface that
// SetMetricsCollector accepts.
type fakeMetricsCollector struct {
	latency   time.Duration
	bandwidth int64
}

func (f fakeMetricsCollector) GetAverageLatency() time.Duration { return f.latency }
func (f fakeMetricsCollector) GetAverageBandwidth() int64       { return f.bandwidth }

// TestAutoDetectLoop_FiresHandlerOnModeChange proves the AutoDetectLoop
// change-detection branch actually executes when the detected mode differs from
// the previous mode.
//
// Discrimination: DetectMode assigns md.currentMode internally before returning,
// so the pre-fix comparison `mode != md.GetCurrentMode()` was always false and
// the branch (and any handler placed in it) could never fire. With the fix
// (capture oldMode via GetCurrentMode BEFORE calling DetectMode) the branch runs
// and the handler is invoked with (previous, new). Reverting the fix makes this
// test hang until the 2s timeout and fail.
func TestAutoDetectLoop_FiresHandlerOnModeChange(t *testing.T) {
	md := NewModeDetector()

	// New detectors start in ModeHybrid. A 100ms average latency crosses the
	// 50ms internet threshold, so the first detection cycle switches to
	// ModeInternet -- a genuine change from the initial mode.
	md.SetMetricsCollector(fakeMetricsCollector{
		latency:   100 * time.Millisecond,
		bandwidth: 10e9,
	})

	if got := md.GetCurrentMode(); got != ModeHybrid {
		t.Fatalf("precondition: expected initial mode %v, got %v", ModeHybrid, got)
	}

	type change struct{ from, to NetworkMode }
	changes := make(chan change, 4)
	md.SetModeChangeHandler(func(oldMode, newMode NetworkMode) {
		select {
		case changes <- change{from: oldMode, to: newMode}:
		default:
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go md.AutoDetectLoop(ctx, 5*time.Millisecond)

	select {
	case c := <-changes:
		if c.from != ModeHybrid {
			t.Errorf("handler old mode = %v, want %v", c.from, ModeHybrid)
		}
		if c.to != ModeInternet {
			t.Errorf("handler new mode = %v, want %v", c.to, ModeInternet)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("mode-change handler never fired: AutoDetectLoop change-detection branch did not execute")
	}
}
