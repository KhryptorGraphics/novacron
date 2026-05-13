package upgrade

import (
	"context"
	"testing"
	"time"
)

type staticModeMetrics struct {
	latency   time.Duration
	bandwidth int64
}

func (m staticModeMetrics) GetAverageLatency() time.Duration {
	return m.latency
}

func (m staticModeMetrics) GetAverageBandwidth() int64 {
	return m.bandwidth
}

func TestModeDetectorUsesMetricsCollector(t *testing.T) {
	tests := []struct {
		name      string
		metrics   staticModeMetrics
		wantMode  NetworkMode
		startMode NetworkMode
	}{
		{
			name: "datacenter",
			metrics: staticModeMetrics{
				latency:   5 * time.Millisecond,
				bandwidth: 2_000_000_000,
			},
			wantMode:  ModeDatacenter,
			startMode: ModeHybrid,
		},
		{
			name: "internet_by_latency",
			metrics: staticModeMetrics{
				latency:   75 * time.Millisecond,
				bandwidth: 2_000_000_000,
			},
			wantMode:  ModeInternet,
			startMode: ModeHybrid,
		},
		{
			name: "internet_by_bandwidth",
			metrics: staticModeMetrics{
				latency:   5 * time.Millisecond,
				bandwidth: 100_000_000,
			},
			wantMode:  ModeInternet,
			startMode: ModeHybrid,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			detector := NewModeDetector()
			detector.ForceMode(tt.startMode)
			detector.SetMetricsCollector(tt.metrics)

			if got := detector.DetectMode(context.Background()); got != tt.wantMode {
				t.Fatalf("DetectMode() = %s, want %s", got, tt.wantMode)
			}
		})
	}
}

func TestModeDetectorDefaultsToHybridWithoutMeasurements(t *testing.T) {
	detector := NewModeDetector()

	if got := detector.DetectMode(context.Background()); got != ModeHybrid {
		t.Fatalf("DetectMode() = %s, want %s", got, ModeHybrid)
	}
}

func TestAutoDetectLoopReportsModeChanges(t *testing.T) {
	detector := NewModeDetector()
	detector.ForceMode(ModeInternet)
	detector.SetMetricsCollector(staticModeMetrics{
		latency:   5 * time.Millisecond,
		bandwidth: 2_000_000_000,
	})

	changes := make(chan [2]NetworkMode, 1)
	detector.SetModeChangeHandler(func(previous, current NetworkMode) {
		changes <- [2]NetworkMode{previous, current}
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go detector.AutoDetectLoop(ctx, 10*time.Millisecond)

	select {
	case change := <-changes:
		if change[0] != ModeInternet || change[1] != ModeDatacenter {
			t.Fatalf("mode change = %s -> %s, want %s -> %s", change[0], change[1], ModeInternet, ModeDatacenter)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for mode change callback")
	}
}
