package monitoring

import "testing"

func TestNetworkTelemetryMeasureLatencyUsesRecordedSample(t *testing.T) {
	nt := NewNetworkTelemetry()
	nt.UpdateTopology(&NetworkTopology{
		Links: []NetworkLink{
			{Source: "a", Destination: "b", Latency: 25},
		},
	})
	nt.RecordLatency("a", "b", 12.5, "TCP")

	if got := nt.measureLatency("a", "b"); got != 12.5 {
		t.Fatalf("measureLatency() = %f, want recorded sample 12.5", got)
	}
}

func TestNetworkTelemetryMeasureLatencyUsesTopology(t *testing.T) {
	nt := NewNetworkTelemetry()
	nt.UpdateTopology(&NetworkTopology{
		Links: []NetworkLink{
			{Source: "a", Destination: "b", Latency: 25},
		},
	})

	if got := nt.measureLatency("a", "b"); got != 25 {
		t.Fatalf("measureLatency() = %f, want topology latency 25", got)
	}
}

func TestNetworkTelemetryMeasureLatencyUsesTunnelHealth(t *testing.T) {
	nt := NewNetworkTelemetry()
	nt.UpdateTunnelHealth("tun-1", "a", "b", TunnelUp, map[string]interface{}{
		"latency": float64(33),
	})

	if got := nt.measureLatency("a", "b"); got != 33 {
		t.Fatalf("measureLatency() = %f, want tunnel latency 33", got)
	}
}

func TestNetworkTelemetryMeasureLatencyUnknownPair(t *testing.T) {
	nt := NewNetworkTelemetry()

	if got := nt.measureLatency("unknown-a", "unknown-b"); got != 0 {
		t.Fatalf("measureLatency() = %f, want 0 for unobserved pair", got)
	}
}
