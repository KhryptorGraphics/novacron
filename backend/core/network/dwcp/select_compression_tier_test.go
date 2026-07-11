package dwcp

import "testing"

// TestSelectCompressionTier_LatencyThresholds closes a real coverage gap
// left by the migration_adapter_roundtrip_test.go suite (novacron-lce):
// every round trip there runs over 127.0.0.1, where AMST.Connect now
// measures real dial RTT and calls UpdateMetrics with it (fixed
// alongside novacron-38p's benchmark work) - but loopback's real RTT is
// sub-millisecond and truncates to 0ms, so selectCompressionTier still
// always returns CompressionLocal there in practice — none of those
// tests ever exercise
// CompressionRegional or CompressionGlobal. That matters specifically
// because HDE.CompressMemory only engages EnableQuantization's quantize()
// (irreversibly lossy, no dequantize anywhere in Decompress) at
// tier == CompressionGlobal: the flag is off by construction today
// (NewMigrationAdapter forces EnableQuantization=false), so the round-trip
// suite passing proves nothing about that path specifically — a future
// change re-enabling EnableQuantization would slip past every existing
// migration test silently, since none of them ever reach Global. This
// test exercises selectCompressionTier directly against its documented
// latency thresholds (migration_adapter.go: <5ms none-skip, <10ms local,
// <50ms regional, else global) so the None-skip and Global-tier selection
// paths both stay covered,
// independent of what tier any specific round-trip test happens to hit.
func TestSelectCompressionTier_LatencyThresholds(t *testing.T) {
	adapter, err := NewMigrationAdapter(MigrationAdapterConfig{EnableDWCP: true})
	if err != nil {
		t.Fatalf("NewMigrationAdapter failed: %v", err)
	}
	defer adapter.Close()

	cases := []struct {
		name      string
		latencyMs int64
		want      CompressionLevel
	}{
		{"lan_zero_latency", 0, CompressionLevelNone},
		{"lan_just_under_none_ceiling", 4, CompressionLevelNone},
		{"none_local_boundary", 5, CompressionLocal},
		{"just_under_local_ceiling", 9, CompressionLocal},
		{"local_regional_boundary", 10, CompressionRegional},
		{"mid_regional", 30, CompressionRegional},
		{"just_under_regional_ceiling", 49, CompressionRegional},
		{"regional_global_boundary", 50, CompressionGlobal},
		{"high_wan_latency", 200, CompressionGlobal},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			amst, err := NewAMST(AMSTConfig{MinStreams: 1, MaxStreams: 1, InitialStreams: 1})
			if err != nil {
				t.Fatalf("NewAMST failed: %v", err)
			}
			defer amst.Close()
			amst.UpdateMetrics(c.latencyMs, 0, 0)

			conn := &MigrationConnection{AMST: amst}
			got := adapter.selectCompressionTier(conn)
			if got != c.want {
				t.Errorf("selectCompressionTier(latency=%dms) = %v, want %v", c.latencyMs, got, c.want)
			}
		})
	}
}
