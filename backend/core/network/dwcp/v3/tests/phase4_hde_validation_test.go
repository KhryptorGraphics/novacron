package tests

import (
	"fmt"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// TestPhase4_HDE_Validation_Real replaces the HDE_Validation subtest that
// used to live inside TestPhase4_FinalIntegrationValidation (now removed
// from there — see that function's t.Skip comment). The original subtest
// asserted against simulateCompression, a stub that computes
// `ratio := 1.0 + level*0.3` and returns a same-size zero-filled slice
// without ever reading its data argument — at level=3 (Regional) that is
// exactly 1.9, deterministically, for any input, making the "Regional
// compression should achieve > 2x" assertion fail for a reason unrelated
// to real DWCP compression (confirmed via novacron-38p benchmark work,
// novacron-v4y). This test asserts the same claims against the real
// dwcp.HDE component instead.
//
// RegionalLevel/GlobalLevel are pinned explicitly (not left at the zero
// value) — NewHDE only overrides RegionalLevel when it is <0 or >22, so
// an unset zero value silently compresses "regional" at the same speed
// as "local" (zstd SpeedFastest), making the tier comparison meaningless
// (see benchmark_hde_compression_test.go's doc comment for the same
// gotcha, discovered in the same session).
func TestPhase4_HDE_Validation_Real(t *testing.T) {
	hde, err := dwcp.NewHDE(dwcp.HDEConfig{RegionalLevel: 3, GlobalLevel: 9})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	// Same 1MB compressible pattern as the original (now-removed)
	// HDE_Validation subtest.
	testData := make([]byte, 1024*1024)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	tiers := []struct {
		name      string
		level     dwcp.CompressionLevel
		minRatio  float64
		criterion string
	}{
		{"local", dwcp.CompressionLocal, 1.5, "Local compression should achieve > 1.5x"},
		{"regional", dwcp.CompressionRegional, 2.0, "Regional compression should achieve > 2x"},
		{"global", dwcp.CompressionGlobal, 3.0, "Global compression should achieve > 3x"},
	}

	for _, tier := range tiers {
		t.Run(tier.name, func(t *testing.T) {
			compressed, cerr := hde.CompressMemory(fmt.Sprintf("vm-validate-%s", tier.name), testData, tier.level)
			if cerr != nil {
				t.Fatalf("CompressMemory failed: %v", cerr)
			}
			ratio := float64(len(testData)) / float64(len(compressed))
			t.Logf("%s compression ratio: %.2fx (real HDE.CompressMemory, %d -> %d bytes)",
				tier.name, ratio, len(testData), len(compressed))
			if ratio <= tier.minRatio {
				t.Errorf("%s: got %.2fx, want > %.2fx (%s)", tier.name, ratio, tier.minRatio, tier.criterion)
			}
		})
	}

	// Delta encoding efficiency: real baseline-vs-delta comparison, not
	// simulateDeltaEncoding's raw byte-diff count. Compress the same
	// 10%-modified buffer twice - once against an established baseline
	// (delta-encoded) and once fresh (no baseline, independent vmID) -
	// and compare compressed sizes. This is what "delta should reduce
	// transferred size by > 80% when 10% of data changed" actually means
	// end-to-end, matching the pattern established in
	// BenchmarkMigrationPipelinePrimitivesDeltaReuse
	// (benchmark_migration_time_test.go) earlier in this session.
	t.Run("delta_encoding_efficiency", func(t *testing.T) {
		deltaHDE, derr := dwcp.NewHDE(dwcp.HDEConfig{GlobalLevel: 9, EnableDelta: true, BlockSize: 4096, DeltaThreshold: 0.7})
		if derr != nil {
			t.Fatalf("NewHDE failed: %v", derr)
		}
		defer deltaHDE.Close()

		base := generateVMMemoryData(len(testData))
		modified := make([]byte, len(base))
		copy(modified, base)
		modifiedCount := len(base) / 10 // 10% of data, matching the original subtest
		for i := 0; i < modifiedCount; i++ {
			modified[i] = ^modified[i]
		}

		// Fresh compression (independent vmID => no baseline to delta against).
		fresh, ferr := deltaHDE.CompressMemory("vm-delta-fresh", modified, dwcp.CompressionGlobal)
		if ferr != nil {
			t.Fatalf("fresh CompressMemory failed: %v", ferr)
		}

		// Baseline-then-delta: same vmID sees the unmodified data first,
		// then the 10%-modified data, so the second call can delta
		// against the stored baseline.
		const deltaVMID = "vm-delta-baseline"
		if _, err := deltaHDE.CompressMemory(deltaVMID, base, dwcp.CompressionGlobal); err != nil {
			t.Fatalf("baseline CompressMemory failed: %v", err)
		}
		delta, derr2 := deltaHDE.CompressMemory(deltaVMID, modified, dwcp.CompressionGlobal)
		if derr2 != nil {
			t.Fatalf("delta CompressMemory failed: %v", derr2)
		}

		efficiency := 1.0 - float64(len(delta))/float64(len(fresh))
		t.Logf("Delta encoding efficiency: %.1f%% smaller than fresh compression (fresh=%d bytes, delta=%d bytes)",
			efficiency*100, len(fresh), len(delta))
		if efficiency <= 0.8 {
			t.Errorf("delta encoding: got %.1f%% reduction vs fresh compression, want > 80%%", efficiency*100)
		}
	})
}
