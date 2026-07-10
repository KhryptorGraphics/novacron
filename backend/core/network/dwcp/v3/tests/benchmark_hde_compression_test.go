package tests

import (
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp"
)

// BenchmarkHDECompressionByEntropy measures real compression ratio across
// distinct entropy profiles at each compression tier.
//
// RegionalLevel is pinned to 3, matching the struct doc comment's stated
// default: NewHDE only overrides RegionalLevel when it is <0 or >22
// (hde.go:163-165), so an unset zero value silently stays 0 and the
// "regional" tier would otherwise compress at zstd SpeedFastest — the same
// speed as "local" — instead of the intended SpeedDefault, making any
// local-vs-regional comparison meaningless.
func BenchmarkHDECompressionByEntropy(b *testing.B) {
	dataPatterns := []struct {
		name string
		gen  func() []byte
	}{
		{"zeros", func() []byte { return make([]byte, 1024*1024) }},
		{"random", func() []byte { d := make([]byte, 1024*1024); rand.Read(d); return d }},
		{"text_repeating", func() []byte {
			d := make([]byte, 1024*1024)
			base := []byte("Lorem ipsum dolor sit amet, consectetur adipiscing elit. ")
			for i := 0; i < len(d); i += len(base) {
				n := copy(d[i:], base)
				if n == 0 {
					break
				}
			}
			return d
		}},
		{"mixed_vm_memory", func() []byte { return generateVMMemoryData(1024 * 1024) }},
	}

	tiers := []struct {
		name  string
		level dwcp.CompressionLevel
	}{
		{"local", dwcp.CompressionLocal},
		{"regional", dwcp.CompressionRegional},
		{"global", dwcp.CompressionGlobal},
	}

	for _, pattern := range dataPatterns {
		for _, tier := range tiers {
			b.Run(fmt.Sprintf("pattern_%s_tier_%s", pattern.name, tier.name), func(b *testing.B) {
				hde, err := dwcp.NewHDE(dwcp.HDEConfig{RegionalLevel: 3, GlobalLevel: 9})
				if err != nil {
					b.Fatal(err)
				}
				defer hde.Close()

				data := pattern.gen()
				b.SetBytes(int64(len(data)))
				b.ResetTimer()

				var totalCompressed int64
				for range b.N {
					compressed, err := hde.CompressMemory("vm-bench", data, tier.level)
					if err != nil {
						b.Fatal(err)
					}
					totalCompressed += int64(len(compressed))
				}
				avgCompressed := totalCompressed / int64(b.N)
				if avgCompressed > 0 {
					b.ReportMetric(float64(len(data))/float64(avgCompressed), "ratio")
				}
			})
		}
	}
}

// BenchmarkRegionalCompressionDetail isolates whether delta encoding,
// dictionary compression, or the raw zstd level move the regional-tier
// compression ratio. Uses a fixed data mix (60% zero pages, 40% random)
// approximating realistic VM memory across all configs so only the
// HDEConfig knobs vary.
//
// The "1.9x measured vs 2.0x target" framing this investigation started
// from is NOT a real measurement: it traces to
// TestPhase4_FinalIntegrationValidation/.../HDE_Validation
// (phase4_final_validation_test.go:63-82, quarantined via t.Skip citing
// this exact number) calling simulateCompression(data, level)
// (phase4_final_validation_test.go:350-354), a closed-form formula —
// ratio := 1.0 + float64(level)*0.3 — that never inspects data content at
// all. At level=3 this is exactly 1.9 for any input, deterministically. It
// never calls dwcp.HDE or zstd. The benchmarks below are the first real
// HDE.CompressMemory regional-tier measurements taken in this codebase;
// see novacron-38p for the full write-up and real numbers.
//
// Two verified correctness fixes vs. a naive version of this benchmark:
//
//  1. Every config sets RegionalLevel explicitly, not GlobalLevel — see the
//     RegionalLevel doc comment on BenchmarkHDECompressionByEntropy above.
//     Without it, every subtest here would silently compress at zstd
//     SpeedFastest regardless of the "level 3" a GlobalLevel:3 typo implies.
//  2. Every iteration uses a unique vmID (hde.go:229-256): CompressMemory
//     unconditionally overwrites the stored per-vmID baseline with whatever
//     it was just called with. A fixed vmID plus byte-identical data across
//     b.N iterations means every iteration after the first diffs the data
//     against itself (a zero-size delta), collapsing the reported "ratio"
//     into a meaningless, inflated number reflecting unchanged-data reuse,
//     not first-copy regional compression.
//
// Beyond the plan's original delta/dictionary knob axis, this also sweeps
// RegionalLevel across the three zstd speed buckets EncoderLevelFromZstd
// does not already cover via the level=3 configs above (SpeedFastest,
// SpeedBetterCompression, SpeedBestCompression — encoder_options.go in
// klauspost/compress@v1.18.1), added because CompressMemory never calls
// compressWithDict (only CompressDisk does, hde.go) and delta cannot help a
// guaranteed single-pass compression — so the delta/dict axis alone cannot
// answer "what would close the gap"; RegionalLevel is the only lever that
// can, empirically, for this API.
func BenchmarkRegionalCompressionDetail(b *testing.B) {
	configs := []struct {
		name string
		cfg  dwcp.HDEConfig
	}{
		{"no_delta_no_dict", dwcp.HDEConfig{RegionalLevel: 3}},
		{"delta_only", dwcp.HDEConfig{RegionalLevel: 3, EnableDelta: true, BlockSize: 4096, DeltaThreshold: 0.7}},
		{"delta_and_dict", dwcp.HDEConfig{RegionalLevel: 3, EnableDelta: true, EnableDictionary: true, DictSize: 1024, TrainingSamples: 100, BlockSize: 4096, DeltaThreshold: 0.7}},
		{"delta_and_dict_aggressive", dwcp.HDEConfig{RegionalLevel: 3, EnableDelta: true, EnableDictionary: true, DictSize: 4096, TrainingSamples: 500, BlockSize: 4096, DeltaThreshold: 0.5}},
		{"regional_level_1_fastest", dwcp.HDEConfig{RegionalLevel: 1}},
		{"regional_level_7_better", dwcp.HDEConfig{RegionalLevel: 7}},
		{"regional_level_15_best", dwcp.HDEConfig{RegionalLevel: 15}},
	}

	data := make([]byte, 1024*1024)
	rand.Read(data)
	for i := 0; i < len(data); i += 4096 {
		if (i/4096)%5 < 3 { // 60% of 4KB pages become zero pages
			end := i + 4096
			if end > len(data) {
				end = len(data)
			}
			for j := i; j < end; j++ {
				data[j] = 0
			}
		}
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			hde, err := dwcp.NewHDE(cfg.cfg)
			if err != nil {
				b.Fatal(err)
			}
			defer hde.Close()

			b.SetBytes(int64(len(data)))
			b.ResetTimer()

			var totalCompressed int64
			for i := range b.N {
				compressed, err := hde.CompressMemory(fmt.Sprintf("vm-bench-%d", i), data, dwcp.CompressionRegional)
				if err != nil {
					b.Fatal(err)
				}
				totalCompressed += int64(len(compressed))
			}
			avgCompressed := totalCompressed / int64(b.N)
			if avgCompressed > 0 {
				b.ReportMetric(float64(len(data))/float64(avgCompressed), "ratio")
			}
		})
	}
}
