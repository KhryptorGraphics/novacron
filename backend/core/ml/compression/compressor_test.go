package compression

import (
	"testing"
)

func TestQuantization(t *testing.T) {
	config := &CompressionConfig{
		QuantizationBits: 8,
		MaxAccuracyLoss:  0.05,
	}

	compressor := NewModelCompressor(config)

	weights := [][]float64{
		{0.1, 0.2, 0.3},
		{0.4, 0.5, 0.6},
		{0.7, 0.8, 0.9},
	}

	result, compressed, err := compressor.quantize(weights)
	if err != nil {
		t.Fatalf("quantization failed: %v", err)
	}

	if result.CompressionRatio < 2.0 {
		t.Errorf("expected compression ratio > 2.0, got %.2f", result.CompressionRatio)
	}

	t.Logf("Quantization: %.2fx compression, %.4f accuracy loss",
		result.CompressionRatio, result.AccuracyLoss)

	if len(compressed) != len(weights) {
		t.Error("compressed weights dimension mismatch")
	}
}

func TestPruning(t *testing.T) {
	config := &CompressionConfig{
		PruningRatio:    0.5,
		MaxAccuracyLoss: 0.1,
	}

	compressor := NewModelCompressor(config)

	weights := [][]float64{
		{0.1, 0.2, 0.001},
		{0.002, 0.5, 0.003},
		{0.7, 0.001, 0.9},
	}

	result, pruned, err := compressor.prune(weights)
	if err != nil {
		t.Fatalf("pruning failed: %v", err)
	}

	zeroCount := 0
	for i := range pruned {
		for j := range pruned[i] {
			if pruned[i][j] == 0 {
				zeroCount++
			}
		}
	}

	expectedZeros := int(float64(len(weights)*len(weights[0])) * config.PruningRatio)
	if zeroCount < expectedZeros-1 {
		t.Errorf("expected ~%d zeros, got %d", expectedZeros, zeroCount)
	}

	t.Logf("Pruning: %.2fx compression, %d zeros",
		result.CompressionRatio, zeroCount)
}

func TestLowRankFactorization(t *testing.T) {
	config := &CompressionConfig{
		TargetCompression: 5.0,
	}

	compressor := NewModelCompressor(config)

	weights := [][]float64{
		{1, 2, 3, 4},
		{2, 4, 6, 8},
		{3, 6, 9, 12},
	}

	result, compressed, err := compressor.lowRankFactorization(weights)
	if err != nil {
		t.Fatalf("low-rank factorization failed: %v", err)
	}

	if result.CompressionRatio < 1.0 {
		t.Error("compression ratio should be >= 1.0")
	}

	t.Logf("Low-rank: %.2fx compression, rank=%d",
		result.CompressionRatio, int(result.Metrics["rank"]))

	if len(compressed) != len(weights) || len(compressed[0]) != len(weights[0]) {
		t.Error("reconstructed weights dimension mismatch")
	}
}
