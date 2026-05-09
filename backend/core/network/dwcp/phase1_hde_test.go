package dwcp_test

import (
	"bytes"
	"crypto/rand"
	"sync"
	"testing"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/compression"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
)

func TestPhase1_HDECompressionRoundTrip(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.CompressionLevel = 9

	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	vmMemory := bytes.Repeat([]byte("VM_MEMORY_PAGE_ZEROED_"), 512*1024)
	encoded, err := encoder.Encode("vm-compression-test", vmMemory)
	require.NoError(t, err)
	assert.Greater(t, encoded.CompressionRatio(), 1.0)

	decoded, err := encoder.Decode("vm-compression-test", encoded)
	require.NoError(t, err)
	assert.True(t, bytes.Equal(vmMemory, decoded), "Decompression should be lossless")
}

func TestPhase1_HDEDeltaRoundTrip(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	config := compression.DefaultDeltaEncodingConfig()
	config.DeltaAlgorithm = "bsdiff"

	encoder, err := compression.NewDeltaEncoder(config, logger)
	require.NoError(t, err)
	defer encoder.Close()

	stateKey := "vm-disk"
	baseDisk := make([]byte, 4*1024*1024)
	_, _ = rand.Read(baseDisk)

	baseEncoded, err := encoder.Encode(stateKey, baseDisk)
	require.NoError(t, err)
	assert.False(t, baseEncoded.IsDelta, "First encode should create a baseline")

	modifiedDisk := make([]byte, len(baseDisk))
	copy(modifiedDisk, baseDisk)
	for i := 0; i < len(modifiedDisk)/100; i++ {
		offset := i * 100
		modifiedDisk[offset] = ^modifiedDisk[offset]
	}

	deltaEncoded, err := encoder.Encode(stateKey, modifiedDisk)
	require.NoError(t, err)
	assert.True(t, deltaEncoded.IsDelta, "Second encode should use delta encoding")

	decoded, err := encoder.Decode(stateKey, deltaEncoded)
	require.NoError(t, err)
	assert.True(t, bytes.Equal(modifiedDisk, decoded), "Delta reconstruction should be accurate")
}

func TestPhase1_HDEMetrics(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	encoder, err := compression.NewDeltaEncoder(compression.DefaultDeltaEncodingConfig(), logger)
	require.NoError(t, err)
	defer encoder.Close()

	testData := make([]byte, 4*1024*1024)
	_, _ = rand.Read(testData)

	_, err = encoder.Encode("metrics-test", testData)
	require.NoError(t, err)

	metrics := encoder.GetMetrics()
	assert.Equal(t, uint64(1), metrics["total_encoded"].(uint64))
	assert.Equal(t, 1, metrics["baseline_count"].(int))
	assert.Contains(t, metrics, "compression_level")
}

func TestPhase1_HDEConcurrency(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	encoder, err := compression.NewDeltaEncoder(compression.DefaultDeltaEncodingConfig(), logger)
	require.NoError(t, err)
	defer encoder.Close()

	const numGoroutines = 20
	const dataSize = 512 * 1024

	var wg sync.WaitGroup
	errChan := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			data := make([]byte, dataSize)
			_, _ = rand.Read(data)

			if _, err := encoder.Encode("concurrent-"+string(rune('A'+id)), data); err != nil {
				errChan <- err
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		require.NoError(t, err)
	}

	metrics := encoder.GetMetrics()
	assert.Equal(t, uint64(numGoroutines), metrics["total_encoded"].(uint64))
}
