package testing

import (
	"crypto/rand"
	"encoding/binary"
	"math"
	mathrand "math/rand"
	"time"
)

// WorkloadGenerator generates realistic VM memory workloads
type WorkloadGenerator struct {
	pattern WorkloadPattern
	size    int64
	seed    int64
}

// NewWorkloadGenerator creates a new workload generator
func NewWorkloadGenerator(pattern WorkloadPattern, size int64) *WorkloadGenerator {
	return &WorkloadGenerator{
		pattern: pattern,
		size:    size,
		seed:    time.Now().UnixNano(),
	}
}

// GenerateVMMemory generates VM memory data based on the pattern
func (wg *WorkloadGenerator) GenerateVMMemory(size int64) []byte {
	data := make([]byte, size)

	switch wg.pattern {
	case PatternRealWorld:
		wg.fillWithRealisticData(data)
	case PatternBursty:
		wg.fillWithBurstyData(data)
	case PatternSinusoidal:
		wg.fillWithSinusoidalData(data)
	default:
		wg.fillWithConstantData(data)
	}

	return data
}

// fillWithRealisticData simulates real VM memory patterns
func (wg *WorkloadGenerator) fillWithRealisticData(data []byte) {
	size := len(data)
	offset := 0

	// 1. Zero pages (30% - common in VMs)
	zeroRatio := 0.3
	zeroBytes := int(float64(size) * zeroRatio)
	// Zero pages are already zero-initialized
	offset += zeroBytes

	// 2. Repetitive patterns (40% - OS code, libraries, page cache)
	repetitiveRatio := 0.4
	repetitiveBytes := int(float64(size) * repetitiveRatio)
	wg.fillRepetitivePattern(data[offset:offset+repetitiveBytes])
	offset += repetitiveBytes

	// 3. Low-entropy data (20% - text, logs, structured data)
	lowEntropyRatio := 0.2
	lowEntropyBytes := int(float64(size) * lowEntropyRatio)
	wg.fillLowEntropyData(data[offset:offset+lowEntropyBytes])
	offset += lowEntropyBytes

	// 4. High-entropy data (10% - encrypted data, random working set)
	highEntropyBytes := size - offset
	rand.Read(data[offset:offset+highEntropyBytes])
}

// fillRepetitivePattern fills with repetitive patterns
func (wg *WorkloadGenerator) fillRepetitivePattern(data []byte) {
	patterns := [][]byte{
		[]byte("KERNEL_CODE_SECTION_"),
		[]byte("LIBRARY_FUNCTION_XX_"),
		[]byte("PAGE_CACHE_ENTRY_000"),
		[]byte("STACK_FRAME_POINTER_"),
	}

	patternIdx := 0
	offset := 0

	for offset < len(data) {
		pattern := patterns[patternIdx%len(patterns)]
		copySize := len(pattern)
		if offset+copySize > len(data) {
			copySize = len(data) - offset
		}
		copy(data[offset:offset+copySize], pattern)
		offset += copySize
		patternIdx++
	}
}

// fillLowEntropyData fills with low-entropy data (text-like)
func (wg *WorkloadGenerator) fillLowEntropyData(data []byte) {
	// Simulate text data with limited alphabet
	alphabet := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \n"
	rng := mathrand.New(mathrand.NewSource(wg.seed))

	for i := 0; i < len(data); i++ {
		data[i] = alphabet[rng.Intn(len(alphabet))]
	}
}

// fillWithBurstyData fills with bursty patterns
func (wg *WorkloadGenerator) fillWithBurstyData(data []byte) {
	rng := mathrand.New(mathrand.NewSource(wg.seed))
	offset := 0
	burstSize := 64 * 1024 // 64 KB bursts

	for offset < len(data) {
		// Random burst or zero
		if rng.Float64() < 0.3 { // 30% bursts
			size := burstSize
			if offset+size > len(data) {
				size = len(data) - offset
			}
			rand.Read(data[offset : offset+size])
			offset += size
		} else {
			// Zero region
			size := burstSize * 2
			if offset+size > len(data) {
				size = len(data) - offset
			}
			offset += size
		}
	}
}

// fillWithSinusoidalData fills with sinusoidal patterns
func (wg *WorkloadGenerator) fillWithSinusoidalData(data []byte) {
	for i := 0; i < len(data); i++ {
		// Create sinusoidal pattern
		phase := float64(i) / 1000.0
		value := 128 + 127*math.Sin(phase)
		data[i] = byte(value)
	}
}

// fillWithConstantData fills with constant random data
func (wg *WorkloadGenerator) fillWithConstantData(data []byte) {
	rand.Read(data)
}

// GenerateMemorySnapshot generates a complete memory snapshot
func (wg *WorkloadGenerator) GenerateMemorySnapshot(vmID string, size int64) *MemorySnapshot {
	data := wg.GenerateVMMemory(size)

	return &MemorySnapshot{
		VMID:      vmID,
		Timestamp: time.Now(),
		Size:      size,
		Data:      data,
		Metadata: MemoryMetadata{
			ZeroPages:      wg.countZeroPages(data),
			CompressRatio:  wg.estimateCompressibility(data),
			Entropy:        wg.calculateEntropy(data),
		},
	}
}

// MemorySnapshot represents a VM memory snapshot
type MemorySnapshot struct {
	VMID      string
	Timestamp time.Time
	Size      int64
	Data      []byte
	Metadata  MemoryMetadata
}

// MemoryMetadata contains metadata about memory
type MemoryMetadata struct {
	ZeroPages     int
	CompressRatio float64
	Entropy       float64
}

// countZeroPages counts the number of zero pages (4KB)
func (wg *WorkloadGenerator) countZeroPages(data []byte) int {
	pageSize := 4096
	zeroPages := 0

	for offset := 0; offset < len(data); offset += pageSize {
		end := offset + pageSize
		if end > len(data) {
			end = len(data)
		}

		isZero := true
		for i := offset; i < end; i++ {
			if data[i] != 0 {
				isZero = false
				break
			}
		}

		if isZero {
			zeroPages++
		}
	}

	return zeroPages
}

// estimateCompressibility estimates compression ratio
func (wg *WorkloadGenerator) estimateCompressibility(data []byte) float64 {
	// Simple estimation based on zero bytes and repetition
	zeroCount := 0
	for _, b := range data {
		if b == 0 {
			zeroCount++
		}
	}

	zeroRatio := float64(zeroCount) / float64(len(data))

	// Rough estimation: more zeros = better compression
	// Real-world VMs typically compress 3-20x
	estimatedRatio := 3.0 + (zeroRatio * 17.0)

	return estimatedRatio
}

// calculateEntropy calculates Shannon entropy
func (wg *WorkloadGenerator) calculateEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0
	}

	// Count byte frequencies
	freq := make([]int, 256)
	for _, b := range data {
		freq[b]++
	}

	// Calculate entropy
	entropy := 0.0
	size := float64(len(data))

	for _, count := range freq {
		if count > 0 {
			p := float64(count) / size
			entropy -= p * math.Log2(p)
		}
	}

	return entropy
}

// WorkloadScheduler schedules workload generation
type WorkloadScheduler struct {
	generator  *WorkloadGenerator
	workload   *Workload
	operations chan *WorkloadOperation
}

// WorkloadOperation represents a single operation
type WorkloadOperation struct {
	ID        int
	Type      WorkloadType
	VMSize    int64
	Timestamp time.Time
	Source    string
	Target    string
}

// NewWorkloadScheduler creates a new workload scheduler
func NewWorkloadScheduler(workload *Workload) *WorkloadScheduler {
	generator := NewWorkloadGenerator(workload.Pattern, workload.VMSize)

	return &WorkloadScheduler{
		generator:  generator,
		workload:   workload,
		operations: make(chan *WorkloadOperation, workload.Operations),
	}
}

// Schedule schedules workload operations
func (ws *WorkloadScheduler) Schedule() {
	defer close(ws.operations)

	switch ws.workload.Pattern {
	case PatternBursty:
		ws.scheduleBursty()
	case PatternSinusoidal:
		ws.scheduleSinusoidal()
	case PatternConstant:
		ws.scheduleConstant()
	default:
		ws.scheduleConstant()
	}
}

// scheduleConstant schedules operations at constant rate
func (ws *WorkloadScheduler) scheduleConstant() {
	interval := ws.workload.ThinkTime
	if interval == 0 {
		interval = 5 * time.Second
	}

	for i := 0; i < ws.workload.Operations; i++ {
		op := &WorkloadOperation{
			ID:        i,
			Type:      ws.workload.Type,
			VMSize:    ws.workload.VMSize,
			Timestamp: time.Now(),
		}
		ws.operations <- op

		if i < ws.workload.Operations-1 {
			time.Sleep(interval)
		}
	}
}

// scheduleBursty schedules operations in bursts
func (ws *WorkloadScheduler) scheduleBursty() {
	burstSize := ws.workload.Concurrency * 2
	burstInterval := ws.workload.ThinkTime * 5

	for i := 0; i < ws.workload.Operations; {
		// Send burst
		for j := 0; j < burstSize && i < ws.workload.Operations; j++ {
			op := &WorkloadOperation{
				ID:        i,
				Type:      ws.workload.Type,
				VMSize:    ws.workload.VMSize,
				Timestamp: time.Now(),
			}
			ws.operations <- op
			i++
		}

		// Wait before next burst
		if i < ws.workload.Operations {
			time.Sleep(burstInterval)
		}
	}
}

// scheduleSinusoidal schedules operations in sinusoidal pattern
func (ws *WorkloadScheduler) scheduleSinusoidal() {
	baseInterval := ws.workload.ThinkTime
	if baseInterval == 0 {
		baseInterval = 5 * time.Second
	}

	for i := 0; i < ws.workload.Operations; i++ {
		op := &WorkloadOperation{
			ID:        i,
			Type:      ws.workload.Type,
			VMSize:    ws.workload.VMSize,
			Timestamp: time.Now(),
		}
		ws.operations <- op

		if i < ws.workload.Operations-1 {
			// Vary interval sinusoidally
			phase := float64(i) / float64(ws.workload.Operations) * 2 * math.Pi
			factor := 0.5 + 0.5*math.Sin(phase)
			interval := time.Duration(float64(baseInterval) * factor)
			time.Sleep(interval)
		}
	}
}

// GetOperations returns the operations channel
func (ws *WorkloadScheduler) GetOperations() <-chan *WorkloadOperation {
	return ws.operations
}

// GenerateNetworkTraffic generates network traffic pattern
func (wg *WorkloadGenerator) GenerateNetworkTraffic(duration time.Duration, targetBandwidth int) []TrafficSample {
	samples := make([]TrafficSample, 0)
	interval := 100 * time.Millisecond
	numSamples := int(duration / interval)

	rng := mathrand.New(mathrand.NewSource(wg.seed))

	for i := 0; i < numSamples; i++ {
		var bandwidth int

		switch wg.pattern {
		case PatternConstant:
			bandwidth = targetBandwidth
		case PatternBursty:
			if rng.Float64() < 0.2 {
				bandwidth = int(float64(targetBandwidth) * 1.5)
			} else {
				bandwidth = int(float64(targetBandwidth) * 0.5)
			}
		case PatternSinusoidal:
			phase := float64(i) / float64(numSamples) * 2 * math.Pi
			factor := 0.5 + 0.5*math.Sin(phase)
			bandwidth = int(float64(targetBandwidth) * factor)
		default:
			bandwidth = targetBandwidth
		}

		samples = append(samples, TrafficSample{
			Timestamp: time.Now().Add(time.Duration(i) * interval),
			Bandwidth: bandwidth,
		})
	}

	return samples
}

// TrafficSample represents a traffic measurement sample
type TrafficSample struct {
	Timestamp time.Time
	Bandwidth int // Mbps
}

// Random utilities for generating realistic data
func generateRandomIP() string {
	var ip [4]byte
	binary.LittleEndian.PutUint32(ip[:], mathrand.Uint32())
	return fmt.Sprintf("%d.%d.%d.%d", ip[0], ip[1], ip[2], ip[3])
}

func generateRandomMAC() string {
	mac := make([]byte, 6)
	rand.Read(mac)
	return fmt.Sprintf("%02x:%02x:%02x:%02x:%02x:%02x",
		mac[0], mac[1], mac[2], mac[3], mac[4], mac[5])
}
