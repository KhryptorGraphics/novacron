// Package dwcp implements the Distributed WAN Communication Protocol for NovaCron
package dwcp

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"
	"sync/atomic"
	"time"
)

// MigrationAdapter provides DWCP-optimized migration capabilities
// High-level API for VM migration service integration
type MigrationAdapter struct {
	// Core components
	amst   *AMST
	hde    *HDE
	config MigrationAdapterConfig

	// Connection management
	connections map[string]*MigrationConnection
	connPool    sync.Pool

	// Baseline management
	vmBaselines map[string]*VMBaseline

	// Performance metrics
	migrationsCompleted   atomic.Int64
	migrationsFailed      atomic.Int64
	totalBytesTransferred atomic.Int64
	averageSpeedup        atomic.Value // float64

	// Synchronization
	mu     sync.RWMutex
	ctx    context.Context
	cancel context.CancelFunc
}

// MigrationAdapterConfig contains configuration for the migration adapter
type MigrationAdapterConfig struct {
	// DWCP settings
	EnableDWCP     bool // Enable DWCP optimization (default: true)
	EnableFallback bool // Enable fallback to standard TCP (default: true)

	// AMST configuration
	AMSTConfig AMSTConfig

	// HDE configuration
	HDEConfig HDEConfig

	// Network settings
	ListenPort        int           // Port for incoming migrations (default: 9876)
	ConnectionTimeout time.Duration // Connection timeout (default: 30s)

	// Performance targets
	TargetSpeedup  float64 // Target speedup over baseline (default: 2.5x)
	MaxMemoryUsage int64   // Maximum memory for caching (default: 2GB)

	// Monitoring
	MetricsInterval time.Duration // Metrics collection interval (default: 10s)

	// Receive-side hooks. Invoked by receiveMemory/receiveDisk once a
	// migration has been fully received, reassembled, and (for the DWCP
	// path) decompressed. Optional: if nil, received data is logged and
	// counted in metrics but otherwise discarded — this package is a
	// network transport, not a hypervisor integration; applying received
	// state to a live VM is the caller's responsibility (see
	// backend/core/migration/orchestrator_dwcp.go for the intended
	// caller). data is the fully reconstructed original bytes (already
	// decompressed for the DWCP path) — never partial, never compressed.
	OnMemoryReceived func(vmID string, data []byte)
	OnDiskReceived   func(vmID string, blockID int, data []byte)
}

// Wire envelope written directly on the raw net.Conn, immediately after
// connect and before any payload-specific framing, by both the standard
// and DWCP send paths (migrateMemoryStandard/migrateDiskStandard/
// MigrateVMMemory/MigrateVMDisk) and read by handleIncomingMigration
// before dispatch. Format: [protocol:1][vmIDLen:2 big-endian][vmID].
//
// Replaces the pre-fix single type byte, which was read unconditionally
// by handleIncomingMigration but never actually written by ANY sender:
// migrateMemoryStandard/migrateDiskStandard send an 8-byte size header
// with no type prefix at all, and the DWCP path (via AMST.Connect+
// Transfer) never wrote one either — so the receive side has never been
// able to correctly frame either path (see novacron-lce).
const (
	migrateProtoStandardMemory byte = 0
	migrateProtoStandardDisk   byte = 1
	migrateProtoDWCPMemory     byte = 2
	migrateProtoDWCPDisk       byte = 3
)

// writeMigrateEnvelope writes the protocol+vmID preamble to conn.
func writeMigrateEnvelope(conn net.Conn, protocol byte, vmID string) error {
	if len(vmID) > 65535 {
		return fmt.Errorf("vmID too long for envelope: %d bytes", len(vmID))
	}
	buf := make([]byte, 0, 3+len(vmID))
	buf = append(buf, protocol)
	buf = binary.BigEndian.AppendUint16(buf, uint16(len(vmID)))
	buf = append(buf, vmID...)
	_, err := conn.Write(buf)
	return err
}

// readMigrateEnvelope reads the protocol+vmID preamble from conn.
func readMigrateEnvelope(conn net.Conn) (protocol byte, vmID string, err error) {
	header := make([]byte, 3)
	if _, err := io.ReadFull(conn, header); err != nil {
		return 0, "", fmt.Errorf("failed to read envelope header: %w", err)
	}
	protocol = header[0]
	vmIDLen := binary.BigEndian.Uint16(header[1:3])
	vmIDBytes := make([]byte, vmIDLen)
	if vmIDLen > 0 {
		if _, err := io.ReadFull(conn, vmIDBytes); err != nil {
			return 0, "", fmt.Errorf("failed to read envelope vmID: %w", err)
		}
	}
	return protocol, string(vmIDBytes), nil
}

// MigrationConnection represents a DWCP migration connection
type MigrationConnection struct {
	ID               string
	SourceHost       string
	TargetHost       string
	AMST             *AMST
	StartTime        time.Time
	State            MigrationState
	BytesTransferred int64
	mu               sync.Mutex
}

// MigrationState represents the state of a migration
type MigrationState int

const (
	MigrationStateInit MigrationState = iota
	MigrationStateConnecting
	MigrationStateTransferring
	MigrationStateVerifying
	MigrationStateCompleted
	MigrationStateFailed
)

// VMBaseline stores VM state baselines for delta encoding
type VMBaseline struct {
	VMID           string
	MemoryBaseline []byte
	DiskBaselines  map[int][]byte // Block ID to baseline data
	LastUpdated    time.Time
	mu             sync.RWMutex
}

// NewMigrationAdapter creates a new DWCP migration adapter
func NewMigrationAdapter(config MigrationAdapterConfig) (*MigrationAdapter, error) {
	// Set defaults
	if config.ListenPort <= 0 {
		config.ListenPort = 9876
	}
	if config.ConnectionTimeout <= 0 {
		config.ConnectionTimeout = 30 * time.Second
	}
	if config.TargetSpeedup <= 0 {
		config.TargetSpeedup = 2.5
	}
	if config.MaxMemoryUsage <= 0 {
		config.MaxMemoryUsage = 2 * 1024 * 1024 * 1024 // 2GB
	}
	if config.MetricsInterval <= 0 {
		config.MetricsInterval = 10 * time.Second
	}

	// The receive path (receiveMemory/receiveDisk) round-trips through
	// HDE.Decompress, which cannot correctly reverse everything
	// HDE.CompressMemory/CompressDisk can produce:
	//   - EnableDelta: Decompress's isDelta branch returns the raw delta
	//     bytes as-is, not memory reconstructed against a baseline
	//     (hde.go Decompress, "For now, return the decompressed delta").
	//   - EnableDictionary: CompressDisk dict-compresses THEN plain
	//     zstd-compresses on top (double compression) whenever a trained
	//     dictionary exists; Decompress only reverses the outer plain
	//     layer, never the dictionary layer (hde.go CompressDisk).
	//   - EnableQuantization: quantize() masks off low bits of every byte
	//     at tier CompressionGlobal — irreversible information loss with
	//     no dequantize step anywhere in Decompress (hde.go quantize).
	// All three are forced off here so the receive path is correct by
	// construction, independent of what a caller passes — see novacron-o0e.
	config.HDEConfig.EnableDelta = false
	config.HDEConfig.EnableDictionary = false
	config.HDEConfig.EnableQuantization = false

	// The receive path correlates one accepted net.Conn to exactly one
	// migration by relying on AMST using exactly one stream per Transfer
	// (see the envelope-preamble comment on migrateEnvelope). Multi-stream
	// receive-side session correlation is real follow-up work, not solved
	// here — see the "N-stream migration receive" bd issue filed alongside
	// this fix. MinStreams/MaxStreams must ALL be forced, not just
	// InitialStreams: NewAMST floors InitialStreams up to MinStreams
	// (default 4) when MinStreams is unset (amst.go), and EnableAdaptive's
	// optimize() would otherwise drift streamCount back toward MinStreams
	// on live traffic (amst.go optimize()).
	config.AMSTConfig.MinStreams = 1
	config.AMSTConfig.MaxStreams = 1
	config.AMSTConfig.InitialStreams = 1
	config.AMSTConfig.EnableAdaptive = false

	ctx, cancel := context.WithCancel(context.Background())

	adapter := &MigrationAdapter{
		config:      config,
		connections: make(map[string]*MigrationConnection),
		vmBaselines: make(map[string]*VMBaseline),
		ctx:         ctx,
		cancel:      cancel,
	}

	// Initialize average speedup
	adapter.averageSpeedup.Store(float64(1.0))

	// Create AMST instance if DWCP is enabled
	if config.EnableDWCP {
		amst, err := NewAMST(config.AMSTConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create AMST: %w", err)
		}
		adapter.amst = amst

		// Create HDE instance
		hde, err := NewHDE(config.HDEConfig)
		if err != nil {
			return nil, fmt.Errorf("failed to create HDE: %w", err)
		}
		adapter.hde = hde
	}

	// Create connection pool
	adapter.connPool = sync.Pool{
		New: func() interface{} {
			return &MigrationConnection{}
		},
	}

	// Start metrics collector
	go adapter.metricsLoop()

	return adapter, nil
}

// MigrateVMMemory migrates VM memory using DWCP optimization
func (adapter *MigrationAdapter) MigrateVMMemory(ctx context.Context, vmID string, memoryData []byte, targetHost string, progressCallback func(int64)) error {
	if !adapter.config.EnableDWCP {
		return adapter.migrateMemoryStandard(ctx, vmID, memoryData, targetHost, progressCallback)
	}

	startTime := time.Now()
	originalSize := int64(len(memoryData))

	// Create connection (always fresh — see createConnection)
	conn, err := adapter.createConnection(ctx, vmID, targetHost)
	if err != nil {
		if adapter.config.EnableFallback {
			// Fallback to standard migration
			return adapter.migrateMemoryStandard(ctx, vmID, memoryData, targetHost, progressCallback)
		}
		return fmt.Errorf("failed to establish DWCP connection: %w", err)
	}

	// Determine compression tier based on network latency
	tier := adapter.selectCompressionTier(conn)

	// Compress memory with HDE
	compressed, err := adapter.hde.CompressMemory(vmID, memoryData, tier)
	if err != nil {
		adapter.CleanupConnection(vmID, targetHost)
		return fmt.Errorf("memory compression failed: %w", err)
	}

	compressionRatio := float64(originalSize) / float64(len(compressed))
	fmt.Printf("DWCP: Memory compressed from %d to %d bytes (%.2fx compression)\n",
		originalSize, len(compressed), compressionRatio)

	// Write the envelope on the connection's single stream immediately
	// before AMST.Transfer chunk data flows — deliberately AFTER
	// compression, not before: the receiver's very next read blocks on
	// the first AMST chunk header, and writing the envelope before a
	// (potentially long, for large payloads at high compression levels)
	// CompressMemory call would make the receiver wait out that entire
	// compression time against its ReadTimeout with nothing having gone
	// wrong. So the receiver's envelope-read is immediately followed by
	// data that is already fully ready to send.
	streamConn, err := conn.AMST.singleStreamConn()
	if err != nil {
		adapter.CleanupConnection(vmID, targetHost)
		return fmt.Errorf("failed to get connection stream: %w", err)
	}
	if err := writeMigrateEnvelope(streamConn, migrateProtoDWCPMemory, vmID); err != nil {
		adapter.CleanupConnection(vmID, targetHost)
		return fmt.Errorf("failed to send envelope: %w", err)
	}

	// Transfer using AMST
	err = conn.AMST.Transfer(ctx, compressed, func(transferred int64) {
		// Scale progress based on compression ratio
		if progressCallback != nil {
			actualProgress := int64(float64(transferred) * compressionRatio)
			if actualProgress > originalSize {
				actualProgress = originalSize
			}
			progressCallback(actualProgress)
		}
	})

	if err != nil {
		conn.State = MigrationStateFailed
		adapter.migrationsFailed.Add(1)
		adapter.CleanupConnection(vmID, targetHost)
		return fmt.Errorf("AMST transfer failed: %w", err)
	}

	// Update metrics
	duration := time.Since(startTime)
	throughput := float64(originalSize) / duration.Seconds()
	conn.BytesTransferred += originalSize
	adapter.totalBytesTransferred.Add(originalSize)

	// Calculate speedup
	baselineThroughput := 20 * 1024 * 1024 // 20 MB/s baseline
	speedup := throughput / float64(baselineThroughput)
	adapter.updateAverageSpeedup(speedup)

	fmt.Printf("DWCP: Memory migration completed in %.2fs (%.2f MB/s, %.2fx speedup)\n",
		duration.Seconds(), throughput/1024/1024, speedup)

	// Store baseline for future migrations
	adapter.storeMemoryBaseline(vmID, memoryData)

	// Close and evict this connection: the receive side correlates one
	// accepted net.Conn to exactly one migration and cannot support a
	// second Transfer reusing the same connection (see createConnection).
	adapter.CleanupConnection(vmID, targetHost)

	return nil
}

// MigrateVMDisk migrates VM disk blocks using DWCP optimization
func (adapter *MigrationAdapter) MigrateVMDisk(ctx context.Context, vmID string, diskBlocks map[int][]byte, targetHost string, progressCallback func(int64)) error {
	if !adapter.config.EnableDWCP {
		return adapter.migrateDiskStandard(ctx, vmID, diskBlocks, targetHost, progressCallback)
	}

	startTime := time.Now()
	totalSize := int64(0)
	for _, block := range diskBlocks {
		totalSize += int64(len(block))
	}

	// Create connection (always fresh — see createConnection)
	conn, err := adapter.createConnection(ctx, vmID, targetHost)
	if err != nil {
		if adapter.config.EnableFallback {
			return adapter.migrateDiskStandard(ctx, vmID, diskBlocks, targetHost, progressCallback)
		}
		return fmt.Errorf("failed to establish DWCP connection: %w", err)
	}

	// Determine compression tier
	tier := adapter.selectCompressionTier(conn)

	// Process blocks in parallel
	type compressedBlock struct {
		ID   int
		Data []byte
	}

	compressedChan := make(chan compressedBlock, len(diskBlocks))
	errChan := make(chan error, len(diskBlocks))

	var wg sync.WaitGroup
	for blockID, blockData := range diskBlocks {
		wg.Add(1)
		go func(id int, data []byte) {
			defer wg.Done()

			// Compress block with HDE
			compressed, err := adapter.hde.CompressDisk(vmID, data, id, tier)
			if err != nil {
				errChan <- fmt.Errorf("block %d compression failed: %w", id, err)
				return
			}

			compressedChan <- compressedBlock{
				ID:   id,
				Data: compressed,
			}
		}(blockID, blockData)
	}

	wg.Wait()
	close(compressedChan)
	close(errChan)

	// Check for compression errors
	for err := range errChan {
		if err != nil {
			adapter.CleanupConnection(vmID, targetHost)
			return err
		}
	}

	// Collect compressed blocks
	compressedBlocks := make([]byte, 0)
	blockCount := 0
	for block := range compressedChan {
		// Add block header
		header := make([]byte, 8)
		binary.BigEndian.PutUint32(header[0:4], uint32(block.ID))
		binary.BigEndian.PutUint32(header[4:8], uint32(len(block.Data)))
		compressedBlocks = append(compressedBlocks, header...)
		compressedBlocks = append(compressedBlocks, block.Data...)
		blockCount++
	}

	compressionRatio := float64(totalSize) / float64(len(compressedBlocks))
	fmt.Printf("DWCP: Disk compressed from %d to %d bytes (%.2fx compression)\n",
		totalSize, len(compressedBlocks), compressionRatio)

	// Write the envelope on the connection's single stream immediately
	// before AMST.Transfer chunk data flows — deliberately after all
	// block compression, not before; see the matching comment in
	// MigrateVMMemory for why.
	streamConn, err := conn.AMST.singleStreamConn()
	if err != nil {
		adapter.CleanupConnection(vmID, targetHost)
		return fmt.Errorf("failed to get connection stream: %w", err)
	}
	if err := writeMigrateEnvelope(streamConn, migrateProtoDWCPDisk, vmID); err != nil {
		adapter.CleanupConnection(vmID, targetHost)
		return fmt.Errorf("failed to send envelope: %w", err)
	}

	// Transfer using AMST
	transferred := atomic.Int64{}
	err = conn.AMST.Transfer(ctx, compressedBlocks, func(bytes int64) {
		transferred.Store(bytes)
		if progressCallback != nil {
			actualProgress := int64(float64(bytes) * compressionRatio)
			if actualProgress > totalSize {
				actualProgress = totalSize
			}
			progressCallback(actualProgress)
		}
	})

	if err != nil {
		conn.State = MigrationStateFailed
		adapter.migrationsFailed.Add(1)
		adapter.CleanupConnection(vmID, targetHost)
		return fmt.Errorf("disk transfer failed: %w", err)
	}

	// Update metrics
	duration := time.Since(startTime)
	throughput := float64(totalSize) / duration.Seconds()
	conn.BytesTransferred += totalSize
	adapter.totalBytesTransferred.Add(totalSize)

	// Calculate speedup
	baselineThroughput := 15 * 1024 * 1024 // 15 MB/s baseline for disk
	speedup := throughput / float64(baselineThroughput)
	adapter.updateAverageSpeedup(speedup)

	fmt.Printf("DWCP: Disk migration completed in %.2fs (%.2f MB/s, %.2fx speedup)\n",
		duration.Seconds(), throughput/1024/1024, speedup)

	// Store baselines for future migrations
	adapter.storeDiskBaselines(vmID, diskBlocks)

	// Close and evict this connection — see the matching comment in
	// MigrateVMMemory.
	adapter.CleanupConnection(vmID, targetHost)

	return nil
}

// createConnection always creates and connects a fresh MigrationConnection.
// Renamed from getOrCreateConnection: this package no longer reuses a
// cached connection across multiple migrations. AMST.Connect/Transfer's
// wire format carries no session or migration identifier (see the
// migrateEnvelope comment above), so the receive side (handleIncomingMigration)
// can only correctly correlate one accepted net.Conn to exactly one
// migration when the sender establishes exactly one fresh connection per
// migration and MigrateVMMemory/MigrateVMDisk close it via
// CleanupConnection immediately after Transfer succeeds — see
// NewMigrationAdapter's single-stream AMSTConfig override for the other
// half of this invariant.
func (adapter *MigrationAdapter) createConnection(ctx context.Context, vmID string, targetHost string) (*MigrationConnection, error) {
	adapter.mu.Lock()
	defer adapter.mu.Unlock()

	connID := fmt.Sprintf("%s-%s", vmID, targetHost)

	// Create new connection
	conn := adapter.connPool.Get().(*MigrationConnection)
	conn.ID = connID
	conn.SourceHost = "localhost" // Would be determined dynamically
	conn.TargetHost = targetHost
	conn.StartTime = time.Now()
	conn.State = MigrationStateConnecting
	conn.BytesTransferred = 0

	// Create new AMST instance for this connection. adapter.config.AMSTConfig
	// was forced to exactly one stream in NewMigrationAdapter.
	amst, err := NewAMST(adapter.config.AMSTConfig)
	if err != nil {
		adapter.connPool.Put(conn)
		return nil, fmt.Errorf("failed to create AMST: %w", err)
	}

	// Connect to target
	port := adapter.config.ListenPort
	err = amst.Connect(ctx, targetHost, port)
	if err != nil {
		adapter.connPool.Put(conn)
		return nil, fmt.Errorf("failed to connect: %w", err)
	}

	conn.AMST = amst
	conn.State = MigrationStateTransferring

	// Store connection (briefly, for the in-flight migration's metrics/
	// cleanup visibility — MigrateVMMemory/MigrateVMDisk remove it via
	// CleanupConnection right after Transfer succeeds).
	adapter.connections[connID] = conn

	return conn, nil
}

// selectCompressionTier selects the appropriate compression tier based on network conditions
func (adapter *MigrationAdapter) selectCompressionTier(conn *MigrationConnection) CompressionLevel {
	// Get latency from AMST metrics
	metrics := conn.AMST.GetMetrics()
	latency := metrics["latency_ms"].(int64)

	// Select tier based on latency
	if latency < 10 {
		return CompressionLocal // Fast local network
	} else if latency < 50 {
		return CompressionRegional // Regional network
	}
	return CompressionGlobal // WAN/Internet
}

// storeMemoryBaseline stores memory baseline for future delta encoding
func (adapter *MigrationAdapter) storeMemoryBaseline(vmID string, memoryData []byte) {
	adapter.mu.Lock()
	defer adapter.mu.Unlock()

	baseline, exists := adapter.vmBaselines[vmID]
	if !exists {
		baseline = &VMBaseline{
			VMID:          vmID,
			DiskBaselines: make(map[int][]byte),
		}
		adapter.vmBaselines[vmID] = baseline
	}

	baseline.mu.Lock()
	baseline.MemoryBaseline = memoryData
	baseline.LastUpdated = time.Now()
	baseline.mu.Unlock()
}

// storeDiskBaselines stores disk baselines for future delta encoding
func (adapter *MigrationAdapter) storeDiskBaselines(vmID string, diskBlocks map[int][]byte) {
	adapter.mu.Lock()
	defer adapter.mu.Unlock()

	baseline, exists := adapter.vmBaselines[vmID]
	if !exists {
		baseline = &VMBaseline{
			VMID:          vmID,
			DiskBaselines: make(map[int][]byte),
		}
		adapter.vmBaselines[vmID] = baseline
	}

	baseline.mu.Lock()
	for blockID, blockData := range diskBlocks {
		baseline.DiskBaselines[blockID] = blockData
	}
	baseline.LastUpdated = time.Now()
	baseline.mu.Unlock()
}

// migrateMemoryStandard performs standard TCP migration without DWCP
func (adapter *MigrationAdapter) migrateMemoryStandard(ctx context.Context, vmID string, memoryData []byte, targetHost string, progressCallback func(int64)) error {
	// Standard TCP transfer without optimization
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", targetHost, adapter.config.ListenPort),
		adapter.config.ConnectionTimeout)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close()

	if err := writeMigrateEnvelope(conn, migrateProtoStandardMemory, vmID); err != nil {
		return fmt.Errorf("failed to send envelope: %w", err)
	}

	// Send data size
	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(len(memoryData)))
	if _, err := conn.Write(header); err != nil {
		return fmt.Errorf("failed to send header: %w", err)
	}

	// Send data in chunks
	chunkSize := 64 * 1024 // 64KB chunks
	totalSent := int64(0)

	for offset := 0; offset < len(memoryData); offset += chunkSize {
		end := offset + chunkSize
		if end > len(memoryData) {
			end = len(memoryData)
		}

		n, err := conn.Write(memoryData[offset:end])
		if err != nil {
			return fmt.Errorf("failed to send data: %w", err)
		}

		totalSent += int64(n)
		if progressCallback != nil {
			progressCallback(totalSent)
		}
	}

	return nil
}

// migrateDiskStandard performs standard TCP disk migration without DWCP
func (adapter *MigrationAdapter) migrateDiskStandard(ctx context.Context, vmID string, diskBlocks map[int][]byte, targetHost string, progressCallback func(int64)) error {
	// Standard TCP transfer without optimization
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", targetHost, adapter.config.ListenPort),
		adapter.config.ConnectionTimeout)
	if err != nil {
		return fmt.Errorf("failed to connect: %w", err)
	}
	defer conn.Close()

	if err := writeMigrateEnvelope(conn, migrateProtoStandardDisk, vmID); err != nil {
		return fmt.Errorf("failed to send envelope: %w", err)
	}

	totalSize := int64(0)
	for _, block := range diskBlocks {
		totalSize += int64(len(block))
	}

	// Send total size
	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(totalSize))
	if _, err := conn.Write(header); err != nil {
		return fmt.Errorf("failed to send header: %w", err)
	}

	// Send blocks
	totalSent := int64(0)
	for blockID, blockData := range diskBlocks {
		// Send block header
		blockHeader := make([]byte, 8)
		binary.BigEndian.PutUint32(blockHeader[0:4], uint32(blockID))
		binary.BigEndian.PutUint32(blockHeader[4:8], uint32(len(blockData)))

		if _, err := conn.Write(blockHeader); err != nil {
			return fmt.Errorf("failed to send block header: %w", err)
		}

		// Send block data
		n, err := conn.Write(blockData)
		if err != nil {
			return fmt.Errorf("failed to send block data: %w", err)
		}

		totalSent += int64(n)
		if progressCallback != nil {
			progressCallback(totalSent)
		}
	}

	return nil
}

// updateAverageSpeedup updates the running average speedup
func (adapter *MigrationAdapter) updateAverageSpeedup(speedup float64) {
	// Exponential moving average
	current := adapter.averageSpeedup.Load().(float64)
	newAverage := current*0.8 + speedup*0.2
	adapter.averageSpeedup.Store(newAverage)
}

// metricsLoop periodically collects and reports metrics
func (adapter *MigrationAdapter) metricsLoop() {
	ticker := time.NewTicker(adapter.config.MetricsInterval)
	defer ticker.Stop()

	for {
		select {
		case <-adapter.ctx.Done():
			return
		case <-ticker.C:
			adapter.collectMetrics()
		}
	}
}

// collectMetrics collects current metrics
func (adapter *MigrationAdapter) collectMetrics() {
	completed := adapter.migrationsCompleted.Load()
	failed := adapter.migrationsFailed.Load()
	total := completed + failed

	if total == 0 {
		return
	}

	successRate := float64(completed) / float64(total)
	avgSpeedup := adapter.averageSpeedup.Load().(float64)
	totalBytes := adapter.totalBytesTransferred.Load()

	fmt.Printf("DWCP Migration Metrics - Success Rate: %.2f%%, Avg Speedup: %.2fx, Total: %.2f GB\n",
		successRate*100, avgSpeedup, float64(totalBytes)/1024/1024/1024)

	// Report AMST metrics if available
	if adapter.amst != nil {
		amstMetrics := adapter.amst.GetMetrics()
		fmt.Printf("  AMST: Streams: %d, Transfer Rate: %.2f MB/s, Latency: %dms\n",
			amstMetrics["active_streams"],
			float64(amstMetrics["transfer_rate"].(int64))/1024/1024,
			amstMetrics["latency_ms"])
	}

	// Report HDE metrics if available
	if adapter.hde != nil {
		hdeMetrics := adapter.hde.GetMetrics()
		fmt.Printf("  HDE: Compression Ratio: %.2fx, Delta Hit Rate: %.2f%%\n",
			hdeMetrics["compression_ratio"],
			hdeMetrics["delta_hit_rate"].(float64)*100)
	}
}

// CleanupConnection cleans up a migration connection
func (adapter *MigrationAdapter) CleanupConnection(vmID string, targetHost string) error {
	adapter.mu.Lock()
	defer adapter.mu.Unlock()

	connID := fmt.Sprintf("%s-%s", vmID, targetHost)
	conn, exists := adapter.connections[connID]
	if !exists {
		return nil
	}

	// Close AMST connection
	if conn.AMST != nil {
		conn.AMST.Close()
	}

	// Mark as completed
	if conn.State == MigrationStateTransferring {
		conn.State = MigrationStateCompleted
		adapter.migrationsCompleted.Add(1)
	}

	// Return to pool
	delete(adapter.connections, connID)
	adapter.connPool.Put(conn)

	return nil
}

// TrainDictionary trains a compression dictionary for a specific VM type
func (adapter *MigrationAdapter) TrainDictionary(vmType string, samples [][]byte) error {
	if adapter.hde == nil {
		return errors.New("HDE not initialized")
	}

	return adapter.hde.TrainDictionary(vmType, samples)
}

// GetMetrics returns adapter metrics
func (adapter *MigrationAdapter) GetMetrics() map[string]interface{} {
	adapter.mu.RLock()
	activeConnections := len(adapter.connections)
	baselineCount := len(adapter.vmBaselines)
	adapter.mu.RUnlock()

	metrics := map[string]interface{}{
		"migrations_completed":    adapter.migrationsCompleted.Load(),
		"migrations_failed":       adapter.migrationsFailed.Load(),
		"total_bytes_transferred": adapter.totalBytesTransferred.Load(),
		"average_speedup":         adapter.averageSpeedup.Load(),
		"active_connections":      activeConnections,
		"baseline_count":          baselineCount,
		"dwcp_enabled":            adapter.config.EnableDWCP,
		"fallback_enabled":        adapter.config.EnableFallback,
	}

	// Add AMST metrics
	if adapter.amst != nil {
		metrics["amst"] = adapter.amst.GetMetrics()
	}

	// Add HDE metrics
	if adapter.hde != nil {
		metrics["hde"] = adapter.hde.GetMetrics()
	}

	return metrics
}

// Close closes the adapter and releases resources
func (adapter *MigrationAdapter) Close() error {
	adapter.cancel()

	// Close all connections
	adapter.mu.Lock()
	for _, conn := range adapter.connections {
		if conn.AMST != nil {
			conn.AMST.Close()
		}
	}
	adapter.mu.Unlock()

	// Close AMST
	if adapter.amst != nil {
		adapter.amst.Close()
	}

	// Close HDE
	if adapter.hde != nil {
		adapter.hde.Close()
	}

	return nil
}

// ListenForMigrations starts a listener for incoming migrations
func (adapter *MigrationAdapter) ListenForMigrations(ctx context.Context) error {
	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", adapter.config.ListenPort))
	if err != nil {
		return fmt.Errorf("failed to start listener: %w", err)
	}
	defer listener.Close()

	fmt.Printf("DWCP Migration Adapter listening on port %d\n", adapter.config.ListenPort)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		conn, err := listener.Accept()
		if err != nil {
			continue
		}

		go adapter.handleIncomingMigration(ctx, conn)
	}
}

// handleIncomingMigration handles an incoming migration connection. Reads
// the wire envelope (see writeMigrateEnvelope) to determine which of the
// four send paths (standard/DWCP x memory/disk) wrote this connection,
// then dispatches to the matching receive function.
func (adapter *MigrationAdapter) handleIncomingMigration(ctx context.Context, conn net.Conn) {
	defer conn.Close()

	// Bound the envelope read so a connection that never sends one (or a
	// dead/stalled sender) cannot leak this goroutine forever.
	if adapter.config.ConnectionTimeout > 0 {
		conn.SetReadDeadline(time.Now().Add(adapter.config.ConnectionTimeout))
	}
	protocol, vmID, err := readMigrateEnvelope(conn)
	conn.SetReadDeadline(time.Time{})
	if err != nil {
		fmt.Printf("DWCP: failed to read migration envelope: %v\n", err)
		return
	}

	switch protocol {
	case migrateProtoStandardMemory:
		adapter.receiveMemoryStandard(conn, vmID)
	case migrateProtoStandardDisk:
		adapter.receiveDiskStandard(conn, vmID)
	case migrateProtoDWCPMemory:
		adapter.receiveMemoryDWCP(ctx, conn, vmID)
	case migrateProtoDWCPDisk:
		adapter.receiveDiskDWCP(ctx, conn, vmID)
	default:
		fmt.Printf("DWCP: unknown migration protocol byte: %d\n", protocol)
	}
}

// receiveMemoryStandard receives memory sent by migrateMemoryStandard:
// an 8-byte big-endian size header followed by that many raw bytes.
func (adapter *MigrationAdapter) receiveMemoryStandard(conn net.Conn, vmID string) {
	data, err := readSizePrefixedPayload(conn, adapter.config.ConnectionTimeout, adapter.config.MaxMemoryUsage)
	if err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("DWCP: standard memory receive failed for vmID %s: %v\n", vmID, err)
		return
	}
	adapter.migrationsCompleted.Add(1)
	adapter.totalBytesTransferred.Add(int64(len(data)))
	if adapter.config.OnMemoryReceived != nil {
		adapter.config.OnMemoryReceived(vmID, data)
		return
	}
	fmt.Printf("DWCP: received %d bytes of standard memory for vmID %s (no OnMemoryReceived callback configured)\n", len(data), vmID)
}

// receiveDiskStandard receives disk blocks sent by migrateDiskStandard:
// an 8-byte total-data-size header, then repeated
// [blockID:4][blockLen:4]+data until that many data bytes are read.
func (adapter *MigrationAdapter) receiveDiskStandard(conn net.Conn, vmID string) {
	blocks, err := readSizePrefixedDiskBlocks(conn, adapter.config.ConnectionTimeout, adapter.config.MaxMemoryUsage)
	if err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("DWCP: standard disk receive failed for vmID %s: %v\n", vmID, err)
		return
	}
	totalBytes := int64(0)
	for blockID, data := range blocks {
		totalBytes += int64(len(data))
		if adapter.config.OnDiskReceived != nil {
			adapter.config.OnDiskReceived(vmID, blockID, data)
		}
	}
	adapter.migrationsCompleted.Add(1)
	adapter.totalBytesTransferred.Add(totalBytes)
	if adapter.config.OnDiskReceived == nil {
		fmt.Printf("DWCP: received %d standard disk blocks (%d bytes) for vmID %s (no OnDiskReceived callback configured)\n", len(blocks), totalBytes, vmID)
	}
}

// receiveMemoryDWCP receives memory sent by MigrateVMMemory: AMST-framed,
// HDE-compressed bytes on the connection's single stream.
func (adapter *MigrationAdapter) receiveMemoryDWCP(ctx context.Context, conn net.Conn, vmID string) {
	if adapter.hde == nil {
		fmt.Printf("DWCP: received DWCP memory migration for vmID %s but HDE is not initialized\n", vmID)
		return
	}
	compressed, err := receiveViaSingleStreamAMST(ctx, conn, adapter.config.ConnectionTimeout)
	if err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("DWCP: memory AMST receive failed for vmID %s: %v\n", vmID, err)
		return
	}
	data, err := adapter.hdeDecompressChecked(compressed, "memory", vmID)
	if err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("DWCP: %v\n", err)
		return
	}
	adapter.migrationsCompleted.Add(1)
	adapter.totalBytesTransferred.Add(int64(len(data)))
	if adapter.config.OnMemoryReceived != nil {
		adapter.config.OnMemoryReceived(vmID, data)
		return
	}
	fmt.Printf("DWCP: received %d bytes of DWCP memory for vmID %s (no OnMemoryReceived callback configured)\n", len(data), vmID)
}

// receiveDiskDWCP receives disk blocks sent by MigrateVMDisk: AMST-framed
// bytes on the connection's single stream, reassembling into the
// [blockID:4][blockLen:4]+HDE-compressed-data layout MigrateVMDisk wrote.
func (adapter *MigrationAdapter) receiveDiskDWCP(ctx context.Context, conn net.Conn, vmID string) {
	if adapter.hde == nil {
		fmt.Printf("DWCP: received DWCP disk migration for vmID %s but HDE is not initialized\n", vmID)
		return
	}
	compressedBlocks, err := receiveViaSingleStreamAMST(ctx, conn, adapter.config.ConnectionTimeout)
	if err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("DWCP: disk AMST receive failed for vmID %s: %v\n", vmID, err)
		return
	}

	totalBytes := int64(0)
	blockCount := 0
	offset := 0
	for offset < len(compressedBlocks) {
		if offset+8 > len(compressedBlocks) {
			adapter.migrationsFailed.Add(1)
			fmt.Printf("DWCP: disk receive for vmID %s: truncated block header at offset %d\n", vmID, offset)
			return
		}
		blockID := int(binary.BigEndian.Uint32(compressedBlocks[offset : offset+4]))
		blockLen := int(binary.BigEndian.Uint32(compressedBlocks[offset+4 : offset+8]))
		offset += 8
		if blockLen < 0 || offset+blockLen > len(compressedBlocks) {
			adapter.migrationsFailed.Add(1)
			fmt.Printf("DWCP: disk receive for vmID %s: truncated block %d data\n", vmID, blockID)
			return
		}
		compressedBlockData := compressedBlocks[offset : offset+blockLen]
		offset += blockLen

		blockData, err := adapter.hdeDecompressChecked(compressedBlockData, fmt.Sprintf("disk block %d", blockID), vmID)
		if err != nil {
			adapter.migrationsFailed.Add(1)
			fmt.Printf("DWCP: %v\n", err)
			return
		}
		totalBytes += int64(len(blockData))
		blockCount++
		if adapter.config.OnDiskReceived != nil {
			adapter.config.OnDiskReceived(vmID, blockID, blockData)
		}
	}

	adapter.migrationsCompleted.Add(1)
	adapter.totalBytesTransferred.Add(totalBytes)
	if adapter.config.OnDiskReceived == nil {
		fmt.Printf("DWCP: received %d DWCP disk blocks (%d bytes) for vmID %s (no OnDiskReceived callback configured)\n", blockCount, totalBytes, vmID)
	}
}

// hdeDecompressChecked decompresses a received HDE packet, refusing
// delta-encoded packets rather than silently returning a corrupt
// reconstruction. EnableDelta is forced off for adapter.hde
// (NewMigrationAdapter), so no sender using this adapter's own
// CompressMemory/CompressDisk should ever produce packet[0]==1 — this is
// defense in depth for if that invariant is ever violated, since
// HDE.Decompress's isDelta branch returns the raw delta bytes as-is, not
// memory reconstructed against a baseline (see hde.go Decompress).
func (adapter *MigrationAdapter) hdeDecompressChecked(compressed []byte, what, vmID string) ([]byte, error) {
	if len(compressed) > 0 && compressed[0] == 1 {
		return nil, fmt.Errorf("refusing to decompress delta-encoded %s for vmID %s (delta is disabled for this adapter's HDE)", what, vmID)
	}
	data, err := adapter.hde.Decompress(compressed)
	if err != nil {
		return nil, fmt.Errorf("%s decompress failed for vmID %s: %w", what, vmID, err)
	}
	return data, nil
}

// stallReadChunkSize bounds each individual read in readFullChunked, so a
// read-deadline refresh (see readFullChunked) happens often enough to
// detect a stalled peer without capping total transfer duration.
const stallReadChunkSize = 64 * 1024

// readFullChunked reads exactly len(buf) bytes from conn in
// stallReadChunkSize pieces, refreshing conn's read deadline before each
// piece instead of setting one deadline for the whole read. This makes
// readTimeout a stall-detection window (max time between chunks of
// forward progress) rather than a cap on total transfer duration — a
// single deadline set once before a multi-GB io.ReadFull would make
// ConnectionTimeout (default 30s) a hard ceiling on total standard-path
// transfer time regardless of how much data is actively flowing, timing
// out large payloads on slower-but-healthy links. Matches the model
// AMST.Receive already uses per chunk. readTimeout <= 0 disables the
// deadline entirely.
func readFullChunked(conn net.Conn, buf []byte, readTimeout time.Duration) error {
	for offset := 0; offset < len(buf); offset += stallReadChunkSize {
		end := offset + stallReadChunkSize
		if end > len(buf) {
			end = len(buf)
		}
		if readTimeout > 0 {
			conn.SetReadDeadline(time.Now().Add(readTimeout))
		}
		if _, err := io.ReadFull(conn, buf[offset:end]); err != nil {
			return err
		}
	}
	return nil
}

// readSizePrefixedPayload reads the wire format migrateMemoryStandard
// writes: an 8-byte big-endian size header, then that many raw bytes.
// readTimeout is a stall-detection window, refreshed before every
// stallReadChunkSize piece (see readFullChunked) — not a cap on total
// transfer time. maxSize caps the wire-supplied size before allocating —
// an unbounded make([]byte, size) driven directly by an 8-byte header on
// the wire would let a garbage or hostile size value attempt a multi-GB
// allocation and crash the process.
func readSizePrefixedPayload(conn net.Conn, readTimeout time.Duration, maxSize int64) ([]byte, error) {
	header := make([]byte, 8)
	if err := readFullChunked(conn, header, readTimeout); err != nil {
		return nil, fmt.Errorf("failed to read size header: %w", err)
	}
	size := binary.BigEndian.Uint64(header)
	if maxSize > 0 && size > uint64(maxSize) {
		return nil, fmt.Errorf("payload size %d exceeds maximum %d", size, maxSize)
	}
	data := make([]byte, size)
	if err := readFullChunked(conn, data, readTimeout); err != nil {
		return nil, fmt.Errorf("failed to read %d-byte payload: %w", size, err)
	}
	return data, nil
}

// readSizePrefixedDiskBlocks reads the wire format migrateDiskStandard
// writes: an 8-byte total-data-size header (sum of block data lengths,
// excluding per-block headers), then repeated
// [blockID:4][blockLen:4]+data until that many data bytes are read.
// readTimeout/maxSize as in readSizePrefixedPayload — maxSize bounds
// both the total and any single block's size.
func readSizePrefixedDiskBlocks(conn net.Conn, readTimeout time.Duration, maxSize int64) (map[int][]byte, error) {
	header := make([]byte, 8)
	if err := readFullChunked(conn, header, readTimeout); err != nil {
		return nil, fmt.Errorf("failed to read total-size header: %w", err)
	}
	totalSize := binary.BigEndian.Uint64(header)
	if maxSize > 0 && totalSize > uint64(maxSize) {
		return nil, fmt.Errorf("total disk size %d exceeds maximum %d", totalSize, maxSize)
	}

	blocks := make(map[int][]byte)
	var received uint64
	for received < totalSize {
		blockHeader := make([]byte, 8)
		if err := readFullChunked(conn, blockHeader, readTimeout); err != nil {
			return nil, fmt.Errorf("failed to read block header: %w", err)
		}
		blockID := int(binary.BigEndian.Uint32(blockHeader[0:4]))
		blockLen := binary.BigEndian.Uint32(blockHeader[4:8])
		if maxSize > 0 && uint64(blockLen) > uint64(maxSize) {
			return nil, fmt.Errorf("block %d size %d exceeds maximum %d", blockID, blockLen, maxSize)
		}
		blockData := make([]byte, blockLen)
		if err := readFullChunked(conn, blockData, readTimeout); err != nil {
			return nil, fmt.Errorf("failed to read block %d data: %w", blockID, err)
		}
		blocks[blockID] = blockData
		received += uint64(blockLen)
	}
	return blocks, nil
}

// receiveViaSingleStreamAMST reconstructs a Transfer payload from conn by
// registering it as the sole stream of a throwaway, receive-only AMST
// instance. Only valid for a single-stream sender (see
// NewMigrationAdapter's AMSTConfig override) — with exactly one stream,
// AMST.Receive's per-stream completion check (totalReceived >= totalSize)
// fires on that same stream's own last chunk read, so there is no other
// stream left blocking on a header that will never arrive. readTimeout,
// when positive, bounds how long a single header/chunk read may block,
// so a stalled or misbehaving sender cannot hang this goroutine forever.
func receiveViaSingleStreamAMST(ctx context.Context, conn net.Conn, readTimeout time.Duration) ([]byte, error) {
	amst, err := NewAMST(AMSTConfig{MinStreams: 1, MaxStreams: 1, InitialStreams: 1, ReadTimeout: readTimeout})
	if err != nil {
		return nil, fmt.Errorf("failed to create receive-side AMST: %w", err)
	}
	defer amst.Close()

	amst.mu.Lock()
	amst.streams = append(amst.streams, &Stream{
		id:         "recv-0",
		conn:       conn,
		amst:       amst,
		active:     true,
		lastActive: time.Now(),
	})
	amst.activeStreams.Add(1)
	amst.mu.Unlock()

	return amst.Receive(ctx, nil)
}
