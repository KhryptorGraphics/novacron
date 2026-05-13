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

const (
	migrationTypeMemory byte = iota
	migrationTypeDisk
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

	// Get or create connection
	conn, err := adapter.getOrCreateConnection(ctx, vmID, targetHost)
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
		return fmt.Errorf("memory compression failed: %w", err)
	}

	compressionRatio := float64(originalSize) / float64(len(compressed))
	fmt.Printf("DWCP: Memory compressed from %d to %d bytes (%.2fx compression)\n",
		originalSize, len(compressed), compressionRatio)

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

	// Get or create connection
	conn, err := adapter.getOrCreateConnection(ctx, vmID, targetHost)
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

	return nil
}

// getOrCreateConnection gets an existing connection or creates a new one
func (adapter *MigrationAdapter) getOrCreateConnection(ctx context.Context, vmID string, targetHost string) (*MigrationConnection, error) {
	adapter.mu.Lock()
	defer adapter.mu.Unlock()

	connID := fmt.Sprintf("%s-%s", vmID, targetHost)

	// Check for existing connection
	if conn, exists := adapter.connections[connID]; exists && conn.State != MigrationStateFailed {
		return conn, nil
	}

	// Create new connection
	conn := adapter.connPool.Get().(*MigrationConnection)
	conn.ID = connID
	conn.SourceHost = "localhost" // Would be determined dynamically
	conn.TargetHost = targetHost
	conn.StartTime = time.Now()
	conn.State = MigrationStateConnecting
	conn.BytesTransferred = 0

	// Create new AMST instance for this connection
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

	// Store connection
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

	if err := writeMigrationHeader(conn, migrationTypeMemory, vmID); err != nil {
		return fmt.Errorf("failed to send migration header: %w", err)
	}

	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(len(memoryData)))
	if err := writeAll(conn, header); err != nil {
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

		chunk := memoryData[offset:end]
		if err := writeAll(conn, chunk); err != nil {
			return fmt.Errorf("failed to send data: %w", err)
		}

		totalSent += int64(len(chunk))
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

	totalSize := int64(0)
	for _, block := range diskBlocks {
		totalSize += int64(len(block))
	}

	if err := writeMigrationHeader(conn, migrationTypeDisk, vmID); err != nil {
		return fmt.Errorf("failed to send migration header: %w", err)
	}

	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(totalSize))
	if err := writeAll(conn, header); err != nil {
		return fmt.Errorf("failed to send header: %w", err)
	}

	// Send blocks
	totalSent := int64(0)
	for blockID, blockData := range diskBlocks {
		// Send block header
		blockHeader := make([]byte, 8)
		binary.BigEndian.PutUint32(blockHeader[0:4], uint32(blockID))
		binary.BigEndian.PutUint32(blockHeader[4:8], uint32(len(blockData)))

		if err := writeAll(conn, blockHeader); err != nil {
			return fmt.Errorf("failed to send block header: %w", err)
		}

		if err := writeAll(conn, blockData); err != nil {
			return fmt.Errorf("failed to send block data: %w", err)
		}

		totalSent += int64(len(blockData))
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

		go adapter.handleIncomingMigration(conn)
	}
}

// handleIncomingMigration handles an incoming migration connection
func (adapter *MigrationAdapter) handleIncomingMigration(conn net.Conn) {
	defer conn.Close()

	// Read migration type
	typeBuf := make([]byte, 1)
	if _, err := io.ReadFull(conn, typeBuf); err != nil {
		return
	}

	switch typeBuf[0] {
	case migrationTypeMemory:
		adapter.receiveMemory(conn)
	case migrationTypeDisk:
		adapter.receiveDisk(conn)
	default:
		fmt.Printf("Unknown migration type: %d\n", typeBuf[0])
	}
}

// receiveMemory receives memory data from a migration
func (adapter *MigrationAdapter) receiveMemory(conn net.Conn) {
	vmID, err := readMigrationVMID(conn)
	if err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("Failed to read memory migration VM ID: %v\n", err)
		return
	}

	header := make([]byte, 8)
	if _, err := io.ReadFull(conn, header); err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("Failed to read memory migration header: %v\n", err)
		return
	}

	size := int64(binary.BigEndian.Uint64(header))
	if size < 0 || size > adapter.config.MaxMemoryUsage {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("Invalid memory migration size for VM %s: %d\n", vmID, size)
		return
	}

	memoryData := make([]byte, size)
	if _, err := io.ReadFull(conn, memoryData); err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("Failed to read memory migration data for VM %s: %v\n", vmID, err)
		return
	}

	adapter.storeMemoryBaseline(vmID, memoryData)
	adapter.totalBytesTransferred.Add(size)
	adapter.migrationsCompleted.Add(1)
}

// receiveDisk receives disk data from a migration
func (adapter *MigrationAdapter) receiveDisk(conn net.Conn) {
	vmID, err := readMigrationVMID(conn)
	if err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("Failed to read disk migration VM ID: %v\n", err)
		return
	}

	header := make([]byte, 8)
	if _, err := io.ReadFull(conn, header); err != nil {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("Failed to read disk migration header: %v\n", err)
		return
	}

	totalSize := int64(binary.BigEndian.Uint64(header))
	if totalSize < 0 || totalSize > adapter.config.MaxMemoryUsage {
		adapter.migrationsFailed.Add(1)
		fmt.Printf("Invalid disk migration size for VM %s: %d\n", vmID, totalSize)
		return
	}

	blocks := make(map[int][]byte)
	received := int64(0)
	blockHeader := make([]byte, 8)
	for received < totalSize {
		if _, err := io.ReadFull(conn, blockHeader); err != nil {
			adapter.migrationsFailed.Add(1)
			fmt.Printf("Failed to read disk block header for VM %s: %v\n", vmID, err)
			return
		}

		blockID := int(binary.BigEndian.Uint32(blockHeader[0:4]))
		blockSize := int64(binary.BigEndian.Uint32(blockHeader[4:8]))
		if blockSize <= 0 || blockSize > totalSize-received {
			adapter.migrationsFailed.Add(1)
			fmt.Printf("Invalid disk block size for VM %s block %d: %d\n", vmID, blockID, blockSize)
			return
		}

		blockData := make([]byte, blockSize)
		if _, err := io.ReadFull(conn, blockData); err != nil {
			adapter.migrationsFailed.Add(1)
			fmt.Printf("Failed to read disk block data for VM %s block %d: %v\n", vmID, blockID, err)
			return
		}

		blocks[blockID] = blockData
		received += blockSize
	}

	adapter.storeDiskBaselines(vmID, blocks)
	adapter.totalBytesTransferred.Add(totalSize)
	adapter.migrationsCompleted.Add(1)
}

func writeMigrationHeader(conn net.Conn, migrationType byte, vmID string) error {
	if len(vmID) == 0 {
		return errors.New("vmID is required")
	}
	if len(vmID) > 65535 {
		return fmt.Errorf("vmID too long: %d bytes", len(vmID))
	}

	header := make([]byte, 3)
	header[0] = migrationType
	binary.BigEndian.PutUint16(header[1:3], uint16(len(vmID)))
	if err := writeAll(conn, header); err != nil {
		return err
	}
	return writeAll(conn, []byte(vmID))
}

func readMigrationVMID(conn net.Conn) (string, error) {
	header := make([]byte, 2)
	if _, err := io.ReadFull(conn, header); err != nil {
		return "", err
	}

	length := int(binary.BigEndian.Uint16(header))
	if length == 0 {
		return "", errors.New("empty VM ID")
	}

	vmID := make([]byte, length)
	if _, err := io.ReadFull(conn, vmID); err != nil {
		return "", err
	}

	return string(vmID), nil
}

func writeAll(conn net.Conn, data []byte) error {
	for len(data) > 0 {
		n, err := conn.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}
