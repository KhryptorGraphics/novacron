// Package dwcp implements the Distributed WAN Communication Protocol for NovaCron
package dwcp

import (
	"container/list"
	"context"
	"crypto/rand"
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

	// DWCP multi-stream session correlation (novacron-hpa): accumulates
	// independently-accepted connections (one per AMST stream) that carry
	// the same sender-generated sessionID until all of them arrive — see
	// handleIncomingDWCPStream/registerDWCPStream/completeDWCPSession.
	sessionMu       sync.Mutex
	pendingSessions map[string]*pendingDWCPSession

	// Baseline management
	vmBaselines map[string]*VMBaseline
	// baselineLRU orders vmBaselines by recency for MaxMemoryUsage-bounded
	// eviction (novacron-y45); baselineBytes tracks total retained baseline
	// bytes. Both guarded by adapter.mu.
	baselineLRU   *list.List
	baselineBytes int64

	// Performance metrics
	migrationsCompleted   atomic.Int64
	migrationsFailed      atomic.Int64
	totalBytesTransferred atomic.Int64
	// avgThroughputVsBaseline is throughput / a HARDCODED reference
	// value (20 MB/s memory, 15 MB/s disk - not a measurement of this
	// deployment's actual non-DWCP throughput on the same link). Renamed
	// from "averageSpeedup"/"average_speedup" - that name and its "Nx
	// speedup" log line were misleading: novacron-38p's benchmark work
	// found this reported 30-103x "speedup" on runs where DWCP was
	// actually measured 1.02-2.29x SLOWER than uncompressed transfer on
	// the same link (see BenchmarkMigrationAdapterEndToEnd,
	// BenchmarkMigrationWANBandwidthConstrained). This field answers
	// "how does this throughput compare to a fixed reference point,"
	// not "how does DWCP compare to standard on this link" - do not
	// read it as the latter.
	avgThroughputVsBaseline atomic.Value // float64

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

// dwcpSessionIDLen is the length in bytes of the random session identifier
// used to correlate the N independently-accepted TCP connections (AMST
// streams) belonging to one logical DWCP Transfer on the receive side
// (novacron-hpa). 16 random bytes is plenty to avoid collisions between
// concurrent migrations hitting the same listener at once — it does not
// need to be cryptographically unguessable, only unique.
const dwcpSessionIDLen = 16

// writeDWCPStreamEnvelope writes the full per-stream header for one AMST
// stream belonging to a DWCP migration onto conn: the standard protocol+
// vmID envelope (byte-identical to what the non-DWCP path writes, so
// handleIncomingMigration's first read is unchanged for every protocol),
// immediately followed by the session correlation fields
// [sessionID][streamIndex][streamCount] that every stream of an N-stream
// Transfer carries so the receiver can regroup them into one logical
// Receive (see handleIncomingDWCPStream). Every stream of one Transfer
// writes the SAME protocol/vmID/sessionID/streamCount and a unique
// streamIndex in [0, streamCount).
func writeDWCPStreamEnvelope(conn net.Conn, protocol byte, vmID string, sessionID [dwcpSessionIDLen]byte, streamIndex, streamCount int) error {
	if err := writeMigrateEnvelope(conn, protocol, vmID); err != nil {
		return err
	}
	buf := make([]byte, 0, dwcpSessionIDLen+4)
	buf = append(buf, sessionID[:]...)
	buf = binary.BigEndian.AppendUint16(buf, uint16(streamIndex))
	buf = binary.BigEndian.AppendUint16(buf, uint16(streamCount))
	_, err := conn.Write(buf)
	return err
}

// readDWCPStreamCorrelation reads the fields writeDWCPStreamEnvelope wrote
// immediately after the protocol+vmID envelope, which the caller must
// already have consumed via readMigrateEnvelope.
func readDWCPStreamCorrelation(conn net.Conn) (sessionID string, streamIndex, streamCount int, err error) {
	buf := make([]byte, dwcpSessionIDLen+4)
	if _, err := io.ReadFull(conn, buf); err != nil {
		return "", 0, 0, fmt.Errorf("failed to read stream correlation header: %w", err)
	}
	sessionID = string(buf[:dwcpSessionIDLen])
	streamIndex = int(binary.BigEndian.Uint16(buf[dwcpSessionIDLen : dwcpSessionIDLen+2]))
	streamCount = int(binary.BigEndian.Uint16(buf[dwcpSessionIDLen+2 : dwcpSessionIDLen+4]))
	return sessionID, streamIndex, streamCount, nil
}

// writeDWCPEnvelopeToAllStreams generates a fresh session ID and writes the
// per-stream envelope (see writeDWCPStreamEnvelope) to every active stream
// of amstInst, so the receiver can correlate all of them — however many
// there are, including exactly one — back to a single logical Transfer.
// Must be called after amstInst.Connect and before amstInst.Transfer.
func writeDWCPEnvelopeToAllStreams(amstInst *AMST, protocol byte, vmID string) error {
	conns := amstInst.activeStreamConns()
	if len(conns) == 0 {
		return errors.New("no active AMST streams to write envelope to")
	}
	var sessionID [dwcpSessionIDLen]byte
	if _, err := rand.Read(sessionID[:]); err != nil {
		return fmt.Errorf("failed to generate session ID: %w", err)
	}
	streamCount := len(conns)
	for i, c := range conns {
		if err := writeDWCPStreamEnvelope(c, protocol, vmID, sessionID, i, streamCount); err != nil {
			return fmt.Errorf("failed to write stream envelope to stream %d/%d: %w", i, streamCount, err)
		}
	}
	return nil
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
	// retainedBytes (len(MemoryBaseline)+sum(DiskBaselines)) and lruElem (this
	// entry's node in adapter.baselineLRU) are accounting fields for
	// MaxMemoryUsage-bounded eviction (novacron-y45). They are guarded by
	// adapter.mu, NOT by the VMBaseline.mu above.
	retainedBytes int64
	lruElem       *list.Element
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
	//   - EnableDictionary: dictionary compression is now internally correct
	//     and reversible (novacron-976 — CompressMemory/CompressDisk emit a
	//     single dictionary-compressed frame that HDE.Decompress reverses via
	//     a dictionary-aware decoder). It stays off HERE only because the
	//     receiver is a DIFFERENT process whose HDE has not trained the same
	//     dictionary: this transport does not distribute dictionaries, so the
	//     receive-side Decompress would fail with "no dictionary decoder for
	//     dict ID". Enabling it requires shipping the dictionary to the peer.
	//   - EnableQuantization: quantize() masks off low bits of every byte
	//     at tier CompressionGlobal — irreversible information loss with
	//     no dequantize step anywhere in Decompress (hde.go quantize).
	// All three are forced off here so the receive path is correct by
	// construction, independent of what a caller passes — see novacron-o0e.
	//
	// EnableQuantization=false is now LOAD-BEARING, not just defensive:
	// before AMST.Connect measured real dial RTT (added alongside this
	// session's WAN benchmark work), selectCompressionTier always saw
	// latency_ms==0 and could never select CompressionGlobal — the only
	// tier where quantize() runs — so this flag was previously a second,
	// practically-unreachable layer behind that. Real latency now makes
	// Global tier reachable, making this the SOLE remaining guard against
	// irreversible quantization on the receive path. Do not remove this
	// line without first adding a matching dequantize step to
	// hde.go Decompress.
	config.HDEConfig.EnableDelta = false
	config.HDEConfig.EnableDictionary = false
	config.HDEConfig.EnableQuantization = false

	// The receive path now correlates N independently-accepted net.Conns
	// (one per AMST stream) back to a single logical migration via an
	// explicit session ID written on every stream at connect time
	// (writeDWCPStreamEnvelope/handleIncomingDWCPStream, novacron-hpa) —
	// so, unlike before, AMSTConfig no longer needs to be forced to
	// exactly one stream to be correct. Default to single-stream ONLY
	// when the caller hasn't opted into adaptive multi-stream
	// (EnableAdaptive) and hasn't explicitly configured stream counts —
	// this keeps every existing caller that leaves AMSTConfig zero-valued
	// on today's proven single-connection behavior, while callers that
	// already configure real multi-stream (e.g.
	// backend/core/migration/orchestrator_dwcp.go, which sets
	// MinStreams/MaxStreams/InitialStreams/EnableAdaptive=true and had
	// every one of those silently clamped to 1 stream by the old
	// unconditional force) finally get what they asked for.
	if !config.AMSTConfig.EnableAdaptive {
		if config.AMSTConfig.MinStreams <= 0 {
			config.AMSTConfig.MinStreams = 1
		}
		if config.AMSTConfig.MaxStreams <= 0 {
			config.AMSTConfig.MaxStreams = 1
		}
		if config.AMSTConfig.InitialStreams <= 0 {
			config.AMSTConfig.InitialStreams = 1
		}
	}

	ctx, cancel := context.WithCancel(context.Background())

	adapter := &MigrationAdapter{
		config:          config,
		connections:     make(map[string]*MigrationConnection),
		pendingSessions: make(map[string]*pendingDWCPSession),
		vmBaselines:     make(map[string]*VMBaseline),
		baselineLRU:     list.New(),
		ctx:             ctx,
		cancel:          cancel,
	}

	// Initialize avgThroughputVsBaseline
	adapter.avgThroughputVsBaseline.Store(float64(1.0))

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

	// Write the envelope (+ session correlation header, so N>1 streams can
	// be regrouped on receive) on every one of this connection's AMST
	// streams immediately before AMST.Transfer chunk data flows —
	// deliberately AFTER compression, not before: the receiver's very
	// next read blocks on the envelope/correlation header, and writing it
	// before a (potentially long, for large payloads at high compression
	// levels) CompressMemory call would make the receiver wait out that
	// entire compression time against its ReadTimeout with nothing having
	// gone wrong. So the receiver's envelope-read is immediately followed
	// by data that is already fully ready to send.
	if err := writeDWCPEnvelopeToAllStreams(conn.AMST, migrateProtoDWCPMemory, vmID); err != nil {
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
	conn.BytesTransferred += originalSize
	adapter.totalBytesTransferred.Add(originalSize)

	// Guard against duration.Seconds() == 0 (coarse clock resolution,
	// or a pathologically fast transfer) - an unguarded division here
	// produces +Inf, which permanently poisons avgThroughputVsBaseline's
	// exponential moving average (Inf*0.2 + x*0.8 == Inf forever, no
	// recovery short of process restart). Skips only the metric/log
	// block, not storeMemoryBaseline/CleanupConnection below - an early
	// return here would leak the AMST connection and skip baseline
	// storage, a worse bug than the one being guarded against.
	if duration > 0 {
		throughput := float64(originalSize) / duration.Seconds()

		// throughputVsBaseline compares this migration's throughput against
		// a HARDCODED reference point, not against this deployment's actual
		// non-DWCP throughput on the same link - see avgThroughputVsBaseline's
		// doc comment. Do not read this as "DWCP vs standard".
		baselineThroughput := 20 * 1024 * 1024 // 20 MB/s reference point
		throughputVsBaseline := throughput / float64(baselineThroughput)
		adapter.updateAvgThroughputVsBaseline(throughputVsBaseline)

		fmt.Printf("DWCP: Memory migration completed in %.2fs (%.2f MB/s, %.2fx vs %d MB/s reference - NOT a standard-vs-DWCP comparison)\n",
			duration.Seconds(), throughput/1024/1024, throughputVsBaseline, baselineThroughput/1024/1024)
	} else {
		fmt.Printf("DWCP: Memory migration completed with unmeasurable duration (%d bytes) - skipping throughput metric\n", originalSize)
	}

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

	// Write the envelope (+ session correlation header) to every one of
	// this connection's AMST streams immediately before AMST.Transfer
	// chunk data flows — deliberately after all block compression, not
	// before; see the matching comment in MigrateVMMemory for why.
	if err := writeDWCPEnvelopeToAllStreams(conn.AMST, migrateProtoDWCPDisk, vmID); err != nil {
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
	conn.BytesTransferred += totalSize
	adapter.totalBytesTransferred.Add(totalSize)

	// Guard against duration.Seconds() == 0 - see the matching comment
	// in MigrateVMMemory for why (unguarded division -> +Inf ->
	// permanently poisoned avgThroughputVsBaseline). Skips only the
	// metric/log block, not storeDiskBaselines/CleanupConnection below.
	if duration > 0 {
		throughput := float64(totalSize) / duration.Seconds()

		// throughputVsBaseline compares this migration's throughput against
		// a HARDCODED reference point, not against this deployment's actual
		// non-DWCP throughput on the same link - see avgThroughputVsBaseline's
		// doc comment. Do not read this as "DWCP vs standard".
		baselineThroughput := 15 * 1024 * 1024 // 15 MB/s reference point for disk
		throughputVsBaseline := throughput / float64(baselineThroughput)
		adapter.updateAvgThroughputVsBaseline(throughputVsBaseline)

		fmt.Printf("DWCP: Disk migration completed in %.2fs (%.2f MB/s, %.2fx vs %d MB/s reference - NOT a standard-vs-DWCP comparison)\n",
			duration.Seconds(), throughput/1024/1024, throughputVsBaseline, baselineThroughput/1024/1024)
	} else {
		fmt.Printf("DWCP: Disk migration completed with unmeasurable duration (%d bytes) - skipping throughput metric\n", totalSize)
	}

	// Store baselines for future migrations
	adapter.storeDiskBaselines(vmID, diskBlocks)

	// Close and evict this connection — see the matching comment in
	// MigrateVMMemory.
	adapter.CleanupConnection(vmID, targetHost)

	return nil
}

// createConnection always creates and connects a fresh MigrationConnection.
// Renamed from getOrCreateConnection: this package no longer reuses a
// cached connection across multiple migrations — each migration dials a
// brand new set of AMST streams (however many AMSTConfig calls for) and
// MigrateVMMemory/MigrateVMDisk close all of them via CleanupConnection
// immediately after Transfer succeeds. The receive side no longer depends
// on "exactly one connection per migration" to know which accepted
// net.Conns belong together — see writeDWCPStreamEnvelope/
// handleIncomingDWCPStream for the session-ID-based correlation that
// replaced that invariant (novacron-hpa).
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

	// Create new AMST instance for this connection, using whatever stream
	// count/adaptive settings NewMigrationAdapter resolved (defaults to
	// single-stream unless the caller opted into more — see there).
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

	// Select tier based on latency. On a fast/LAN link, bandwidth is not the
	// bottleneck and compression CPU is not recovered — measured 2.2x SLOWER
	// than sending uncompressed (novacron-94l) — so skip compression entirely
	// below the fast-link threshold. Compression only pays off once the link
	// is slow enough that fewer bytes on the wire outweigh the compression CPU
	// (a decisive 2.6-2.78x win on WAN-bandwidth-constrained links).
	switch {
	case latency < 5:
		return CompressionLevelNone // Fast/LAN: don't compress
	case latency < 10:
		return CompressionLocal // Fast local network
	case latency < 50:
		return CompressionRegional // Regional network
	default:
		return CompressionGlobal // WAN/Internet
	}
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
	delta := int64(len(memoryData)) - int64(len(baseline.MemoryBaseline))
	baseline.MemoryBaseline = memoryData
	baseline.LastUpdated = time.Now()
	baseline.mu.Unlock()

	// Track retained bytes and enforce the MaxMemoryUsage cap (novacron-y45).
	baseline.retainedBytes += delta
	adapter.baselineBytes += delta
	adapter.touchBaselineLocked(baseline)
	adapter.evictBaselinesLocked()
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
	var delta int64
	for blockID, blockData := range diskBlocks {
		delta += int64(len(blockData)) - int64(len(baseline.DiskBaselines[blockID]))
		baseline.DiskBaselines[blockID] = blockData
	}
	baseline.LastUpdated = time.Now()
	baseline.mu.Unlock()

	// Track retained bytes and enforce the MaxMemoryUsage cap (novacron-y45).
	baseline.retainedBytes += delta
	adapter.baselineBytes += delta
	adapter.touchBaselineLocked(baseline)
	adapter.evictBaselinesLocked()
}

// touchBaselineLocked records b as most-recently-used in the eviction LRU.
// Caller MUST hold adapter.mu (novacron-y45).
func (adapter *MigrationAdapter) touchBaselineLocked(b *VMBaseline) {
	if b.lruElem == nil {
		b.lruElem = adapter.baselineLRU.PushFront(b.VMID)
	} else {
		adapter.baselineLRU.MoveToFront(b.lruElem)
	}
}

// evictBaselinesLocked evicts least-recently-used VM baselines until the total
// retained baseline bytes fit within config.MaxMemoryUsage. Caller MUST hold
// adapter.mu. Bounds adapter.vmBaselines against high VM churn: baselines were
// previously stored per unique vmID with no eviction and no reader, a pure
// memory leak (novacron-y45).
func (adapter *MigrationAdapter) evictBaselinesLocked() {
	for adapter.baselineBytes > adapter.config.MaxMemoryUsage && adapter.baselineLRU.Len() > 0 {
		oldest := adapter.baselineLRU.Back()
		vmID := oldest.Value.(string)
		if b, ok := adapter.vmBaselines[vmID]; ok {
			adapter.baselineBytes -= b.retainedBytes
			delete(adapter.vmBaselines, vmID)
		}
		adapter.baselineLRU.Remove(oldest)
	}
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

// updateAvgThroughputVsBaseline updates the running average
// throughput-vs-fixed-reference-point ratio (not a DWCP-vs-standard
// speedup - see avgThroughputVsBaseline's doc comment)
func (adapter *MigrationAdapter) updateAvgThroughputVsBaseline(ratio float64) {
	// Exponential moving average
	current := adapter.avgThroughputVsBaseline.Load().(float64)
	newAverage := current*0.8 + ratio*0.2
	adapter.avgThroughputVsBaseline.Store(newAverage)
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
	avgThroughputVsBaseline := adapter.avgThroughputVsBaseline.Load().(float64)
	totalBytes := adapter.totalBytesTransferred.Load()

	fmt.Printf("DWCP Migration Metrics - Success Rate: %.2f%%, Avg Throughput vs Reference: %.2fx (NOT vs standard), Total: %.2f GB\n",
		successRate*100, avgThroughputVsBaseline, float64(totalBytes)/1024/1024/1024)

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
	baselineBytes := adapter.baselineBytes
	adapter.mu.RUnlock()

	metrics := map[string]interface{}{
		"migrations_completed":             adapter.migrationsCompleted.Load(),
		"migrations_failed":                adapter.migrationsFailed.Load(),
		"total_bytes_transferred":          adapter.totalBytesTransferred.Load(),
		"avg_throughput_vs_reference_ratio": adapter.avgThroughputVsBaseline.Load(),
		"active_connections":               activeConnections,
		"baseline_count":                   baselineCount,
		"baseline_bytes":                   baselineBytes,
		"dwcp_enabled":                     adapter.config.EnableDWCP,
		"fallback_enabled":                 adapter.config.EnableFallback,
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

	// Close any DWCP multi-stream sessions still waiting for the rest of
	// their streams to arrive.
	adapter.sessionMu.Lock()
	for sessionID, session := range adapter.pendingSessions {
		for _, c := range session.conns {
			if c != nil {
				c.Close()
			}
		}
		if session.timer != nil {
			session.timer.Stop()
		}
		delete(adapter.pendingSessions, sessionID)
	}
	adapter.sessionMu.Unlock()

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
// then dispatches to the matching receive function — directly for
// standard paths (always exactly one connection per migration); via DWCP
// multi-stream session correlation for DWCP paths, since a DWCP migration
// may span N independently-accepted connections (see
// handleIncomingDWCPStream).
func (adapter *MigrationAdapter) handleIncomingMigration(ctx context.Context, conn net.Conn) {
	// Bound the envelope read so a connection that never sends one (or a
	// dead/stalled sender) cannot leak this goroutine forever.
	if adapter.config.ConnectionTimeout > 0 {
		conn.SetReadDeadline(time.Now().Add(adapter.config.ConnectionTimeout))
	}
	protocol, vmID, err := readMigrateEnvelope(conn)
	conn.SetReadDeadline(time.Time{})
	if err != nil {
		conn.Close()
		fmt.Printf("DWCP: failed to read migration envelope: %v\n", err)
		return
	}

	switch protocol {
	case migrateProtoStandardMemory:
		defer conn.Close()
		adapter.receiveMemoryStandard(conn, vmID)
	case migrateProtoStandardDisk:
		defer conn.Close()
		adapter.receiveDiskStandard(conn, vmID)
	case migrateProtoDWCPMemory, migrateProtoDWCPDisk:
		// conn ownership transfers to the DWCP session machinery from
		// here: it is either handed to a still-forming session (closed
		// later by that session's AMST.Close() once Receive() completes,
		// or by expireDWCPSession on timeout) or, if this call completes
		// the session, closed the same way once completeDWCPSession's
		// Receive() returns.
		adapter.handleIncomingDWCPStream(ctx, conn, protocol, vmID)
	default:
		conn.Close()
		fmt.Printf("DWCP: unknown migration protocol byte: %d\n", protocol)
	}
}

// pendingDWCPSession accumulates the N independently-accepted net.Conns
// (one per AMST stream) that share one sessionID, written by the sender
// via writeDWCPStreamEnvelope, until all streamCount of them have arrived
// — see handleIncomingDWCPStream/registerDWCPStream.
type pendingDWCPSession struct {
	protocol    byte
	vmID        string
	streamCount int
	conns       []net.Conn // indexed by streamIndex; nil until that stream arrives
	received    int
	timer       *time.Timer // fires expireDWCPSession if the session never completes
}

// handleIncomingDWCPStream reads the session correlation header (written
// immediately after the protocol+vmID envelope handleIncomingMigration
// already consumed) from one accepted DWCP connection, registers it into
// its migration's pendingDWCPSession, and — only for whichever goroutine's
// registration completes that session — proceeds to actually receive the
// migration. Every other stream's goroutine returns immediately without
// closing conn: ownership has transferred to the pending session (or, on
// error, conn is closed here).
func (adapter *MigrationAdapter) handleIncomingDWCPStream(ctx context.Context, conn net.Conn, protocol byte, vmID string) {
	if adapter.config.ConnectionTimeout > 0 {
		conn.SetReadDeadline(time.Now().Add(adapter.config.ConnectionTimeout))
	}
	sessionID, streamIndex, streamCount, err := readDWCPStreamCorrelation(conn)
	conn.SetReadDeadline(time.Time{})
	if err != nil {
		conn.Close()
		fmt.Printf("DWCP: failed to read stream correlation for vmID %s: %v\n", vmID, err)
		return
	}

	session, complete, err := adapter.registerDWCPStream(sessionID, streamCount, protocol, vmID, streamIndex, conn)
	if err != nil {
		conn.Close()
		fmt.Printf("DWCP: %v\n", err)
		return
	}
	if !complete {
		return
	}

	adapter.completeDWCPSession(ctx, session)
}

// registerDWCPStream adds conn to the pending session for sessionID
// (creating it, and arming its expiry timer, on first arrival), returning
// the session and whether this call completed it (all streamCount streams
// now registered). Once complete, the session is removed from
// pendingSessions and its timer stopped — the caller becomes solely
// responsible for it.
func (adapter *MigrationAdapter) registerDWCPStream(sessionID string, streamCount int, protocol byte, vmID string, streamIndex int, conn net.Conn) (*pendingDWCPSession, bool, error) {
	if streamCount <= 0 || streamIndex < 0 || streamIndex >= streamCount {
		return nil, false, fmt.Errorf("invalid DWCP stream index/count for vmID %s: index=%d count=%d", vmID, streamIndex, streamCount)
	}

	adapter.sessionMu.Lock()
	defer adapter.sessionMu.Unlock()

	session, ok := adapter.pendingSessions[sessionID]
	if !ok {
		session = &pendingDWCPSession{
			protocol:    protocol,
			vmID:        vmID,
			streamCount: streamCount,
			conns:       make([]net.Conn, streamCount),
		}
		adapter.pendingSessions[sessionID] = session
		if adapter.config.ConnectionTimeout > 0 {
			session.timer = time.AfterFunc(adapter.config.ConnectionTimeout, func() {
				adapter.expireDWCPSession(sessionID)
			})
		}
	}
	if session.streamCount != streamCount {
		return nil, false, fmt.Errorf("DWCP stream count mismatch for vmID %s: session expects %d, stream reports %d", vmID, session.streamCount, streamCount)
	}
	if session.conns[streamIndex] != nil {
		return nil, false, fmt.Errorf("duplicate DWCP stream index %d for vmID %s", streamIndex, vmID)
	}

	session.conns[streamIndex] = conn
	session.received++
	complete := session.received == session.streamCount
	if complete {
		delete(adapter.pendingSessions, sessionID)
		if session.timer != nil {
			session.timer.Stop()
		}
	}
	return session, complete, nil
}

// expireDWCPSession evicts and closes a session's partially-arrived
// streams if it is still pending ConnectionTimeout after its first stream
// arrived — guards against leaking accepted connections (and the
// goroutines blocked registering them) forever when a sender dials fewer
// streams than it announces, or some of its connections never reach this
// listener at all.
func (adapter *MigrationAdapter) expireDWCPSession(sessionID string) {
	adapter.sessionMu.Lock()
	session, ok := adapter.pendingSessions[sessionID]
	if ok {
		delete(adapter.pendingSessions, sessionID)
	}
	adapter.sessionMu.Unlock()
	if !ok {
		return // already completed (or expired) concurrently
	}

	fmt.Printf("DWCP: session for vmID %s timed out with %d of %d streams received\n", session.vmID, session.received, session.streamCount)
	for _, c := range session.conns {
		if c != nil {
			c.Close()
		}
	}
}

// completeDWCPSession wraps a fully-arrived session's connections in a
// receive-only AMST instance (mirroring how Connect wires up a dialed
// stream: id/conn/active/lastActive set, appended to amst.streams,
// activeStreams incremented) and dispatches to the matching receive
// function. Streams that end up receiving zero chunks (N > actual chunk
// count) are handled by AMST.Receive itself: the sender closes every one
// of its streams right after a successful Transfer regardless of how many
// chunks each carried (CleanupConnection -> AMST.Close), so a zero-chunk
// stream's first header read cleanly hits io.EOF (AMST.Receive's
// `err == io.EOF` branch) instead of blocking forever.
func (adapter *MigrationAdapter) completeDWCPSession(ctx context.Context, session *pendingDWCPSession) {
	amstInst, err := NewAMST(AMSTConfig{
		MinStreams:     session.streamCount,
		MaxStreams:     session.streamCount,
		InitialStreams: session.streamCount,
		ReadTimeout:    adapter.config.ConnectionTimeout,
	})
	if err != nil {
		fmt.Printf("DWCP: failed to create receive-side AMST for vmID %s: %v\n", session.vmID, err)
		for _, c := range session.conns {
			c.Close()
		}
		return
	}

	amstInst.mu.Lock()
	for i, c := range session.conns {
		amstInst.streams = append(amstInst.streams, &Stream{
			id:         fmt.Sprintf("recv-%d", i),
			conn:       c,
			amst:       amstInst,
			active:     true,
			lastActive: time.Now(),
		})
		amstInst.activeStreams.Add(1)
	}
	amstInst.mu.Unlock()

	switch session.protocol {
	case migrateProtoDWCPMemory:
		adapter.receiveMemoryDWCP(ctx, amstInst, session.vmID)
	case migrateProtoDWCPDisk:
		adapter.receiveDiskDWCP(ctx, amstInst, session.vmID)
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

// receiveMemoryDWCP receives memory sent by MigrateVMMemory: HDE-compressed
// bytes, AMST-framed across amstInst's (possibly many) streams. amstInst is
// already fully populated with every stream of this migration by the time
// this is called — see completeDWCPSession, which builds it from a
// fully-arrived pendingDWCPSession before dispatching here.
func (adapter *MigrationAdapter) receiveMemoryDWCP(ctx context.Context, amstInst *AMST, vmID string) {
	defer amstInst.Close()

	if adapter.hde == nil {
		fmt.Printf("DWCP: received DWCP memory migration for vmID %s but HDE is not initialized\n", vmID)
		return
	}
	compressed, err := amstInst.Receive(ctx, nil)
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
// bytes across amstInst's (possibly many) streams, reassembling into the
// [blockID:4][blockLen:4]+HDE-compressed-data layout MigrateVMDisk wrote.
// amstInst is already fully populated — see receiveMemoryDWCP's doc
// comment (same contract).
func (adapter *MigrationAdapter) receiveDiskDWCP(ctx context.Context, amstInst *AMST, vmID string) {
	defer amstInst.Close()

	if adapter.hde == nil {
		fmt.Printf("DWCP: received DWCP disk migration for vmID %s but HDE is not initialized\n", vmID)
		return
	}
	compressedBlocks, err := amstInst.Receive(ctx, nil)
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
