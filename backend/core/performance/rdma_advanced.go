// RDMA Advanced Optimization for DWCP v3
//
// Implements advanced RDMA features including:
// - Dynamically Connected (DC) transport
// - Extended Reliable Connected (XRC) transport
// - RDMA write/read optimization
// - Memory registration optimization
// - Adaptive RDMA configuration
//
// Phase 7: Extreme Performance Optimization
// Target: Sub-microsecond latency, 100+ Gbps throughput

package performance

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"
)

// RDMA Transport Types
type TransportType int

const (
	TransportRC  TransportType = iota // Reliable Connected
	TransportUC                        // Unreliable Connected
	TransportRD                        // Reliable Datagram
	TransportUD                        // Unreliable Datagram
	TransportDC                        // Dynamically Connected
	TransportXRC                       // Extended Reliable Connected
)

// RDMA Operation Types
type OpType int

const (
	OpSend OpType = iota
	OpRecv
	OpWrite
	OpRead
	OpAtomic
	OpFetchAdd
	OpCompareSwap
)

// Memory Registration Cache
type MemoryRegion struct {
	addr   uintptr
	length uint64
	lkey   uint32
	rkey   uint32
	handle unsafe.Pointer
	refCnt atomic.Int64
	flags  uint32
}

// RDMA Queue Pair Configuration
type QueuePairConfig struct {
	transport      TransportType
	maxSendWR      uint32
	maxRecvWR      uint32
	maxSendSGE     uint32
	maxRecvSGE     uint32
	maxInlineData  uint32
	qpNum          uint32
	signalAll      bool
	sqPollInterval time.Duration
}

// RDMA Completion Queue
type CompletionQueue struct {
	cqe         []WorkCompletion
	size        uint32
	head        atomic.Uint32
	tail        atomic.Uint32
	events      chan WorkCompletion
	pollMode    bool
	eventMode   bool
	armRequired bool
}

// Work Completion Entry
type WorkCompletion struct {
	wrID      uint64
	status    CompletionStatus
	opcode    OpType
	byteLen   uint32
	qpNum     uint32
	srcQP     uint32
	timestamp uint64
}

// Completion Status
type CompletionStatus int

const (
	StatusSuccess CompletionStatus = iota
	StatusLocalLengthError
	StatusLocalQPOpError
	StatusLocalProtectionError
	StatusWRFlushError
	StatusMemoryWindowBindError
	StatusBadResponseError
	StatusLocalAccessError
	StatusRemoteInvalidRequestError
	StatusRemoteAccessError
	StatusRemoteOpError
	StatusRetryExceeded
	StatusRNRRetryExceeded
)

// Advanced RDMA Manager
type RDMAManager struct {
	ctx              context.Context
	cancel           context.CancelFunc
	mu               sync.RWMutex
	queuePairs       map[uint32]*QueuePair
	memoryRegions    map[uintptr]*MemoryRegion
	mrCache          *MemoryRegistrationCache
	completionQueues map[uint32]*CompletionQueue
	transport        TransportType
	stats            *RDMAStats
	adaptiveConfig   *AdaptiveConfig
	dcTargets        map[string]*DCTarget
	xrcDomains       map[uint32]*XRCDomain
}

// Queue Pair
type QueuePair struct {
	qpNum          uint32
	config         QueuePairConfig
	sendCQ         *CompletionQueue
	recvCQ         *CompletionQueue
	sendQueue      []WorkRequest
	recvQueue      []WorkRequest
	state          QPState
	sqHead         atomic.Uint32
	sqTail         atomic.Uint32
	rqHead         atomic.Uint32
	rqTail         atomic.Uint32
	inlineThreshold uint32
	signalInterval  uint32
	opCounter      atomic.Uint64
}

// Queue Pair State
type QPState int

const (
	QPStateReset QPState = iota
	QPStateInit
	QPStateRTR // Ready to Receive
	QPStateRTS // Ready to Send
	QPStateError
)

// Work Request
type WorkRequest struct {
	wrID       uint64
	opcode     OpType
	flags      uint32
	sgeList    []ScatterGatherElement
	remoteAddr uint64
	rkey       uint32
	compareAdd uint64
	swap       uint64
	timestamp  uint64
}

// Scatter/Gather Element
type ScatterGatherElement struct {
	addr   uint64
	length uint32
	lkey   uint32
}

// Memory Registration Cache
type MemoryRegistrationCache struct {
	mu      sync.RWMutex
	cache   map[uintptr]*MemoryRegion
	lruList []uintptr
	maxSize int
	hits    atomic.Uint64
	misses  atomic.Uint64
}

// RDMA Statistics
type RDMAStats struct {
	sendOps        atomic.Uint64
	recvOps        atomic.Uint64
	writeOps       atomic.Uint64
	readOps        atomic.Uint64
	atomicOps      atomic.Uint64
	completions    atomic.Uint64
	errors         atomic.Uint64
	retries        atomic.Uint64
	inlineHits     atomic.Uint64
	zerocopyhits   atomic.Uint64
	totalBytes     atomic.Uint64
	totalLatencyNs atomic.Uint64
	pollCycles     atomic.Uint64
}

// Adaptive Configuration
type AdaptiveConfig struct {
	mu                    sync.RWMutex
	currentInlineThreshold uint32
	currentSignalInterval  uint32
	currentPollInterval    time.Duration
	latencyTarget         time.Duration
	throughputTarget      uint64
	measurements          []PerformanceMeasurement
	adjustmentInterval    time.Duration
	lastAdjustment        time.Time
}

// Performance Measurement
type PerformanceMeasurement struct {
	timestamp      time.Time
	latency        time.Duration
	throughput     uint64
	cpuUtilization float64
	queueDepth     uint32
	retryRate      float64
}

// DC (Dynamically Connected) Target
type DCTarget struct {
	dcKey      uint64
	targetQPN  uint32
	targetLID  uint16
	targetGID  [16]byte
	pathMTU    uint32
	refCount   atomic.Int32
	lastUsed   time.Time
	connection unsafe.Pointer
}

// XRC (Extended Reliable Connected) Domain
type XRCDomain struct {
	xrcdNum    uint32
	sharedRQ   *ReceiveQueue
	qpList     []*QueuePair
	srqContext unsafe.Pointer
	mu         sync.RWMutex
}

// Shared Receive Queue
type ReceiveQueue struct {
	srqNum     uint32
	maxWR      uint32
	maxSGE     uint32
	limit      uint32
	head       atomic.Uint32
	tail       atomic.Uint32
	recvQueue  []WorkRequest
	mu         sync.Mutex
}

// NewRDMAManager creates a new advanced RDMA manager
func NewRDMAManager(ctx context.Context, transport TransportType) (*RDMAManager, error) {
	ctx, cancel := context.WithCancel(ctx)

	rm := &RDMAManager{
		ctx:              ctx,
		cancel:           cancel,
		queuePairs:       make(map[uint32]*QueuePair),
		memoryRegions:    make(map[uintptr]*MemoryRegion),
		completionQueues: make(map[uint32]*CompletionQueue),
		transport:        transport,
		stats:            &RDMAStats{},
		dcTargets:        make(map[string]*DCTarget),
		xrcDomains:       make(map[uint32]*XRCDomain),
	}

	// Initialize memory registration cache
	rm.mrCache = &MemoryRegistrationCache{
		cache:   make(map[uintptr]*MemoryRegion),
		lruList: make([]uintptr, 0, 1024),
		maxSize: 1024,
	}

	// Initialize adaptive configuration
	rm.adaptiveConfig = &AdaptiveConfig{
		currentInlineThreshold: 256,
		currentSignalInterval:  64,
		currentPollInterval:    100 * time.Microsecond,
		latencyTarget:         10 * time.Microsecond,
		throughputTarget:      100 * 1024 * 1024 * 1024, // 100 Gbps
		measurements:          make([]PerformanceMeasurement, 0, 1000),
		adjustmentInterval:    1 * time.Second,
		lastAdjustment:        time.Now(),
	}

	// Start adaptive tuning
	go rm.adaptiveTuningLoop()

	// Start statistics collection
	go rm.statsCollectionLoop()

	return rm, nil
}

// CreateQueuePair creates a new queue pair with specified configuration
func (rm *RDMAManager) CreateQueuePair(config QueuePairConfig) (*QueuePair, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	// Create send completion queue
	sendCQ, err := rm.createCompletionQueue(config.maxSendWR, true)
	if err != nil {
		return nil, fmt.Errorf("failed to create send CQ: %w", err)
	}

	// Create receive completion queue
	recvCQ, err := rm.createCompletionQueue(config.maxRecvWR, false)
	if err != nil {
		return nil, fmt.Errorf("failed to create recv CQ: %w", err)
	}

	qp := &QueuePair{
		qpNum:           config.qpNum,
		config:          config,
		sendCQ:          sendCQ,
		recvCQ:          recvCQ,
		sendQueue:       make([]WorkRequest, config.maxSendWR),
		recvQueue:       make([]WorkRequest, config.maxRecvWR),
		state:           QPStateReset,
		inlineThreshold: config.maxInlineData,
		signalInterval:  rm.adaptiveConfig.currentSignalInterval,
	}

	rm.queuePairs[qp.qpNum] = qp

	return qp, nil
}

// RegisterMemory registers a memory region for RDMA operations
func (rm *RDMAManager) RegisterMemory(addr uintptr, length uint64, flags uint32) (*MemoryRegion, error) {
	// Check cache first
	if mr := rm.mrCache.lookup(addr, length); mr != nil {
		rm.stats.inlineHits.Add(1)
		mr.refCnt.Add(1)
		return mr, nil
	}

	rm.mrCache.misses.Add(1)

	// Register new memory region
	mr := &MemoryRegion{
		addr:   addr,
		length: length,
		flags:  flags,
	}

	// Simulate hardware registration (in production, use actual verbs)
	mr.lkey = uint32(addr & 0xFFFFFFFF)
	mr.rkey = uint32((addr >> 32) & 0xFFFFFFFF)
	mr.handle = unsafe.Pointer(addr)
	mr.refCnt.Store(1)

	rm.mu.Lock()
	rm.memoryRegions[addr] = mr
	rm.mu.Unlock()

	// Add to cache
	rm.mrCache.insert(mr)

	return mr, nil
}

// PostSend posts a send work request to the queue pair
func (rm *RDMAManager) PostSend(qpNum uint32, wr WorkRequest) error {
	rm.mu.RLock()
	qp, exists := rm.queuePairs[qpNum]
	rm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("queue pair %d not found", qpNum)
	}

	if qp.state != QPStateRTS {
		return fmt.Errorf("queue pair not in RTS state")
	}

	// Check if we should use inline
	totalSize := uint32(0)
	for _, sge := range wr.sgeList {
		totalSize += sge.length
	}

	useInline := totalSize <= qp.inlineThreshold
	if useInline {
		wr.flags |= 0x01 // Inline flag
		rm.stats.inlineHits.Add(1)
	}

	// Determine if we need completion
	opNum := qp.opCounter.Add(1)
	needCompletion := (opNum % uint64(qp.signalInterval)) == 0 || qp.config.signalAll
	if needCompletion {
		wr.flags |= 0x02 // Signaled flag
	}

	// Add timestamp
	wr.timestamp = uint64(time.Now().UnixNano())

	// Post to send queue
	tail := qp.sqTail.Load()
	if tail-qp.sqHead.Load() >= qp.config.maxSendWR {
		return fmt.Errorf("send queue full")
	}

	qp.sendQueue[tail%qp.config.maxSendWR] = wr
	qp.sqTail.Store(tail + 1)

	// Update statistics
	switch wr.opcode {
	case OpSend:
		rm.stats.sendOps.Add(1)
	case OpWrite:
		rm.stats.writeOps.Add(1)
	case OpRead:
		rm.stats.readOps.Add(1)
	case OpAtomic, OpFetchAdd, OpCompareSwap:
		rm.stats.atomicOps.Add(1)
	}

	rm.stats.totalBytes.Add(uint64(totalSize))

	// Process send queue (simulate hardware processing)
	go rm.processSendQueue(qp)

	return nil
}

// PostRecv posts a receive work request to the queue pair
func (rm *RDMAManager) PostRecv(qpNum uint32, wr WorkRequest) error {
	rm.mu.RLock()
	qp, exists := rm.queuePairs[qpNum]
	rm.mu.RUnlock()

	if !exists {
		return fmt.Errorf("queue pair %d not found", qpNum)
	}

	if qp.state != QPStateRTR && qp.state != QPStateRTS {
		return fmt.Errorf("queue pair not ready to receive")
	}

	tail := qp.rqTail.Load()
	if tail-qp.rqHead.Load() >= qp.config.maxRecvWR {
		return fmt.Errorf("recv queue full")
	}

	qp.recvQueue[tail%qp.config.maxRecvWR] = wr
	qp.rqTail.Store(tail + 1)

	rm.stats.recvOps.Add(1)

	return nil
}

// PollCompletion polls the completion queue for completed operations
func (rm *RDMAManager) PollCompletion(cqNum uint32, maxEntries int) ([]WorkCompletion, error) {
	rm.mu.RLock()
	cq, exists := rm.completionQueues[cqNum]
	rm.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("completion queue %d not found", cqNum)
	}

	completions := make([]WorkCompletion, 0, maxEntries)

	for len(completions) < maxEntries {
		head := cq.head.Load()
		tail := cq.tail.Load()

		if head >= tail {
			break
		}

		wc := cq.cqe[head%cq.size]
		completions = append(completions, wc)

		cq.head.Store(head + 1)
		rm.stats.completions.Add(1)

		// Calculate latency
		if wc.timestamp > 0 {
			latency := uint64(time.Now().UnixNano()) - wc.timestamp
			rm.stats.totalLatencyNs.Add(latency)
		}
	}

	rm.stats.pollCycles.Add(1)

	return completions, nil
}

// RDMAWrite performs an RDMA write operation
func (rm *RDMAManager) RDMAWrite(qpNum uint32, localAddr uintptr, length uint32,
	lkey uint32, remoteAddr uint64, rkey uint32) error {

	wr := WorkRequest{
		wrID:       uint64(time.Now().UnixNano()),
		opcode:     OpWrite,
		flags:      0,
		remoteAddr: remoteAddr,
		rkey:       rkey,
		sgeList: []ScatterGatherElement{
			{
				addr:   uint64(localAddr),
				length: length,
				lkey:   lkey,
			},
		},
	}

	return rm.PostSend(qpNum, wr)
}

// RDMARead performs an RDMA read operation
func (rm *RDMAManager) RDMARead(qpNum uint32, localAddr uintptr, length uint32,
	lkey uint32, remoteAddr uint64, rkey uint32) error {

	wr := WorkRequest{
		wrID:       uint64(time.Now().UnixNano()),
		opcode:     OpRead,
		flags:      0x02, // Always signal reads
		remoteAddr: remoteAddr,
		rkey:       rkey,
		sgeList: []ScatterGatherElement{
			{
				addr:   uint64(localAddr),
				length: length,
				lkey:   lkey,
			},
		},
	}

	return rm.PostSend(qpNum, wr)
}

// CreateDCTarget creates a Dynamically Connected target
func (rm *RDMAManager) CreateDCTarget(targetID string, qpn uint32, lid uint16, gid [16]byte) (*DCTarget, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if rm.transport != TransportDC {
		return nil, fmt.Errorf("DC transport not enabled")
	}

	target := &DCTarget{
		dcKey:     uint64(time.Now().UnixNano()),
		targetQPN: qpn,
		targetLID: lid,
		targetGID: gid,
		pathMTU:   4096,
		lastUsed:  time.Now(),
	}

	target.refCount.Store(1)
	rm.dcTargets[targetID] = target

	return target, nil
}

// CreateXRCDomain creates an Extended Reliable Connected domain
func (rm *RDMAManager) CreateXRCDomain(xrcdNum uint32) (*XRCDomain, error) {
	rm.mu.Lock()
	defer rm.mu.Unlock()

	if rm.transport != TransportXRC {
		return nil, fmt.Errorf("XRC transport not enabled")
	}

	// Create shared receive queue
	srq := &ReceiveQueue{
		srqNum:    xrcdNum,
		maxWR:     1024,
		maxSGE:    16,
		limit:     512,
		recvQueue: make([]WorkRequest, 1024),
	}

	domain := &XRCDomain{
		xrcdNum:  xrcdNum,
		sharedRQ: srq,
		qpList:   make([]*QueuePair, 0, 64),
	}

	rm.xrcDomains[xrcdNum] = domain

	return domain, nil
}

// Adaptive tuning loop
func (rm *RDMAManager) adaptiveTuningLoop() {
	ticker := time.NewTicker(rm.adaptiveConfig.adjustmentInterval)
	defer ticker.Stop()

	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.performAdaptiveTuning()
		}
	}
}

// Perform adaptive tuning based on performance metrics
func (rm *RDMAManager) performAdaptiveTuning() {
	rm.adaptiveConfig.mu.Lock()
	defer rm.adaptiveConfig.mu.Unlock()

	// Calculate average latency
	completions := rm.stats.completions.Load()
	if completions == 0 {
		return
	}

	avgLatency := time.Duration(rm.stats.totalLatencyNs.Load() / completions)

	// Calculate throughput
	duration := time.Since(rm.adaptiveConfig.lastAdjustment)
	throughput := rm.stats.totalBytes.Load() / uint64(duration.Seconds())

	// Record measurement
	measurement := PerformanceMeasurement{
		timestamp:  time.Now(),
		latency:    avgLatency,
		throughput: throughput,
	}

	rm.adaptiveConfig.measurements = append(rm.adaptiveConfig.measurements, measurement)
	if len(rm.adaptiveConfig.measurements) > 1000 {
		rm.adaptiveConfig.measurements = rm.adaptiveConfig.measurements[1:]
	}

	// Adjust inline threshold
	if avgLatency > rm.adaptiveConfig.latencyTarget {
		// Increase inline threshold to reduce latency
		rm.adaptiveConfig.currentInlineThreshold = min(
			rm.adaptiveConfig.currentInlineThreshold+64, 512)
	} else if throughput < rm.adaptiveConfig.throughputTarget {
		// Decrease inline threshold to improve throughput
		rm.adaptiveConfig.currentInlineThreshold = max(
			rm.adaptiveConfig.currentInlineThreshold-64, 64)
	}

	// Adjust signal interval
	hitRate := float64(rm.stats.inlineHits.Load()) / float64(rm.stats.sendOps.Load())
	if hitRate < 0.8 {
		rm.adaptiveConfig.currentSignalInterval = min(
			rm.adaptiveConfig.currentSignalInterval*2, 128)
	} else if hitRate > 0.95 {
		rm.adaptiveConfig.currentSignalInterval = max(
			rm.adaptiveConfig.currentSignalInterval/2, 16)
	}

	rm.adaptiveConfig.lastAdjustment = time.Now()
}

// Statistics collection loop
func (rm *RDMAManager) statsCollectionLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-rm.ctx.Done():
			return
		case <-ticker.C:
			rm.printStatistics()
		}
	}
}

// Print statistics
func (rm *RDMAManager) printStatistics() {
	completions := rm.stats.completions.Load()
	if completions == 0 {
		return
	}

	avgLatency := time.Duration(rm.stats.totalLatencyNs.Load() / completions)
	throughput := rm.stats.totalBytes.Load()

	fmt.Printf("\n=== RDMA Statistics ===\n")
	fmt.Printf("Send ops: %d\n", rm.stats.sendOps.Load())
	fmt.Printf("Recv ops: %d\n", rm.stats.recvOps.Load())
	fmt.Printf("Write ops: %d\n", rm.stats.writeOps.Load())
	fmt.Printf("Read ops: %d\n", rm.stats.readOps.Load())
	fmt.Printf("Atomic ops: %d\n", rm.stats.atomicOps.Load())
	fmt.Printf("Completions: %d\n", completions)
	fmt.Printf("Errors: %d\n", rm.stats.errors.Load())
	fmt.Printf("Average latency: %v\n", avgLatency)
	fmt.Printf("Total throughput: %.2f GB/s\n", float64(throughput)/(1024*1024*1024))
	fmt.Printf("Inline hit rate: %.2f%%\n",
		float64(rm.stats.inlineHits.Load())/float64(rm.stats.sendOps.Load())*100)
	fmt.Printf("MR cache hit rate: %.2f%%\n",
		float64(rm.mrCache.hits.Load())/float64(rm.mrCache.hits.Load()+rm.mrCache.misses.Load())*100)
	fmt.Printf("=======================\n\n")
}

// Helper functions for memory registration cache
func (mrc *MemoryRegistrationCache) lookup(addr uintptr, length uint64) *MemoryRegion {
	mrc.mu.RLock()
	defer mrc.mu.RUnlock()

	mr, exists := mrc.cache[addr]
	if exists && mr.length >= length {
		mrc.hits.Add(1)
		return mr
	}

	return nil
}

func (mrc *MemoryRegistrationCache) insert(mr *MemoryRegion) {
	mrc.mu.Lock()
	defer mrc.mu.Unlock()

	if len(mrc.cache) >= mrc.maxSize {
		// Evict LRU entry
		if len(mrc.lruList) > 0 {
			oldest := mrc.lruList[0]
			delete(mrc.cache, oldest)
			mrc.lruList = mrc.lruList[1:]
		}
	}

	mrc.cache[mr.addr] = mr
	mrc.lruList = append(mrc.lruList, mr.addr)
}

// Process send queue (simulated hardware processing)
func (rm *RDMAManager) processSendQueue(qp *QueuePair) {
	head := qp.sqHead.Load()
	tail := qp.sqTail.Load()

	if head >= tail {
		return
	}

	wr := qp.sendQueue[head%qp.config.maxSendWR]

	// Simulate processing delay
	time.Sleep(100 * time.Nanosecond)

	// Create completion
	wc := WorkCompletion{
		wrID:      wr.wrID,
		status:    StatusSuccess,
		opcode:    wr.opcode,
		byteLen:   0,
		qpNum:     qp.qpNum,
		timestamp: wr.timestamp,
	}

	for _, sge := range wr.sgeList {
		wc.byteLen += sge.length
	}

	// Post to completion queue
	cqTail := qp.sendCQ.tail.Load()
	qp.sendCQ.cqe[cqTail%qp.sendCQ.size] = wc
	qp.sendCQ.tail.Store(cqTail + 1)

	qp.sqHead.Store(head + 1)
	rm.stats.zerocopyhits.Add(1)
}

// Create completion queue
func (rm *RDMAManager) createCompletionQueue(size uint32, pollMode bool) (*CompletionQueue, error) {
	cq := &CompletionQueue{
		cqe:       make([]WorkCompletion, size),
		size:      size,
		pollMode:  pollMode,
		eventMode: !pollMode,
	}

	if !pollMode {
		cq.events = make(chan WorkCompletion, size)
	}

	cqNum := uint32(len(rm.completionQueues))
	rm.completionQueues[cqNum] = cq

	return cq, nil
}

func min(a, b uint32) uint32 {
	if a < b {
		return a
	}
	return b
}

func max(a, b uint32) uint32 {
	if a > b {
		return a
	}
	return b
}

// Close cleans up all RDMA resources
func (rm *RDMAManager) Close() error {
	rm.cancel()

	rm.mu.Lock()
	defer rm.mu.Unlock()

	// Cleanup queue pairs
	for _, qp := range rm.queuePairs {
		qp.state = QPStateReset
	}

	// Cleanup memory regions
	for _, mr := range rm.memoryRegions {
		mr.refCnt.Store(0)
	}

	return nil
}
