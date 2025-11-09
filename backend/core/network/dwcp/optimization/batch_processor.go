package optimization

import (
	"fmt"
	"net"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// BatchProcessor processes operations in batches for efficiency
type BatchProcessor struct {
	batchSize  int
	buffer     [][]byte
	timeout    time.Duration
	fd         int
	mu         sync.Mutex
	flushTimer *time.Timer
}

// NewBatchProcessor creates a batch processor
func NewBatchProcessor(batchSize int, timeout time.Duration, fd int) *BatchProcessor {
	bp := &BatchProcessor{
		batchSize: batchSize,
		buffer:    make([][]byte, 0, batchSize),
		timeout:   timeout,
		fd:        fd,
	}

	if timeout > 0 {
		bp.flushTimer = time.AfterFunc(timeout, bp.Flush)
	}

	return bp
}

// AddToBatch adds data to batch
func (bp *BatchProcessor) AddToBatch(data []byte) error {
	bp.mu.Lock()
	bp.buffer = append(bp.buffer, data)

	if len(bp.buffer) >= bp.batchSize {
		batch := bp.buffer
		bp.buffer = make([][]byte, 0, bp.batchSize)
		bp.mu.Unlock()

		// Reset timer
		if bp.flushTimer != nil {
			bp.flushTimer.Reset(bp.timeout)
		}

		return bp.processBatch(batch)
	}

	bp.mu.Unlock()
	return nil
}

// processBatch processes a full batch using writev
func (bp *BatchProcessor) processBatch(batch [][]byte) error {
	if len(batch) == 0 {
		return nil
	}

	// Use writev for vectorized I/O
	iovecs := make([]syscall.Iovec, len(batch))
	for i, data := range batch {
		iovecs[i] = syscall.Iovec{
			Base: &data[0],
			Len:  uint64(len(data)),
		}
	}

	_, _, err := syscall.Syscall(
		syscall.SYS_WRITEV,
		uintptr(bp.fd),
		uintptr(unsafe.Pointer(&iovecs[0])),
		uintptr(len(iovecs)),
	)

	if err != 0 {
		return fmt.Errorf("writev failed: %v", err)
	}

	return nil
}

// Flush flushes pending data
func (bp *BatchProcessor) Flush() error {
	bp.mu.Lock()
	batch := bp.buffer
	bp.buffer = make([][]byte, 0, bp.batchSize)
	bp.mu.Unlock()

	return bp.processBatch(batch)
}

// Close closes the batch processor
func (bp *BatchProcessor) Close() error {
	if bp.flushTimer != nil {
		bp.flushTimer.Stop()
	}
	return bp.Flush()
}

// BatchReader reads in batches using readv
type BatchReader struct {
	batchSize int
	buffers   [][]byte
	fd        int
}

// NewBatchReader creates a batch reader
func NewBatchReader(batchSize, bufferSize int, fd int) *BatchReader {
	buffers := make([][]byte, batchSize)
	for i := range buffers {
		buffers[i] = make([]byte, bufferSize)
	}

	return &BatchReader{
		batchSize: batchSize,
		buffers:   buffers,
		fd:        fd,
	}
}

// Read reads multiple buffers at once
func (br *BatchReader) Read() ([][]byte, error) {
	iovecs := make([]syscall.Iovec, len(br.buffers))
	for i, buf := range br.buffers {
		iovecs[i] = syscall.Iovec{
			Base: &buf[0],
			Len:  uint64(len(buf)),
		}
	}

	n, _, err := syscall.Syscall(
		syscall.SYS_READV,
		uintptr(br.fd),
		uintptr(unsafe.Pointer(&iovecs[0])),
		uintptr(len(iovecs)),
	)

	if err != 0 {
		return nil, fmt.Errorf("readv failed: %v", err)
	}

	// Return only filled buffers
	result := make([][]byte, 0, br.batchSize)
	remaining := int(n)
	for _, buf := range br.buffers {
		if remaining <= 0 {
			break
		}
		size := len(buf)
		if size > remaining {
			size = remaining
		}
		result = append(result, buf[:size])
		remaining -= size
	}

	return result, nil
}

// BatchSender sends multiple packets in one syscall
type BatchSender struct {
	conn      *net.UDPConn
	batchSize int
	messages  []syscall.Iovec
	addrs     []syscall.RawSockaddrInet4
}

// NewBatchSender creates a batch UDP sender
func NewBatchSender(conn *net.UDPConn, batchSize int) (*BatchSender, error) {
	return &BatchSender{
		conn:      conn,
		batchSize: batchSize,
		messages:  make([]syscall.Iovec, 0, batchSize),
		addrs:     make([]syscall.RawSockaddrInet4, 0, batchSize),
	}, nil
}

// Send sends a packet (batched)
func (bs *BatchSender) Send(data []byte, addr *net.UDPAddr) error {
	// Convert address
	var rawAddr syscall.RawSockaddrInet4
	rawAddr.Family = syscall.AF_INET
	rawAddr.Port = uint16(addr.Port)<<8 | uint16(addr.Port)>>8
	copy(rawAddr.Addr[:], addr.IP.To4())

	bs.messages = append(bs.messages, syscall.Iovec{
		Base: &data[0],
		Len:  uint64(len(data)),
	})
	bs.addrs = append(bs.addrs, rawAddr)

	if len(bs.messages) >= bs.batchSize {
		return bs.Flush()
	}

	return nil
}

// Flush sends all pending packets
func (bs *BatchSender) Flush() error {
	if len(bs.messages) == 0 {
		return nil
	}

	// Use sendmmsg for batch sending (Linux 3.0+)
	file, err := bs.conn.File()
	if err != nil {
		return err
	}
	defer file.Close()

	// TODO: Implement sendmmsg syscall
	// For now, fall back to individual sends
	for i := range bs.messages {
		_, _, err := syscall.Syscall6(
			syscall.SYS_SENDTO,
			file.Fd(),
			uintptr(unsafe.Pointer(bs.messages[i].Base)),
			uintptr(bs.messages[i].Len),
			0,
			uintptr(unsafe.Pointer(&bs.addrs[i])),
			unsafe.Sizeof(bs.addrs[i]),
		)
		if err != 0 {
			return fmt.Errorf("sendto failed: %v", err)
		}
	}

	bs.messages = bs.messages[:0]
	bs.addrs = bs.addrs[:0]

	return nil
}

// BatchAllocator allocates multiple objects at once
type BatchAllocator struct {
	objectSize int
	batchSize  int
	pool       *ObjectPool
}

// NewBatchAllocator creates a batch allocator
func NewBatchAllocator(objectSize, batchSize int) *BatchAllocator {
	return &BatchAllocator{
		objectSize: objectSize,
		batchSize:  batchSize,
		pool:       NewObjectPool(),
	}
}

// AllocateBatch allocates multiple objects
func (ba *BatchAllocator) AllocateBatch() [][]byte {
	batch := make([][]byte, ba.batchSize)
	for i := range batch {
		batch[i] = ba.pool.Get(ba.objectSize)
	}
	return batch
}

// FreeBatch frees multiple objects
func (ba *BatchAllocator) FreeBatch(batch [][]byte) {
	for _, obj := range batch {
		ba.pool.Put(obj)
	}
}

// CoalescingBuffer coalesces small writes into larger ones
type CoalescingBuffer struct {
	buffer   []byte
	capacity int
	size     int
	mu       sync.Mutex
	writer   func([]byte) error
}

// NewCoalescingBuffer creates a coalescing buffer
func NewCoalescingBuffer(capacity int, writer func([]byte) error) *CoalescingBuffer {
	return &CoalescingBuffer{
		buffer:   make([]byte, capacity),
		capacity: capacity,
		writer:   writer,
	}
}

// Write writes data, coalescing if possible
func (cb *CoalescingBuffer) Write(data []byte) error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	// If data fits in buffer, append it
	if cb.size+len(data) <= cb.capacity {
		copy(cb.buffer[cb.size:], data)
		cb.size += len(data)
		return nil
	}

	// Buffer full, flush and write
	if cb.size > 0 {
		if err := cb.writer(cb.buffer[:cb.size]); err != nil {
			return err
		}
		cb.size = 0
	}

	// If data is larger than buffer, write directly
	if len(data) > cb.capacity {
		return cb.writer(data)
	}

	// Otherwise, buffer it
	copy(cb.buffer, data)
	cb.size = len(data)
	return nil
}

// Flush flushes the buffer
func (cb *CoalescingBuffer) Flush() error {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.size == 0 {
		return nil
	}

	err := cb.writer(cb.buffer[:cb.size])
	cb.size = 0
	return err
}

// PipelinedProcessor processes data in pipeline stages
type PipelinedProcessor struct {
	stages   []func([]byte) []byte
	workers  int
	inputCh  chan []byte
	outputCh chan []byte
	done     chan struct{}
}

// NewPipelinedProcessor creates a pipelined processor
func NewPipelinedProcessor(stages []func([]byte) []byte, workers int) *PipelinedProcessor {
	pp := &PipelinedProcessor{
		stages:   stages,
		workers:  workers,
		inputCh:  make(chan []byte, workers*2),
		outputCh: make(chan []byte, workers*2),
		done:     make(chan struct{}),
	}

	pp.start()
	return pp
}

// start initializes pipeline workers
func (pp *PipelinedProcessor) start() {
	for i := 0; i < pp.workers; i++ {
		go pp.worker()
	}
}

// worker processes data through pipeline stages
func (pp *PipelinedProcessor) worker() {
	for {
		select {
		case data := <-pp.inputCh:
			// Process through all stages
			result := data
			for _, stage := range pp.stages {
				result = stage(result)
			}
			pp.outputCh <- result
		case <-pp.done:
			return
		}
	}
}

// Process processes data through pipeline
func (pp *PipelinedProcessor) Process(data []byte) {
	pp.inputCh <- data
}

// Result returns processed data
func (pp *PipelinedProcessor) Result() []byte {
	return <-pp.outputCh
}

// Close closes the processor
func (pp *PipelinedProcessor) Close() {
	close(pp.done)
	close(pp.inputCh)
	close(pp.outputCh)
}
