package lockfree

import (
	"sync/atomic"
	"unsafe"
)

// LockFreeRingBuffer implements a high-performance ring buffer
type LockFreeRingBuffer struct {
	buffer    []unsafe.Pointer
	mask      uint64
	head      uint64
	tail      uint64
	_padding1 [56]byte // Prevent false sharing
}

// NewLockFreeRingBuffer creates a ring buffer (size must be power of 2)
func NewLockFreeRingBuffer(size uint64) *LockFreeRingBuffer {
	if size&(size-1) != 0 {
		panic("size must be power of 2")
	}

	return &LockFreeRingBuffer{
		buffer: make([]unsafe.Pointer, size),
		mask:   size - 1,
		head:   0,
		tail:   0,
	}
}

// Push adds an item to the ring buffer
func (rb *LockFreeRingBuffer) Push(value interface{}) bool {
	for {
		head := atomic.LoadUint64(&rb.head)
		tail := atomic.LoadUint64(&rb.tail)

		// Check if buffer is full
		if (head-tail) >= uint64(len(rb.buffer)) {
			return false
		}

		idx := head & rb.mask
		if atomic.CompareAndSwapUint64(&rb.head, head, head+1) {
			atomic.StorePointer(&rb.buffer[idx], unsafe.Pointer(&value))
			return true
		}
	}
}

// Pop removes an item from the ring buffer
func (rb *LockFreeRingBuffer) Pop() (interface{}, bool) {
	for {
		tail := atomic.LoadUint64(&rb.tail)
		head := atomic.LoadUint64(&rb.head)

		// Check if buffer is empty
		if tail >= head {
			return nil, false
		}

		idx := tail & rb.mask
		if atomic.CompareAndSwapUint64(&rb.tail, tail, tail+1) {
			ptr := atomic.LoadPointer(&rb.buffer[idx])
			if ptr == nil {
				continue
			}
			value := *(*interface{})(ptr)
			atomic.StorePointer(&rb.buffer[idx], nil)
			return value, true
		}
	}
}

// Len returns approximate buffer length
func (rb *LockFreeRingBuffer) Len() int {
	head := atomic.LoadUint64(&rb.head)
	tail := atomic.LoadUint64(&rb.tail)
	return int(head - tail)
}

// Cap returns buffer capacity
func (rb *LockFreeRingBuffer) Cap() int {
	return len(rb.buffer)
}

// ByteRingBuffer is optimized for byte slices
type ByteRingBuffer struct {
	buffer []byte
	mask   uint64
	head   uint64
	tail   uint64
}

// NewByteRingBuffer creates a byte-optimized ring buffer
func NewByteRingBuffer(size uint64) *ByteRingBuffer {
	if size&(size-1) != 0 {
		panic("size must be power of 2")
	}

	return &ByteRingBuffer{
		buffer: make([]byte, size),
		mask:   size - 1,
		head:   0,
		tail:   0,
	}
}

// Write writes bytes to ring buffer
func (brb *ByteRingBuffer) Write(data []byte) int {
	written := 0
	for written < len(data) {
		head := atomic.LoadUint64(&brb.head)
		tail := atomic.LoadUint64(&brb.tail)
		available := uint64(len(brb.buffer)) - (head - tail)

		if available == 0 {
			break // Buffer full
		}

		toWrite := len(data) - written
		if toWrite > int(available) {
			toWrite = int(available)
		}

		// Calculate write position and handle wrap-around
		pos := head & brb.mask
		firstChunk := int(uint64(len(brb.buffer)) - pos)
		if firstChunk > toWrite {
			firstChunk = toWrite
		}

		copy(brb.buffer[pos:], data[written:written+firstChunk])
		if firstChunk < toWrite {
			// Wrap around
			copy(brb.buffer, data[written+firstChunk:written+toWrite])
		}

		if atomic.CompareAndSwapUint64(&brb.head, head, head+uint64(toWrite)) {
			written += toWrite
		}
	}

	return written
}

// Read reads bytes from ring buffer
func (brb *ByteRingBuffer) Read(data []byte) int {
	read := 0
	for read < len(data) {
		tail := atomic.LoadUint64(&brb.tail)
		head := atomic.LoadUint64(&brb.head)
		available := head - tail

		if available == 0 {
			break // Buffer empty
		}

		toRead := len(data) - read
		if toRead > int(available) {
			toRead = int(available)
		}

		// Calculate read position and handle wrap-around
		pos := tail & brb.mask
		firstChunk := int(uint64(len(brb.buffer)) - pos)
		if firstChunk > toRead {
			firstChunk = toRead
		}

		copy(data[read:], brb.buffer[pos:pos+uint64(firstChunk)])
		if firstChunk < toRead {
			// Wrap around
			copy(data[read+firstChunk:], brb.buffer[:toRead-firstChunk])
		}

		if atomic.CompareAndSwapUint64(&brb.tail, tail, tail+uint64(toRead)) {
			read += toRead
		}
	}

	return read
}

// Available returns available space
func (brb *ByteRingBuffer) Available() int {
	head := atomic.LoadUint64(&brb.head)
	tail := atomic.LoadUint64(&brb.tail)
	return int(uint64(len(brb.buffer)) - (head - tail))
}

// Len returns bytes available to read
func (brb *ByteRingBuffer) Len() int {
	head := atomic.LoadUint64(&brb.head)
	tail := atomic.LoadUint64(&brb.tail)
	return int(head - tail)
}

// SPSC (Single Producer Single Consumer) ring buffer - fastest variant
type SPSCRingBuffer struct {
	buffer    []unsafe.Pointer
	mask      uint64
	_padding1 [56]byte
	head      uint64
	_padding2 [56]byte
	tail      uint64
	_padding3 [56]byte
}

// NewSPSCRingBuffer creates an SPSC ring buffer
func NewSPSCRingBuffer(size uint64) *SPSCRingBuffer {
	if size&(size-1) != 0 {
		panic("size must be power of 2")
	}

	return &SPSCRingBuffer{
		buffer: make([]unsafe.Pointer, size),
		mask:   size - 1,
		head:   0,
		tail:   0,
	}
}

// Push adds item (single producer only)
func (spsc *SPSCRingBuffer) Push(value interface{}) bool {
	head := atomic.LoadUint64(&spsc.head)
	tail := atomic.LoadUint64(&spsc.tail)

	if (head - tail) >= uint64(len(spsc.buffer)) {
		return false // Full
	}

	idx := head & spsc.mask
	atomic.StorePointer(&spsc.buffer[idx], unsafe.Pointer(&value))
	atomic.StoreUint64(&spsc.head, head+1)
	return true
}

// Pop removes item (single consumer only)
func (spsc *SPSCRingBuffer) Pop() (interface{}, bool) {
	tail := atomic.LoadUint64(&spsc.tail)
	head := atomic.LoadUint64(&spsc.head)

	if tail >= head {
		return nil, false // Empty
	}

	idx := tail & spsc.mask
	ptr := atomic.LoadPointer(&spsc.buffer[idx])
	value := *(*interface{})(ptr)
	atomic.StorePointer(&spsc.buffer[idx], nil)
	atomic.StoreUint64(&spsc.tail, tail+1)
	return value, true
}

// Len returns buffer length
func (spsc *SPSCRingBuffer) Len() int {
	head := atomic.LoadUint64(&spsc.head)
	tail := atomic.LoadUint64(&spsc.tail)
	return int(head - tail)
}

// BatchRingBuffer supports batch operations for better throughput
type BatchRingBuffer struct {
	buffer []unsafe.Pointer
	mask   uint64
	head   uint64
	tail   uint64
}

// NewBatchRingBuffer creates a batch-optimized ring buffer
func NewBatchRingBuffer(size uint64) *BatchRingBuffer {
	if size&(size-1) != 0 {
		panic("size must be power of 2")
	}

	return &BatchRingBuffer{
		buffer: make([]unsafe.Pointer, size),
		mask:   size - 1,
		head:   0,
		tail:   0,
	}
}

// PushBatch adds multiple items atomically
func (brb *BatchRingBuffer) PushBatch(values []interface{}) int {
	count := len(values)
	if count == 0 {
		return 0
	}

	for {
		head := atomic.LoadUint64(&brb.head)
		tail := atomic.LoadUint64(&brb.tail)
		available := uint64(len(brb.buffer)) - (head - tail)

		if available == 0 {
			return 0
		}

		toPush := count
		if uint64(toPush) > available {
			toPush = int(available)
		}

		if atomic.CompareAndSwapUint64(&brb.head, head, head+uint64(toPush)) {
			for i := 0; i < toPush; i++ {
				idx := (head + uint64(i)) & brb.mask
				atomic.StorePointer(&brb.buffer[idx], unsafe.Pointer(&values[i]))
			}
			return toPush
		}
	}
}

// PopBatch removes multiple items atomically
func (brb *BatchRingBuffer) PopBatch(maxCount int) []interface{} {
	if maxCount == 0 {
		return nil
	}

	for {
		tail := atomic.LoadUint64(&brb.tail)
		head := atomic.LoadUint64(&brb.head)
		available := head - tail

		if available == 0 {
			return nil
		}

		toPop := maxCount
		if uint64(toPop) > available {
			toPop = int(available)
		}

		if atomic.CompareAndSwapUint64(&brb.tail, tail, tail+uint64(toPop)) {
			result := make([]interface{}, toPop)
			for i := 0; i < toPop; i++ {
				idx := (tail + uint64(i)) & brb.mask
				ptr := atomic.LoadPointer(&brb.buffer[idx])
				result[i] = *(*interface{})(ptr)
				atomic.StorePointer(&brb.buffer[idx], nil)
			}
			return result
		}
	}
}
