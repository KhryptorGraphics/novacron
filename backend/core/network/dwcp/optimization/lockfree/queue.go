package lockfree

import (
	"sync/atomic"
	"unsafe"
)

// LockFreeQueue implements a Michael-Scott lock-free queue
type LockFreeQueue struct {
	head   unsafe.Pointer
	tail   unsafe.Pointer
	length int64
}

// node represents a queue node
type node struct {
	value interface{}
	next  unsafe.Pointer
}

// NewLockFreeQueue creates a new lock-free queue
func NewLockFreeQueue() *LockFreeQueue {
	n := &node{}
	ptr := unsafe.Pointer(n)
	return &LockFreeQueue{
		head:   ptr,
		tail:   ptr,
		length: 0,
	}
}

// Enqueue adds an item to the queue
func (q *LockFreeQueue) Enqueue(value interface{}) {
	n := &node{value: value}
	nPtr := unsafe.Pointer(n)

	for {
		tail := atomic.LoadPointer(&q.tail)
		next := atomic.LoadPointer(&(*node)(tail).next)

		// Check if tail is still consistent
		if tail == atomic.LoadPointer(&q.tail) {
			if next == nil {
				// Try to link new node
				if atomic.CompareAndSwapPointer(&(*node)(tail).next, next, nPtr) {
					// Enqueue succeeded, try to swing tail
					atomic.CompareAndSwapPointer(&q.tail, tail, nPtr)
					atomic.AddInt64(&q.length, 1)
					return
				}
			} else {
				// Tail was not pointing to last node, help advance it
				atomic.CompareAndSwapPointer(&q.tail, tail, next)
			}
		}
	}
}

// Dequeue removes and returns an item from the queue
func (q *LockFreeQueue) Dequeue() (interface{}, bool) {
	for {
		head := atomic.LoadPointer(&q.head)
		tail := atomic.LoadPointer(&q.tail)
		next := atomic.LoadPointer(&(*node)(head).next)

		// Check if head is still consistent
		if head == atomic.LoadPointer(&q.head) {
			if head == tail {
				// Queue is empty or tail is falling behind
				if next == nil {
					return nil, false
				}
				// Tail is falling behind, help advance it
				atomic.CompareAndSwapPointer(&q.tail, tail, next)
			} else {
				// Read value before CAS to avoid race
				value := (*node)(next).value
				// Try to swing head to next node
				if atomic.CompareAndSwapPointer(&q.head, head, next) {
					atomic.AddInt64(&q.length, -1)
					return value, true
				}
			}
		}
	}
}

// Len returns the approximate queue length
func (q *LockFreeQueue) Len() int {
	return int(atomic.LoadInt64(&q.length))
}

// IsEmpty checks if queue is empty
func (q *LockFreeQueue) IsEmpty() bool {
	head := atomic.LoadPointer(&q.head)
	next := atomic.LoadPointer(&(*node)(head).next)
	return next == nil
}

// Peek returns the front item without removing it
func (q *LockFreeQueue) Peek() (interface{}, bool) {
	for {
		head := atomic.LoadPointer(&q.head)
		tail := atomic.LoadPointer(&q.tail)
		next := atomic.LoadPointer(&(*node)(head).next)

		if head == atomic.LoadPointer(&q.head) {
			if head == tail {
				if next == nil {
					return nil, false
				}
				atomic.CompareAndSwapPointer(&q.tail, tail, next)
			} else {
				return (*node)(next).value, true
			}
		}
	}
}

// LockFreeBoundedQueue implements a bounded lock-free queue using ring buffer
type LockFreeBoundedQueue struct {
	buffer []unsafe.Pointer
	mask   uint64
	head   uint64
	tail   uint64
}

// NewLockFreeBoundedQueue creates a bounded queue (size must be power of 2)
func NewLockFreeBoundedQueue(size uint64) *LockFreeBoundedQueue {
	if size&(size-1) != 0 {
		panic("size must be power of 2")
	}

	return &LockFreeBoundedQueue{
		buffer: make([]unsafe.Pointer, size),
		mask:   size - 1,
		head:   0,
		tail:   0,
	}
}

// Enqueue adds item to bounded queue
func (q *LockFreeBoundedQueue) Enqueue(value interface{}) bool {
	for {
		head := atomic.LoadUint64(&q.head)
		tail := atomic.LoadUint64(&q.tail)

		// Check if queue is full
		if head-tail >= uint64(len(q.buffer)) {
			return false
		}

		idx := head & q.mask
		if atomic.CompareAndSwapUint64(&q.head, head, head+1) {
			// Store value at index
			atomic.StorePointer(&q.buffer[idx], unsafe.Pointer(&value))
			return true
		}
	}
}

// Dequeue removes item from bounded queue
func (q *LockFreeBoundedQueue) Dequeue() (interface{}, bool) {
	for {
		tail := atomic.LoadUint64(&q.tail)
		head := atomic.LoadUint64(&q.head)

		// Check if queue is empty
		if tail >= head {
			return nil, false
		}

		idx := tail & q.mask
		if atomic.CompareAndSwapUint64(&q.tail, tail, tail+1) {
			// Load value from index
			ptr := atomic.LoadPointer(&q.buffer[idx])
			if ptr == nil {
				continue
			}
			value := *(*interface{})(ptr)
			// Clear the slot
			atomic.StorePointer(&q.buffer[idx], nil)
			return value, true
		}
	}
}

// Len returns approximate queue length
func (q *LockFreeBoundedQueue) Len() int {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)
	return int(head - tail)
}

// Cap returns queue capacity
func (q *LockFreeBoundedQueue) Cap() int {
	return len(q.buffer)
}

// IsFull checks if queue is full
func (q *LockFreeBoundedQueue) IsFull() bool {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)
	return head-tail >= uint64(len(q.buffer))
}

// IsEmpty checks if queue is empty
func (q *LockFreeBoundedQueue) IsEmpty() bool {
	head := atomic.LoadUint64(&q.head)
	tail := atomic.LoadUint64(&q.tail)
	return tail >= head
}

// Multi-producer, multi-consumer queue optimized for high throughput
type MPMCQueue struct {
	buffer    []atomicSlot
	mask      uint64
	headSeq   uint64
	tailSeq   uint64
	_padding1 [56]byte // Cache line padding
	enqueueSeq uint64
	_padding2 [56]byte
	dequeueSeq uint64
	_padding3 [56]byte
}

type atomicSlot struct {
	sequence uint64
	value    unsafe.Pointer
}

// NewMPMCQueue creates a multi-producer multi-consumer queue
func NewMPMCQueue(size uint64) *MPMCQueue {
	if size&(size-1) != 0 {
		panic("size must be power of 2")
	}

	buffer := make([]atomicSlot, size)
	for i := range buffer {
		atomic.StoreUint64(&buffer[i].sequence, uint64(i))
	}

	return &MPMCQueue{
		buffer:     buffer,
		mask:       size - 1,
		headSeq:    0,
		tailSeq:    0,
		enqueueSeq: 0,
		dequeueSeq: 0,
	}
}

// Enqueue adds item using MPMC protocol
func (q *MPMCQueue) Enqueue(value interface{}) bool {
	for {
		pos := atomic.LoadUint64(&q.enqueueSeq)
		idx := pos & q.mask
		slot := &q.buffer[idx]
		seq := atomic.LoadUint64(&slot.sequence)
		diff := int64(seq) - int64(pos)

		if diff == 0 {
			if atomic.CompareAndSwapUint64(&q.enqueueSeq, pos, pos+1) {
				atomic.StorePointer(&slot.value, unsafe.Pointer(&value))
				atomic.StoreUint64(&slot.sequence, pos+1)
				return true
			}
		} else if diff < 0 {
			return false // Queue full
		}
	}
}

// Dequeue removes item using MPMC protocol
func (q *MPMCQueue) Dequeue() (interface{}, bool) {
	for {
		pos := atomic.LoadUint64(&q.dequeueSeq)
		idx := pos & q.mask
		slot := &q.buffer[idx]
		seq := atomic.LoadUint64(&slot.sequence)
		diff := int64(seq) - int64(pos+1)

		if diff == 0 {
			if atomic.CompareAndSwapUint64(&q.dequeueSeq, pos, pos+1) {
				ptr := atomic.LoadPointer(&slot.value)
				value := *(*interface{})(ptr)
				atomic.StoreUint64(&slot.sequence, pos+q.mask+1)
				return value, true
			}
		} else if diff < 0 {
			return nil, false // Queue empty
		}
	}
}
