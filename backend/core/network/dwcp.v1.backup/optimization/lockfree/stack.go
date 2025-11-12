package lockfree

import (
	"sync/atomic"
	"unsafe"
)

// LockFreeStack implements Treiber's lock-free stack
type LockFreeStack struct {
	head   unsafe.Pointer
	length int64
}

// stackNode represents a stack node
type stackNode struct {
	value interface{}
	next  unsafe.Pointer
}

// NewLockFreeStack creates a new lock-free stack
func NewLockFreeStack() *LockFreeStack {
	return &LockFreeStack{
		head:   nil,
		length: 0,
	}
}

// Push adds an item to the stack
func (s *LockFreeStack) Push(value interface{}) {
	n := &stackNode{value: value}

	for {
		oldHead := atomic.LoadPointer(&s.head)
		n.next = oldHead

		if atomic.CompareAndSwapPointer(&s.head, oldHead, unsafe.Pointer(n)) {
			atomic.AddInt64(&s.length, 1)
			return
		}
	}
}

// Pop removes and returns the top item
func (s *LockFreeStack) Pop() (interface{}, bool) {
	for {
		oldHead := atomic.LoadPointer(&s.head)
		if oldHead == nil {
			return nil, false
		}

		n := (*stackNode)(oldHead)
		newHead := atomic.LoadPointer(&n.next)

		if atomic.CompareAndSwapPointer(&s.head, oldHead, newHead) {
			atomic.AddInt64(&s.length, -1)
			return n.value, true
		}
	}
}

// Peek returns the top item without removing it
func (s *LockFreeStack) Peek() (interface{}, bool) {
	head := atomic.LoadPointer(&s.head)
	if head == nil {
		return nil, false
	}
	return (*stackNode)(head).value, true
}

// Len returns approximate stack length
func (s *LockFreeStack) Len() int {
	return int(atomic.LoadInt64(&s.length))
}

// IsEmpty checks if stack is empty
func (s *LockFreeStack) IsEmpty() bool {
	return atomic.LoadPointer(&s.head) == nil
}

// Clear removes all items
func (s *LockFreeStack) Clear() {
	atomic.StorePointer(&s.head, nil)
	atomic.StoreInt64(&s.length, 0)
}
