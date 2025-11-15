package optimization

import (
	"sync"
	"sync/atomic"

	"github.com/khryptorgraphics/novacron/backend/core/network/dwcp/optimization/lockfree"
)

// ObjectPool provides size-class based memory pooling
type ObjectPool struct {
	pools     []sync.Pool
	sizeClass []int
	stats     poolStats
	enableGC  bool
}

type poolStats struct {
	allocations   uint64
	deallocations uint64
	hits          uint64
	misses        uint64
}

// NewObjectPool creates a new object pool with predefined size classes
func NewObjectPool() *ObjectPool {
	// Size classes optimized for network packets and DWCP messages
	sizeClasses := []int{
		64,      // Small control messages
		128,     // Headers
		256,     // Small data packets
		512,     // Medium packets
		1024,    // Standard MTU fragments
		2048,    // Larger fragments
		4096,    // Page-size buffers
		8192,    // Jumbo frame fragments
		16384,   // Large messages
		32768,   // Very large messages
		65536,   // Max UDP packet
		131072,  // 128KB buffers
		262144,  // 256KB buffers
		524288,  // 512KB buffers
		1048576, // 1MB buffers
	}

	pools := make([]sync.Pool, len(sizeClasses))
	for i, size := range sizeClasses {
		size := size // Capture for closure
		pools[i] = sync.Pool{
			New: func() interface{} {
				return make([]byte, size)
			},
		}
	}

	return &ObjectPool{
		pools:     pools,
		sizeClass: sizeClasses,
		enableGC:  true,
	}
}

// Get retrieves a buffer of at least the requested size
func (op *ObjectPool) Get(size int) []byte {
	atomic.AddUint64(&op.stats.allocations, 1)

	idx := op.findSizeClass(size)
	if idx < 0 {
		// Size too large for pooling
		atomic.AddUint64(&op.stats.misses, 1)
		return make([]byte, size)
	}

	atomic.AddUint64(&op.stats.hits, 1)
	buf := op.pools[idx].Get().([]byte)
	return buf[:size]
}

// Put returns a buffer to the pool
func (op *ObjectPool) Put(buf []byte) {
	if buf == nil {
		return
	}

	atomic.AddUint64(&op.stats.deallocations, 1)

	idx := op.findSizeClass(cap(buf))
	if idx >= 0 {
		// Reset buffer
		buf = buf[:cap(buf)]
		for i := range buf {
			buf[i] = 0
		}
		op.pools[idx].Put(buf)
	}
}

// findSizeClass returns the index of the smallest size class >= size
func (op *ObjectPool) findSizeClass(size int) int {
	// Binary search for size class
	left, right := 0, len(op.sizeClass)-1

	for left <= right {
		mid := (left + right) / 2
		if op.sizeClass[mid] == size {
			return mid
		} else if op.sizeClass[mid] < size {
			left = mid + 1
		} else {
			right = mid - 1
		}
	}

	if left < len(op.sizeClass) {
		return left
	}
	return -1
}

// Stats returns pool statistics
func (op *ObjectPool) Stats() PoolStats {
	return PoolStats{
		Allocations:   atomic.LoadUint64(&op.stats.allocations),
		Deallocations: atomic.LoadUint64(&op.stats.deallocations),
		Hits:          atomic.LoadUint64(&op.stats.hits),
		Misses:        atomic.LoadUint64(&op.stats.misses),
		HitRate:       op.hitRate(),
	}
}

func (op *ObjectPool) hitRate() float64 {
	hits := atomic.LoadUint64(&op.stats.hits)
	total := hits + atomic.LoadUint64(&op.stats.misses)
	if total == 0 {
		return 0
	}
	return float64(hits) / float64(total)
}

// PoolStats contains pool statistics
type PoolStats struct {
	Allocations   uint64
	Deallocations uint64
	Hits          uint64
	Misses        uint64
	HitRate       float64
}

// SlabAllocator provides slab-based allocation for fixed-size objects
type SlabAllocator struct {
	objectSize int
	slabs      []*slab
	freeLists  *LockFreeStack
	mu         sync.RWMutex
}

type slab struct {
	memory    []byte
	freeCount int
	objects   int
}

// NewSlabAllocator creates a slab allocator for fixed-size objects
func NewSlabAllocator(objectSize, objectsPerSlab int) *SlabAllocator {
	sa := &SlabAllocator{
		objectSize: objectSize,
		slabs:      make([]*slab, 0, 16),
		freeLists:  NewLockFreeStack(),
	}

	// Pre-allocate first slab
	sa.allocateSlab(objectsPerSlab)

	return sa
}

func (sa *SlabAllocator) allocateSlab(objects int) *slab {
	s := &slab{
		memory:    make([]byte, objects*sa.objectSize),
		freeCount: objects,
		objects:   objects,
	}

	// Add all objects to free list
	for i := 0; i < objects; i++ {
		offset := i * sa.objectSize
		sa.freeLists.Push(&s.memory[offset])
	}

	sa.mu.Lock()
	sa.slabs = append(sa.slabs, s)
	sa.mu.Unlock()

	return s
}

// Allocate gets an object from the slab
func (sa *SlabAllocator) Allocate() []byte {
	if obj, ok := sa.freeLists.Pop(); ok {
		return (*obj.(*[]byte))[:sa.objectSize]
	}

	// No free objects, allocate new slab
	sa.allocateSlab(1024)
	if obj, ok := sa.freeLists.Pop(); ok {
		return (*obj.(*[]byte))[:sa.objectSize]
	}

	return nil
}

// Free returns an object to the slab
func (sa *SlabAllocator) Free(obj []byte) {
	if len(obj) != sa.objectSize {
		return
	}

	// Clear memory
	for i := range obj {
		obj[i] = 0
	}

	sa.freeLists.Push(&obj)
}

// TLSPool provides thread-local storage pool for reduced contention
type TLSPool struct {
	local sync.Pool
	size  int
}

// NewTLSPool creates a thread-local storage pool
func NewTLSPool(size int) *TLSPool {
	return &TLSPool{
		local: sync.Pool{
			New: func() interface{} {
				return make([]byte, size)
			},
		},
		size: size,
	}
}

// Get retrieves a buffer from TLS pool
func (tp *TLSPool) Get() []byte {
	return tp.local.Get().([]byte)
}

// Put returns a buffer to TLS pool
func (tp *TLSPool) Put(buf []byte) {
	if cap(buf) == tp.size {
		tp.local.Put(buf[:cap(buf)])
	}
}

// HugePageAllocator allocates memory using huge pages for better TLB performance
type HugePageAllocator struct {
	pageSize int
	pools    map[int]*sync.Pool
	mu       sync.RWMutex
}

// NewHugePageAllocator creates an allocator using huge pages
func NewHugePageAllocator() *HugePageAllocator {
	return &HugePageAllocator{
		pageSize: 2 * 1024 * 1024, // 2MB huge pages
		pools:    make(map[int]*sync.Pool),
	}
}

// Allocate gets memory from huge page pool
func (hpa *HugePageAllocator) Allocate(size int) []byte {
	// Round up to page size
	pages := (size + hpa.pageSize - 1) / hpa.pageSize
	allocSize := pages * hpa.pageSize

	hpa.mu.RLock()
	pool, exists := hpa.pools[allocSize]
	hpa.mu.RUnlock()

	if !exists {
		hpa.mu.Lock()
		pool = &sync.Pool{
			New: func() interface{} {
				// Try to allocate with huge pages
				buf, err := NewZeroCopyBuffer(allocSize)
				if err != nil {
					return make([]byte, allocSize)
				}
				return buf.Bytes()
			},
		}
		hpa.pools[allocSize] = pool
		hpa.mu.Unlock()
	}

	return pool.Get().([]byte)[:size]
}

// Free returns memory to huge page pool
func (hpa *HugePageAllocator) Free(buf []byte) {
	if buf == nil {
		return
	}

	allocSize := cap(buf)
	hpa.mu.RLock()
	pool, exists := hpa.pools[allocSize]
	hpa.mu.RUnlock()

	if exists {
		pool.Put(buf[:cap(buf)])
	}
}

// LockFreeStack is imported from lockfree package
type LockFreeStack = lockfree.LockFreeStack

// NewLockFreeStack creates a new lock-free stack
func NewLockFreeStack() *LockFreeStack {
	return lockfree.NewLockFreeStack()
}
