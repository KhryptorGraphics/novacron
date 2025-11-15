package optimization

import (
	"sync/atomic"
	"unsafe"
)

//go:linkname prefetch runtime.prefetch
func prefetch(addr unsafe.Pointer)

//go:linkname prefetchw runtime.prefetchw
func prefetchw(addr unsafe.Pointer)

// PrefetchMode defines prefetch strategy
type PrefetchMode int

const (
	PrefetchRead PrefetchMode = iota
	PrefetchWrite
	PrefetchReadWrite
)

// Prefetcher provides cache prefetching operations
type Prefetcher struct {
	distance int // Prefetch distance in bytes
	mode     PrefetchMode
}

// NewPrefetcher creates a new prefetcher
func NewPrefetcher(distance int, mode PrefetchMode) *Prefetcher {
	return &Prefetcher{
		distance: distance,
		mode:     mode,
	}
}

// PrefetchSlice prefetches a byte slice into cache
func (p *Prefetcher) PrefetchSlice(data []byte) {
	if len(data) == 0 {
		return
	}

	// Prefetch in cache line chunks (64 bytes)
	cacheLineSize := 64
	for i := 0; i < len(data); i += cacheLineSize {
		if p.mode == PrefetchWrite {
			prefetchw(unsafe.Pointer(&data[i]))
		} else {
			prefetch(unsafe.Pointer(&data[i]))
		}
	}
}

// PrefetchAhead prefetches data ahead of current position
func (p *Prefetcher) PrefetchAhead(data []byte, pos int) {
	prefetchPos := pos + p.distance
	if prefetchPos < len(data) {
		if p.mode == PrefetchWrite {
			prefetchw(unsafe.Pointer(&data[prefetchPos]))
		} else {
			prefetch(unsafe.Pointer(&data[prefetchPos]))
		}
	}
}

// StreamingPrefetch prefetches for streaming access pattern
func StreamingPrefetch(data []byte, chunkSize int, process func([]byte)) {
	for i := 0; i < len(data); i += chunkSize {
		// Prefetch next chunk
		nextChunk := i + chunkSize
		if nextChunk < len(data) {
			prefetch(unsafe.Pointer(&data[nextChunk]))
		}

		// Process current chunk
		end := i + chunkSize
		if end > len(data) {
			end = len(data)
		}
		process(data[i:end])
	}
}

// CacheAligned represents a cache-line aligned value
type CacheAligned struct {
	_   [64]byte // Padding before
	val interface{}
	_   [64]byte // Padding after
}

// NewCacheAligned creates a cache-aligned value
func NewCacheAligned(val interface{}) *CacheAligned {
	return &CacheAligned{val: val}
}

// Get returns the value
func (ca *CacheAligned) Get() interface{} {
	return ca.val
}

// Set sets the value
func (ca *CacheAligned) Set(val interface{}) {
	ca.val = val
}

// CacheLinePadding provides padding to prevent false sharing
type CacheLinePadding [64]byte

// PaddedInt64 is a cache-line padded int64
type PaddedInt64 struct {
	_   CacheLinePadding
	val int64
	_   CacheLinePadding
}

// NewPaddedInt64 creates a padded int64
func NewPaddedInt64(val int64) *PaddedInt64 {
	return &PaddedInt64{val: val}
}

// Get returns the value
func (p *PaddedInt64) Get() int64 {
	return atomic.LoadInt64(&p.val)
}

// Set sets the value
func (p *PaddedInt64) Set(val int64) {
	atomic.StoreInt64(&p.val, val)
}

// Add atomically adds delta
func (p *PaddedInt64) Add(delta int64) int64 {
	return atomic.AddInt64(&p.val, delta)
}

// PaddedUint64 is a cache-line padded uint64
type PaddedUint64 struct {
	_   CacheLinePadding
	val uint64
	_   CacheLinePadding
}

// NewPaddedUint64 creates a padded uint64
func NewPaddedUint64(val uint64) *PaddedUint64 {
	return &PaddedUint64{val: val}
}

// Get returns the value
func (p *PaddedUint64) Get() uint64 {
	return atomic.LoadUint64(&p.val)
}

// Set sets the value
func (p *PaddedUint64) Set(val uint64) {
	atomic.StoreUint64(&p.val, val)
}

// Add atomically adds delta
func (p *PaddedUint64) Add(delta uint64) uint64 {
	return atomic.AddUint64(&p.val, delta)
}

// StrideAccess optimizes stride-based memory access
type StrideAccess struct {
	data     []byte
	stride   int
	prefetch *Prefetcher
}

// NewStrideAccess creates a stride access optimizer
func NewStrideAccess(data []byte, stride int) *StrideAccess {
	return &StrideAccess{
		data:     data,
		stride:   stride,
		prefetch: NewPrefetcher(stride*4, PrefetchRead),
	}
}

// Access accesses data with stride prefetching
func (sa *StrideAccess) Access(index int, fn func([]byte)) {
	pos := index * sa.stride
	if pos >= len(sa.data) {
		return
	}

	// Prefetch ahead
	sa.prefetch.PrefetchAhead(sa.data, pos)

	// Access current data
	end := pos + sa.stride
	if end > len(sa.data) {
		end = len(sa.data)
	}
	fn(sa.data[pos:end])
}

// TemporalLocality hints for cache management
type TemporalLocality int

const (
	TemporalNone   TemporalLocality = iota // No temporal locality
	TemporalLow                            // Low temporal locality
	TemporalMedium                         // Medium temporal locality
	TemporalHigh                           // High temporal locality
)

// NonTemporalLoad performs non-temporal load (bypass cache)
func NonTemporalLoad(addr unsafe.Pointer) uint64 {
	// This would use MOVNTDQA instruction on x86
	// For now, fallback to regular load
	return *(*uint64)(addr)
}

// NonTemporalStore performs non-temporal store (bypass cache)
func NonTemporalStore(addr unsafe.Pointer, val uint64) {
	// This would use MOVNTI instruction on x86
	// For now, fallback to regular store
	*(*uint64)(addr) = val
}

// StreamLoad loads data with streaming hint
func StreamLoad(data []byte) {
	for i := 0; i < len(data); i += 8 {
		if i+8 <= len(data) {
			addr := unsafe.Pointer(&data[i])
			_ = NonTemporalLoad(addr)
		}
	}
}

// StreamStore stores data with streaming hint
func StreamStore(data []byte, val byte) {
	for i := 0; i < len(data); i += 8 {
		if i+8 <= len(data) {
			addr := unsafe.Pointer(&data[i])
			NonTemporalStore(addr, uint64(val))
		}
	}
}

// CopyWithPrefetch copies data with prefetching
func CopyWithPrefetch(dst, src []byte, chunkSize int) int {
	if len(dst) < len(src) {
		return 0
	}

	copied := 0
	for i := 0; i < len(src); i += chunkSize {
		// Prefetch next chunk
		nextPos := i + chunkSize
		if nextPos < len(src) {
			prefetch(unsafe.Pointer(&src[nextPos]))
			prefetchw(unsafe.Pointer(&dst[nextPos]))
		}

		// Copy current chunk
		end := i + chunkSize
		if end > len(src) {
			end = len(src)
		}
		n := copy(dst[i:], src[i:end])
		copied += n
	}

	return copied
}

// SoftwarePrefetch performs software prefetching
type SoftwarePrefetch struct {
	lookahead int
	lineSize  int
}

// NewSoftwarePrefetch creates a software prefetcher
func NewSoftwarePrefetch(lookahead, lineSize int) *SoftwarePrefetch {
	return &SoftwarePrefetch{
		lookahead: lookahead,
		lineSize:  lineSize,
	}
}

// PrefetchRange prefetches a range of addresses
func (sp *SoftwarePrefetch) PrefetchRange(base unsafe.Pointer, offset, length int) {
	lines := (length + sp.lineSize - 1) / sp.lineSize
	for i := 0; i < lines; i++ {
		addr := unsafe.Pointer(uintptr(base) + uintptr(offset+i*sp.lineSize))
		prefetch(addr)
	}
}

// HugePageOptimizer optimizes for huge page usage
type HugePageOptimizer struct {
	pageSize int
}

// NewHugePageOptimizer creates a huge page optimizer
func NewHugePageOptimizer(pageSize int) *HugePageOptimizer {
	return &HugePageOptimizer{
		pageSize: pageSize,
	}
}

// AlignToHugePage aligns size to huge page boundary
func (hpo *HugePageOptimizer) AlignToHugePage(size int) int {
	return ((size + hpo.pageSize - 1) / hpo.pageSize) * hpo.pageSize
}

// IsAligned checks if address is huge page aligned
func (hpo *HugePageOptimizer) IsAligned(addr uintptr) bool {
	return addr&uintptr(hpo.pageSize-1) == 0
}
