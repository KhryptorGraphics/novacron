package bullshark

import (
	"sort"
	"sync"
)

// OrderingEngine provides deterministic transaction ordering
type OrderingEngine struct {
	dag *DAG
	mu  sync.RWMutex

	// Ordering cache
	orderedCache map[int][]Transaction
	cacheSize    int
}

// NewOrderingEngine creates a new transaction ordering engine
func NewOrderingEngine(dag *DAG, cacheSize int) *OrderingEngine {
	return &OrderingEngine{
		dag:          dag,
		orderedCache: make(map[int][]Transaction),
		cacheSize:    cacheSize,
	}
}

// OrderTransactions orders transactions from a set of vertices
func (oe *OrderingEngine) OrderTransactions(vertices []*Vertex) ([]Transaction, error) {
	oe.mu.Lock()
	defer oe.mu.Unlock()

	// Use topological sort for deterministic ordering
	sorted, err := oe.topologicalOrder(vertices)
	if err != nil {
		return nil, err
	}

	// Extract transactions in order
	ordered := make([]Transaction, 0)
	txSeen := make(map[string]bool)

	for _, v := range sorted {
		for _, tx := range v.Txs {
			// Deduplicate transactions
			if !txSeen[tx.ID] {
				ordered = append(ordered, tx)
				txSeen[tx.ID] = true
			}
		}
	}

	return ordered, nil
}

// topologicalOrder performs topological sort on vertices
func (oe *OrderingEngine) topologicalOrder(vertices []*Vertex) ([]*Vertex, error) {
	// Build dependency graph
	inDegree := make(map[string]int)
	adjacency := make(map[string][]*Vertex)
	vertexMap := make(map[string]*Vertex)

	for _, v := range vertices {
		vertexMap[v.ID] = v
		inDegree[v.ID] = 0
	}

	for _, v := range vertices {
		for _, parent := range v.Parents {
			if _, exists := vertexMap[parent.ID]; exists {
				adjacency[parent.ID] = append(adjacency[parent.ID], v)
				inDegree[v.ID]++
			}
		}
	}

	// Kahn's algorithm
	queue := make([]*Vertex, 0)
	for id, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, vertexMap[id])
		}
	}

	// Sort queue by timestamp for determinism
	sort.Slice(queue, func(i, j int) bool {
		return queue[i].Timestamp.Before(queue[j].Timestamp)
	})

	sorted := make([]*Vertex, 0, len(vertices))

	for len(queue) > 0 {
		v := queue[0]
		queue = queue[1:]
		sorted = append(sorted, v)

		for _, child := range adjacency[v.ID] {
			inDegree[child.ID]--
			if inDegree[child.ID] == 0 {
				queue = append(queue, child)
			}
		}

		// Re-sort queue for determinism
		sort.Slice(queue, func(i, j int) bool {
			return queue[i].Timestamp.Before(queue[j].Timestamp)
		})
	}

	return sorted, nil
}

// OrderByRound orders transactions from a specific round
func (oe *OrderingEngine) OrderByRound(round int) ([]Transaction, error) {
	oe.mu.RLock()

	// Check cache
	if cached, exists := oe.orderedCache[round]; exists {
		oe.mu.RUnlock()
		return cached, nil
	}
	oe.mu.RUnlock()

	// Get vertices for round
	vertices := oe.dag.GetVerticesByRound(round)

	// Order transactions
	ordered, err := oe.OrderTransactions(vertices)
	if err != nil {
		return nil, err
	}

	// Cache result
	oe.mu.Lock()
	oe.orderedCache[round] = ordered

	// Evict old cache entries
	if len(oe.orderedCache) > oe.cacheSize {
		oe.evictOldest()
	}
	oe.mu.Unlock()

	return ordered, nil
}

// evictOldest removes the oldest entry from cache
func (oe *OrderingEngine) evictOldest() {
	if len(oe.orderedCache) == 0 {
		return
	}

	minRound := int(^uint(0) >> 1) // Max int
	for round := range oe.orderedCache {
		if round < minRound {
			minRound = round
		}
	}

	delete(oe.orderedCache, minRound)
}

// OrderBatch orders a batch of transactions using parallel processing
func (oe *OrderingEngine) OrderBatch(batches [][]*Vertex) ([][]Transaction, error) {
	results := make([][]Transaction, len(batches))
	errors := make([]error, len(batches))

	var wg sync.WaitGroup
	for i, batch := range batches {
		wg.Add(1)
		go func(idx int, vertices []*Vertex) {
			defer wg.Done()
			ordered, err := oe.OrderTransactions(vertices)
			results[idx] = ordered
			errors[idx] = err
		}(i, batch)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}

	return results, nil
}

// PriorityOrdering orders transactions with priority consideration
type PriorityOrdering struct {
	engine   *OrderingEngine
	priority map[string]int
}

// NewPriorityOrdering creates a priority-aware ordering engine
func NewPriorityOrdering(engine *OrderingEngine) *PriorityOrdering {
	return &PriorityOrdering{
		engine:   engine,
		priority: make(map[string]int),
	}
}

// SetPriority sets priority for a transaction
func (po *PriorityOrdering) SetPriority(txID string, priority int) {
	po.priority[txID] = priority
}

// OrderWithPriority orders transactions considering priority
func (po *PriorityOrdering) OrderWithPriority(vertices []*Vertex) ([]Transaction, error) {
	// Get base ordering
	ordered, err := po.engine.OrderTransactions(vertices)
	if err != nil {
		return nil, err
	}

	// Sort by priority (higher priority first)
	sort.SliceStable(ordered, func(i, j int) bool {
		priI, existsI := po.priority[ordered[i].ID]
		priJ, existsJ := po.priority[ordered[j].ID]

		if !existsI && !existsJ {
			return false
		}
		if !existsI {
			return false
		}
		if !existsJ {
			return true
		}

		return priI > priJ
	})

	return ordered, nil
}

// FastOrdering provides optimized ordering for high throughput
type FastOrdering struct {
	batchSize int
	workers   int
}

// NewFastOrdering creates a fast ordering engine
func NewFastOrdering(batchSize, workers int) *FastOrdering {
	return &FastOrdering{
		batchSize: batchSize,
		workers:   workers,
	}
}

// OrderFast orders transactions with parallel processing
func (fo *FastOrdering) OrderFast(vertices []*Vertex) ([]Transaction, error) {
	// Partition vertices into batches
	batches := fo.partition(vertices)

	// Process batches in parallel
	results := make([][]Transaction, len(batches))
	var wg sync.WaitGroup
	semaphore := make(chan struct{}, fo.workers)

	for i, batch := range batches {
		wg.Add(1)
		go func(idx int, verts []*Vertex) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			// Extract transactions from batch
			txs := make([]Transaction, 0)
			for _, v := range verts {
				txs = append(txs, v.Txs...)
			}

			// Sort by timestamp
			sort.Slice(txs, func(i, j int) bool {
				return txs[i].Timestamp.Before(txs[j].Timestamp)
			})

			results[idx] = txs
		}(i, batch)
	}

	wg.Wait()

	// Merge results
	merged := make([]Transaction, 0)
	for _, result := range results {
		merged = append(merged, result...)
	}

	return merged, nil
}

// partition splits vertices into batches
func (fo *FastOrdering) partition(vertices []*Vertex) [][]*Vertex {
	if len(vertices) == 0 {
		return nil
	}

	batchCount := (len(vertices) + fo.batchSize - 1) / fo.batchSize
	batches := make([][]*Vertex, batchCount)

	for i := 0; i < len(vertices); i++ {
		batchIdx := i / fo.batchSize
		batches[batchIdx] = append(batches[batchIdx], vertices[i])
	}

	return batches
}
