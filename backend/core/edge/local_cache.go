package edge

import (
	"container/list"
	"sync"
	"time"
)

// CacheEntry represents a cached item
type CacheEntry struct {
	Key       string
	Value     interface{}
	CreatedAt time.Time
	AccessedAt time.Time
	ExpiresAt time.Time
	Size      int64
	Hits      int64
}

// LocalCache implements an LRU cache for edge agents
type LocalCache struct {
	maxSize   int64
	currentSize int64
	items     map[string]*list.Element
	lruList   *list.List
	mutex     sync.RWMutex
	
	// Statistics
	hits   int64
	misses int64
	evictions int64
}

// NewLocalCache creates a new local cache
func NewLocalCache(maxSize int64) *LocalCache {
	return &LocalCache{
		maxSize: maxSize,
		items:   make(map[string]*list.Element),
		lruList: list.New(),
	}
}

// Start starts the cache (no-op for local cache)
func (c *LocalCache) Start() error {
	return nil
}

// Stop stops the cache
func (c *LocalCache) Stop() error {
	c.Clear()
	return nil
}

// Get retrieves an item from cache
func (c *LocalCache) Get(key string) (interface{}, bool) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	element, exists := c.items[key]
	if !exists {
		c.misses++
		return nil, false
	}
	
	entry := element.Value.(*CacheEntry)
	
	// Check expiration
	if !entry.ExpiresAt.IsZero() && time.Now().After(entry.ExpiresAt) {
		c.removeElement(element)
		c.misses++
		return nil, false
	}
	
	// Update access time and move to front
	entry.AccessedAt = time.Now()
	entry.Hits++
	c.lruList.MoveToFront(element)
	c.hits++
	
	return entry.Value, true
}

// Set stores an item in cache
func (c *LocalCache) Set(key string, value interface{}, ttl time.Duration) error {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	now := time.Now()
	var expiresAt time.Time
	if ttl > 0 {
		expiresAt = now.Add(ttl)
	}
	
	// Calculate size (simplified)
	size := c.calculateSize(value)
	
	// Check if item already exists
	if element, exists := c.items[key]; exists {
		entry := element.Value.(*CacheEntry)
		oldSize := entry.Size
		
		// Update existing entry
		entry.Value = value
		entry.AccessedAt = now
		entry.ExpiresAt = expiresAt
		entry.Size = size
		
		c.currentSize = c.currentSize - oldSize + size
		c.lruList.MoveToFront(element)
	} else {
		// Create new entry
		entry := &CacheEntry{
			Key:        key,
			Value:      value,
			CreatedAt:  now,
			AccessedAt: now,
			ExpiresAt:  expiresAt,
			Size:       size,
			Hits:       0,
		}
		
		element := c.lruList.PushFront(entry)
		c.items[key] = element
		c.currentSize += size
	}
	
	// Ensure we don't exceed max size
	c.evictIfNeeded()
	
	return nil
}

// Delete removes an item from cache
func (c *LocalCache) Delete(key string) bool {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	element, exists := c.items[key]
	if !exists {
		return false
	}
	
	c.removeElement(element)
	return true
}

// Clear removes all items from cache
func (c *LocalCache) Clear() {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	
	c.items = make(map[string]*list.Element)
	c.lruList = list.New()
	c.currentSize = 0
}

// Stats returns cache statistics
func (c *LocalCache) Stats() map[string]interface{} {
	c.mutex.RLock()
	defer c.mutex.RUnlock()
	
	hitRate := float64(0)
	if c.hits+c.misses > 0 {
		hitRate = float64(c.hits) / float64(c.hits+c.misses) * 100
	}
	
	return map[string]interface{}{
		"hits":       c.hits,
		"misses":     c.misses,
		"hit_rate":   hitRate,
		"evictions":  c.evictions,
		"items":      len(c.items),
		"size_bytes": c.currentSize,
		"max_size":   c.maxSize,
		"utilization": float64(c.currentSize) / float64(c.maxSize) * 100,
	}
}

// evictIfNeeded removes items if cache is over capacity
func (c *LocalCache) evictIfNeeded() {
	for c.currentSize > c.maxSize && c.lruList.Len() > 0 {
		// Remove least recently used item
		element := c.lruList.Back()
		if element != nil {
			c.removeElement(element)
			c.evictions++
		}
	}
}

// removeElement removes an element from both map and list
func (c *LocalCache) removeElement(element *list.Element) {
	entry := element.Value.(*CacheEntry)
	delete(c.items, entry.Key)
	c.lruList.Remove(element)
	c.currentSize -= entry.Size
}

// calculateSize estimates the size of a value
func (c *LocalCache) calculateSize(value interface{}) int64 {
	// Simplified size calculation
	switch v := value.(type) {
	case string:
		return int64(len(v))
	case []byte:
		return int64(len(v))
	default:
		return 64 // Default size for other types
	}
}

// MetricsCache specializes the cache for metrics storage
type MetricsCache struct {
	*LocalCache
	maxEntries int
}

// NewMetricsCache creates a metrics-specific cache
func NewMetricsCache(maxSize int64) *MetricsCache {
	return &MetricsCache{
		LocalCache: NewLocalCache(maxSize),
		maxEntries: 10000, // Maximum number of metric entries
	}
}

// Add adds a metric snapshot to cache
func (mc *MetricsCache) Add(snapshot *MetricSnapshot) error {
	key := snapshot.AgentID + "_" + snapshot.Timestamp.Format(time.RFC3339)
	ttl := 24 * time.Hour // Keep metrics for 24 hours
	
	return mc.Set(key, snapshot, ttl)
}

// GetRecent retrieves recent metrics for an agent
func (mc *MetricsCache) GetRecent(agentID string, duration time.Duration) []*MetricSnapshot {
	mc.mutex.RLock()
	defer mc.mutex.RUnlock()
	
	var snapshots []*MetricSnapshot
	cutoff := time.Now().Add(-duration)
	
	// Iterate through cache items
	for _, element := range mc.items {
		entry := element.Value.(*CacheEntry)
		if snapshot, ok := entry.Value.(*MetricSnapshot); ok {
			if snapshot.AgentID == agentID && snapshot.Timestamp.After(cutoff) {
				snapshots = append(snapshots, snapshot)
			}
		}
	}
	
	return snapshots
}

// EdgeTaskManager manages task execution on edge nodes
type EdgeTaskManager struct {
	maxTasks    int
	activeTasks map[string]*EdgeTask
	taskQueue   []*EdgeTask
	mutex       sync.RWMutex
	
	// Execution control
	acceptingTasks bool
	stopChan       chan struct{}
	workers        int
}

// NewEdgeTaskManager creates a new task manager
func NewEdgeTaskManager(maxTasks int) *EdgeTaskManager {
	return &EdgeTaskManager{
		maxTasks:       maxTasks,
		activeTasks:    make(map[string]*EdgeTask),
		taskQueue:      make([]*EdgeTask, 0),
		acceptingTasks: true,
		stopChan:       make(chan struct{}),
		workers:        4, // Default number of worker goroutines
	}
}

// Start starts the task manager
func (tm *EdgeTaskManager) Start() error {
	// Start worker goroutines
	for i := 0; i < tm.workers; i++ {
		go tm.worker()
	}
	return nil
}

// Stop stops the task manager
func (tm *EdgeTaskManager) Stop() error {
	close(tm.stopChan)
	return nil
}

// CanHandle checks if the manager can handle a task type
func (tm *EdgeTaskManager) CanHandle(taskType string) bool {
	supportedTypes := []string{
		"compute",
		"analytics",
		"cache_update",
		"metric_aggregation",
		"data_processing",
		"ml_inference",
	}
	
	for _, supportedType := range supportedTypes {
		if taskType == supportedType {
			return true
		}
	}
	return false
}

// ExecuteTask executes a single task
func (tm *EdgeTaskManager) ExecuteTask(task *EdgeTask) (map[string]interface{}, error) {
	switch task.Type {
	case "compute":
		return tm.executeComputeTask(task)
	case "analytics":
		return tm.executeAnalyticsTask(task)
	case "cache_update":
		return tm.executeCacheUpdateTask(task)
	case "metric_aggregation":
		return tm.executeMetricAggregationTask(task)
	case "data_processing":
		return tm.executeDataProcessingTask(task)
	case "ml_inference":
		return tm.executeMLInferenceTask(task)
	default:
		return nil, fmt.Errorf("unsupported task type: %s", task.Type)
	}
}

// GetQueueLength returns the current queue length
func (tm *EdgeTaskManager) GetQueueLength() int {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	return len(tm.taskQueue)
}

// GetEstimatedWaitTime estimates wait time for a task
func (tm *EdgeTaskManager) GetEstimatedWaitTime(task *EdgeTask) time.Duration {
	tm.mutex.RLock()
	defer tm.mutex.RUnlock()
	
	// Simple estimation based on queue length and average task time
	queueLength := len(tm.taskQueue)
	averageTaskTime := 10 * time.Second // Estimated average
	
	return time.Duration(queueLength) * averageTaskTime
}

// StopAcceptingTasks stops accepting new tasks
func (tm *EdgeTaskManager) StopAcceptingTasks() {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	tm.acceptingTasks = false
}

// WaitForCompletion waits for all active tasks to complete
func (tm *EdgeTaskManager) WaitForCompletion() {
	for {
		tm.mutex.RLock()
		activeCount := len(tm.activeTasks)
		tm.mutex.RUnlock()
		
		if activeCount == 0 {
			break
		}
		
		time.Sleep(100 * time.Millisecond)
	}
}

// ForceStop forcefully stops all tasks
func (tm *EdgeTaskManager) ForceStop() {
	tm.mutex.Lock()
	defer tm.mutex.Unlock()
	
	// Mark all active tasks as cancelled
	for _, task := range tm.activeTasks {
		task.Status = TaskStatusCancelled
	}
	
	// Clear queue
	tm.taskQueue = tm.taskQueue[:0]
}

// Task execution implementations

func (tm *EdgeTaskManager) executeComputeTask(task *EdgeTask) (map[string]interface{}, error) {
	// Simulate compute task execution
	time.Sleep(time.Duration(100+len(task.ID)) * time.Millisecond)
	
	return map[string]interface{}{
		"result":           "compute_completed",
		"execution_time_ms": 100 + len(task.ID),
		"cpu_usage":        0.75,
		"memory_usage":     task.ResourceReqs.MemoryMB,
	}, nil
}

func (tm *EdgeTaskManager) executeAnalyticsTask(task *EdgeTask) (map[string]interface{}, error) {
	// Simulate analytics processing
	time.Sleep(200 * time.Millisecond)
	
	return map[string]interface{}{
		"result":        "analytics_completed",
		"records_processed": 1000,
		"anomalies_detected": 3,
		"processing_rate":   5000.0, // records per second
	}, nil
}

func (tm *EdgeTaskManager) executeCacheUpdateTask(task *EdgeTask) (map[string]interface{}, error) {
	// Simulate cache update
	time.Sleep(50 * time.Millisecond)
	
	return map[string]interface{}{
		"result":          "cache_updated",
		"entries_updated": 100,
		"cache_hit_rate":  0.85,
	}, nil
}

func (tm *EdgeTaskManager) executeMetricAggregationTask(task *EdgeTask) (map[string]interface{}, error) {
	// Simulate metric aggregation
	time.Sleep(150 * time.Millisecond)
	
	return map[string]interface{}{
		"result":            "metrics_aggregated",
		"metrics_processed": 500,
		"aggregation_period": "5m",
		"data_points":       2500,
	}, nil
}

func (tm *EdgeTaskManager) executeDataProcessingTask(task *EdgeTask) (map[string]interface{}, error) {
	// Simulate data processing
	time.Sleep(300 * time.Millisecond)
	
	return map[string]interface{}{
		"result":         "data_processed",
		"bytes_processed": 1024 * 1024, // 1MB
		"transformation": "json_to_parquet",
		"compression_ratio": 0.65,
	}, nil
}

func (tm *EdgeTaskManager) executeMLInferenceTask(task *EdgeTask) (map[string]interface{}, error) {
	// Simulate ML inference
	time.Sleep(80 * time.Millisecond)
	
	return map[string]interface{}{
		"result":           "inference_completed",
		"model_version":    "1.2.3",
		"inference_time_ms": 80,
		"confidence":       0.92,
		"predictions":      []float64{0.1, 0.7, 0.2},
	}, nil
}

// worker is a background goroutine that processes tasks
func (tm *EdgeTaskManager) worker() {
	for {
		select {
		case <-tm.stopChan:
			return
		default:
			// Get next task from queue
			tm.mutex.Lock()
			if len(tm.taskQueue) == 0 {
				tm.mutex.Unlock()
				time.Sleep(100 * time.Millisecond)
				continue
			}
			
			task := tm.taskQueue[0]
			tm.taskQueue = tm.taskQueue[1:]
			tm.activeTasks[task.ID] = task
			tm.mutex.Unlock()
			
			// Execute task
			task.Status = TaskStatusRunning
			task.StartedAt = time.Now()
			
			result, err := tm.ExecuteTask(task)
			
			task.CompletedAt = time.Now()
			if err != nil {
				task.Status = TaskStatusFailed
				task.Error = err.Error()
			} else {
				task.Status = TaskStatusCompleted
				task.Result = result
			}
			
			// Remove from active tasks
			tm.mutex.Lock()
			delete(tm.activeTasks, task.ID)
			tm.mutex.Unlock()
		}
	}
}