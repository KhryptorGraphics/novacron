package vm

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/pkg/database"
)

// OptimizedHandler handles VM API requests with query optimization
type OptimizedHandler struct {
	db         *database.OptimizedDB
	vmRepo     *database.OptimizedVMRepository
	cache      *ResponseCache
}

// ResponseCache implements response caching for API endpoints
type ResponseCache struct {
	cache map[string]*CachedResponse
	ttl   time.Duration
}

// CachedResponse represents a cached HTTP response
type CachedResponse struct {
	Data      interface{}
	Headers   map[string]string
	ExpiresAt time.Time
}

// NewOptimizedHandler creates an optimized VM API handler
func NewOptimizedHandler(db *database.OptimizedDB) *OptimizedHandler {
	return &OptimizedHandler{
		db:     db,
		vmRepo: database.NewOptimizedVMRepository(db),
		cache: &ResponseCache{
			cache: make(map[string]*CachedResponse),
			ttl:   30 * time.Second,
		},
	}
}

// ListVMsOptimized handles GET /vms with query optimization
// Target: <200ms response time (from 1.2s)
func (h *OptimizedHandler) ListVMsOptimized(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	// Check response cache
	cacheKey := r.URL.String()
	if cached := h.cache.Get(cacheKey); cached != nil {
		w.Header().Set("X-Cache-Hit", "true")
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeJSON(w, http.StatusOK, cached.Data)
		return
	}

	// Parse query parameters
	q := r.URL.Query()
	filters := make(map[string]interface{})

	// Pagination
	page := 1
	pageSize := 20
	if p := q.Get("page"); p != "" {
		if val, err := strconv.Atoi(p); err == nil && val > 0 {
			page = val
		}
	}
	if ps := q.Get("pageSize"); ps != "" {
		if val, err := strconv.Atoi(ps); err == nil && val > 0 && val <= 100 {
			pageSize = val
		}
	}

	filters["limit"] = pageSize
	filters["offset"] = (page - 1) * pageSize

	// Filtering
	if state := q.Get("state"); state != "" {
		filters["state"] = state
	}
	if nodeID := q.Get("nodeId"); nodeID != "" {
		filters["node_id"] = nodeID
	}
	if orgID := q.Get("organizationId"); orgID != "" {
		filters["organization_id"] = orgID
	}

	// Use optimized query with materialized view
	vms, err := h.vmRepo.ListVMsFast(ctx, filters)
	if err != nil {
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeError(w, http.StatusInternalServerError, "database_error", err.Error())
		return
	}

	// Get total count for pagination (cached separately)
	countQuery := `SELECT COUNT(*) FROM mv_vm_listing WHERE 1=1`
	var totalCount int
	if err := h.db.Get(&totalCount, countQuery); err == nil {
		totalPages := (totalCount + pageSize - 1) / pageSize
		pagination := map[string]interface{}{
			"page":       page,
			"pageSize":   pageSize,
			"total":      totalCount,
			"totalPages": totalPages,
		}
		pjson, _ := json.Marshal(pagination)
		w.Header().Set("X-Pagination", string(pjson))
	}

	// Cache response
	h.cache.Set(cacheKey, vms)

	// Send response
	w.Header().Set("X-Cache-Hit", "false")
	w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
	writeJSON(w, http.StatusOK, vms)
}

// GetDashboardOptimized handles GET /dashboard with query optimization
// Target: <500ms response time (from 3.4s)
func (h *OptimizedHandler) GetDashboardOptimized(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	// Get organization ID from context/auth
	orgID := r.URL.Query().Get("organizationId")
	if orgID == "" {
		orgID = "default" // fallback
	}

	// Check cache
	cacheKey := fmt.Sprintf("dashboard_%s", orgID)
	if cached := h.cache.Get(cacheKey); cached != nil {
		w.Header().Set("X-Cache-Hit", "true")
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeJSON(w, http.StatusOK, cached.Data)
		return
	}

	// Use pre-computed dashboard stats from materialized view
	stats, err := h.vmRepo.GetDashboardStats(ctx, orgID)
	if err != nil {
		// Fallback to aggregated query if materialized view fails
		stats = h.getDashboardStatsFallback(ctx, orgID)
	}

	// Add real-time metrics using optimized queries
	realTimeMetrics := h.getRealTimeMetrics(ctx)
	stats["realtime"] = realTimeMetrics

	// Cache response
	h.cache.Set(cacheKey, stats)

	// Send response
	w.Header().Set("X-Cache-Hit", "false")
	w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
	writeJSON(w, http.StatusOK, stats)
}

// GetVMMetricsOptimized handles GET /vms/{id}/metrics with optimization
// Target: <200ms response time (from 2.1s for real-time monitoring)
func (h *OptimizedHandler) GetVMMetricsOptimized(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	vars := mux.Vars(r)
	vmID := vars["id"]

	// Parse time range
	q := r.URL.Query()
	var startTime, endTime time.Time

	if s := q.Get("start"); s != "" {
		if t, err := time.Parse(time.RFC3339, s); err == nil {
			startTime = t
		}
	}
	if e := q.Get("end"); e != "" {
		if t, err := time.Parse(time.RFC3339, e); err == nil {
			endTime = t
		}
	}

	// Default to last hour if not specified
	if startTime.IsZero() {
		startTime = time.Now().Add(-1 * time.Hour)
	}
	if endTime.IsZero() {
		endTime = time.Now()
	}

	// Check cache
	cacheKey := fmt.Sprintf("metrics_%s_%d_%d", vmID, startTime.Unix(), endTime.Unix())
	if cached := h.cache.Get(cacheKey); cached != nil {
		w.Header().Set("X-Cache-Hit", "true")
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeJSON(w, http.StatusOK, cached.Data)
		return
	}

	// Use optimized metrics query
	metrics, err := h.vmRepo.GetVMMetricsOptimized(ctx, vmID, startTime, endTime)
	if err != nil {
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeError(w, http.StatusInternalServerError, "database_error", err.Error())
		return
	}

	// Process metrics for response
	response := map[string]interface{}{
		"vm_id":      vmID,
		"start_time": startTime,
		"end_time":   endTime,
		"metrics":    metrics,
		"summary":    h.calculateMetricsSummary(metrics),
	}

	// Cache response (shorter TTL for real-time data)
	h.cache.SetWithTTL(cacheKey, response, 10*time.Second)

	// Send response
	w.Header().Set("X-Cache-Hit", "false")
	w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
	writeJSON(w, http.StatusOK, response)
}

// GetNodeCapacityOptimized handles GET /nodes/capacity with optimization
func (h *OptimizedHandler) GetNodeCapacityOptimized(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	// Check cache
	cacheKey := "node_capacity"
	if cached := h.cache.Get(cacheKey); cached != nil {
		w.Header().Set("X-Cache-Hit", "true")
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeJSON(w, http.StatusOK, cached.Data)
		return
	}

	// Use materialized view for node capacity
	capacity, err := h.vmRepo.GetNodeCapacity(ctx)
	if err != nil {
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeError(w, http.StatusInternalServerError, "database_error", err.Error())
		return
	}

	// Cache response
	h.cache.Set(cacheKey, capacity)

	// Send response
	w.Header().Set("X-Cache-Hit", "false")
	w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
	writeJSON(w, http.StatusOK, capacity)
}

// BulkMetricsIngestion handles POST /metrics/bulk for efficient metric ingestion
func (h *OptimizedHandler) BulkMetricsIngestion(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	ctx := r.Context()

	var metrics []database.VMMetric
	if err := json.NewDecoder(r.Body).Decode(&metrics); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_payload", err.Error())
		return
	}

	// Use bulk insert with COPY for optimal performance
	if err := h.vmRepo.BulkInsertMetrics(ctx, metrics); err != nil {
		w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
		writeError(w, http.StatusInternalServerError, "database_error", err.Error())
		return
	}

	// Invalidate related caches
	h.cache.InvalidatePattern("metrics_")
	h.cache.InvalidatePattern("dashboard_")

	w.Header().Set("X-Response-Time", fmt.Sprintf("%dms", time.Since(start).Milliseconds()))
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"message": "Metrics ingested successfully",
		"count":   len(metrics),
	})
}

// Helper methods

func (h *OptimizedHandler) getDashboardStatsFallback(ctx context.Context, orgID string) map[string]interface{} {
	// Fallback aggregation query if materialized view is unavailable
	query := `
		WITH vm_stats AS (
			SELECT 
				COUNT(*) as total_vms,
				COUNT(*) FILTER (WHERE state = 'running') as running_vms,
				SUM(cpu_cores) as total_cpu,
				SUM(memory_mb) as total_memory
			FROM vms
			WHERE organization_id = $1
		),
		node_stats AS (
			SELECT 
				COUNT(*) as total_nodes,
				COUNT(*) FILTER (WHERE status = 'online') as online_nodes
			FROM nodes
		)
		SELECT * FROM vm_stats CROSS JOIN node_stats`

	var stats map[string]interface{}
	if err := h.db.GetContext(ctx, &stats, query, orgID); err != nil {
		return make(map[string]interface{})
	}
	return stats
}

func (h *OptimizedHandler) getRealTimeMetrics(ctx context.Context) map[string]interface{} {
	// Get latest metrics from last 5 minutes
	query := `
		WITH latest AS (
			SELECT DISTINCT ON (vm_id)
				vm_id,
				cpu_usage,
				memory_percent
			FROM vm_metrics
			WHERE timestamp > NOW() - INTERVAL '5 minutes'
			ORDER BY vm_id, timestamp DESC
		)
		SELECT 
			AVG(cpu_usage) as avg_cpu,
			AVG(memory_percent) as avg_memory,
			MAX(cpu_usage) as max_cpu,
			MAX(memory_percent) as max_memory
		FROM latest`

	var metrics map[string]interface{}
	if err := h.db.GetContext(ctx, &metrics, query); err != nil {
		return make(map[string]interface{})
	}
	return metrics
}

func (h *OptimizedHandler) calculateMetricsSummary(metrics []*database.VMMetric) map[string]interface{} {
	if len(metrics) == 0 {
		return nil
	}

	var totalCPU, totalMem float64
	var maxCPU, maxMem float64

	for _, m := range metrics {
		totalCPU += m.CPUUsage
		totalMem += m.MemoryPercent
		if m.CPUUsage > maxCPU {
			maxCPU = m.CPUUsage
		}
		if m.MemoryPercent > maxMem {
			maxMem = m.MemoryPercent
		}
	}

	return map[string]interface{}{
		"avg_cpu":    totalCPU / float64(len(metrics)),
		"avg_memory": totalMem / float64(len(metrics)),
		"max_cpu":    maxCPU,
		"max_memory": maxMem,
		"samples":    len(metrics),
	}
}

// Cache helper methods

func (c *ResponseCache) Get(key string) *CachedResponse {
	if entry, exists := c.cache[key]; exists {
		if time.Now().Before(entry.ExpiresAt) {
			return entry
		}
		delete(c.cache, key)
	}
	return nil
}

func (c *ResponseCache) Set(key string, data interface{}) {
	c.cache[key] = &CachedResponse{
		Data:      data,
		ExpiresAt: time.Now().Add(c.ttl),
	}
}

func (c *ResponseCache) SetWithTTL(key string, data interface{}, ttl time.Duration) {
	c.cache[key] = &CachedResponse{
		Data:      data,
		ExpiresAt: time.Now().Add(ttl),
	}
}

func (c *ResponseCache) InvalidatePattern(pattern string) {
	for key := range c.cache {
		if strings.Contains(key, pattern) {
			delete(c.cache, key)
		}
	}
}

// RegisterOptimizedRoutes registers optimized VM API routes
func (h *OptimizedHandler) RegisterOptimizedRoutes(router *mux.Router) {
	// Optimized endpoints
	router.HandleFunc("/v2/vms", h.ListVMsOptimized).Methods("GET")
	router.HandleFunc("/v2/dashboard", h.GetDashboardOptimized).Methods("GET")
	router.HandleFunc("/v2/vms/{id}/metrics", h.GetVMMetricsOptimized).Methods("GET")
	router.HandleFunc("/v2/nodes/capacity", h.GetNodeCapacityOptimized).Methods("GET")
	router.HandleFunc("/v2/metrics/bulk", h.BulkMetricsIngestion).Methods("POST")
	
	// Admin endpoints
	router.HandleFunc("/v2/admin/cache/stats", h.GetCacheStats).Methods("GET")
	router.HandleFunc("/v2/admin/cache/clear", h.ClearCache).Methods("POST")
}

// GetCacheStats returns cache statistics
func (h *OptimizedHandler) GetCacheStats(w http.ResponseWriter, r *http.Request) {
	stats := h.db.GetCacheStats()
	stats["response_cache_size"] = len(h.cache.cache)
	writeJSON(w, http.StatusOK, stats)
}

// ClearCache clears all caches
func (h *OptimizedHandler) ClearCache(w http.ResponseWriter, r *http.Request) {
	// Clear response cache
	h.cache.cache = make(map[string]*CachedResponse)
	
	// Note: Query cache in database layer should be cleared separately if needed
	
	writeJSON(w, http.StatusOK, map[string]string{
		"message": "Cache cleared successfully",
	})
}