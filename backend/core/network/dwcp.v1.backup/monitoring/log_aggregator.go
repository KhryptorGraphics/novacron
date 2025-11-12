package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// LogLevel defines log severity levels
type LogLevel int

const (
	LogLevelDebug LogLevel = iota
	LogLevelInfo
	LogLevelWarn
	LogLevelError
	LogLevelFatal
)

func (l LogLevel) String() string {
	switch l {
	case LogLevelDebug:
		return "DEBUG"
	case LogLevelInfo:
		return "INFO"
	case LogLevelWarn:
		return "WARN"
	case LogLevelError:
		return "ERROR"
	case LogLevelFatal:
		return "FATAL"
	default:
		return "UNKNOWN"
	}
}

// LogEntry represents a structured log entry
type LogEntry struct {
	Timestamp   time.Time              `json:"timestamp"`
	Level       LogLevel               `json:"level"`
	Message     string                 `json:"message"`
	Region      string                 `json:"region"`
	Service     string                 `json:"service"`
	TraceID     string                 `json:"trace_id,omitempty"`
	SpanID      string                 `json:"span_id,omitempty"`
	Fields      map[string]interface{} `json:"fields,omitempty"`
	StackTrace  string                 `json:"stack_trace,omitempty"`
}

// LogAggregator aggregates logs from multiple regions
type LogAggregator struct {
	mu sync.RWMutex

	// Storage
	logs         []*LogEntry
	maxLogs      int
	retentionPeriod time.Duration

	// Indexing
	byRegion     map[string][]*LogEntry
	byService    map[string][]*LogEntry
	byTraceID    map[string][]*LogEntry

	// Configuration
	elasticURL   string
	indexName    string
}

// LogFilter filters log entries
type LogFilter struct {
	Level      LogLevel
	Region     string
	Service    string
	TraceID    string
	StartTime  time.Time
	EndTime    time.Time
	SearchTerm string
	Limit      int
}

// NewLogAggregator creates a new log aggregator
func NewLogAggregator(elasticURL, indexName string) *LogAggregator {
	return &LogAggregator{
		logs:            make([]*LogEntry, 0),
		maxLogs:         1000000, // 1M logs in memory
		retentionPeriod: 30 * 24 * time.Hour, // 30 days
		byRegion:        make(map[string][]*LogEntry),
		byService:       make(map[string][]*LogEntry),
		byTraceID:       make(map[string][]*LogEntry),
		elasticURL:      elasticURL,
		indexName:       indexName,
	}
}

// Ingest ingests a log entry
func (la *LogAggregator) Ingest(entry *LogEntry) error {
	la.mu.Lock()
	defer la.mu.Unlock()

	// Add to main storage
	la.logs = append(la.logs, entry)

	// Update indexes
	la.byRegion[entry.Region] = append(la.byRegion[entry.Region], entry)
	la.byService[entry.Service] = append(la.byService[entry.Service], entry)
	if entry.TraceID != "" {
		la.byTraceID[entry.TraceID] = append(la.byTraceID[entry.TraceID], entry)
	}

	// Enforce max logs
	if len(la.logs) > la.maxLogs {
		// Remove oldest
		removed := la.logs[0]
		la.logs = la.logs[1:]

		// Update indexes
		la.removeFromIndex(removed)
	}

	// Send to Elasticsearch
	if la.elasticURL != "" {
		go la.sendToElasticsearch(entry)
	}

	return nil
}

// Search searches logs with filters
func (la *LogAggregator) Search(filter LogFilter) []*LogEntry {
	la.mu.RLock()
	defer la.mu.RUnlock()

	var results []*LogEntry

	// Use indexes for efficiency
	var candidates []*LogEntry

	if filter.Region != "" {
		candidates = la.byRegion[filter.Region]
	} else if filter.Service != "" {
		candidates = la.byService[filter.Service]
	} else if filter.TraceID != "" {
		candidates = la.byTraceID[filter.TraceID]
	} else {
		candidates = la.logs
	}

	// Apply filters
	for _, entry := range candidates {
		if la.matchesFilter(entry, filter) {
			results = append(results, entry)

			if filter.Limit > 0 && len(results) >= filter.Limit {
				break
			}
		}
	}

	return results
}

// matchesFilter checks if entry matches filter
func (la *LogAggregator) matchesFilter(entry *LogEntry, filter LogFilter) bool {
	if filter.Level > 0 && entry.Level < filter.Level {
		return false
	}

	if filter.Region != "" && entry.Region != filter.Region {
		return false
	}

	if filter.Service != "" && entry.Service != filter.Service {
		return false
	}

	if filter.TraceID != "" && entry.TraceID != filter.TraceID {
		return false
	}

	if !filter.StartTime.IsZero() && entry.Timestamp.Before(filter.StartTime) {
		return false
	}

	if !filter.EndTime.IsZero() && entry.Timestamp.After(filter.EndTime) {
		return false
	}

	if filter.SearchTerm != "" {
		// Simple substring search (production would use full-text search)
		found := false
		if containsString(entry.Message, filter.SearchTerm) {
			found = true
		}
		for _, v := range entry.Fields {
			if str, ok := v.(string); ok && containsString(str, filter.SearchTerm) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

// GetLogsByTrace retrieves all logs for a trace
func (la *LogAggregator) GetLogsByTrace(traceID string) []*LogEntry {
	la.mu.RLock()
	defer la.mu.RUnlock()

	entries, ok := la.byTraceID[traceID]
	if !ok {
		return nil
	}

	result := make([]*LogEntry, len(entries))
	copy(result, entries)
	return result
}

// CorrelateWithTrace correlates log entries with a trace
func (la *LogAggregator) CorrelateWithTrace(traceID string) []*LogEntry {
	return la.GetLogsByTrace(traceID)
}

// Cleanup removes old log entries
func (la *LogAggregator) Cleanup() {
	la.mu.Lock()
	defer la.mu.Unlock()

	cutoff := time.Now().Add(-la.retentionPeriod)

	var kept []*LogEntry
	for _, entry := range la.logs {
		if entry.Timestamp.After(cutoff) {
			kept = append(kept, entry)
		} else {
			la.removeFromIndex(entry)
		}
	}

	la.logs = kept
}

// removeFromIndex removes entry from indexes
func (la *LogAggregator) removeFromIndex(entry *LogEntry) {
	// Remove from region index
	if entries, ok := la.byRegion[entry.Region]; ok {
		la.byRegion[entry.Region] = removeEntry(entries, entry)
	}

	// Remove from service index
	if entries, ok := la.byService[entry.Service]; ok {
		la.byService[entry.Service] = removeEntry(entries, entry)
	}

	// Remove from trace index
	if entry.TraceID != "" {
		if entries, ok := la.byTraceID[entry.TraceID]; ok {
			la.byTraceID[entry.TraceID] = removeEntry(entries, entry)
		}
	}
}

// sendToElasticsearch sends log entry to Elasticsearch
func (la *LogAggregator) sendToElasticsearch(entry *LogEntry) error {
	// Convert to JSON
	data, err := json.Marshal(entry)
	if err != nil {
		return fmt.Errorf("failed to marshal log entry: %w", err)
	}

	// Send to Elasticsearch (simplified)
	// Production would use Elasticsearch client library
	url := fmt.Sprintf("%s/%s/_doc", la.elasticURL, la.indexName)
	_ = url
	_ = data

	// Implementation would POST to Elasticsearch
	return nil
}

// StartAutoCleanup starts automatic cleanup
func (la *LogAggregator) StartAutoCleanup(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			la.Cleanup()
		}
	}
}

// GetStatistics returns log statistics
func (la *LogAggregator) GetStatistics() map[string]interface{} {
	la.mu.RLock()
	defer la.mu.RUnlock()

	levelCounts := make(map[string]int)
	for _, entry := range la.logs {
		levelCounts[entry.Level.String()]++
	}

	return map[string]interface{}{
		"total_logs":     len(la.logs),
		"by_level":       levelCounts,
		"regions":        len(la.byRegion),
		"services":       len(la.byService),
		"traces":         len(la.byTraceID),
	}
}

// Helper functions

func containsString(s, substr string) bool {
	// Simple contains check
	return len(s) >= len(substr) && findSubstring(s, substr)
}

func findSubstring(s, substr string) bool {
	if len(substr) == 0 {
		return true
	}
	if len(s) < len(substr) {
		return false
	}
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func removeEntry(entries []*LogEntry, toRemove *LogEntry) []*LogEntry {
	var result []*LogEntry
	for _, entry := range entries {
		if entry != toRemove {
			result = append(result, entry)
		}
	}
	return result
}

// StructuredLogger provides structured logging interface
type StructuredLogger struct {
	aggregator *LogAggregator
	region     string
	service    string
}

// NewStructuredLogger creates a new structured logger
func NewStructuredLogger(aggregator *LogAggregator, region, service string) *StructuredLogger {
	return &StructuredLogger{
		aggregator: aggregator,
		region:     region,
		service:    service,
	}
}

// Debug logs debug message
func (sl *StructuredLogger) Debug(message string, fields map[string]interface{}) {
	sl.log(LogLevelDebug, message, fields)
}

// Info logs info message
func (sl *StructuredLogger) Info(message string, fields map[string]interface{}) {
	sl.log(LogLevelInfo, message, fields)
}

// Warn logs warning message
func (sl *StructuredLogger) Warn(message string, fields map[string]interface{}) {
	sl.log(LogLevelWarn, message, fields)
}

// Error logs error message
func (sl *StructuredLogger) Error(message string, fields map[string]interface{}) {
	sl.log(LogLevelError, message, fields)
}

// Fatal logs fatal message
func (sl *StructuredLogger) Fatal(message string, fields map[string]interface{}) {
	sl.log(LogLevelFatal, message, fields)
}

// log logs a message
func (sl *StructuredLogger) log(level LogLevel, message string, fields map[string]interface{}) {
	entry := &LogEntry{
		Timestamp: time.Now(),
		Level:     level,
		Message:   message,
		Region:    sl.region,
		Service:   sl.service,
		Fields:    fields,
	}

	sl.aggregator.Ingest(entry)
}
