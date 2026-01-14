package scheduler

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

)

var (
	lifetimeDataPointsCollected = promauto.NewCounter(prometheus.CounterOpts{
		Name: "lifetime_data_points_collected_total",
		Help: "Total number of lifetime data points collected",
	})

	lifetimeActualDistribution = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "lifetime_actual_distribution_seconds",
		Help:    "Histogram of actual VM lifetimes in seconds",
		Buckets: prometheus.ExponentialBuckets(3600, 2, 25), // 1h to ~1 month
	})
)

// LifetimeRecord captures VM lifecycle data for training
type LifetimeRecord struct {
	VMID           string                              `json:"vm_id"`
	CreatedAt      time.Time                           `json:"created_at"`
	DeletedAt      *time.Time                          `json:"deleted_at,omitempty"`
	ActualLifetime time.Duration                       `json:"actual_lifetime,omitempty"`
	Features       LifetimeFeatures `json:"features"`
}

// LifetimeDataCollector tracks VM lifecycle events and collects training data
type LifetimeDataCollector struct {
	mu        sync.RWMutex
	activeVMs map[string]*LifetimeRecord // VMID -> record
	completed []LifetimeRecord
}

// NewLifetimeDataCollector creates a new collector
func NewLifetimeDataCollector() *LifetimeDataCollector {
	return &LifetimeDataCollector{
		activeVMs: make(map[string]*LifetimeRecord),
	}
}

// OnVMCreated records a new VM creation event
func (c *LifetimeDataCollector) OnVMCreated(vmID string, createdAt time.Time, features LifetimeFeatures) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.activeVMs[vmID] = &LifetimeRecord{
		VMID:      vmID,
		CreatedAt: createdAt,
		Features:  features,
	}
}

// OnVMDeleted records a VM deletion event and computes actual lifetime
func (c *LifetimeDataCollector) OnVMDeleted(vmID string, deletedAt time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if rec, ok := c.activeVMs[vmID]; ok {
		rec.DeletedAt = &deletedAt
		rec.ActualLifetime = deletedAt.Sub(rec.CreatedAt)

		c.completed = append(c.completed, *rec)
		delete(c.activeVMs, vmID)

		lifetimeDataPointsCollected.Inc()
		lifetimeActualDistribution.Observe(float64(rec.ActualLifetime.Seconds()))
	}
}

// ExportTrainingData exports filtered and sampled data to JSON file
func (c *LifetimeDataCollector) ExportTrainingData(outputPath string, startDate, endDate time.Time) error {
	c.mu.RLock()
	defer c.mu.RUnlock()

	var filtered []LifetimeRecord
	for _, rec := range c.completed {
		if rec.CreatedAt.After(startDate) && rec.CreatedAt.Before(endDate) {
			filtered = append(filtered, rec)
		}
	}

	// Sampling strategy: all VMs >1h lifetime, or 10% of short-lived
	var sampled []LifetimeRecord
	shortLivedCount := 0
	for _, rec := range filtered {
		if rec.ActualLifetime > time.Hour {
			sampled = append(sampled, rec)
		} else {
			shortLivedCount++
		}
	}

	sampleSize := shortLivedCount / 10
	for i, rec := range filtered {
		if rec.ActualLifetime <= time.Hour && len(sampled) < len(filtered)-sampleSize+1 && i%10 == 0 {
			sampled = append(sampled, rec)
		}
		if len(sampled) >= sampleSize {
			break
		}
	}

	data, err := json.MarshalIndent(sampled, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal data: %w", err)
	}

	if err := os.WriteFile(outputPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// GetActiveVMCount returns number of active tracked VMs
func (c *LifetimeDataCollector) GetActiveVMCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.activeVMs)
}

// GetCompletedCount returns number of completed records
func (c *LifetimeDataCollector) GetCompletedCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.completed)
}
