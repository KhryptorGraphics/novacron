package dr

import (
	"sync"
	"time"
)

// MetricsCollector collects DR metrics
type MetricsCollector struct {
	recoveryMetrics *RecoveryMetrics
	backupMetrics   *BackupMetrics
	restoreMetrics  *RestoreMetrics
	mu              sync.RWMutex
}

// NewMetricsCollector creates a metrics collector
func NewMetricsCollector() *MetricsCollector {
	return &MetricsCollector{
		recoveryMetrics: &RecoveryMetrics{},
		backupMetrics:   &BackupMetrics{},
		restoreMetrics:  &RestoreMetrics{},
	}
}

// RecordFailover records a failover event
func (mc *MetricsCollector) RecordFailover(duration time.Duration, success bool) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	if duration < mc.recoveryMetrics.RTO || mc.recoveryMetrics.RTO == 0 {
		mc.recoveryMetrics.RTO = duration
	}

	if success {
		mc.recoveryMetrics.FailoverSuccessRate =
			(mc.recoveryMetrics.FailoverSuccessRate*0.9 + 1.0*0.1)
	} else {
		mc.recoveryMetrics.FailoverSuccessRate *= 0.9
	}

	mc.recoveryMetrics.LastIncident = time.Now()
}

// RecordBackup records a backup event
func (mc *MetricsCollector) RecordBackup(sizeBytes int64, duration time.Duration, success bool) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.backupMetrics.TotalBackups++

	if success {
		mc.backupMetrics.SuccessfulBackups++
		mc.backupMetrics.TotalSizeBytes += sizeBytes
		mc.backupMetrics.LastBackupTime = time.Now()

		if mc.backupMetrics.AvgDuration == 0 {
			mc.backupMetrics.AvgDuration = duration
		} else {
			mc.backupMetrics.AvgDuration =
				(mc.backupMetrics.AvgDuration*9 + duration) / 10
		}
	} else {
		mc.backupMetrics.FailedBackups++
	}
}

// RecordRestore records a restore event
func (mc *MetricsCollector) RecordRestore(bytesRestored int64, duration time.Duration, success bool) {
	mc.mu.Lock()
	defer mc.mu.Unlock()

	mc.restoreMetrics.TotalRestores++

	if success {
		mc.restoreMetrics.SuccessfulRestores++
		mc.restoreMetrics.TotalBytesRestored += bytesRestored
		mc.restoreMetrics.LastRestoreTime = time.Now()

		if mc.restoreMetrics.AvgDuration == 0 {
			mc.restoreMetrics.AvgDuration = duration
		} else {
			mc.restoreMetrics.AvgDuration =
				(mc.restoreMetrics.AvgDuration*9 + duration) / 10
		}
	} else {
		mc.restoreMetrics.FailedRestores++
	}
}

// GetMetrics returns all metrics
func (mc *MetricsCollector) GetMetrics() map[string]interface{} {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	return map[string]interface{}{
		"recovery": mc.recoveryMetrics,
		"backup":   mc.backupMetrics,
		"restore":  mc.restoreMetrics,
	}
}

// CalculateSuccessRate calculates overall success rate
func (mc *MetricsCollector) CalculateSuccessRate() float64 {
	mc.mu.RLock()
	defer mc.mu.RUnlock()

	backupRate := 0.0
	if mc.backupMetrics.TotalBackups > 0 {
		backupRate = float64(mc.backupMetrics.SuccessfulBackups) /
			float64(mc.backupMetrics.TotalBackups)
	}

	restoreRate := 0.0
	if mc.restoreMetrics.TotalRestores > 0 {
		restoreRate = float64(mc.restoreMetrics.SuccessfulRestores) /
			float64(mc.restoreMetrics.TotalRestores)
	}

	failoverRate := mc.recoveryMetrics.FailoverSuccessRate

	// Weighted average
	return (backupRate*0.4 + restoreRate*0.4 + failoverRate*0.2)
}
