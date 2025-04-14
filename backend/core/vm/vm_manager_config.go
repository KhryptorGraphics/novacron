package vm

import "time"

// DefaultVMManagerConfig returns the default VM manager configuration
func DefaultVMManagerConfig() VMManagerConfig {
	return VMManagerConfig{
		MaxVMs:                 100,
		MaxVMsPerUser:          10,
		MaxCPUPerVM:            8,
		MaxMemoryPerVM:         16384, // 16 GB
		MaxDiskPerVM:           1024,  // 1 TB
		VMStartTimeout:         30 * time.Second,
		VMStopTimeout:          30 * time.Second,
		VMDeleteTimeout:        60 * time.Second,
		VMRestartTimeout:       60 * time.Second,
		VMPauseTimeout:         30 * time.Second,
		VMResumeTimeout:        30 * time.Second,
		VMSnapshotTimeout:      60 * time.Second,
		VMBackupTimeout:        60 * time.Second,
		VMRestoreTimeout:       60 * time.Second,
		VMCloneTimeout:         60 * time.Second,
		VMResizeTimeout:        60 * time.Second,
		VMAttachDiskTimeout:    30 * time.Second,
		VMDetachDiskTimeout:    30 * time.Second,
		VMAttachNetworkTimeout: 30 * time.Second,
		VMDetachNetworkTimeout: 30 * time.Second,
		VMConsoleTimeout:       30 * time.Second,
		VMLogTimeout:           30 * time.Second,
		VMMetricsTimeout:       30 * time.Second,
		VMHealthTimeout:        30 * time.Second,
		VMMonitorTimeout:       30 * time.Second,
		VMScheduleTimeout:      30 * time.Second,
		VMClusterTimeout:       60 * time.Second,
		VMSecurityTimeout:      30 * time.Second,
		VMStorageTimeout:       30 * time.Second,
		VMNetworkTimeout:       30 * time.Second,
		VMBackupRetention:      30 * 24 * time.Hour, // 30 days
		VMSnapshotRetention:    7 * 24 * time.Hour,  // 7 days
		VMLogRetention:         7 * 24 * time.Hour,  // 7 days
		VMMetricsRetention:     7 * 24 * time.Hour,  // 7 days
		VMHealthRetention:      7 * 24 * time.Hour,  // 7 days
		VMMonitorRetention:     7 * 24 * time.Hour,  // 7 days
		VMEventRetention:       7 * 24 * time.Hour,  // 7 days
	}
}
