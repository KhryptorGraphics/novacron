package vm

// GetSnapshotManager returns the VM snapshot manager
func (m *VMManager) GetSnapshotManager() *VMSnapshotManager {
	return m.snapshotManager
}

// GetBackupManager returns the VM backup manager
func (m *VMManager) GetBackupManager() *VMBackupManager {
	return m.backupManager
}

// GetClusterManager returns the VM cluster manager
func (m *VMManager) GetClusterManager() *VMClusterManager {
	return m.clusterManager
}

// GetNetworkManager returns the VM network manager
func (m *VMManager) GetNetworkManager() *VMNetworkManager {
	return m.networkManager
}

// GetStorageManager returns the VM storage manager
func (m *VMManager) GetStorageManager() *VMStorageManager {
	return m.storageManager
}

// GetSecurityManager returns the VM security manager
func (m *VMManager) GetSecurityManager() *VMSecurityManager {
	return m.securityManager
}

// GetHealthManager returns the VM health manager
func (m *VMManager) GetHealthManager() *VMHealthManager {
	return m.healthManager
}

// GetMonitor returns the VM monitor
func (m *VMManager) GetMonitor() *VMMonitor {
	return m.monitor
}

// GetMetricsCollector returns the VM metrics collector
func (m *VMManager) GetMetricsCollector() *VMMetricsCollector {
	return m.metricsCollector
}

// GetScheduler returns the VM scheduler
func (m *VMManager) GetScheduler() *VMScheduler {
	return m.scheduler
}

// SetSnapshotManager sets the VM snapshot manager
func (m *VMManager) SetSnapshotManager(snapshotManager *VMSnapshotManager) {
	m.snapshotManager = snapshotManager
}

// SetBackupManager sets the VM backup manager
func (m *VMManager) SetBackupManager(backupManager *VMBackupManager) {
	m.backupManager = backupManager
}

// SetClusterManager sets the VM cluster manager
func (m *VMManager) SetClusterManager(clusterManager *VMClusterManager) {
	m.clusterManager = clusterManager
}

// SetNetworkManager sets the VM network manager
func (m *VMManager) SetNetworkManager(networkManager *VMNetworkManager) {
	m.networkManager = networkManager
}

// SetStorageManager sets the VM storage manager
func (m *VMManager) SetStorageManager(storageManager *VMStorageManager) {
	m.storageManager = storageManager
}

// SetSecurityManager sets the VM security manager
func (m *VMManager) SetSecurityManager(securityManager *VMSecurityManager) {
	m.securityManager = securityManager
}

// SetHealthManager sets the VM health manager
func (m *VMManager) SetHealthManager(healthManager *VMHealthManager) {
	m.healthManager = healthManager
}

// SetMonitor sets the VM monitor
func (m *VMManager) SetMonitor(monitor *VMMonitor) {
	m.monitor = monitor
}

// SetMetricsCollector sets the VM metrics collector
func (m *VMManager) SetMetricsCollector(metricsCollector *VMMetricsCollector) {
	m.metricsCollector = metricsCollector
}

// SetScheduler sets the VM scheduler
func (m *VMManager) SetScheduler(scheduler *VMScheduler) {
	m.scheduler = scheduler
}
