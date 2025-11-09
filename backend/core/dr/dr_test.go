package dr

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDROrchestrator(t *testing.T) {
	config := DefaultDRConfig()
	config.PrimaryRegion = "us-east-1"
	config.SecondaryRegions = []string{"us-west-2", "eu-west-1"}

	orchestrator, err := NewOrchestrator(config)
	require.NoError(t, err)
	require.NotNil(t, orchestrator)

	// Start orchestrator
	err = orchestrator.Start()
	require.NoError(t, err)

	// Get initial status
	status := orchestrator.GetStatus()
	assert.Equal(t, StateNormal, status.State)
	assert.Equal(t, "us-east-1", status.PrimaryRegion)

	// Stop orchestrator
	err = orchestrator.Stop()
	require.NoError(t, err)
}

func TestFailoverExecution(t *testing.T) {
	config := DefaultDRConfig()
	config.AutoFailover = true

	orchestrator, err := NewOrchestrator(config)
	require.NoError(t, err)

	err = orchestrator.Start()
	require.NoError(t, err)
	defer orchestrator.Stop()

	// Trigger failover
	event := &FailureEvent{
		ID:           "test-failure-1",
		Type:         FailureTypeRegion,
		Severity:     10,
		DetectedAt:   time.Now(),
		AffectedZone: "us-east-1",
		Description:  "Test region failure",
		AutoFailover: true,
	}

	err = orchestrator.ReportFailure(event)
	require.NoError(t, err)

	// Wait for failover to initiate
	time.Sleep(2 * time.Second)

	// Verify state changed
	status := orchestrator.GetStatus()
	assert.True(t, status.State == StateFailingOver || status.State == StateRecovery)
}

func TestBackupSystem(t *testing.T) {
	config := DefaultDRConfig()

	backupSys, err := NewBackupSystem(config)
	require.NoError(t, err)
	require.NotNil(t, backupSys)

	ctx := context.Background()

	// Start backup system
	err = backupSys.Start(ctx)
	require.NoError(t, err)

	// Initiate manual backup
	backupID, err := backupSys.InitiateBackup(ctx, BackupTypeFull)
	require.NoError(t, err)
	assert.NotEmpty(t, backupID)

	// Wait for backup to start
	time.Sleep(500 * time.Millisecond)

	// Get backup status
	job, err := backupSys.GetBackupStatus(backupID)
	require.NoError(t, err)
	assert.Equal(t, "running", job.Status)

	// Stop backup system
	err = backupSys.Stop()
	require.NoError(t, err)
}

func TestRestoreSystem(t *testing.T) {
	config := DefaultDRConfig()

	backupSys, err := NewBackupSystem(config)
	require.NoError(t, err)

	restoreSys := NewRestoreSystem(config, backupSys)
	require.NotNil(t, restoreSys)

	ctx := context.Background()

	// Initiate restore
	target := RestoreTarget{
		Type:         "vm",
		TargetID:     "test-vm-1",
		TargetRegion: "us-west-2",
	}

	restoreID, err := restoreSys.RestoreFromBackup(ctx, "backup-12345", target)
	require.NoError(t, err)
	assert.NotEmpty(t, restoreID)

	// Wait for restore to start
	time.Sleep(500 * time.Millisecond)

	// Get restore status
	job, err := restoreSys.GetRestoreStatus(restoreID)
	require.NoError(t, err)
	assert.NotNil(t, job)
}

func TestSplitBrainPrevention(t *testing.T) {
	config := DefaultDRConfig()

	sbp, err := NewSplitBrainPreventionSystem(config)
	require.NoError(t, err)
	require.NotNil(t, sbp)

	ctx := context.Background()

	// Start split-brain prevention
	err = sbp.Start(ctx)
	require.NoError(t, err)

	// Check quorum
	err = sbp.CheckQuorum(ctx)
	assert.NoError(t, err)

	// Get quorum status
	status := sbp.GetQuorumStatus()
	assert.NotNil(t, status)
	assert.True(t, status["has_quorum"].(bool))
}

func TestHealthMonitor(t *testing.T) {
	config := DefaultDRConfig()

	hm := NewHealthMonitor(config)
	require.NotNil(t, hm)

	ctx := context.Background()

	// Start health monitor
	err := hm.Start(ctx)
	require.NoError(t, err)

	// Wait for health checks
	time.Sleep(2 * time.Second)

	// Get global health
	health := hm.GetGlobalHealth()
	assert.NotNil(t, health)
	assert.True(t, health.HealthScore > 0)

	// Get region health
	regionHealth, err := hm.GetRegionHealth(config.PrimaryRegion)
	require.NoError(t, err)
	assert.Equal(t, config.PrimaryRegion, regionHealth.RegionID)
}

func TestRegionalFailover(t *testing.T) {
	config := DefaultDRConfig()

	rfm, err := NewRegionalFailoverManager(config)
	require.NoError(t, err)

	// Select target region
	target, err := rfm.SelectTargetRegion("us-east-1")
	require.NoError(t, err)
	assert.Contains(t, config.SecondaryRegions, target)

	// Sync state
	err = rfm.SyncState("us-east-1", target)
	assert.NoError(t, err)

	// Redirect traffic
	err = rfm.RedirectTraffic(target)
	assert.NoError(t, err)

	// Validate failover
	err = rfm.ValidateFailover(target)
	assert.NoError(t, err)
}

func TestIntegrityChecker(t *testing.T) {
	config := DefaultDRConfig()

	ic := NewIntegrityChecker(config)
	require.NotNil(t, ic)

	// Test corruption detection
	data1 := []byte("test data")
	data2 := []byte("corrupted data")

	// First time - store checksum
	err := ic.DetectCorruption("vm_state", "vm-1", data1)
	assert.NoError(t, err)

	// Same data - no corruption
	err = ic.DetectCorruption("vm_state", "vm-1", data1)
	assert.NoError(t, err)

	// Different data - corruption detected
	err = ic.DetectCorruption("vm_state", "vm-1", data2)
	assert.Error(t, err)

	// Check violations
	violations := ic.GetViolations()
	assert.Len(t, violations, 1)
}

func TestDRAPI(t *testing.T) {
	config := DefaultDRConfig()

	orchestrator, err := NewOrchestrator(config)
	require.NoError(t, err)

	err = orchestrator.Start()
	require.NoError(t, err)
	defer orchestrator.Stop()

	api := NewDRAPI(orchestrator)
	require.NotNil(t, api)

	// Get DR status
	status, err := api.GetDRStatus()
	require.NoError(t, err)
	assert.NotNil(t, status)

	// Get metrics
	backupMetrics := api.GetBackupMetrics()
	assert.NotNil(t, backupMetrics)

	restoreMetrics := api.GetRestoreMetrics()
	assert.NotNil(t, restoreMetrics)

	recoveryMetrics := api.GetRecoveryMetrics()
	assert.NotNil(t, recoveryMetrics)
}

func TestBackupRetention(t *testing.T) {
	config := DefaultDRConfig()

	policy := config.RetentionPolicy
	assert.Equal(t, 7, policy.HourlyRetentionDays)
	assert.Equal(t, 30, policy.DailyRetentionDays)
	assert.Equal(t, 90, policy.WeeklyRetentionDays)
	assert.Equal(t, 365, policy.MonthlyRetentionDays)
	assert.Equal(t, 7, policy.YearlyRetentionYears)
}

func TestConfigValidation(t *testing.T) {
	config := DefaultDRConfig()

	// Valid config
	err := config.Validate()
	assert.NoError(t, err)

	// Invalid RTO
	config.RTO = -1
	err = config.Validate()
	assert.Error(t, err)

	config.RTO = 30 * time.Minute

	// Invalid RPO
	config.RPO = 0
	err = config.Validate()
	assert.Error(t, err)

	config.RPO = 5 * time.Minute

	// RPO > RTO
	config.RPO = 60 * time.Minute
	err = config.Validate()
	assert.Error(t, err)
}

func TestMetricsCollection(t *testing.T) {
	mc := NewMetricsCollector()

	// Record failover
	mc.RecordFailover(2*time.Minute, true)

	// Record backup
	mc.RecordBackup(10*1024*1024*1024, 5*time.Minute, true)

	// Record restore
	mc.RecordRestore(10*1024*1024*1024, 3*time.Minute, true)

	// Get metrics
	metrics := mc.GetMetrics()
	assert.NotNil(t, metrics)

	// Calculate success rate
	rate := mc.CalculateSuccessRate()
	assert.True(t, rate > 0)
}

func BenchmarkFailoverDetection(b *testing.B) {
	config := DefaultDRConfig()
	orchestrator, _ := NewOrchestrator(config)
	orchestrator.Start()
	defer orchestrator.Stop()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		event := &FailureEvent{
			ID:           "bench-failure",
			Type:         FailureTypeNode,
			Severity:     5,
			DetectedAt:   time.Now(),
			AffectedZone: "us-east-1",
			Description:  "Benchmark failure",
		}
		orchestrator.ReportFailure(event)
	}
}

func BenchmarkBackupInitiation(b *testing.B) {
	config := DefaultDRConfig()
	backupSys, _ := NewBackupSystem(config)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		backupSys.InitiateBackup(ctx, BackupTypeIncremental)
	}
}

func BenchmarkHealthCheck(b *testing.B) {
	config := DefaultDRConfig()
	hm := NewHealthMonitor(config)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		hm.GetGlobalHealth()
	}
}

func BenchmarkQuorumCheck(b *testing.B) {
	config := DefaultDRConfig()
	sbp, _ := NewSplitBrainPreventionSystem(config)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sbp.CheckQuorum(ctx)
	}
}
