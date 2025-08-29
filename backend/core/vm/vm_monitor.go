package vm

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"
)

// MonitoringConfig represents VM monitoring configuration
type MonitoringConfig struct {
	Enabled           bool              `json:"enabled"`
	IntervalSeconds   int               `json:"interval_seconds"`
	ResourceThreshold ResourceThreshold `json:"resource_threshold"`
	AlertConfig       AlertConfig       `json:"alert_config"`
}

// ResourceThreshold represents resource usage thresholds
type ResourceThreshold struct {
	CPUWarningPercent     float64 `json:"cpu_warning_percent"`
	CPUCriticalPercent    float64 `json:"cpu_critical_percent"`
	MemoryWarningPercent  float64 `json:"memory_warning_percent"`
	MemoryCriticalPercent float64 `json:"memory_critical_percent"`
	DiskWarningPercent    float64 `json:"disk_warning_percent"`
	DiskCriticalPercent   float64 `json:"disk_critical_percent"`
}

// AlertConfig represents alert configuration
type AlertConfig struct {
	Enabled         bool     `json:"enabled"`
	EmailRecipients []string `json:"email_recipients,omitempty"`
	WebhookURL      string   `json:"webhook_url,omitempty"`
	AlertOnWarning  bool     `json:"alert_on_warning"`
	AlertOnCritical bool     `json:"alert_on_critical"`
	AlertOnRecover  bool     `json:"alert_on_recover"`
}

// AlertLevel represents the level of an alert
type AlertLevel string

const (
	// AlertLevelInfo represents an informational alert
	AlertLevelInfo AlertLevel = "info"

	// AlertLevelWarning represents a warning alert
	AlertLevelWarning AlertLevel = "warning"

	// AlertLevelCritical represents a critical alert
	AlertLevelCritical AlertLevel = "critical"
)

// Alert represents a VM alert
type Alert struct {
	ID         string     `json:"id"`
	VMID       string     `json:"vm_id"`
	Level      AlertLevel `json:"level"`
	Message    string     `json:"message"`
	Timestamp  time.Time  `json:"timestamp"`
	Resolved   bool       `json:"resolved"`
	ResolvedAt *time.Time `json:"resolved_at,omitempty"`
}

// VMMonitor monitors VMs
type VMMonitor struct {
	vmManager     *VMManager
	eventManager  *VMEventManager
	config        MonitoringConfig
	alerts        map[string][]*Alert
	alertsMutex   sync.RWMutex
	ctx           context.Context
	cancel        context.CancelFunc
	alertHandlers []AlertHandler
}

// AlertHandler is a function that handles alerts
type AlertHandler func(alert *Alert)

// NewVMMonitor creates a new VM monitor
func NewVMMonitor(vmManager *VMManager, eventManager *VMEventManager, config MonitoringConfig) *VMMonitor {
	ctx, cancel := context.WithCancel(context.Background())

	return &VMMonitor{
		vmManager:    vmManager,
		eventManager: eventManager,
		config:       config,
		alerts:       make(map[string][]*Alert),
		ctx:          ctx,
		cancel:       cancel,
	}
}

// Start starts the VM monitor
func (m *VMMonitor) Start() {
	if !m.config.Enabled {
		log.Println("VM monitoring is disabled")
		return
	}

	log.Println("Starting VM monitor")

	// Register default alert handlers
	m.RegisterAlertHandler(LogAlertHandler)

	// Start the monitoring loop
	go m.monitorLoop()
}

// Stop stops the VM monitor
func (m *VMMonitor) Stop() {
	log.Println("Stopping VM monitor")
	m.cancel()
}

// RegisterAlertHandler registers an alert handler
func (m *VMMonitor) RegisterAlertHandler(handler AlertHandler) {
	m.alertHandlers = append(m.alertHandlers, handler)
}

// GetAlerts returns all alerts for a VM
func (m *VMMonitor) GetAlerts(vmID string) []*Alert {
	m.alertsMutex.RLock()
	defer m.alertsMutex.RUnlock()

	alerts, exists := m.alerts[vmID]
	if !exists {
		return []*Alert{}
	}

	// Create a copy of the alerts
	result := make([]*Alert, len(alerts))
	copy(result, alerts)

	return result
}

// GetActiveAlerts returns all active (unresolved) alerts for a VM
func (m *VMMonitor) GetActiveAlerts(vmID string) []*Alert {
	m.alertsMutex.RLock()
	defer m.alertsMutex.RUnlock()

	alerts, exists := m.alerts[vmID]
	if !exists {
		return []*Alert{}
	}

	// Filter active alerts
	result := make([]*Alert, 0)
	for _, alert := range alerts {
		if !alert.Resolved {
			result = append(result, alert)
		}
	}

	return result
}

// ResolveAlert resolves an alert
func (m *VMMonitor) ResolveAlert(vmID, alertID string) error {
	m.alertsMutex.Lock()
	defer m.alertsMutex.Unlock()

	alerts, exists := m.alerts[vmID]
	if !exists {
		return fmt.Errorf("no alerts found for VM %s", vmID)
	}

	for _, alert := range alerts {
		if alert.ID == alertID && !alert.Resolved {
			now := time.Now()
			alert.Resolved = true
			alert.ResolvedAt = &now

			// Emit recovery alert if configured
			if m.config.AlertConfig.AlertOnRecover {
				recoveryAlert := &Alert{
					ID:         fmt.Sprintf("recovery-%s", alertID),
					VMID:       vmID,
					Level:      AlertLevelInfo,
					Message:    fmt.Sprintf("Recovered from alert: %s", alert.Message),
					Timestamp:  now,
					Resolved:   true,
					ResolvedAt: &now,
				}

				// Handle recovery alert
				for _, handler := range m.alertHandlers {
					handler(recoveryAlert)
				}
			}

			return nil
		}
	}

	return fmt.Errorf("alert %s not found for VM %s", alertID, vmID)
}

// monitorLoop monitors VMs periodically
func (m *VMMonitor) monitorLoop() {
	interval := time.Duration(m.config.IntervalSeconds) * time.Second
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-m.ctx.Done():
			return
		case <-ticker.C:
			m.monitorVMs()
		}
	}
}

// monitorVMs monitors all running VMs
func (m *VMMonitor) monitorVMs() {
	// Get all VMs
	vms := m.vmManager.ListVMs()

	// Monitor each VM
	for _, vm := range vms {
		// Skip VMs that are not running
		if vm.State() != StateRunning {
			continue
		}

		// Create a context with timeout
		ctx, cancel := context.WithTimeout(m.ctx, 5*time.Second)

		// Monitor VM
		m.monitorVM(ctx, vm)

		cancel()
	}
}

// monitorVM monitors a single VM
func (m *VMMonitor) monitorVM(ctx context.Context, vm *VM) {
	// Get VM process info
	processInfo := vm.GetProcessInfo()

	// Check CPU usage
	if processInfo.CPUUsagePercent >= m.config.ResourceThreshold.CPUCriticalPercent {
		m.createAlert(vm, AlertLevelCritical, fmt.Sprintf("CPU usage is critical: %.2f%%", processInfo.CPUUsagePercent))
	} else if processInfo.CPUUsagePercent >= m.config.ResourceThreshold.CPUWarningPercent {
		m.createAlert(vm, AlertLevelWarning, fmt.Sprintf("CPU usage is high: %.2f%%", processInfo.CPUUsagePercent))
	} else {
		// Check if there are any active CPU alerts to resolve
		m.resolveResourceAlerts(vm.ID(), "CPU")
	}

	// Check memory usage
	memoryPercent := float64(processInfo.MemoryUsageMB) / float64(vm.config.MemoryMB) * 100
	if memoryPercent >= m.config.ResourceThreshold.MemoryCriticalPercent {
		m.createAlert(vm, AlertLevelCritical, fmt.Sprintf("Memory usage is critical: %.2f%% (%d MB)", memoryPercent, processInfo.MemoryUsageMB))
	} else if memoryPercent >= m.config.ResourceThreshold.MemoryWarningPercent {
		m.createAlert(vm, AlertLevelWarning, fmt.Sprintf("Memory usage is high: %.2f%% (%d MB)", memoryPercent, processInfo.MemoryUsageMB))
	} else {
		// Check if there are any active memory alerts to resolve
		m.resolveResourceAlerts(vm.ID(), "Memory")
	}

	// In a real implementation, we would also check disk usage
}

// createAlert creates a new alert
func (m *VMMonitor) createAlert(vm *VM, level AlertLevel, message string) {
	// Check if alerting is enabled
	if !m.config.AlertConfig.Enabled {
		return
	}

	// Check if we should alert based on level
	if level == AlertLevelWarning && !m.config.AlertConfig.AlertOnWarning {
		return
	}

	if level == AlertLevelCritical && !m.config.AlertConfig.AlertOnCritical {
		return
	}

	// Check if there's already an active alert with the same message
	m.alertsMutex.RLock()
	alerts, exists := m.alerts[vm.ID()]
	if exists {
		for _, alert := range alerts {
			if !alert.Resolved && alert.Message == message {
				m.alertsMutex.RUnlock()
				return
			}
		}
	}
	m.alertsMutex.RUnlock()

	// Create alert
	alert := &Alert{
		ID:        fmt.Sprintf("alert-%d", time.Now().UnixNano()),
		VMID:      vm.ID(),
		Level:     level,
		Message:   message,
		Timestamp: time.Now(),
		Resolved:  false,
	}

	// Store alert
	m.alertsMutex.Lock()
	if _, exists := m.alerts[vm.ID()]; !exists {
		m.alerts[vm.ID()] = make([]*Alert, 0)
	}
	m.alerts[vm.ID()] = append(m.alerts[vm.ID()], alert)
	m.alertsMutex.Unlock()

	// Handle alert
	for _, handler := range m.alertHandlers {
		handler(alert)
	}

	// Emit event
	m.eventManager.EmitEvent(VMEvent{
		Type:      VMEventError,
		VM:        *vm,
		Timestamp: time.Now(),
		NodeID:    vm.GetNodeID(),
		Message:   fmt.Sprintf("Alert: %s", message),
		Data: map[string]interface{}{
			"alert_id":    alert.ID,
			"alert_level": alert.Level,
		},
	})
}

// resolveResourceAlerts resolves active resource alerts for a VM
func (m *VMMonitor) resolveResourceAlerts(vmID, resourceType string) {
	m.alertsMutex.Lock()
	defer m.alertsMutex.Unlock()

	alerts, exists := m.alerts[vmID]
	if !exists {
		return
	}

	now := time.Now()
	for _, alert := range alerts {
		if !alert.Resolved && (alert.Message == fmt.Sprintf("%s usage is critical", resourceType) || alert.Message == fmt.Sprintf("%s usage is high", resourceType)) {
			alert.Resolved = true
			alert.ResolvedAt = &now

			// Emit recovery alert if configured
			if m.config.AlertConfig.AlertOnRecover {
				recoveryAlert := &Alert{
					ID:         fmt.Sprintf("recovery-%s", alert.ID),
					VMID:       vmID,
					Level:      AlertLevelInfo,
					Message:    fmt.Sprintf("Recovered from alert: %s", alert.Message),
					Timestamp:  now,
					Resolved:   true,
					ResolvedAt: &now,
				}

				// Handle recovery alert
				for _, handler := range m.alertHandlers {
					handler(recoveryAlert)
				}
			}
		}
	}
}

// LogAlertHandler logs alerts
func LogAlertHandler(alert *Alert) {
	if alert.Resolved {
		log.Printf("[VM Alert] [RESOLVED] %s: %s", alert.Level, alert.Message)
	} else {
		log.Printf("[VM Alert] %s: %s", alert.Level, alert.Message)
	}
}
