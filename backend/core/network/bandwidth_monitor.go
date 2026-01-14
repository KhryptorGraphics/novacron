package network

import (
	"bufio"
	"context"
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/vishvananda/netlink"
	"go.uber.org/zap"
)

type BandwidthMeasurement struct {
	InterfaceName string            `json:"interface_name"`
	Timestamp     time.Time         `json:"timestamp"`
	RXBytes       uint64            `json:"rx_bytes"`
	TXBytes       uint64            `json:"tx_bytes"`
	RXPackets     uint64            `json:"rx_packets"`
	TXPackets     uint64            `json:"tx_packets"`
	RXErrors      uint64            `json:"rx_errors"`
	TXErrors      uint64            `json:"tx_errors"`
	RXDropped     uint64            `json:"rx_dropped"`
	TXDropped     uint64            `json:"tx_dropped"`
	RXRate        float64           `json:"rx_rate_bps"`
	TXRate        float64           `json:"tx_rate_bps"`
	Utilization   float64           `json:"utilization_percent"`
	LinkSpeed     uint64            `json:"link_speed_bps"`
	LinkStatus    bool              `json:"link_status"`
	Metadata      map[string]string `json:"metadata"`
}

type BandwidthThreshold struct {
	InterfaceName     string  `json:"interface_name"`
	WarningThreshold  float64 `json:"warning_threshold_percent"`
	CriticalThreshold float64 `json:"critical_threshold_percent"`
	AbsoluteLimit     uint64  `json:"absolute_limit_bps"`
	Enabled           bool    `json:"enabled"`
}

type BandwidthAlert struct {
	ID            string                 `json:"id"`
	InterfaceName string                 `json:"interface_name"`
	Severity      string                 `json:"severity"`
	Message       string                 `json:"message"`
	Timestamp     time.Time              `json:"timestamp"`
	CurrentValue  float64                `json:"current_value"`
	ThresholdValue float64               `json:"threshold_value"`
	Metadata      map[string]interface{} `json:"metadata"`
}

type BandwidthAlertHandler interface {
	HandleAlert(alert *BandwidthAlert) error
}

type InterfaceMonitor struct {
	name         string
	measurements []BandwidthMeasurement
	lastMeasure  *BandwidthMeasurement
	thresholds   []BandwidthThreshold
	mu           sync.RWMutex
	maxHistory   int
}

type BandwidthMonitorConfig struct {
	MonitoringInterval     time.Duration         `json:"monitoring_interval"`
	HistoryRetention       time.Duration         `json:"history_retention"`
	Interfaces             []string              `json:"interfaces"`
	DefaultThresholds      []BandwidthThreshold  `json:"default_thresholds"`
	AlertHandlers          []BandwidthAlertHandler
	EnableQoSHooks         bool                  `json:"enable_qos_hooks"`
	MaxHistoryPoints       int                   `json:"max_history_points"`
	SlidingWindowDuration  time.Duration         `json:"sliding_window_duration"`
}

type BandwidthMonitor struct {
	config       *BandwidthMonitorConfig
	interfaces   map[string]*InterfaceMonitor
	alertHandlers []BandwidthAlertHandler
	qosHooks     []func(string, float64)
	logger       *zap.Logger
	ctx          context.Context
	cancel       context.CancelFunc
	mu           sync.RWMutex
	lastAlerts   map[string]time.Time
	alertsMutex  sync.RWMutex
	running      bool
}

func NewBandwidthMonitor(config *BandwidthMonitorConfig, logger *zap.Logger) *BandwidthMonitor {
	ctx, cancel := context.WithCancel(context.Background())
	
	if config.MonitoringInterval == 0 {
		config.MonitoringInterval = 5 * time.Second
	}
	if config.HistoryRetention == 0 {
		config.HistoryRetention = 24 * time.Hour
	}
	if config.MaxHistoryPoints == 0 {
		config.MaxHistoryPoints = 1000
	}
	if config.SlidingWindowDuration == 0 {
		config.SlidingWindowDuration = 3 * config.MonitoringInterval
	}

	bm := &BandwidthMonitor{
		config:        config,
		interfaces:    make(map[string]*InterfaceMonitor),
		alertHandlers: config.AlertHandlers,
		qosHooks:      make([]func(string, float64), 0),
		logger:        logger,
		ctx:           ctx,
		cancel:        cancel,
		lastAlerts:    make(map[string]time.Time),
	}

	return bm
}

func (bm *BandwidthMonitor) Start() error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if bm.running {
		return fmt.Errorf("bandwidth monitor is already running")
	}

	if len(bm.config.Interfaces) == 0 {
		interfaces, err := bm.discoverInterfaces()
		if err != nil {
			return fmt.Errorf("failed to discover interfaces: %w", err)
		}
		bm.config.Interfaces = interfaces
	}

	for _, ifaceName := range bm.config.Interfaces {
		monitor := &InterfaceMonitor{
			name:         ifaceName,
			measurements: make([]BandwidthMeasurement, 0, bm.config.MaxHistoryPoints),
			thresholds:   bm.config.DefaultThresholds,
			maxHistory:   bm.config.MaxHistoryPoints,
		}
		bm.interfaces[ifaceName] = monitor
	}

	bm.running = true
	go bm.monitoringLoop()

	bm.logger.Info("Bandwidth monitor started", 
		zap.Strings("interfaces", bm.config.Interfaces),
		zap.Duration("interval", bm.config.MonitoringInterval))

	return nil
}

func (bm *BandwidthMonitor) Stop() error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if !bm.running {
		return nil
	}

	bm.cancel()
	bm.running = false

	bm.logger.Info("Bandwidth monitor stopped")
	return nil
}

func (bm *BandwidthMonitor) monitoringLoop() {
	ticker := time.NewTicker(bm.config.MonitoringInterval)
	defer ticker.Stop()

	for {
		select {
		case <-bm.ctx.Done():
			return
		case <-ticker.C:
			bm.collectMetrics()
		}
	}
}

func (bm *BandwidthMonitor) collectMetrics() {
	bm.mu.RLock()
	interfaces := make([]*InterfaceMonitor, 0, len(bm.interfaces))
	for _, iface := range bm.interfaces {
		interfaces = append(interfaces, iface)
	}
	bm.mu.RUnlock()

	for _, iface := range interfaces {
		measurement, err := bm.measureInterface(iface.name)
		if err != nil {
			bm.logger.Warn("Failed to measure interface", 
				zap.String("interface", iface.name),
				zap.Error(err))
			continue
		}

		iface.mu.Lock()
		
		if iface.lastMeasure != nil {
			timeDelta := measurement.Timestamp.Sub(iface.lastMeasure.Timestamp).Seconds()
			if timeDelta > 0 {
				// Store instantaneous rates for metadata if needed - add nil check
				if measurement.Metadata == nil {
					measurement.Metadata = make(map[string]string)
				}
				instantRXRate := float64(measurement.RXBytes-iface.lastMeasure.RXBytes) * 8 / timeDelta
				instantTXRate := float64(measurement.TXBytes-iface.lastMeasure.TXBytes) * 8 / timeDelta
				measurement.Metadata["instant_rx_rate"] = fmt.Sprintf("%.2f", instantRXRate)
				measurement.Metadata["instant_tx_rate"] = fmt.Sprintf("%.2f", instantTXRate)
				
				// Initially set to instantaneous rates, will be overridden by smoothed rates
				measurement.RXRate = instantRXRate
				measurement.TXRate = instantTXRate
				
				if measurement.LinkSpeed > 0 {
					totalRate := measurement.RXRate + measurement.TXRate
					measurement.Utilization = (totalRate / float64(measurement.LinkSpeed)) * 100
				}
			}
		}

		iface.measurements = append(iface.measurements, *measurement)
		// Prune old entries based on HistoryRetention
		if bm.config.HistoryRetention > 0 {
			cutoff := time.Now().Add(-bm.config.HistoryRetention)
			var pruned []BandwidthMeasurement
			for _, m := range iface.measurements {
				if m.Timestamp.After(cutoff) {
					pruned = append(pruned, m)
				}
			}
			iface.measurements = pruned
		}
		if len(iface.measurements) > iface.maxHistory {
			iface.measurements = iface.measurements[1:]
		}
		
		// Calculate smoothed rates using sliding window
		if len(iface.measurements) > 1 {
			effectiveWindow := bm.config.SlidingWindowDuration
			if effectiveWindow == 0 {
				effectiveWindow = 3 * bm.config.MonitoringInterval
			}
			iface.mu.Unlock()
			rxbps, txbps := bm.windowedRate(iface, effectiveWindow)
			iface.mu.Lock()
			
			if rxbps > 0 || txbps > 0 {
				measurement.RXRate = rxbps
				measurement.TXRate = txbps
				
				if measurement.LinkSpeed > 0 {
					totalRate := measurement.RXRate + measurement.TXRate
					measurement.Utilization = (totalRate / float64(measurement.LinkSpeed)) * 100
				}
			}
		}
		
		iface.lastMeasure = measurement
		iface.mu.Unlock()

		bm.checkThresholds(iface.name, measurement)
	}
}

func (bm *BandwidthMonitor) measureInterface(ifaceName string) (*BandwidthMeasurement, error) {
	measurement := &BandwidthMeasurement{
		InterfaceName: ifaceName,
		Timestamp:     time.Now(),
		Metadata:      make(map[string]string),
	}

	stats, err := bm.getInterfaceStats(ifaceName)
	if err != nil {
		return nil, fmt.Errorf("failed to get interface stats: %w", err)
	}

	measurement.RXBytes = stats["rx_bytes"]
	measurement.TXBytes = stats["tx_bytes"]
	measurement.RXPackets = stats["rx_packets"]
	measurement.TXPackets = stats["tx_packets"]
	measurement.RXErrors = stats["rx_errs"]
	measurement.TXErrors = stats["tx_errs"]
	measurement.RXDropped = stats["rx_drop"]
	measurement.TXDropped = stats["tx_drop"]

	link, err := netlink.LinkByName(ifaceName)
	if err != nil {
		bm.logger.Warn("Failed to get link info", 
			zap.String("interface", ifaceName),
			zap.Error(err))
	} else {
		attrs := link.Attrs()
		measurement.LinkStatus = attrs.Flags&net.FlagUp != 0
		
		// Get link speed from /sys/class/net/<iface>/speed
		speed, err := bm.getLinkSpeedBps(ifaceName)
		if err != nil {
			measurement.LinkSpeed = 1_000_000_000 // Default 1Gbps
		} else {
			measurement.LinkSpeed = speed
		}

		measurement.Metadata["mtu"] = strconv.Itoa(attrs.MTU)
		measurement.Metadata["mac"] = attrs.HardwareAddr.String()
		measurement.Metadata["index"] = strconv.Itoa(attrs.Index)
	}

	return measurement, nil
}

func (bm *BandwidthMonitor) getInterfaceStats(ifaceName string) (map[string]uint64, error) {
	file, err := os.Open("/proc/net/dev")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.Contains(line, ifaceName+":") {
			continue
		}

		parts := strings.Fields(line)
		if len(parts) < 17 {
			continue
		}

		stats := make(map[string]uint64)
		fields := []string{
			"rx_bytes", "rx_packets", "rx_errs", "rx_drop", "rx_fifo",
			"rx_frame", "rx_compressed", "rx_multicast",
			"tx_bytes", "tx_packets", "tx_errs", "tx_drop", "tx_fifo",
			"tx_colls", "tx_carrier", "tx_compressed",
		}

		for i, field := range fields {
			if i+1 < len(parts) {
				if val, err := strconv.ParseUint(parts[i+1], 10, 64); err == nil {
					stats[field] = val
				}
			}
		}

		return stats, nil
	}

	return nil, fmt.Errorf("interface %s not found in /proc/net/dev", ifaceName)
}

func (bm *BandwidthMonitor) checkThresholds(ifaceName string, measurement *BandwidthMeasurement) {
	bm.mu.RLock()
	iface := bm.interfaces[ifaceName]
	bm.mu.RUnlock()

	if iface == nil {
		return
	}

	iface.mu.RLock()
	thresholds := iface.thresholds
	iface.mu.RUnlock()

	// Rate limiting policy:
	// - Rate limit alerts to prevent spamming (default: 1 minute between similar alerts)
	// - Alert keys include interface name, alert category (util/abs), and severity
	// - This prevents conflation of different alert types while still providing effective rate limiting
	
	for _, threshold := range thresholds {
		if !threshold.Enabled {
			continue
		}
		
		// Apply thresholds where InterfaceName matches exactly OR is a wildcard "*"
		// Also support simple glob prefix matches (e.g., "eth*" matches "eth0", "eth1", etc.)
		matches := false
		if threshold.InterfaceName == ifaceName || threshold.InterfaceName == "*" {
			matches = true
		} else if strings.HasSuffix(threshold.InterfaceName, "*") {
			prefix := strings.TrimSuffix(threshold.InterfaceName, "*")
			if strings.HasPrefix(ifaceName, prefix) {
				matches = true
			}
		}
		
		if !matches {
			continue
		}

		var alert *BandwidthAlert
		var alertKey string
		
		// Check utilization thresholds
		if measurement.Utilization >= threshold.CriticalThreshold {
			alertKey = fmt.Sprintf("%s_util_critical", ifaceName)

			bm.alertsMutex.RLock()
			lastAlert, exists := bm.lastAlerts[alertKey]
			bm.alertsMutex.RUnlock()

			if !exists || time.Since(lastAlert) >= time.Minute {
				alert = &BandwidthAlert{
					ID:             fmt.Sprintf("bandwidth_critical_%s_%d", ifaceName, time.Now().Unix()),
					InterfaceName:  ifaceName,
					Severity:       "critical",
					Message:        fmt.Sprintf("Critical bandwidth utilization on %s: %.2f%% (threshold: %.2f%%)", ifaceName, measurement.Utilization, threshold.CriticalThreshold),
					Timestamp:      time.Now(),
					CurrentValue:   measurement.Utilization,
					ThresholdValue: threshold.CriticalThreshold,
					Metadata: map[string]interface{}{
						"rx_rate": measurement.RXRate,
						"tx_rate": measurement.TXRate,
						"link_speed": measurement.LinkSpeed,
					},
				}
			}
		} else if measurement.Utilization >= threshold.WarningThreshold {
			alertKey = fmt.Sprintf("%s_util_warning", ifaceName)

			bm.alertsMutex.RLock()
			lastAlert, exists := bm.lastAlerts[alertKey]
			bm.alertsMutex.RUnlock()

			if !exists || time.Since(lastAlert) >= time.Minute {
				alert = &BandwidthAlert{
					ID:             fmt.Sprintf("bandwidth_warning_%s_%d", ifaceName, time.Now().Unix()),
					InterfaceName:  ifaceName,
					Severity:       "warning",
					Message:        fmt.Sprintf("Warning bandwidth utilization on %s: %.2f%% (threshold: %.2f%%)", ifaceName, measurement.Utilization, threshold.WarningThreshold),
					Timestamp:      time.Now(),
					CurrentValue:   measurement.Utilization,
					ThresholdValue: threshold.WarningThreshold,
					Metadata: map[string]interface{}{
						"rx_rate": measurement.RXRate,
						"tx_rate": measurement.TXRate,
						"link_speed": measurement.LinkSpeed,
					},
				}
			}
		}
		
		if alert != nil {
			bm.handleAlert(alert)
			bm.alertsMutex.Lock()
			bm.lastAlerts[alertKey] = time.Now()
			bm.alertsMutex.Unlock()

			if bm.config.EnableQoSHooks && alert.Severity == "critical" {
				bm.triggerQoSHooks(ifaceName, measurement.Utilization)
			}
			// Reset for absolute limit check
			alert = nil
		}

		// Check absolute bandwidth limit if specified
		if threshold.AbsoluteLimit > 0 {
			currentBandwidth := uint64(measurement.RXRate + measurement.TXRate)
			if currentBandwidth > threshold.AbsoluteLimit {
				alertKey = fmt.Sprintf("%s_abs_critical", ifaceName)

				bm.alertsMutex.RLock()
				lastAlert, exists := bm.lastAlerts[alertKey]
				bm.alertsMutex.RUnlock()

				if !exists || time.Since(lastAlert) >= time.Minute {
					alert = &BandwidthAlert{
						ID:             fmt.Sprintf("bandwidth_absolute_%s_%d", ifaceName, time.Now().Unix()),
						InterfaceName:  ifaceName,
						Severity:       "critical",
						Message:        fmt.Sprintf("Absolute bandwidth limit exceeded on %s: %d bps (limit: %d bps)", ifaceName, currentBandwidth, threshold.AbsoluteLimit),
						Timestamp:      time.Now(),
						CurrentValue:   float64(currentBandwidth),
						ThresholdValue: float64(threshold.AbsoluteLimit),
						Metadata: map[string]interface{}{
							"rx_rate": measurement.RXRate,
							"tx_rate": measurement.TXRate,
							"link_speed": measurement.LinkSpeed,
						},
					}
					
					if alert != nil {
						bm.handleAlert(alert)
						bm.alertsMutex.Lock()
						bm.lastAlerts[alertKey] = time.Now()
						bm.alertsMutex.Unlock()

						if bm.config.EnableQoSHooks {
							bm.triggerQoSHooks(ifaceName, measurement.Utilization)
						}
					}
				}
			}
		}
	}
}

func (bm *BandwidthMonitor) handleAlert(alert *BandwidthAlert) {
	bm.logger.Warn("Bandwidth alert", 
		zap.String("interface", alert.InterfaceName),
		zap.String("severity", alert.Severity),
		zap.String("message", alert.Message),
		zap.Float64("utilization", alert.CurrentValue))

	for _, handler := range bm.alertHandlers {
		if err := handler.HandleAlert(alert); err != nil {
			bm.logger.Error("Failed to handle bandwidth alert", 
				zap.String("alert_id", alert.ID),
				zap.Error(err))
		}
	}
}

func (bm *BandwidthMonitor) triggerQoSHooks(ifaceName string, utilization float64) {
	for _, hook := range bm.qosHooks {
		hook(ifaceName, utilization)
	}
}

func (bm *BandwidthMonitor) discoverInterfaces() ([]string, error) {
	links, err := netlink.LinkList()
	if err != nil {
		return nil, err
	}

	var interfaces []string
	for _, link := range links {
		attrs := link.Attrs()
		
		if attrs.Flags&net.FlagLoopback != 0 {
			continue
		}

		if strings.HasPrefix(attrs.Name, "lo") ||
		   strings.HasPrefix(attrs.Name, "docker") ||
		   strings.HasPrefix(attrs.Name, "veth") {
			continue
		}

		interfaces = append(interfaces, attrs.Name)
	}

	return interfaces, nil
}

func (bm *BandwidthMonitor) GetCurrentMeasurement(ifaceName string) (*BandwidthMeasurement, error) {
	bm.mu.RLock()
	iface := bm.interfaces[ifaceName]
	bm.mu.RUnlock()

	if iface == nil {
		return nil, fmt.Errorf("interface %s not monitored", ifaceName)
	}

	iface.mu.RLock()
	defer iface.mu.RUnlock()

	if iface.lastMeasure == nil {
		return nil, fmt.Errorf("no measurements available for interface %s", ifaceName)
	}

	return iface.lastMeasure, nil
}

func (bm *BandwidthMonitor) GetHistoricalMeasurements(ifaceName string, since time.Time) ([]BandwidthMeasurement, error) {
	bm.mu.RLock()
	iface := bm.interfaces[ifaceName]
	bm.mu.RUnlock()

	if iface == nil {
		return nil, fmt.Errorf("interface %s not monitored", ifaceName)
	}

	iface.mu.RLock()
	defer iface.mu.RUnlock()

	var measurements []BandwidthMeasurement
	for _, measurement := range iface.measurements {
		if measurement.Timestamp.After(since) || measurement.Timestamp.Equal(since) {
			measurements = append(measurements, measurement)
		}
	}

	return measurements, nil
}

func (bm *BandwidthMonitor) GetAllCurrentMeasurements() map[string]*BandwidthMeasurement {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	measurements := make(map[string]*BandwidthMeasurement)
	for name, iface := range bm.interfaces {
		iface.mu.RLock()
		if iface.lastMeasure != nil {
			measurements[name] = iface.lastMeasure
		}
		iface.mu.RUnlock()
	}

	return measurements
}

func (bm *BandwidthMonitor) SetThreshold(ifaceName string, threshold BandwidthThreshold) error {
	bm.mu.RLock()
	iface := bm.interfaces[ifaceName]
	bm.mu.RUnlock()

	if iface == nil {
		return fmt.Errorf("interface %s not monitored", ifaceName)
	}

	iface.mu.Lock()
	defer iface.mu.Unlock()

	for i, existing := range iface.thresholds {
		if existing.InterfaceName == threshold.InterfaceName {
			iface.thresholds[i] = threshold
			return nil
		}
	}

	iface.thresholds = append(iface.thresholds, threshold)
	return nil
}

func (bm *BandwidthMonitor) AddQoSHook(hook func(string, float64)) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	bm.qosHooks = append(bm.qosHooks, hook)
}

func (bm *BandwidthMonitor) GetNetworkUtilizationSummary() map[string]float64 {
	utilization := make(map[string]float64)
	measurements := bm.GetAllCurrentMeasurements()
	
	for ifaceName, measurement := range measurements {
		utilization[ifaceName] = measurement.Utilization
	}
	
	return utilization
}

type DefaultAlertHandler struct {
	logger *zap.Logger
}

func NewDefaultAlertHandler(logger *zap.Logger) *DefaultAlertHandler {
	return &DefaultAlertHandler{logger: logger}
}

func (h *DefaultAlertHandler) HandleAlert(alert *BandwidthAlert) error {
	h.logger.Warn("Bandwidth Alert", 
		zap.String("id", alert.ID),
		zap.String("interface", alert.InterfaceName),
		zap.String("severity", alert.Severity),
		zap.String("message", alert.Message),
		zap.Float64("current_value", alert.CurrentValue),
		zap.Float64("threshold", alert.ThresholdValue))
	return nil
}

// getLinkSpeedBps reads the link speed from /sys/class/net/<iface>/speed
func (bm *BandwidthMonitor) getLinkSpeedBps(ifaceName string) (uint64, error) {
	speedFile := fmt.Sprintf("/sys/class/net/%s/speed", ifaceName)
	data, err := os.ReadFile(speedFile)
	if err != nil {
		return 0, err
	}
	
	speedStr := strings.TrimSpace(string(data))
	speedMbps, err := strconv.ParseUint(speedStr, 10, 64)
	if err != nil {
		return 0, err
	}
	
	// Convert from Mbps to bps
	return speedMbps * 1_000_000, nil
}

// windowedRate calculates average rates over a time window
func (bm *BandwidthMonitor) windowedRate(iface *InterfaceMonitor, window time.Duration) (rxbps, txbps float64) {
	iface.mu.RLock()
	defer iface.mu.RUnlock()
	
	if len(iface.measurements) < 2 {
		return 0, 0
	}
	
	cutoff := time.Now().Add(-window)
	var rxSum, txSum float64
	var count int
	
	for _, m := range iface.measurements {
		if m.Timestamp.After(cutoff) {
			rxSum += m.RXRate
			txSum += m.TXRate
			count++
		}
	}
	
	if count > 0 {
		return rxSum / float64(count), txSum / float64(count)
	}
	
	return 0, 0
}