package testing

import (
	"fmt"
	"os/exec"
	"strings"
	"sync"
	"time"
)

// TrafficController manages Linux traffic control (tc) for network simulation
type TrafficController struct {
	interface_ string
	mu         sync.Mutex
	applied    map[string]bool // track applied rules
}

// NewTrafficController creates a new traffic controller
func NewTrafficController(iface string) *TrafficController {
	return &TrafficController{
		interface_: iface,
		applied:    make(map[string]bool),
	}
}

// ApplyLatency applies latency using netem
func (tc *TrafficController) ApplyLatency(latency time.Duration, jitter time.Duration) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	// Delete existing qdisc first
	tc.deleteQdisc()

	cmd := exec.Command("tc", "qdisc", "add", "dev", tc.interface_,
		"root", "netem", "delay",
		fmt.Sprintf("%dms", latency.Milliseconds()),
		fmt.Sprintf("%dms", jitter.Milliseconds()),
		"distribution", "normal")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to apply latency: %v, output: %s", err, output)
	}

	tc.applied["latency"] = true
	return nil
}

// ApplyBandwidth applies bandwidth limit using tbf (token bucket filter)
func (tc *TrafficController) ApplyBandwidth(limitMbps int) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	// Delete existing qdisc first
	tc.deleteQdisc()

	// Calculate burst size (recommended: 2 * MTU)
	burstKB := 32

	cmd := exec.Command("tc", "qdisc", "add", "dev", tc.interface_,
		"root", "tbf",
		"rate", fmt.Sprintf("%dmbit", limitMbps),
		"burst", fmt.Sprintf("%dkbit", burstKB),
		"latency", "400ms")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to apply bandwidth: %v, output: %s", err, output)
	}

	tc.applied["bandwidth"] = true
	return nil
}

// ApplyPacketLoss applies packet loss using netem
func (tc *TrafficController) ApplyPacketLoss(lossPercent float64) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	// Delete existing qdisc first
	tc.deleteQdisc()

	cmd := exec.Command("tc", "qdisc", "add", "dev", tc.interface_,
		"root", "netem", "loss",
		fmt.Sprintf("%.2f%%", lossPercent))

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to apply packet loss: %v, output: %s", err, output)
	}

	tc.applied["loss"] = true
	return nil
}

// ApplyComplex applies multiple network conditions using netem
func (tc *TrafficController) ApplyComplex(config *ComplexNetworkConfig) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	// Delete existing qdisc first
	tc.deleteQdisc()

	args := []string{"qdisc", "add", "dev", tc.interface_, "root", "netem"}

	// Add latency
	if config.Latency > 0 {
		args = append(args, "delay",
			fmt.Sprintf("%dms", config.Latency.Milliseconds()),
			fmt.Sprintf("%dms", config.Jitter.Milliseconds()))

		if config.LatencyDistribution != "" {
			args = append(args, "distribution", config.LatencyDistribution)
		}
	}

	// Add packet loss
	if config.PacketLoss > 0 {
		args = append(args, "loss", fmt.Sprintf("%.2f%%", config.PacketLoss))

		if config.PacketLossCorrelation > 0 {
			args = append(args, fmt.Sprintf("%.2f%%", config.PacketLossCorrelation))
		}
	}

	// Add packet corruption
	if config.PacketCorruption > 0 {
		args = append(args, "corrupt", fmt.Sprintf("%.2f%%", config.PacketCorruption))
	}

	// Add packet duplication
	if config.PacketDuplication > 0 {
		args = append(args, "duplicate", fmt.Sprintf("%.2f%%", config.PacketDuplication))
	}

	// Add packet reordering
	if config.PacketReordering > 0 {
		args = append(args, "reorder",
			fmt.Sprintf("%.2f%%", config.PacketReordering),
			fmt.Sprintf("%.2f%%", config.ReorderCorrelation))
	}

	// Add rate limiting (if needed)
	if config.RateLimit > 0 {
		args = append(args, "rate", fmt.Sprintf("%dmbit", config.RateLimit))
	}

	cmd := exec.Command("tc", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to apply complex config: %v, output: %s", err, output)
	}

	tc.applied["complex"] = true
	return nil
}

// ComplexNetworkConfig represents complex network conditions
type ComplexNetworkConfig struct {
	Latency               time.Duration
	Jitter                time.Duration
	LatencyDistribution   string  // normal, pareto, paretonormal
	PacketLoss            float64 // percentage
	PacketLossCorrelation float64 // percentage
	PacketCorruption      float64 // percentage
	PacketDuplication     float64 // percentage
	PacketReordering      float64 // percentage
	ReorderCorrelation    float64 // percentage
	RateLimit             int     // Mbps
}

// ApplyHTB applies Hierarchical Token Bucket for complex bandwidth shaping
func (tc *TrafficController) ApplyHTB(config *HTBConfig) error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	// Delete existing qdisc first
	tc.deleteQdisc()

	// Create root HTB qdisc
	cmd := exec.Command("tc", "qdisc", "add", "dev", tc.interface_,
		"root", "handle", "1:", "htb", "default", "30")
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to create HTB qdisc: %v, output: %s", err, output)
	}

	// Create root class
	cmd = exec.Command("tc", "class", "add", "dev", tc.interface_,
		"parent", "1:", "classid", "1:1", "htb",
		"rate", fmt.Sprintf("%dmbit", config.RootRate))
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("failed to create root class: %v, output: %s", err, output)
	}

	// Create child classes for different traffic types
	for i, childRate := range config.ChildRates {
		classID := fmt.Sprintf("1:%d", 10+i)
		cmd = exec.Command("tc", "class", "add", "dev", tc.interface_,
			"parent", "1:1", "classid", classID, "htb",
			"rate", fmt.Sprintf("%dmbit", childRate),
			"ceil", fmt.Sprintf("%dmbit", config.RootRate))
		if output, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("failed to create child class %s: %v, output: %s", classID, err, output)
		}

		// Add SFQ (Stochastic Fairness Queueing) to each class
		cmd = exec.Command("tc", "qdisc", "add", "dev", tc.interface_,
			"parent", classID, "handle", fmt.Sprintf("%d:", 10+i), "sfq", "perturb", "10")
		if output, err := cmd.CombinedOutput(); err != nil {
			return fmt.Errorf("failed to add SFQ to class %s: %v, output: %s", classID, err, output)
		}
	}

	tc.applied["htb"] = true
	return nil
}

// HTBConfig represents Hierarchical Token Bucket configuration
type HTBConfig struct {
	RootRate   int   // Mbps
	ChildRates []int // Mbps for each child class
}

// Reset removes all traffic control rules
func (tc *TrafficController) Reset() error {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	return tc.deleteQdisc()
}

// deleteQdisc removes the root qdisc (and all associated classes/filters)
func (tc *TrafficController) deleteQdisc() error {
	cmd := exec.Command("tc", "qdisc", "del", "dev", tc.interface_, "root")
	output, err := cmd.CombinedOutput()

	// Ignore "No such file or directory" error (means no qdisc exists)
	if err != nil && !strings.Contains(string(output), "No such file or directory") {
		return fmt.Errorf("failed to delete qdisc: %v, output: %s", err, output)
	}

	// Clear applied rules
	tc.applied = make(map[string]bool)
	return nil
}

// GetStatus returns the current tc configuration
func (tc *TrafficController) GetStatus() (string, error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	cmd := exec.Command("tc", "qdisc", "show", "dev", tc.interface_)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to get status: %v, output: %s", err, output)
	}

	return string(output), nil
}

// GetStatistics returns traffic statistics
func (tc *TrafficController) GetStatistics() (string, error) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	cmd := exec.Command("tc", "-s", "qdisc", "show", "dev", tc.interface_)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to get statistics: %v, output: %s", err, output)
	}

	return string(output), nil
}

// ApplyProfile applies a complete network profile
func (tc *TrafficController) ApplyProfile(profile *NetworkProfile) error {
	config := &ComplexNetworkConfig{
		Latency:               profile.Latency,
		Jitter:                profile.Jitter,
		LatencyDistribution:   profile.LatencyDistribution,
		PacketLoss:            profile.PacketLoss,
		PacketLossCorrelation: profile.PacketLossCorrelation,
		PacketCorruption:      profile.PacketCorruption,
		PacketDuplication:     profile.PacketDuplication,
		PacketReordering:      profile.PacketReordering,
		ReorderCorrelation:    profile.ReorderCorrelation,
		RateLimit:             profile.Bandwidth,
	}

	return tc.ApplyComplex(config)
}

// NetworkProfile represents a complete network profile
type NetworkProfile struct {
	Name                  string
	Latency               time.Duration
	Jitter                time.Duration
	LatencyDistribution   string
	PacketLoss            float64
	PacketLossCorrelation float64
	PacketCorruption      float64
	PacketDuplication     float64
	PacketReordering      float64
	ReorderCorrelation    float64
	Bandwidth             int // Mbps
}

// PredefinedProfiles returns common network profiles
func PredefinedProfiles() map[string]*NetworkProfile {
	return map[string]*NetworkProfile{
		"perfect": {
			Name:      "Perfect Network",
			Latency:   0,
			Jitter:    0,
			Bandwidth: 10000, // 10 Gbps
		},
		"lan": {
			Name:                "Local Area Network",
			Latency:             time.Millisecond * 1,
			Jitter:              time.Microsecond * 100,
			LatencyDistribution: "normal",
			PacketLoss:          0.01,
			Bandwidth:           1000, // 1 Gbps
		},
		"wan-low-latency": {
			Name:                "WAN Low Latency",
			Latency:             time.Millisecond * 20,
			Jitter:              time.Millisecond * 5,
			LatencyDistribution: "normal",
			PacketLoss:          0.1,
			Bandwidth:           1000, // 1 Gbps
		},
		"wan-high-latency": {
			Name:                "WAN High Latency",
			Latency:             time.Millisecond * 100,
			Jitter:              time.Millisecond * 20,
			LatencyDistribution: "pareto",
			PacketLoss:          0.5,
			Bandwidth:           100, // 100 Mbps
		},
		"transcontinental": {
			Name:                "Transcontinental Link",
			Latency:             time.Millisecond * 150,
			Jitter:              time.Millisecond * 30,
			LatencyDistribution: "pareto",
			PacketLoss:          1.0,
			PacketReordering:    0.5,
			Bandwidth:           100, // 100 Mbps
		},
		"satellite": {
			Name:                "Satellite Link",
			Latency:             time.Millisecond * 600,
			Jitter:              time.Millisecond * 50,
			LatencyDistribution: "normal",
			PacketLoss:          2.0,
			PacketReordering:    1.0,
			Bandwidth:           10, // 10 Mbps
		},
		"degraded": {
			Name:                  "Degraded Network",
			Latency:               time.Millisecond * 200,
			Jitter:                time.Millisecond * 100,
			LatencyDistribution:   "pareto",
			PacketLoss:            5.0,
			PacketLossCorrelation: 25.0,
			PacketCorruption:      0.1,
			PacketDuplication:     0.5,
			PacketReordering:      2.0,
			ReorderCorrelation:    25.0,
			Bandwidth:             10, // 10 Mbps
		},
	}
}
