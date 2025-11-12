package upgrade

import (
	"encoding/json"
	"hash/fnv"
	"os"
	"sync"
)

// DWCPFeatureFlags controls DWCP v3 feature rollout
type DWCPFeatureFlags struct {
	mu sync.RWMutex

	// Component-level flags (enable/disable individual v3 components)
	EnableV3Transport   bool `json:"enableV3Transport"`   // AMST v3
	EnableV3Compression bool `json:"enableV3Compression"` // HDE v3
	EnableV3Prediction  bool `json:"enableV3Prediction"`  // PBA v3
	EnableV3StateSync   bool `json:"enableV3StateSync"`   // ASS v3
	EnableV3Consensus   bool `json:"enableV3Consensus"`   // ACP v3
	EnableV3Placement   bool `json:"enableV3Placement"`   // ITP v3

	// Rollout control (gradual percentage-based rollout)
	V3RolloutPercentage int `json:"v3RolloutPercentage"` // 0-100%

	// Emergency killswitch (force all operations to v1)
	ForceV1Mode bool `json:"forceV1Mode"`

	// Advanced control
	EnableHybridMode    bool `json:"enableHybridMode"`    // Allow hybrid mode operation
	EnableModeDetection bool `json:"enableModeDetection"` // Auto-detect network mode
}

// Global feature flags instance
var (
	globalFlags = &DWCPFeatureFlags{
		EnableV3Transport:   false,
		EnableV3Compression: false,
		EnableV3Prediction:  false,
		EnableV3StateSync:   false,
		EnableV3Consensus:   false,
		EnableV3Placement:   false,
		V3RolloutPercentage: 0,
		ForceV1Mode:         false,
		EnableHybridMode:    true,  // Hybrid mode enabled by default
		EnableModeDetection: true,  // Auto-detection enabled by default
	}
	flagsMu sync.RWMutex
)

// GetFeatureFlags returns a copy of current feature flags
func GetFeatureFlags() *DWCPFeatureFlags {
	flagsMu.RLock()
	defer flagsMu.RUnlock()

	// Return copy to prevent external mutation
	return &DWCPFeatureFlags{
		EnableV3Transport:   globalFlags.EnableV3Transport,
		EnableV3Compression: globalFlags.EnableV3Compression,
		EnableV3Prediction:  globalFlags.EnableV3Prediction,
		EnableV3StateSync:   globalFlags.EnableV3StateSync,
		EnableV3Consensus:   globalFlags.EnableV3Consensus,
		EnableV3Placement:   globalFlags.EnableV3Placement,
		V3RolloutPercentage: globalFlags.V3RolloutPercentage,
		ForceV1Mode:         globalFlags.ForceV1Mode,
		EnableHybridMode:    globalFlags.EnableHybridMode,
		EnableModeDetection: globalFlags.EnableModeDetection,
	}
}

// UpdateFeatureFlags updates feature flags (hot-reload, no restart required)
func UpdateFeatureFlags(flags *DWCPFeatureFlags) {
	flagsMu.Lock()
	defer flagsMu.Unlock()

	globalFlags.EnableV3Transport = flags.EnableV3Transport
	globalFlags.EnableV3Compression = flags.EnableV3Compression
	globalFlags.EnableV3Prediction = flags.EnableV3Prediction
	globalFlags.EnableV3StateSync = flags.EnableV3StateSync
	globalFlags.EnableV3Consensus = flags.EnableV3Consensus
	globalFlags.EnableV3Placement = flags.EnableV3Placement
	globalFlags.V3RolloutPercentage = flags.V3RolloutPercentage
	globalFlags.ForceV1Mode = flags.ForceV1Mode
	globalFlags.EnableHybridMode = flags.EnableHybridMode
	globalFlags.EnableModeDetection = flags.EnableModeDetection

	// TODO: Add structured logging
	// log.Infof("Feature flags updated: rollout=%d%%, v3transport=%v, forceV1=%v",
	//     flags.V3RolloutPercentage, flags.EnableV3Transport, flags.ForceV1Mode)
}

// ShouldUseV3 determines if v3 should be used based on rollout percentage
// Uses consistent hashing to ensure same node always gets same decision
func ShouldUseV3(nodeID string) bool {
	flags := GetFeatureFlags()

	// Emergency killswitch - force v1 for all
	if flags.ForceV1Mode {
		return false
	}

	// 0% rollout - use v1 for all
	if flags.V3RolloutPercentage == 0 {
		return false
	}

	// 100% rollout - use v3 for all
	if flags.V3RolloutPercentage == 100 {
		return true
	}

	// Gradual rollout using consistent hashing
	// Same node ID always maps to same bucket (0-99)
	bucket := hashToBucket(nodeID)
	return bucket < flags.V3RolloutPercentage
}

// hashToBucket maps a string to a bucket (0-99) using FNV-1a hash
func hashToBucket(s string) int {
	h := fnv.New32a()
	h.Write([]byte(s))
	return int(h.Sum32() % 100)
}

// IsComponentEnabled checks if a specific v3 component is enabled
func IsComponentEnabled(component string) bool {
	flags := GetFeatureFlags()

	// Emergency killswitch
	if flags.ForceV1Mode {
		return false
	}

	switch component {
	case "transport", "amst":
		return flags.EnableV3Transport
	case "compression", "hde":
		return flags.EnableV3Compression
	case "prediction", "pba":
		return flags.EnableV3Prediction
	case "statesync", "ass":
		return flags.EnableV3StateSync
	case "consensus", "acp":
		return flags.EnableV3Consensus
	case "placement", "itp":
		return flags.EnableV3Placement
	default:
		return false
	}
}

// LoadFromFile loads feature flags from JSON file
func LoadFromFile(filename string) error {
	data, err := os.ReadFile(filename)
	if err != nil {
		return err
	}

	var flags DWCPFeatureFlags
	if err := json.Unmarshal(data, &flags); err != nil {
		return err
	}

	UpdateFeatureFlags(&flags)
	return nil
}

// SaveToFile saves current feature flags to JSON file
func SaveToFile(filename string) error {
	flags := GetFeatureFlags()
	data, err := json.MarshalIndent(flags, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(filename, data, 0644)
}

// EnableAll enables all v3 components with given rollout percentage
func EnableAll(rolloutPercentage int) {
	if rolloutPercentage < 0 {
		rolloutPercentage = 0
	}
	if rolloutPercentage > 100 {
		rolloutPercentage = 100
	}

	UpdateFeatureFlags(&DWCPFeatureFlags{
		EnableV3Transport:   true,
		EnableV3Compression: true,
		EnableV3Prediction:  true,
		EnableV3StateSync:   true,
		EnableV3Consensus:   true,
		EnableV3Placement:   true,
		V3RolloutPercentage: rolloutPercentage,
		ForceV1Mode:         false,
		EnableHybridMode:    true,
		EnableModeDetection: true,
	})
}

// DisableAll forces all operations to v1 (emergency rollback)
func DisableAll() {
	UpdateFeatureFlags(&DWCPFeatureFlags{
		EnableV3Transport:   false,
		EnableV3Compression: false,
		EnableV3Prediction:  false,
		EnableV3StateSync:   false,
		EnableV3Consensus:   false,
		EnableV3Placement:   false,
		V3RolloutPercentage: 0,
		ForceV1Mode:         true,
		EnableHybridMode:    false,
		EnableModeDetection: false,
	})
}

// GetRolloutStats returns statistics about current rollout
func GetRolloutStats() map[string]interface{} {
	flags := GetFeatureFlags()

	return map[string]interface{}{
		"rolloutPercentage": flags.V3RolloutPercentage,
		"forceV1Mode":       flags.ForceV1Mode,
		"enabledComponents": map[string]bool{
			"transport":   flags.EnableV3Transport,
			"compression": flags.EnableV3Compression,
			"prediction":  flags.EnableV3Prediction,
			"statesync":   flags.EnableV3StateSync,
			"consensus":   flags.EnableV3Consensus,
			"placement":   flags.EnableV3Placement,
		},
		"hybridMode":    flags.EnableHybridMode,
		"modeDetection": flags.EnableModeDetection,
	}
}
