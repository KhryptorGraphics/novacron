package tests

import "github.com/khryptorgraphics/novacron/backend/core/network/dwcp/upgrade"

type str = string

func setRolloutPercentage(component string, percentage int) {
	flags := upgrade.GetFeatureFlags()
	flags.V3RolloutPercentage = percentage
	flags.ForceV1Mode = false

	switch component {
	case "hde", "compression":
		flags.EnableV3Compression = percentage > 0
	case "amst", "transport":
		flags.EnableV3Transport = percentage > 0
	}

	upgrade.UpdateFeatureFlags(flags)
}

func intMetric(metrics map[string]interface{}, key string) int {
	switch value := metrics[key].(type) {
	case int:
		return value
	case int32:
		return int(value)
	case int64:
		return int(value)
	default:
		return 0
	}
}

func int64Metric(metrics map[string]interface{}, key string) int64 {
	switch value := metrics[key].(type) {
	case int:
		return int64(value)
	case int32:
		return int64(value)
	case int64:
		return value
	default:
		return 0
	}
}

func float64Metric(metrics map[string]interface{}, key string) float64 {
	switch value := metrics[key].(type) {
	case float64:
		return value
	case float32:
		return float64(value)
	case int:
		return float64(value)
	case int64:
		return float64(value)
	default:
		return 0
	}
}
