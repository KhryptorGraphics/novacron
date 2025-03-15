package cloud

import (
	"context"
	"fmt"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

// NativeMetricCollector is responsible for collecting native metrics from cloud providers
type NativeMetricCollector struct {
	// Provider manager reference
	providerManager *ProviderManager

	// Cache settings
	metricCacheTTL time.Duration
}

// NewNativeMetricCollector creates a new native metric collector
func NewNativeMetricCollector(providerManager *ProviderManager, cacheTTL time.Duration) *NativeMetricCollector {
	return &NativeMetricCollector{
		providerManager: providerManager,
		metricCacheTTL:  cacheTTL,
	}
}

// EnhanceWithNativeMetrics enhances VM stats with provider-specific native metrics
func (c *NativeMetricCollector) EnhanceWithNativeMetrics(
	ctx context.Context,
	providerName string,
	vmID string,
	stats *monitoring.VMStats,
) (*monitoring.VMStats, error) {
	if stats == nil {
		return nil, fmt.Errorf("cannot enhance nil stats")
	}

	// Get the appropriate provider
	provider, err := c.providerManager.GetProvider(providerName)
	if err != nil {
		return nil, fmt.Errorf("failed to get provider %s: %w", providerName, err)
	}

	// Use different enhancement strategies based on provider type
	switch providerName {
	case "aws":
		return c.enhanceAWSMetrics(ctx, provider.(*AWSProvider), vmID, stats)
	case "azure":
		return c.enhanceAzureMetrics(ctx, provider.(*AzureProvider), vmID, stats)
	case "gcp":
		return c.enhanceGCPMetrics(ctx, provider.(*GCPProvider), vmID, stats)
	default:
		return stats, fmt.Errorf("unsupported provider for native metrics: %s", providerName)
	}
}

// enhanceAWSMetrics enhances metrics with AWS-specific data
func (c *NativeMetricCollector) enhanceAWSMetrics(
	ctx context.Context,
	provider *AWSProvider,
	vmID string,
	stats *monitoring.VMStats,
) (*monitoring.VMStats, error) {
	// Get AWS-specific metrics
	metrics, err := provider.GetInstanceMetrics(ctx, vmID)
	if err != nil {
		return stats, fmt.Errorf("failed to get AWS native metrics: %w", err)
	}

	// Convert to EnhancedCloudVMStats for easier manipulation
	enhancedStats := monitoring.ConvertInternalToEnhanced(stats)

	// Add or enhance metric values using AWS-specific metrics
	// CPU metrics
	if cpuUtilization, ok := metrics["CPUUtilization"]; ok {
		// Set base CPU utilization if we have a CPU struct
		if enhancedStats.CPU != nil {
			enhancedStats.CPU.Usage = cpuUtilization
		}

		// Add the raw CloudWatch metric to the native metrics
		enhancedStats.NativeMetrics["aws.cpu.utilization"] = cpuUtilization
	}

	// CPU credit metrics for burstable instances
	if cpuCreditUsage, ok := metrics["CPUCreditUsage"]; ok {
		enhancedStats.NativeMetrics["aws.cpu.credit_usage"] = cpuCreditUsage
	}

	if cpuCreditBalance, ok := metrics["CPUCreditBalance"]; ok {
		enhancedStats.NativeMetrics["aws.cpu.credit_balance"] = cpuCreditBalance
	}

	// Memory metrics
	if memoryUtilization, ok := metrics["MemoryUtilization"]; ok {
		if enhancedStats.Memory != nil {
			enhancedStats.Memory.Usage = memoryUtilization
			enhancedStats.Memory.UsagePercent = memoryUtilization
		}

		enhancedStats.NativeMetrics["aws.memory.utilization"] = memoryUtilization
	}

	// Disk metrics
	if diskReadOps, ok := metrics["DiskReadOps"]; ok {
		enhancedStats.NativeMetrics["aws.disk.read_ops"] = diskReadOps
	}

	if diskWriteOps, ok := metrics["DiskWriteOps"]; ok {
		enhancedStats.NativeMetrics["aws.disk.write_ops"] = diskWriteOps
	}

	if diskReadBytes, ok := metrics["DiskReadBytes"]; ok {
		enhancedStats.NativeMetrics["aws.disk.read_bytes"] = diskReadBytes
	}

	if diskWriteBytes, ok := metrics["DiskWriteBytes"]; ok {
		enhancedStats.NativeMetrics["aws.disk.write_bytes"] = diskWriteBytes
	}

	// Network metrics
	if networkIn, ok := metrics["NetworkIn"]; ok {
		enhancedStats.NativeMetrics["aws.network.in_bytes"] = networkIn
	}

	if networkOut, ok := metrics["NetworkOut"]; ok {
		enhancedStats.NativeMetrics["aws.network.out_bytes"] = networkOut
	}

	if networkPacketsIn, ok := metrics["NetworkPacketsIn"]; ok {
		enhancedStats.NativeMetrics["aws.network.packets_in"] = networkPacketsIn
	}

	if networkPacketsOut, ok := metrics["NetworkPacketsOut"]; ok {
		enhancedStats.NativeMetrics["aws.network.packets_out"] = networkPacketsOut
	}

	// EBS-specific metrics
	if volumeReadOps, ok := metrics["VolumeReadOps"]; ok {
		enhancedStats.NativeMetrics["aws.ebs.read_ops"] = volumeReadOps
	}

	if volumeWriteOps, ok := metrics["VolumeWriteOps"]; ok {
		enhancedStats.NativeMetrics["aws.ebs.write_ops"] = volumeWriteOps
	}

	if volumeQueueLength, ok := metrics["VolumeQueueLength"]; ok {
		enhancedStats.NativeMetrics["aws.ebs.queue_length"] = volumeQueueLength
	}

	// Add provider metadata
	if enhancedStats.Metadata == nil {
		enhancedStats.Metadata = make(map[string]string)
	}
	enhancedStats.Metadata["native_metrics_source"] = "aws_cloudwatch"
	enhancedStats.Metadata["enhanced_timestamp"] = time.Now().UTC().Format(time.RFC3339)

	// Convert back to VMStats and return
	return monitoring.ConvertEnhancedToInternal(enhancedStats), nil
}

// enhanceAzureMetrics enhances metrics with Azure-specific data
func (c *NativeMetricCollector) enhanceAzureMetrics(
	ctx context.Context,
	provider *AzureProvider,
	vmID string,
	stats *monitoring.VMStats,
) (*monitoring.VMStats, error) {
	// Get Azure-specific metrics
	metrics, err := provider.GetVMMetrics(ctx, vmID)
	if err != nil {
		return stats, fmt.Errorf("failed to get Azure native metrics: %w", err)
	}

	// Convert to EnhancedCloudVMStats for easier manipulation
	enhancedStats := monitoring.ConvertInternalToEnhanced(stats)

	// Add Azure-specific CPU metrics
	if cpuPercentage, ok := metrics["Percentage CPU"]; ok {
		// Set base CPU utilization
		if enhancedStats.CPU != nil {
			enhancedStats.CPU.Usage = cpuPercentage
		}
		enhancedStats.NativeMetrics["azure.cpu.percentage"] = cpuPercentage
	}

	// Memory metrics
	if availableMemory, ok := metrics["Available Memory Bytes"]; ok {
		enhancedStats.NativeMetrics["azure.memory.available_bytes"] = availableMemory
	}

	if vmMemory, ok := metrics["VM Memory"]; ok {
		enhancedStats.NativeMetrics["azure.memory.total_bytes"] = vmMemory
	}

	// Calculate memory utilization if both metrics are available
	if availableMemory, availableOk := metrics["Available Memory Bytes"]; availableOk {
		if vmMemory, totalOk := metrics["VM Memory"]; totalOk && vmMemory > 0 {
			memoryUtilization := 100 * (1 - (availableMemory / vmMemory))
			enhancedStats.NativeMetrics["azure.memory.utilization_percent"] = memoryUtilization

			// Update memory stats
			if enhancedStats.Memory != nil {
				enhancedStats.Memory.Usage = memoryUtilization
				enhancedStats.Memory.UsagePercent = memoryUtilization
			}
		}
	}

	// Disk metrics
	if diskReadOps, ok := metrics["Disk Read Operations/Sec"]; ok {
		enhancedStats.NativeMetrics["azure.disk.read_ops_per_second"] = diskReadOps
	}

	if diskWriteOps, ok := metrics["Disk Write Operations/Sec"]; ok {
		enhancedStats.NativeMetrics["azure.disk.write_ops_per_second"] = diskWriteOps
	}

	if diskReadBytes, ok := metrics["Disk Read Bytes/Sec"]; ok {
		enhancedStats.NativeMetrics["azure.disk.read_bytes_per_second"] = diskReadBytes
	}

	if diskWriteBytes, ok := metrics["Disk Write Bytes/Sec"]; ok {
		enhancedStats.NativeMetrics["azure.disk.write_bytes_per_second"] = diskWriteBytes
	}

	// Disk latency metrics
	if diskReadLatency, ok := metrics["Disk Read Latency"]; ok {
		enhancedStats.NativeMetrics["azure.disk.read_latency_ms"] = diskReadLatency
	}

	if diskWriteLatency, ok := metrics["Disk Write Latency"]; ok {
		enhancedStats.NativeMetrics["azure.disk.write_latency_ms"] = diskWriteLatency
	}

	// Network metrics
	if networkInTotal, ok := metrics["Network In Total"]; ok {
		enhancedStats.NativeMetrics["azure.network.in_bytes_total"] = networkInTotal
	}

	if networkOutTotal, ok := metrics["Network Out Total"]; ok {
		enhancedStats.NativeMetrics["azure.network.out_bytes_total"] = networkOutTotal
	}

	// Add provider metadata
	if enhancedStats.Metadata == nil {
		enhancedStats.Metadata = make(map[string]string)
	}
	enhancedStats.Metadata["native_metrics_source"] = "azure_insights"
	enhancedStats.Metadata["enhanced_timestamp"] = time.Now().UTC().Format(time.RFC3339)

	// Convert back to VMStats and return
	return monitoring.ConvertEnhancedToInternal(enhancedStats), nil
}

// enhanceGCPMetrics enhances metrics with GCP-specific data
func (c *NativeMetricCollector) enhanceGCPMetrics(
	ctx context.Context,
	provider *GCPProvider,
	vmID string,
	stats *monitoring.VMStats,
) (*monitoring.VMStats, error) {
	// Get GCP-specific metrics
	metrics, err := provider.GetInstanceMetrics(ctx, vmID)
	if err != nil {
		return stats, fmt.Errorf("failed to get GCP native metrics: %w", err)
	}

	// Convert to EnhancedCloudVMStats for easier manipulation
	enhancedStats := monitoring.ConvertInternalToEnhanced(stats)

	// CPU utilization (GCP returns as 0-1 value)
	if cpuUtilization, ok := metrics["cpu/utilization"]; ok {
		// Set base CPU utilization (convert to percentage)
		if enhancedStats.CPU != nil {
			enhancedStats.CPU.Usage = cpuUtilization * 100
		}
		enhancedStats.NativeMetrics["gcp.cpu.utilization"] = cpuUtilization
		enhancedStats.NativeMetrics["gcp.cpu.utilization_percent"] = cpuUtilization * 100
	}

	// Memory metrics
	if memoryUsedBytes, ok := metrics["memory/used_bytes"]; ok {
		enhancedStats.NativeMetrics["gcp.memory.used_bytes"] = memoryUsedBytes

		// Update memory stats if available
		if memoryTotalBytes, totalOk := metrics["memory/total_bytes"]; totalOk && memoryTotalBytes > 0 {
			memoryUtilization := 100 * (memoryUsedBytes / memoryTotalBytes)
			enhancedStats.NativeMetrics["gcp.memory.utilization_percent"] = memoryUtilization

			// Update memory stats
			if enhancedStats.Memory != nil {
				enhancedStats.Memory.Usage = memoryUtilization
				enhancedStats.Memory.UsagePercent = memoryUtilization
			}
		}
	}

	if memoryTotalBytes, ok := metrics["memory/total_bytes"]; ok {
		enhancedStats.NativeMetrics["gcp.memory.total_bytes"] = memoryTotalBytes
	}

	// Disk metrics
	if diskReadOps, ok := metrics["disk/read_ops_count"]; ok {
		enhancedStats.NativeMetrics["gcp.disk.read_ops_count"] = diskReadOps
	}

	if diskWriteOps, ok := metrics["disk/write_ops_count"]; ok {
		enhancedStats.NativeMetrics["gcp.disk.write_ops_count"] = diskWriteOps
	}

	if diskReadBytes, ok := metrics["disk/read_bytes_count"]; ok {
		enhancedStats.NativeMetrics["gcp.disk.read_bytes_count"] = diskReadBytes
	}

	if diskWriteBytes, ok := metrics["disk/write_bytes_count"]; ok {
		enhancedStats.NativeMetrics["gcp.disk.write_bytes_count"] = diskWriteBytes
	}

	// Network metrics
	if networkRxBytes, ok := metrics["network/received_bytes_count"]; ok {
		enhancedStats.NativeMetrics["gcp.network.received_bytes_count"] = networkRxBytes
	}

	if networkTxBytes, ok := metrics["network/sent_bytes_count"]; ok {
		enhancedStats.NativeMetrics["gcp.network.sent_bytes_count"] = networkTxBytes
	}

	if networkRxPackets, ok := metrics["network/received_packets_count"]; ok {
		enhancedStats.NativeMetrics["gcp.network.received_packets_count"] = networkRxPackets
	}

	if networkTxPackets, ok := metrics["network/sent_packets_count"]; ok {
		enhancedStats.NativeMetrics["gcp.network.sent_packets_count"] = networkTxPackets
	}

	// Add GCP-specific metrics
	if firewallDropped, ok := metrics["firewall/dropped_packets_count"]; ok {
		enhancedStats.NativeMetrics["gcp.firewall.dropped_packets_count"] = firewallDropped
	}

	if uptime, ok := metrics["instance/uptime"]; ok {
		enhancedStats.NativeMetrics["gcp.instance.uptime_seconds"] = uptime
	}

	// Add provider metadata
	if enhancedStats.Metadata == nil {
		enhancedStats.Metadata = make(map[string]string)
	}
	enhancedStats.Metadata["native_metrics_source"] = "gcp_monitoring"
	enhancedStats.Metadata["enhanced_timestamp"] = time.Now().UTC().Format(time.RFC3339)

	// Convert back to VMStats and return
	return monitoring.ConvertEnhancedToInternal(enhancedStats), nil
}
