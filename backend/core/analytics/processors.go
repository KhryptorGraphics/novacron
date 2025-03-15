package analytics

import (
	"fmt"
	"time"

	"github.com/novacron/backend/core/monitoring"
)

// MetricCollector collects metrics from the monitoring system
type MetricCollector struct {
	// ID is the unique identifier for this processor
	ID string

	// Name is the human-readable name of the processor
	Name string

	// Description is a description of the processor
	Description string

	// MetricIDs are the IDs of metrics to collect
	MetricIDs []string

	// StartOffset is the offset from now for the start time
	StartOffset time.Duration

	// EndOffset is the offset from now for the end time
	EndOffset time.Duration

	// OutputPrefix is the prefix for the data keys
	OutputPrefix string

	// TimeResolution is the time resolution for aggregation
	TimeResolution time.Duration
}

// NewMetricCollector creates a new metric collector
func NewMetricCollector(id, name, description string, metricIDs []string) *MetricCollector {
	return &MetricCollector{
		ID:             id,
		Name:           name,
		Description:    description,
		MetricIDs:      metricIDs,
		StartOffset:    time.Hour * 24, // Default: last 24 hours
		EndOffset:      0,              // Default: now
		OutputPrefix:   "metrics.",
		TimeResolution: time.Minute * 5, // Default: 5-minute resolution
	}
}

// Process processes data and updates the context
func (p *MetricCollector) Process(ctx *PipelineContext) error {
	endTime := time.Now().Add(-p.EndOffset)
	startTime := endTime.Add(-p.StartOffset)

	// Process each metric ID
	for _, metricID := range p.MetricIDs {
		// Fetch the metric
		metric, err := ctx.MetricRegistry.GetMetric(metricID)
		if err != nil {
			return fmt.Errorf("failed to get metric %s: %w", metricID, err)
		}

		// Get metric values for the time range
		values := metric.GetValues(startTime, endTime)
		if len(values) == 0 {
			// No data for this metric
			continue
		}

		// Aggregate values based on time resolution if needed
		if p.TimeResolution > 0 {
			values = aggregateMetricValues(values, p.TimeResolution)
		}

		// Store in context
		dataKey := p.OutputPrefix + metricID
		ctx.Data[dataKey] = values

		// Also store the latest value for easy access
		ctx.Data[dataKey+".latest"] = values[len(values)-1]
	}

	return nil
}

// GetMetadata returns metadata about the processor
func (p *MetricCollector) GetMetadata() ProcessorMetadata {
	return ProcessorMetadata{
		ID:              p.ID,
		Name:            p.Name,
		Description:     p.Description,
		RequiredMetrics: p.MetricIDs,
		ProducedData:    getProducedDataKeys(p.OutputPrefix, p.MetricIDs),
	}
}

// VMResourceCollector collects VM resource utilization metrics
type VMResourceCollector struct {
	// ID is the unique identifier for this processor
	ID string

	// Name is the human-readable name of the processor
	Name string

	// Description is a description of the processor
	Description string

	// VMIDs are the IDs of VMs to collect metrics for
	VMIDs []string

	// StartOffset is the offset from now for the start time
	StartOffset time.Duration

	// EndOffset is the offset from now for the end time
	EndOffset time.Duration
}

// NewVMResourceCollector creates a new VM resource collector
func NewVMResourceCollector(id, name, description string, vmIDs []string) *VMResourceCollector {
	return &VMResourceCollector{
		ID:          id,
		Name:        name,
		Description: description,
		VMIDs:       vmIDs,
		StartOffset: time.Hour * 24, // Default: last 24 hours
		EndOffset:   0,              // Default: now
	}
}

// Process processes data and updates the context
func (p *VMResourceCollector) Process(ctx *PipelineContext) error {
	endTime := time.Now().Add(-p.EndOffset)
	startTime := endTime.Add(-p.StartOffset)

	// Prepare structure for VM resource data
	vmResources := make(map[string]map[string][]monitoring.MetricValue)

	// For each VM, collect CPU, memory, disk, and network metrics
	for _, vmID := range p.VMIDs {
		vmData := make(map[string][]monitoring.MetricValue)

		// Collect CPU metrics
		cpuMetricID := fmt.Sprintf("vm.%s.cpu.usage", vmID)
		cpuMetric, err := ctx.MetricRegistry.GetMetric(cpuMetricID)
		if err == nil {
			vmData["cpu"] = cpuMetric.GetValues(startTime, endTime)
		}

		// Collect memory metrics
		memMetricID := fmt.Sprintf("vm.%s.memory.usage", vmID)
		memMetric, err := ctx.MetricRegistry.GetMetric(memMetricID)
		if err == nil {
			vmData["memory"] = memMetric.GetValues(startTime, endTime)
		}

		// Collect disk metrics
		diskMetricID := fmt.Sprintf("vm.%s.disk.usage", vmID)
		diskMetric, err := ctx.MetricRegistry.GetMetric(diskMetricID)
		if err == nil {
			vmData["disk"] = diskMetric.GetValues(startTime, endTime)
		}

		// Collect network metrics
		netMetricID := fmt.Sprintf("vm.%s.network.throughput", vmID)
		netMetric, err := ctx.MetricRegistry.GetMetric(netMetricID)
		if err == nil {
			vmData["network"] = netMetric.GetValues(startTime, endTime)
		}

		vmResources[vmID] = vmData
	}

	// Store in context
	ctx.Data["vm.resources"] = vmResources

	return nil
}

// GetMetadata returns metadata about the processor
func (p *VMResourceCollector) GetMetadata() ProcessorMetadata {
	requiredMetrics := make([]string, 0)
	for _, vmID := range p.VMIDs {
		requiredMetrics = append(requiredMetrics,
			fmt.Sprintf("vm.%s.cpu.usage", vmID),
			fmt.Sprintf("vm.%s.memory.usage", vmID),
			fmt.Sprintf("vm.%s.disk.usage", vmID),
			fmt.Sprintf("vm.%s.network.throughput", vmID),
		)
	}

	return ProcessorMetadata{
		ID:              p.ID,
		Name:            p.Name,
		Description:     p.Description,
		RequiredMetrics: requiredMetrics,
		ProducedData:    []string{"vm.resources"},
	}
}

// SystemLoadProcessor processes system load metrics
type SystemLoadProcessor struct {
	// ID is the unique identifier for this processor
	ID string

	// Name is the human-readable name of the processor
	Name string

	// Description is a description of the processor
	Description string

	// NodeIDs are the IDs of nodes to collect metrics for
	NodeIDs []string
}

// NewSystemLoadProcessor creates a new system load processor
func NewSystemLoadProcessor(id, name, description string, nodeIDs []string) *SystemLoadProcessor {
	return &SystemLoadProcessor{
		ID:          id,
		Name:        name,
		Description: description,
		NodeIDs:     nodeIDs,
	}
}

// Process processes data and updates the context
func (p *SystemLoadProcessor) Process(ctx *PipelineContext) error {
	// Get the last hour of data
	endTime := time.Now()
	startTime := endTime.Add(-time.Hour)

	// Process each node
	nodesLoad := make(map[string]map[string]interface{})

	for _, nodeID := range p.NodeIDs {
		nodeLoad := make(map[string]interface{})

		// Get CPU metrics
		cpuMetricID := fmt.Sprintf("node.%s.cpu.usage", nodeID)
		cpuMetric, err := ctx.MetricRegistry.GetMetric(cpuMetricID)
		if err == nil {
			cpuValues := cpuMetric.GetValues(startTime, endTime)
			if len(cpuValues) > 0 {
				// Calculate average CPU load
				sum := 0.0
				for _, v := range cpuValues {
					sum += v.Value
				}
				nodeLoad["cpu.average"] = sum / float64(len(cpuValues))

				// Calculate peak CPU load
				peak := 0.0
				for _, v := range cpuValues {
					if v.Value > peak {
						peak = v.Value
					}
				}
				nodeLoad["cpu.peak"] = peak
			}
		}

		// Get memory metrics
		memMetricID := fmt.Sprintf("node.%s.memory.usage", nodeID)
		memMetric, err := ctx.MetricRegistry.GetMetric(memMetricID)
		if err == nil {
			memValues := memMetric.GetValues(startTime, endTime)
			if len(memValues) > 0 {
				// Calculate average memory usage
				sum := 0.0
				for _, v := range memValues {
					sum += v.Value
				}
				nodeLoad["memory.average"] = sum / float64(len(memValues))

				// Calculate peak memory usage
				peak := 0.0
				for _, v := range memValues {
					if v.Value > peak {
						peak = v.Value
					}
				}
				nodeLoad["memory.peak"] = peak
			}
		}

		nodesLoad[nodeID] = nodeLoad
	}

	// Store in context
	ctx.Data["system.load"] = nodesLoad

	return nil
}

// GetMetadata returns metadata about the processor
func (p *SystemLoadProcessor) GetMetadata() ProcessorMetadata {
	requiredMetrics := make([]string, 0)
	for _, nodeID := range p.NodeIDs {
		requiredMetrics = append(requiredMetrics,
			fmt.Sprintf("node.%s.cpu.usage", nodeID),
			fmt.Sprintf("node.%s.memory.usage", nodeID),
		)
	}

	return ProcessorMetadata{
		ID:              p.ID,
		Name:            p.Name,
		Description:     p.Description,
		RequiredMetrics: requiredMetrics,
		ProducedData:    []string{"system.load"},
	}
}

// Helper function to aggregate metric values based on time resolution
func aggregateMetricValues(values []monitoring.MetricValue, resolution time.Duration) []monitoring.MetricValue {
	if len(values) == 0 {
		return values
	}

	// Group values by time bucket
	buckets := make(map[int64][]float64)
	for _, v := range values {
		// Calculate bucket timestamp
		bucketTime := v.Timestamp.Unix() / int64(resolution.Seconds()) * int64(resolution.Seconds())
		buckets[bucketTime] = append(buckets[bucketTime], v.Value)
	}

	// Aggregate values in each bucket
	result := make([]monitoring.MetricValue, 0, len(buckets))
	for bucketTime, bucketValues := range buckets {
		// Calculate average
		sum := 0.0
		for _, v := range bucketValues {
			sum += v
		}
		average := sum / float64(len(bucketValues))

		// Create aggregated value
		aggregatedValue := monitoring.MetricValue{
			Timestamp: time.Unix(bucketTime, 0),
			Value:     average,
		}
		result = append(result, aggregatedValue)
	}

	return result
}

// Helper function to generate produced data keys from metric IDs
func getProducedDataKeys(prefix string, metricIDs []string) []string {
	keys := make([]string, 0, len(metricIDs)*2)
	for _, metricID := range metricIDs {
		keys = append(keys, prefix+metricID)
		keys = append(keys, prefix+metricID+".latest")
	}
	return keys
}
