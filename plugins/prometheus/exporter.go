package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/novacron/dwcp-sdk-go"
)

// DWCPExporter collects metrics from DWCP and exposes them for Prometheus
type DWCPExporter struct {
	client *dwcp.Client

	// VM metrics
	vmCount        *prometheus.GaugeVec
	vmCPUUsage     *prometheus.GaugeVec
	vmMemoryUsed   *prometheus.GaugeVec
	vmMemoryTotal  *prometheus.GaugeVec
	vmDiskRead     *prometheus.CounterVec
	vmDiskWrite    *prometheus.CounterVec
	vmNetworkRx    *prometheus.CounterVec
	vmNetworkTx    *prometheus.CounterVec

	// Cluster metrics
	nodeCount      prometheus.Gauge
	nodeCapacity   *prometheus.GaugeVec
	nodeUsage      *prometheus.GaugeVec

	// Migration metrics
	migrationTotal        prometheus.Counter
	migrationDuration     prometheus.Histogram
	migrationDowntime     prometheus.Histogram
	migrationThroughput   prometheus.Histogram

	// Performance metrics
	apiLatency         *prometheus.HistogramVec
	apiErrors          *prometheus.CounterVec
}

// NewDWCPExporter creates a new DWCP exporter
func NewDWCPExporter(client *dwcp.Client) *DWCPExporter {
	return &DWCPExporter{
		client: client,

		// VM metrics
		vmCount: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "dwcp",
				Name:      "vm_count",
				Help:      "Number of VMs by state",
			},
			[]string{"state", "node"},
		),
		vmCPUUsage: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "dwcp",
				Name:      "vm_cpu_usage_percent",
				Help:      "VM CPU usage percentage",
			},
			[]string{"vm_id", "vm_name", "node"},
		),
		vmMemoryUsed: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "dwcp",
				Name:      "vm_memory_used_bytes",
				Help:      "VM memory used in bytes",
			},
			[]string{"vm_id", "vm_name", "node"},
		),
		vmMemoryTotal: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "dwcp",
				Name:      "vm_memory_total_bytes",
				Help:      "VM total memory in bytes",
			},
			[]string{"vm_id", "vm_name", "node"},
		),
		vmDiskRead: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "dwcp",
				Name:      "vm_disk_read_bytes_total",
				Help:      "Total disk bytes read",
			},
			[]string{"vm_id", "vm_name", "node"},
		),
		vmDiskWrite: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "dwcp",
				Name:      "vm_disk_write_bytes_total",
				Help:      "Total disk bytes written",
			},
			[]string{"vm_id", "vm_name", "node"},
		),
		vmNetworkRx: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "dwcp",
				Name:      "vm_network_receive_bytes_total",
				Help:      "Total network bytes received",
			},
			[]string{"vm_id", "vm_name", "node"},
		),
		vmNetworkTx: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "dwcp",
				Name:      "vm_network_transmit_bytes_total",
				Help:      "Total network bytes transmitted",
			},
			[]string{"vm_id", "vm_name", "node"},
		),

		// Cluster metrics
		nodeCount: prometheus.NewGauge(
			prometheus.GaugeOpts{
				Namespace: "dwcp",
				Name:      "node_count",
				Help:      "Number of cluster nodes",
			},
		),
		nodeCapacity: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "dwcp",
				Name:      "node_capacity",
				Help:      "Node capacity for resources",
			},
			[]string{"node", "resource"},
		),
		nodeUsage: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Namespace: "dwcp",
				Name:      "node_usage",
				Help:      "Node resource usage",
			},
			[]string{"node", "resource"},
		),

		// Migration metrics
		migrationTotal: prometheus.NewCounter(
			prometheus.CounterOpts{
				Namespace: "dwcp",
				Name:      "migration_total",
				Help:      "Total number of migrations",
			},
		),
		migrationDuration: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: "dwcp",
				Name:      "migration_duration_seconds",
				Help:      "Migration duration in seconds",
				Buckets:   []float64{1, 5, 10, 30, 60, 120, 300, 600},
			},
		),
		migrationDowntime: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: "dwcp",
				Name:      "migration_downtime_milliseconds",
				Help:      "Migration downtime in milliseconds",
				Buckets:   []float64{10, 50, 100, 500, 1000, 5000, 10000},
			},
		),
		migrationThroughput: prometheus.NewHistogram(
			prometheus.HistogramOpts{
				Namespace: "dwcp",
				Name:      "migration_throughput_mbps",
				Help:      "Migration throughput in MB/s",
				Buckets:   []float64{10, 50, 100, 500, 1000, 5000, 10000},
			},
		),

		// Performance metrics
		apiLatency: prometheus.NewHistogramVec(
			prometheus.HistogramOpts{
				Namespace: "dwcp",
				Name:      "api_latency_milliseconds",
				Help:      "API operation latency in milliseconds",
				Buckets:   []float64{1, 5, 10, 50, 100, 500, 1000},
			},
			[]string{"operation"},
		),
		apiErrors: prometheus.NewCounterVec(
			prometheus.CounterOpts{
				Namespace: "dwcp",
				Name:      "api_errors_total",
				Help:      "Total API errors",
			},
			[]string{"operation", "error_type"},
		),
	}
}

// Register registers all metrics with Prometheus
func (e *DWCPExporter) Register(registry *prometheus.Registry) {
	registry.MustRegister(
		e.vmCount,
		e.vmCPUUsage,
		e.vmMemoryUsed,
		e.vmMemoryTotal,
		e.vmDiskRead,
		e.vmDiskWrite,
		e.vmNetworkRx,
		e.vmNetworkTx,
		e.nodeCount,
		e.nodeCapacity,
		e.nodeUsage,
		e.migrationTotal,
		e.migrationDuration,
		e.migrationDowntime,
		e.migrationThroughput,
		e.apiLatency,
		e.apiErrors,
	)
}

// Collect collects metrics from DWCP
func (e *DWCPExporter) Collect(ctx context.Context) error {
	vmClient := e.client.VM()

	// List all VMs
	start := time.Now()
	vms, err := vmClient.List(ctx, nil)
	latency := time.Since(start).Milliseconds()
	e.apiLatency.WithLabelValues("vm_list").Observe(float64(latency))

	if err != nil {
		e.apiErrors.WithLabelValues("vm_list", "connection").Inc()
		return fmt.Errorf("failed to list VMs: %w", err)
	}

	// Reset VM count
	e.vmCount.Reset()

	// Count VMs by state
	stateCounts := make(map[string]map[string]int)
	for _, vm := range vms {
		if _, exists := stateCounts[string(vm.State)]; !exists {
			stateCounts[string(vm.State)] = make(map[string]int)
		}
		stateCounts[string(vm.State)][vm.Node]++

		// Update per-VM metrics
		e.vmMemoryTotal.WithLabelValues(vm.ID, vm.Name, vm.Node).Set(float64(vm.Config.Memory))

		if vm.Metrics != nil {
			e.vmCPUUsage.WithLabelValues(vm.ID, vm.Name, vm.Node).Set(vm.Metrics.CPUUsage)
			e.vmMemoryUsed.WithLabelValues(vm.ID, vm.Name, vm.Node).Set(float64(vm.Metrics.MemoryUsed))
			e.vmDiskRead.WithLabelValues(vm.ID, vm.Name, vm.Node).Add(float64(vm.Metrics.DiskRead))
			e.vmDiskWrite.WithLabelValues(vm.ID, vm.Name, vm.Node).Add(float64(vm.Metrics.DiskWrite))
			e.vmNetworkRx.WithLabelValues(vm.ID, vm.Name, vm.Node).Add(float64(vm.Metrics.NetworkRx))
			e.vmNetworkTx.WithLabelValues(vm.ID, vm.Name, vm.Node).Add(float64(vm.Metrics.NetworkTx))
		}
	}

	// Update state counts
	for state, nodeCounts := range stateCounts {
		for node, count := range nodeCounts {
			e.vmCount.WithLabelValues(state, node).Set(float64(count))
		}
	}

	return nil
}

// StartCollector starts the metrics collection loop
func (e *DWCPExporter) StartCollector(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := e.Collect(ctx); err != nil {
				log.Printf("Failed to collect metrics: %v", err)
			}
		}
	}
}

func main() {
	// Create DWCP client
	config := dwcp.DefaultConfig()
	config.Address = "localhost"
	config.Port = 9000
	config.APIKey = "prometheus-exporter"

	client, err := dwcp.NewClient(config)
	if err != nil {
		log.Fatalf("Failed to create DWCP client: %v", err)
	}

	ctx := context.Background()
	if err := client.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect to DWCP: %v", err)
	}

	// Create exporter
	exporter := NewDWCPExporter(client)

	// Register metrics
	registry := prometheus.NewRegistry()
	exporter.Register(registry)

	// Start collector
	go exporter.StartCollector(ctx, 15*time.Second)

	// Setup HTTP handler
	http.Handle("/metrics", promhttp.HandlerFor(registry, promhttp.HandlerOpts{}))
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	})

	// Start server
	addr := ":9090"
	log.Printf("Starting Prometheus exporter on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
