// Package metrics provides HTTP exporter for Prometheus metrics
package metrics

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Exporter handles HTTP metrics endpoint
type Exporter struct {
	server *http.Server
	port   int
}

// NewExporter creates a new metrics exporter
func NewExporter(port int) *Exporter {
	if port == 0 {
		port = 9090 // Default Prometheus metrics port
	}

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/health", healthHandler)
	mux.HandleFunc("/ready", readyHandler)

	return &Exporter{
		server: &http.Server{
			Addr:         fmt.Sprintf(":%d", port),
			Handler:      mux,
			ReadTimeout:  5 * time.Second,
			WriteTimeout: 10 * time.Second,
			IdleTimeout:  120 * time.Second,
		},
		port: port,
	}
}

// Start begins serving metrics
func (e *Exporter) Start() error {
	log.Printf("Starting DWCP metrics exporter on port %d", e.port)

	go func() {
		if err := e.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Metrics exporter failed: %v", err)
		}
	}()

	return nil
}

// Stop gracefully shuts down the exporter
func (e *Exporter) Stop(ctx context.Context) error {
	log.Printf("Stopping DWCP metrics exporter")
	return e.server.Shutdown(ctx)
}

// Port returns the configured port
func (e *Exporter) Port() int {
	return e.port
}

// healthHandler responds to health checks
func healthHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"healthy","service":"dwcp-metrics"}`))
}

// readyHandler responds to readiness checks
func readyHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ready","service":"dwcp-metrics"}`))
}
