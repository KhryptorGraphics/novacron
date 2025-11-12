// Package metrics provides HTTP exporter for Prometheus metrics
package metrics

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/rs/zerolog/log"
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
	log.Info().
		Int("port", e.port).
		Msg("Starting DWCP metrics exporter")

	go func() {
		if err := e.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Error().
				Err(err).
				Msg("Metrics exporter failed")
		}
	}()

	return nil
}

// Stop gracefully shuts down the exporter
func (e *Exporter) Stop(ctx context.Context) error {
	log.Info().Msg("Stopping DWCP metrics exporter")
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
