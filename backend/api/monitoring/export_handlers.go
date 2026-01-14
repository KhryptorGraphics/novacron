package monitoring

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/prometheus/common/expfmt"
	"github.com/khryptorgraphics/novacron/backend/core/monitoring"
)

var (
	exportRequestDuration = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Name:    "novacron_monitoring_export_duration_seconds",
		Help:    "Duration of monitoring export requests",
		Buckets: prometheus.DefBuckets,
	}, []string{"format", "type"})

	exportRequestCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_monitoring_export_requests_total",
		Help: "Total number of monitoring export requests",
	}, []string{"format", "type", "status"})

	reportGenerationCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_monitoring_reports_generated_total",
		Help: "Total number of monitoring reports generated",
	}, []string{"format", "type"})

	alertAcknowledgmentCount = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "novacron_monitoring_alerts_acknowledged_total",
		Help: "Total number of alerts acknowledged",
	}, []string{"alert_type", "severity"})
)

// ExportHandler handles monitoring export and reporting endpoints
type ExportHandler struct {
	alertManager   *monitoring.AlertManager
	metricRegistry *monitoring.MetricRegistry
	reportGenerator *monitoring.ReportGenerator
}

// NewExportHandler creates a new export handler
func NewExportHandler(alertManager *monitoring.AlertManager, metricRegistry *monitoring.MetricRegistry, reportGenerator *monitoring.ReportGenerator) *ExportHandler {
	return &ExportHandler{
		alertManager:   alertManager,
		metricRegistry: metricRegistry,
		reportGenerator: reportGenerator,
	}
}

// MetricExportRequest represents metric export parameters
type MetricExportRequest struct {
	Format     string    `json:"format"`         // prometheus, json
	StartTime  time.Time `json:"start_time"`
	EndTime    time.Time `json:"end_time"`
	Metrics    []string  `json:"metrics"`        // specific metrics to export
	Labels     map[string]string `json:"labels"` // label filters
	Resolution string    `json:"resolution"`     // 1m, 5m, 1h, etc.
}

// HistoryRequest represents historical metrics request
type HistoryRequest struct {
	MetricName string            `json:"metric_name"`
	StartTime  time.Time         `json:"start_time"`
	EndTime    time.Time         `json:"end_time"`
	Step       time.Duration     `json:"step"`
	Labels     map[string]string `json:"labels"`
	Aggregation string           `json:"aggregation"` // avg, max, min, sum
}

// AlertAcknowledgmentRequest represents alert acknowledgment request
type AlertAcknowledgmentRequest struct {
	AlertIDs []string `json:"alert_ids" validate:"required"`
	UserID   string   `json:"user_id" validate:"required"`
	Comment  string   `json:"comment,omitempty"`
	Until    *time.Time `json:"until,omitempty"` // acknowledge until specific time
}

// ReportGenerationRequest represents report generation request
type ReportGenerationRequest struct {
	Type        string            `json:"type" validate:"required"`        // system, vm, security, performance
	Format      string            `json:"format" validate:"required"`      // pdf, csv, json
	StartTime   time.Time         `json:"start_time" validate:"required"`
	EndTime     time.Time         `json:"end_time" validate:"required"`
	Recipients  []string          `json:"recipients,omitempty"`
	Template    string            `json:"template,omitempty"`
	Parameters  map[string]interface{} `json:"parameters,omitempty"`
	Schedule    *ReportSchedule   `json:"schedule,omitempty"`
}

// ReportSchedule represents recurring report schedule
type ReportSchedule struct {
	Frequency string `json:"frequency"` // daily, weekly, monthly
	Time      string `json:"time"`      // HH:MM format
	Timezone  string `json:"timezone"`
	Enabled   bool   `json:"enabled"`
}

// RegisterExportRoutes registers monitoring export and reporting routes
func (h *ExportHandler) RegisterExportRoutes(router *mux.Router, require func(string, http.HandlerFunc) http.Handler) {
	exportRouter := router.PathPrefix("/api/v1").Subrouter()

	// Metrics export (viewer+)
	exportRouter.Handle("/metrics/export", require("viewer", h.ExportMetrics)).Methods("GET", "POST")
	exportRouter.Handle("/metrics/history", require("viewer", h.GetMetricHistory)).Methods("GET")
	
	// Alert management (operator+)
	exportRouter.Handle("/alerts/acknowledge", require("operator", h.AcknowledgeAlerts)).Methods("POST")
	
	// Report generation (admin+)
	exportRouter.Handle("/reports/generate", require("admin", h.GenerateReport)).Methods("POST")
	exportRouter.Handle("/reports/{id}", require("viewer", h.GetReport)).Methods("GET")
	exportRouter.Handle("/reports", require("viewer", h.ListReports)).Methods("GET")
}

// ExportMetrics handles GET/POST /api/v1/metrics/export
// @Summary Export metrics
// @Description Export metrics in Prometheus or JSON format with filtering
// @Tags Monitoring
// @Accept json
// @Produce json,text/plain
// @Param format query string false "Export format" Enums(prometheus,json) default(prometheus)
// @Param start_time query string false "Start time (RFC3339)"
// @Param end_time query string false "End time (RFC3339)"
// @Param metrics query string false "Comma-separated metric names"
// @Param request body MetricExportRequest false "Export parameters (for POST)"
// @Success 200 {string} string "Exported metrics"
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/metrics/export [get]
// @Router /api/v1/metrics/export [post]
func (h *ExportHandler) ExportMetrics(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(exportRequestDuration.WithLabelValues("", "metrics"))
	defer timer.ObserveDuration()

	var req MetricExportRequest
	
	if r.Method == "POST" {
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			exportRequestCount.WithLabelValues("", "metrics", "error").Inc()
			writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
			return
		}
	} else {
		// Parse query parameters for GET request
		req = h.parseExportParams(r)
	}

	// Set defaults
	if req.Format == "" {
		req.Format = "prometheus"
	}
	if req.EndTime.IsZero() {
		req.EndTime = time.Now()
	}
	if req.StartTime.IsZero() {
		req.StartTime = req.EndTime.Add(-1 * time.Hour)
	}

	// Validate format
	if req.Format != "prometheus" && req.Format != "json" {
		exportRequestCount.WithLabelValues(req.Format, "metrics", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_FORMAT", "Format must be 'prometheus' or 'json'")
		return
	}

	// Validate time range
	if req.StartTime.After(req.EndTime) {
		exportRequestCount.WithLabelValues(req.Format, "metrics", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_TIME_RANGE", "Start time must be before end time")
		return
	}

	// Check time range limit (max 7 days)
	if req.EndTime.Sub(req.StartTime) > 7*24*time.Hour {
		exportRequestCount.WithLabelValues(req.Format, "metrics", "error").Inc()
		writeError(w, http.StatusBadRequest, "TIME_RANGE_TOO_LARGE", "Time range cannot exceed 7 days")
		return
	}

	// Export metrics based on format
	switch req.Format {
	case "prometheus":
		h.exportPrometheusFormat(w, r, req)
	case "json":
		h.exportJSONFormat(w, r, req)
	}
}

// GetMetricHistory handles GET /api/v1/metrics/history
// @Summary Get historical metrics
// @Description Retrieve historical metrics data with time range and aggregation
// @Tags Monitoring
// @Produce json
// @Param metric_name query string true "Metric name"
// @Param start_time query string true "Start time (RFC3339)"
// @Param end_time query string true "End time (RFC3339)"
// @Param step query string false "Time step (5m, 1h, etc.)" default(5m)
// @Param aggregation query string false "Aggregation function" Enums(avg,max,min,sum) default(avg)
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/metrics/history [get]
func (h *ExportHandler) GetMetricHistory(w http.ResponseWriter, r *http.Request) {
	timer := prometheus.NewTimer(exportRequestDuration.WithLabelValues("json", "history"))
	defer timer.ObserveDuration()

	metricName := r.URL.Query().Get("metric_name")
	if metricName == "" {
		exportRequestCount.WithLabelValues("json", "history", "error").Inc()
		writeError(w, http.StatusBadRequest, "MISSING_METRIC_NAME", "Metric name is required")
		return
	}

	startTimeStr := r.URL.Query().Get("start_time")
	endTimeStr := r.URL.Query().Get("end_time")

	startTime, err := time.Parse(time.RFC3339, startTimeStr)
	if err != nil {
		exportRequestCount.WithLabelValues("json", "history", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_START_TIME", "Invalid start time format")
		return
	}

	endTime, err := time.Parse(time.RFC3339, endTimeStr)
	if err != nil {
		exportRequestCount.WithLabelValues("json", "history", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_END_TIME", "Invalid end time format")
		return
	}

	stepStr := r.URL.Query().Get("step")
	if stepStr == "" {
		stepStr = "5m"
	}

	step, err := time.ParseDuration(stepStr)
	if err != nil {
		exportRequestCount.WithLabelValues("json", "history", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_STEP", "Invalid step duration format")
		return
	}

	aggregation := r.URL.Query().Get("aggregation")
	if aggregation == "" {
		aggregation = "avg"
	}

	// Validate aggregation function
	validAggregations := map[string]bool{"avg": true, "max": true, "min": true, "sum": true}
	if !validAggregations[aggregation] {
		exportRequestCount.WithLabelValues("json", "history", "error").Inc()
		writeError(w, http.StatusBadRequest, "INVALID_AGGREGATION", "Aggregation must be one of: avg, max, min, sum")
		return
	}

	// Get historical data
	history, err := h.metricRegistry.QueryRange(r.Context(), monitoring.QueryRangeRequest{
		MetricName:  metricName,
		StartTime:   startTime,
		EndTime:     endTime,
		Step:        step,
		Aggregation: aggregation,
	})
	if err != nil {
		exportRequestCount.WithLabelValues("json", "history", "error").Inc()
		writeError(w, http.StatusInternalServerError, "QUERY_ERROR", "Failed to query metric history")
		return
	}

	exportRequestCount.WithLabelValues("json", "history", "success").Inc()
	writeJSON(w, http.StatusOK, map[string]interface{}{
		"metric_name":  metricName,
		"start_time":   startTime,
		"end_time":     endTime,
		"step":         step.String(),
		"aggregation":  aggregation,
		"data":         history,
	})
}

// AcknowledgeAlerts handles POST /api/v1/alerts/acknowledge
// @Summary Acknowledge alerts
// @Description Acknowledge one or more alerts to stop notifications
// @Tags Monitoring
// @Accept json
// @Produce json
// @Param request body AlertAcknowledgmentRequest true "Acknowledgment request"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/alerts/acknowledge [post]
func (h *ExportHandler) AcknowledgeAlerts(w http.ResponseWriter, r *http.Request) {
	var req AlertAcknowledgmentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if len(req.AlertIDs) == 0 {
		writeError(w, http.StatusBadRequest, "MISSING_ALERT_IDS", "Alert IDs are required")
		return
	}

	if req.UserID == "" {
		writeError(w, http.StatusBadRequest, "MISSING_USER_ID", "User ID is required")
		return
	}

	// Process acknowledgments
	results := make(map[string]interface{})
	successful := 0
	failed := 0

	for _, alertID := range req.AlertIDs {
		ackReq := monitoring.AcknowledgmentRequest{
			AlertID: alertID,
			UserID:  req.UserID,
			Comment: req.Comment,
			Until:   req.Until,
		}

		if err := h.alertManager.AcknowledgeAlert(r.Context(), ackReq); err != nil {
			results[alertID] = map[string]string{
				"status": "failed",
				"error":  err.Error(),
			}
			failed++
		} else {
			results[alertID] = map[string]string{
				"status": "acknowledged",
			}
			successful++
			
			// Get alert details for metrics
			alert, _ := h.alertManager.GetAlert(r.Context(), alertID)
			alertType := "unknown"
			severity := "unknown"
			if alert != nil {
				alertType = alert.Type
				severity = alert.Severity
			}
			alertAcknowledgmentCount.WithLabelValues(alertType, severity).Inc()
		}
	}

	writeJSON(w, http.StatusOK, map[string]interface{}{
		"acknowledged": successful,
		"failed":       failed,
		"results":      results,
		"timestamp":    time.Now(),
	})
}

// GenerateReport handles POST /api/v1/reports/generate
// @Summary Generate monitoring report
// @Description Generate PDF or CSV reports for monitoring data
// @Tags Monitoring
// @Accept json
// @Produce json
// @Param request body ReportGenerationRequest true "Report generation request"
// @Success 202 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/reports/generate [post]
func (h *ExportHandler) GenerateReport(w http.ResponseWriter, r *http.Request) {
	var req ReportGenerationRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "INVALID_JSON", "Invalid JSON format")
		return
	}

	// Validate request
	if req.Type == "" {
		writeError(w, http.StatusBadRequest, "MISSING_TYPE", "Report type is required")
		return
	}

	if req.Format == "" {
		writeError(w, http.StatusBadRequest, "MISSING_FORMAT", "Report format is required")
		return
	}

	validTypes := map[string]bool{"system": true, "vm": true, "security": true, "performance": true}
	if !validTypes[req.Type] {
		writeError(w, http.StatusBadRequest, "INVALID_TYPE", "Report type must be one of: system, vm, security, performance")
		return
	}

	validFormats := map[string]bool{"pdf": true, "csv": true, "json": true}
	if !validFormats[req.Format] {
		writeError(w, http.StatusBadRequest, "INVALID_FORMAT", "Report format must be one of: pdf, csv, json")
		return
	}

	// Validate time range
	if req.StartTime.After(req.EndTime) {
		writeError(w, http.StatusBadRequest, "INVALID_TIME_RANGE", "Start time must be before end time")
		return
	}

	// Generate report asynchronously
	reportID, err := h.reportGenerator.GenerateReport(r.Context(), monitoring.ReportRequest{
		Type:        req.Type,
		Format:      req.Format,
		StartTime:   req.StartTime,
		EndTime:     req.EndTime,
		Recipients:  req.Recipients,
		Template:    req.Template,
		Parameters:  req.Parameters,
		Schedule:    h.convertReportSchedule(req.Schedule),
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "REPORT_ERROR", "Failed to initiate report generation")
		return
	}

	reportGenerationCount.WithLabelValues(req.Format, req.Type).Inc()

	writeJSON(w, http.StatusAccepted, map[string]interface{}{
		"report_id":   reportID,
		"status":      "generating",
		"type":        req.Type,
		"format":      req.Format,
		"started_at":  time.Now(),
		"message":     "Report generation initiated",
	})
}

// GetReport handles GET /api/v1/reports/{id}
// @Summary Get generated report
// @Description Retrieve a previously generated report
// @Tags Monitoring
// @Produce json,application/pdf,text/csv
// @Param id path string true "Report ID"
// @Success 200 {file} file "Generated report file"
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 404 {object} map[string]interface{} "Report not found"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/reports/{id} [get]
func (h *ExportHandler) GetReport(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	reportID := vars["id"]

	if reportID == "" {
		writeError(w, http.StatusBadRequest, "MISSING_REPORT_ID", "Report ID is required")
		return
	}

	report, err := h.reportGenerator.GetReport(r.Context(), reportID)
	if err != nil {
		if err == monitoring.ErrReportNotFound {
			writeError(w, http.StatusNotFound, "REPORT_NOT_FOUND", "Report not found")
			return
		}
		writeError(w, http.StatusInternalServerError, "INTERNAL_ERROR", "Failed to retrieve report")
		return
	}

	// Set appropriate content type and headers
	switch report.Format {
	case "pdf":
		w.Header().Set("Content-Type", "application/pdf")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"report-%s.pdf\"", reportID))
	case "csv":
		w.Header().Set("Content-Type", "text/csv")
		w.Header().Set("Content-Disposition", fmt.Sprintf("attachment; filename=\"report-%s.csv\"", reportID))
	case "json":
		w.Header().Set("Content-Type", "application/json")
	}

	w.WriteHeader(http.StatusOK)
	w.Write(report.Data)
}

// ListReports handles GET /api/v1/reports
// @Summary List generated reports
// @Description List all generated reports with pagination
// @Tags Monitoring
// @Produce json
// @Param page query int false "Page number" default(1)
// @Param limit query int false "Items per page" default(20)
// @Param type query string false "Filter by report type"
// @Success 200 {object} map[string]interface{}
// @Failure 400 {object} map[string]interface{} "Invalid request"
// @Failure 401 {object} map[string]interface{} "Unauthorized"
// @Failure 500 {object} map[string]interface{} "Internal server error"
// @Router /api/v1/reports [get]
func (h *ExportHandler) ListReports(w http.ResponseWriter, r *http.Request) {
	// Parse pagination parameters
	page := 1
	limit := 20

	if pageStr := r.URL.Query().Get("page"); pageStr != "" {
		if p, err := strconv.Atoi(pageStr); err == nil && p > 0 {
			page = p
		}
	}

	if limitStr := r.URL.Query().Get("limit"); limitStr != "" {
		if l, err := strconv.Atoi(limitStr); err == nil && l > 0 && l <= 100 {
			limit = l
		}
	}

	reportType := r.URL.Query().Get("type")

	reports, total, err := h.reportGenerator.ListReports(r.Context(), monitoring.ListReportsRequest{
		Page:   page,
		Limit:  limit,
		Type:   reportType,
	})
	if err != nil {
		writeError(w, http.StatusInternalServerError, "LIST_ERROR", "Failed to list reports")
		return
	}

	response := map[string]interface{}{
		"reports": reports,
		"pagination": map[string]interface{}{
			"page":        page,
			"limit":       limit,
			"total":       total,
			"total_pages": (total + limit - 1) / limit,
		},
	}

	writeJSON(w, http.StatusOK, response)
}

// Helper functions

func (h *ExportHandler) parseExportParams(r *http.Request) MetricExportRequest {
	req := MetricExportRequest{
		Format: r.URL.Query().Get("format"),
	}

	if startStr := r.URL.Query().Get("start_time"); startStr != "" {
		if t, err := time.Parse(time.RFC3339, startStr); err == nil {
			req.StartTime = t
		}
	}

	if endStr := r.URL.Query().Get("end_time"); endStr != "" {
		if t, err := time.Parse(time.RFC3339, endStr); err == nil {
			req.EndTime = t
		}
	}

	if metrics := r.URL.Query().Get("metrics"); metrics != "" {
		req.Metrics = strings.Split(metrics, ",")
	}

	return req
}

func (h *ExportHandler) exportPrometheusFormat(w http.ResponseWriter, r *http.Request, req MetricExportRequest) {
	exportRequestCount.WithLabelValues("prometheus", "metrics", "success").Inc()

	metrics, err := h.metricRegistry.ExportPrometheus(r.Context(), monitoring.PrometheusExportRequest{
		StartTime: req.StartTime,
		EndTime:   req.EndTime,
		Metrics:   req.Metrics,
		Labels:    req.Labels,
	})
	if err != nil {
		exportRequestCount.WithLabelValues("prometheus", "metrics", "error").Inc()
		writeError(w, http.StatusInternalServerError, "EXPORT_ERROR", "Failed to export metrics")
		return
	}

	w.Header().Set("Content-Type", string(expfmt.FmtText))
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(metrics))
}

func (h *ExportHandler) exportJSONFormat(w http.ResponseWriter, r *http.Request, req MetricExportRequest) {
	exportRequestCount.WithLabelValues("json", "metrics", "success").Inc()

	metrics, err := h.metricRegistry.ExportJSON(r.Context(), monitoring.JSONExportRequest{
		StartTime:  req.StartTime,
		EndTime:    req.EndTime,
		Metrics:    req.Metrics,
		Labels:     req.Labels,
		Resolution: req.Resolution,
	})
	if err != nil {
		exportRequestCount.WithLabelValues("json", "metrics", "error").Inc()
		writeError(w, http.StatusInternalServerError, "EXPORT_ERROR", "Failed to export metrics")
		return
	}

	writeJSON(w, http.StatusOK, metrics)
}

func (h *ExportHandler) convertReportSchedule(schedule *ReportSchedule) *monitoring.ReportSchedule {
	if schedule == nil {
		return nil
	}
	return &monitoring.ReportSchedule{
		Frequency: schedule.Frequency,
		Time:      schedule.Time,
		Timezone:  schedule.Timezone,
		Enabled:   schedule.Enabled,
	}
}