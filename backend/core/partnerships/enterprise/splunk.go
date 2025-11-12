// Package enterprise provides Splunk integration for advanced logging
package enterprise

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// SplunkClient handles Splunk HTTP Event Collector integration
type SplunkClient struct {
	hecURL     string
	token      string
	client     *http.Client
	index      string
	source     string
	sourcetype string
}

// SplunkConfig configures Splunk integration
type SplunkConfig struct {
	HECURL     string
	Token      string
	Index      string
	Source     string
	Sourcetype string
	Timeout    time.Duration
}

// Event represents a Splunk event
type Event struct {
	Time       int64                  `json:"time,omitempty"`
	Host       string                 `json:"host,omitempty"`
	Source     string                 `json:"source,omitempty"`
	Sourcetype string                 `json:"sourcetype,omitempty"`
	Index      string                 `json:"index,omitempty"`
	Event      interface{}            `json:"event"`
	Fields     map[string]interface{} `json:"fields,omitempty"`
}

// SearchResult represents Splunk search results
type SearchResult struct {
	Results []map[string]interface{} `json:"results"`
	Preview bool                     `json:"preview"`
	InitOffset int                   `json:"init_offset"`
	Messages []SearchMessage          `json:"messages"`
}

// SearchMessage represents a search message
type SearchMessage struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// NewSplunkClient creates a new Splunk client
func NewSplunkClient(cfg SplunkConfig) *SplunkClient {
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}

	return &SplunkClient{
		hecURL:     cfg.HECURL,
		token:      cfg.Token,
		index:      cfg.Index,
		source:     cfg.Source,
		sourcetype: cfg.Sourcetype,
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// SendEvent sends a single event to Splunk HEC
func (s *SplunkClient) SendEvent(ctx context.Context, event Event) error {
	// Set defaults if not provided
	if event.Index == "" {
		event.Index = s.index
	}
	if event.Source == "" {
		event.Source = s.source
	}
	if event.Sourcetype == "" {
		event.Sourcetype = s.sourcetype
	}
	if event.Time == 0 {
		event.Time = time.Now().Unix()
	}

	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.hecURL+"/services/collector/event", bytes.NewReader(data))
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Splunk "+s.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send event: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// SendBatch sends multiple events in a batch
func (s *SplunkClient) SendBatch(ctx context.Context, events []Event) error {
	var buffer bytes.Buffer

	for _, event := range events {
		// Set defaults if not provided
		if event.Index == "" {
			event.Index = s.index
		}
		if event.Source == "" {
			event.Source = s.source
		}
		if event.Sourcetype == "" {
			event.Sourcetype = s.sourcetype
		}
		if event.Time == 0 {
			event.Time = time.Now().Unix()
		}

		data, err := json.Marshal(event)
		if err != nil {
			return fmt.Errorf("failed to marshal event: %w", err)
		}

		buffer.Write(data)
		buffer.WriteString("\n")
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.hecURL+"/services/collector/event", &buffer)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Authorization", "Splunk "+s.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to send batch: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	return nil
}

// LogVMOperation logs a VM operation to Splunk
func (s *SplunkClient) LogVMOperation(ctx context.Context, operation string, vmID string, details map[string]interface{}) error {
	event := Event{
		Event: map[string]interface{}{
			"operation":  operation,
			"vm_id":      vmID,
			"details":    details,
			"product":    "novacron",
			"component":  "vm-manager",
		},
		Fields: map[string]interface{}{
			"operation_type": operation,
			"vm_id":          vmID,
			"product":        "novacron",
		},
	}

	return s.SendEvent(ctx, event)
}

// LogMigration logs a migration event
func (s *SplunkClient) LogMigration(ctx context.Context, migrationID string, phase string, status string, metrics map[string]interface{}) error {
	event := Event{
		Event: map[string]interface{}{
			"migration_id": migrationID,
			"phase":        phase,
			"status":       status,
			"metrics":      metrics,
			"product":      "novacron",
			"component":    "migration-engine",
		},
		Fields: map[string]interface{}{
			"migration_id": migrationID,
			"phase":        phase,
			"status":       status,
			"product":      "novacron",
		},
	}

	return s.SendEvent(ctx, event)
}

// LogSecurityEvent logs a security event
func (s *SplunkClient) LogSecurityEvent(ctx context.Context, eventType string, severity string, details map[string]interface{}) error {
	event := Event{
		Event: map[string]interface{}{
			"event_type": eventType,
			"severity":   severity,
			"details":    details,
			"product":    "novacron",
			"component":  "security",
		},
		Fields: map[string]interface{}{
			"event_type": eventType,
			"severity":   severity,
			"product":    "novacron",
			"security":   "true",
		},
	}

	return s.SendEvent(ctx, event)
}

// LogPerformanceMetrics logs performance metrics
func (s *SplunkClient) LogPerformanceMetrics(ctx context.Context, metrics map[string]interface{}) error {
	event := Event{
		Event: map[string]interface{}{
			"metrics":   metrics,
			"product":   "novacron",
			"component": "performance",
		},
		Fields: map[string]interface{}{
			"product":     "novacron",
			"metric_type": "performance",
		},
	}

	return s.SendEvent(ctx, event)
}

// CreateDashboard creates a Splunk dashboard for NovaCron
func (s *SplunkClient) CreateDashboard() string {
	dashboard := `
<dashboard version="1.1">
  <label>NovaCron Operations Dashboard</label>
  <description>Real-time monitoring of NovaCron DWCP v3 operations</description>

  <row>
    <panel>
      <title>VM Operations Over Time</title>
      <chart>
        <search>
          <query>index=novacron sourcetype=vm-operations | timechart count by operation</query>
          <earliest>-24h@h</earliest>
          <latest>now</latest>
        </search>
        <option name="charting.chart">line</option>
      </chart>
    </panel>

    <panel>
      <title>Active Migrations</title>
      <single>
        <search>
          <query>index=novacron sourcetype=migrations status=in_progress | stats count</query>
          <earliest>-24h@h</earliest>
          <latest>now</latest>
        </search>
      </single>
    </panel>
  </row>

  <row>
    <panel>
      <title>Migration Success Rate</title>
      <chart>
        <search>
          <query>index=novacron sourcetype=migrations | stats count by status | eval success_rate=if(status="completed", count, 0)</query>
          <earliest>-7d@d</earliest>
          <latest>now</latest>
        </search>
        <option name="charting.chart">pie</option>
      </chart>
    </panel>

    <panel>
      <title>Security Events</title>
      <table>
        <search>
          <query>index=novacron sourcetype=security | table _time, event_type, severity, details | sort -_time | head 10</query>
          <earliest>-24h@h</earliest>
          <latest>now</latest>
        </search>
      </table>
    </panel>
  </row>

  <row>
    <panel>
      <title>Performance Metrics</title>
      <chart>
        <search>
          <query>index=novacron sourcetype=performance | timechart avg(metrics.cpu_usage) as CPU, avg(metrics.memory_usage) as Memory, avg(metrics.disk_io) as Disk</query>
          <earliest>-4h@h</earliest>
          <latest>now</latest>
        </search>
        <option name="charting.chart">area</option>
      </chart>
    </panel>
  </row>

  <row>
    <panel>
      <title>Top Errors (Last 24h)</title>
      <table>
        <search>
          <query>index=novacron error OR failed | stats count by error_type, component | sort -count | head 20</query>
          <earliest>-24h@h</earliest>
          <latest>now</latest>
        </search>
      </table>
    </panel>

    <panel>
      <title>Cloud Provider Distribution</title>
      <chart>
        <search>
          <query>index=novacron sourcetype=vm-operations | stats count by cloud_provider</query>
          <earliest>-24h@h</earliest>
          <latest>now</latest>
        </search>
        <option name="charting.chart">pie</option>
      </chart>
    </panel>
  </row>
</dashboard>
`
	return dashboard
}

// CreateAlerts creates recommended Splunk alerts
func (s *SplunkClient) CreateAlerts() []string {
	alerts := []string{
		// Critical migration failure
		`search index=novacron sourcetype=migrations status=failed severity=critical | stats count by migration_id`,

		// High error rate
		`search index=novacron error | stats count by component | where count > 100`,

		// Security incidents
		`search index=novacron sourcetype=security severity IN (critical, high)`,

		// Resource exhaustion
		`search index=novacron sourcetype=performance metrics.cpu_usage > 90 OR metrics.memory_usage > 90`,

		// Failed authentication attempts
		`search index=novacron event_type=authentication status=failed | stats count by source_ip | where count > 5`,
	}

	return alerts
}
