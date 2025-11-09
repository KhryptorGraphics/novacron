package testing

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// TestReporter handles test reporting and alerting
type TestReporter struct {
	grafanaURL    string
	prometheusURL string
	slackWebhook  string
	emailConfig   *EmailConfig
}

// EmailConfig configures email notifications
type EmailConfig struct {
	SMTPServer string
	SMTPPort   int
	From       string
	To         []string
	Username   string
	Password   string
}

// TestReport represents a comprehensive test report
type TestReport struct {
	Timestamp       time.Time
	TotalTests      int
	PassedTests     int
	FailedTests     int
	SkippedTests    int
	AverageDuration time.Duration
	SuccessRate     float64
	Scenarios       []*TestResult
	Trends          *TrendData
}

// TrendData represents testing trends
type TrendData struct {
	SuccessRateTrend    string  // "improving", "stable", "declining"
	PerformanceTrend    string
	LatestSuccessRate   float64
	PreviousSuccessRate float64
}

// NewTestReporter creates a new test reporter
func NewTestReporter() *TestReporter {
	return &TestReporter{
		grafanaURL:    "http://localhost:3000",
		prometheusURL: "http://localhost:9090",
		slackWebhook:  "", // Configure from environment
	}
}

// GenerateReport generates a comprehensive test report
func (tr *TestReporter) GenerateReport(results []*TestResult) *TestReport {
	report := &TestReport{
		Timestamp:  time.Now(),
		TotalTests: len(results),
		Scenarios:  results,
	}

	var totalDuration time.Duration

	for _, result := range results {
		if result.Passed {
			report.PassedTests++
		} else {
			report.FailedTests++
		}
		totalDuration += result.Duration
	}

	if report.TotalTests > 0 {
		report.AverageDuration = totalDuration / time.Duration(report.TotalTests)
		report.SuccessRate = float64(report.PassedTests) / float64(report.TotalTests)
	}

	return report
}

// PublishToDashboard publishes metrics to monitoring dashboard
func (tr *TestReporter) PublishToDashboard(report *TestReport) error {
	// Update Prometheus metrics (if Prometheus client is available)
	tr.updatePrometheusMetrics(report)

	// Send to Grafana via API
	return tr.sendToGrafana(report)
}

// updatePrometheusMetrics updates Prometheus metrics
func (tr *TestReporter) updatePrometheusMetrics(report *TestReport) {
	// In a real implementation, this would use Prometheus client library
	fmt.Printf("[METRICS] Total Tests: %d\n", report.TotalTests)
	fmt.Printf("[METRICS] Passed: %d\n", report.PassedTests)
	fmt.Printf("[METRICS] Failed: %d\n", report.FailedTests)
	fmt.Printf("[METRICS] Success Rate: %.2f%%\n", report.SuccessRate*100)
}

// sendToGrafana sends report data to Grafana
func (tr *TestReporter) sendToGrafana(report *TestReport) error {
	// Convert report to Grafana annotation format
	annotation := map[string]interface{}{
		"time":    report.Timestamp.Unix() * 1000, // milliseconds
		"text":    fmt.Sprintf("Test Run: %d passed, %d failed", report.PassedTests, report.FailedTests),
		"tags":    []string{"dwcp", "testing", "automated"},
	}

	data, err := json.Marshal(annotation)
	if err != nil {
		return fmt.Errorf("failed to marshal annotation: %v", err)
	}

	// Send to Grafana API
	url := fmt.Sprintf("%s/api/annotations", tr.grafanaURL)
	resp, err := http.Post(url, "application/json", strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to send to Grafana: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Grafana returned status %d", resp.StatusCode)
	}

	return nil
}

// SendFailureAlert sends alerts when tests fail
func (tr *TestReporter) SendFailureAlert(testRun *TestRun) error {
	message := tr.formatFailureMessage(testRun)

	// Send to Slack
	if tr.slackWebhook != "" {
		if err := tr.sendSlackAlert(message); err != nil {
			fmt.Printf("Failed to send Slack alert: %v\n", err)
		}
	}

	// Send email
	if tr.emailConfig != nil {
		if err := tr.sendEmailAlert(message); err != nil {
			fmt.Printf("Failed to send email alert: %v\n", err)
		}
	}

	return nil
}

// formatFailureMessage formats the failure alert message
func (tr *TestReporter) formatFailureMessage(testRun *TestRun) string {
	var sb strings.Builder

	sb.WriteString("🚨 DWCP Test Failures Detected 🚨\n\n")
	sb.WriteString(fmt.Sprintf("Test Run ID: %s\n", testRun.ID))
	sb.WriteString(fmt.Sprintf("Timestamp: %s\n", testRun.StartTime.Format(time.RFC3339)))
	sb.WriteString(fmt.Sprintf("Total Tests: %d\n", testRun.Summary.TotalTests))
	sb.WriteString(fmt.Sprintf("Failed Tests: %d\n", testRun.Summary.FailedTests))
	sb.WriteString(fmt.Sprintf("Success Rate: %.2f%%\n\n", testRun.Summary.SuccessRate*100))

	sb.WriteString("Failed Scenarios:\n")
	for _, result := range testRun.Results {
		if !result.Passed {
			sb.WriteString(fmt.Sprintf("  - %s (Duration: %v)\n", result.Scenario, result.Duration))
			for _, reason := range result.FailureReasons {
				sb.WriteString(fmt.Sprintf("    * %s\n", reason))
			}
		}
	}

	return sb.String()
}

// sendSlackAlert sends alert to Slack
func (tr *TestReporter) sendSlackAlert(message string) error {
	payload := map[string]interface{}{
		"text": message,
		"attachments": []map[string]interface{}{
			{
				"color": "danger",
				"title": "DWCP Test Failure",
				"text":  message,
			},
		},
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("failed to marshal Slack payload: %v", err)
	}

	resp, err := http.Post(tr.slackWebhook, "application/json", strings.NewReader(string(data)))
	if err != nil {
		return fmt.Errorf("failed to send Slack webhook: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("Slack returned status %d", resp.StatusCode)
	}

	return nil
}

// sendEmailAlert sends alert via email
func (tr *TestReporter) sendEmailAlert(message string) error {
	// In a real implementation, this would use an SMTP library
	fmt.Printf("[EMAIL ALERT] %s\n", message)
	return nil
}

// GenerateHTMLReport generates an HTML report
func (tr *TestReporter) GenerateHTMLReport(report *TestReport) string {
	var sb strings.Builder

	sb.WriteString("<html><head><title>DWCP Test Report</title>")
	sb.WriteString("<style>")
	sb.WriteString("body { font-family: Arial, sans-serif; margin: 20px; }")
	sb.WriteString(".header { background: #2c3e50; color: white; padding: 20px; }")
	sb.WriteString(".summary { margin: 20px 0; }")
	sb.WriteString(".passed { color: green; }")
	sb.WriteString(".failed { color: red; }")
	sb.WriteString("table { width: 100%; border-collapse: collapse; }")
	sb.WriteString("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }")
	sb.WriteString("th { background-color: #4CAF50; color: white; }")
	sb.WriteString("</style></head><body>")

	// Header
	sb.WriteString("<div class='header'>")
	sb.WriteString("<h1>DWCP Multi-Datacenter Test Report</h1>")
	sb.WriteString(fmt.Sprintf("<p>Generated: %s</p>", report.Timestamp.Format(time.RFC1123)))
	sb.WriteString("</div>")

	// Summary
	sb.WriteString("<div class='summary'>")
	sb.WriteString("<h2>Summary</h2>")
	sb.WriteString(fmt.Sprintf("<p>Total Tests: %d</p>", report.TotalTests))
	sb.WriteString(fmt.Sprintf("<p class='passed'>Passed: %d</p>", report.PassedTests))
	sb.WriteString(fmt.Sprintf("<p class='failed'>Failed: %d</p>", report.FailedTests))
	sb.WriteString(fmt.Sprintf("<p>Success Rate: %.2f%%</p>", report.SuccessRate*100))
	sb.WriteString(fmt.Sprintf("<p>Average Duration: %v</p>", report.AverageDuration))
	sb.WriteString("</div>")

	// Detailed Results
	sb.WriteString("<h2>Detailed Results</h2>")
	sb.WriteString("<table>")
	sb.WriteString("<tr><th>Scenario</th><th>Status</th><th>Duration</th><th>Operations</th></tr>")

	for _, result := range report.Scenarios {
		status := "<span class='passed'>PASSED</span>"
		if !result.Passed {
			status = "<span class='failed'>FAILED</span>"
		}

		sb.WriteString("<tr>")
		sb.WriteString(fmt.Sprintf("<td>%s</td>", result.Scenario))
		sb.WriteString(fmt.Sprintf("<td>%s</td>", status))
		sb.WriteString(fmt.Sprintf("<td>%v</td>", result.Duration))
		sb.WriteString(fmt.Sprintf("<td>%d</td>", len(result.Metrics.OperationResults)))
		sb.WriteString("</tr>")
	}

	sb.WriteString("</table>")
	sb.WriteString("</body></html>")

	return sb.String()
}

// ExportToJSON exports report as JSON
func (tr *TestReporter) ExportToJSON(report *TestReport) ([]byte, error) {
	return json.MarshalIndent(report, "", "  ")
}

// ExportToCSV exports report as CSV
func (tr *TestReporter) ExportToCSV(report *TestReport) string {
	var sb strings.Builder

	sb.WriteString("Scenario,Status,Duration,Total Operations,Successful Operations,Failed Operations\n")

	for _, result := range report.Scenarios {
		status := "PASSED"
		if !result.Passed {
			status = "FAILED"
		}

		totalOps := len(result.Metrics.OperationResults)
		successfulOps := 0
		failedOps := 0

		for _, op := range result.Metrics.OperationResults {
			if op.Success {
				successfulOps++
			} else {
				failedOps++
			}
		}

		sb.WriteString(fmt.Sprintf("%s,%s,%v,%d,%d,%d\n",
			result.Scenario, status, result.Duration, totalOps, successfulOps, failedOps))
	}

	return sb.String()
}

// PrintReport prints a formatted report to console
func (tr *TestReporter) PrintReport(report *TestReport) {
	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Println("DWCP Multi-Datacenter Test Report")
	fmt.Println(strings.Repeat("=", 80))
	fmt.Printf("Generated: %s\n", report.Timestamp.Format(time.RFC1123))
	fmt.Println()

	fmt.Println("Summary:")
	fmt.Printf("  Total Tests:      %d\n", report.TotalTests)
	fmt.Printf("  Passed Tests:     %d\n", report.PassedTests)
	fmt.Printf("  Failed Tests:     %d\n", report.FailedTests)
	fmt.Printf("  Success Rate:     %.2f%%\n", report.SuccessRate*100)
	fmt.Printf("  Average Duration: %v\n", report.AverageDuration)
	fmt.Println()

	fmt.Println("Detailed Results:")
	fmt.Println(strings.Repeat("-", 80))

	for _, result := range report.Scenarios {
		status := "✓ PASSED"
		if !result.Passed {
			status = "✗ FAILED"
		}

		fmt.Printf("\n%s %s\n", status, result.Scenario)
		fmt.Printf("  Duration: %v\n", result.Duration)
		fmt.Printf("  Operations: %d\n", len(result.Metrics.OperationResults))

		if !result.Passed && len(result.FailureReasons) > 0 {
			fmt.Println("  Failures:")
			for _, reason := range result.FailureReasons {
				fmt.Printf("    - %s\n", reason)
			}
		}
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
}
