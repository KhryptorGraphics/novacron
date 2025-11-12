// Package enterprise provides enterprise software integrations
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

// ServiceNowClient handles ServiceNow ITSM integration
type ServiceNowClient struct {
	baseURL    string
	username   string
	password   string
	client     *http.Client
}

// ServiceNowConfig configures ServiceNow integration
type ServiceNowConfig struct {
	InstanceURL string
	Username    string
	Password    string
	Timeout     time.Duration
}

// Incident represents a ServiceNow incident
type Incident struct {
	SysID            string    `json:"sys_id,omitempty"`
	Number           string    `json:"number,omitempty"`
	ShortDescription string    `json:"short_description"`
	Description      string    `json:"description,omitempty"`
	Priority         string    `json:"priority"` // 1=Critical, 2=High, 3=Moderate, 4=Low, 5=Planning
	Urgency          string    `json:"urgency"`  // 1=High, 2=Medium, 3=Low
	Impact           string    `json:"impact"`   // 1=High, 2=Medium, 3=Low
	Category         string    `json:"category,omitempty"`
	Subcategory      string    `json:"subcategory,omitempty"`
	AssignmentGroup  string    `json:"assignment_group,omitempty"`
	AssignedTo       string    `json:"assigned_to,omitempty"`
	State            string    `json:"state,omitempty"` // 1=New, 2=In Progress, 3=On Hold, 6=Resolved, 7=Closed
	CallerID         string    `json:"caller_id,omitempty"`
	ContactType      string    `json:"contact_type,omitempty"`
	OpenedAt         time.Time `json:"opened_at,omitempty"`
	ClosedAt         *time.Time `json:"closed_at,omitempty"`
	WorkNotes        string    `json:"work_notes,omitempty"`
	CloseNotes       string    `json:"close_notes,omitempty"`
	ConfigurationItem string   `json:"cmdb_ci,omitempty"`
}

// ChangeRequest represents a ServiceNow change request
type ChangeRequest struct {
	SysID            string    `json:"sys_id,omitempty"`
	Number           string    `json:"number,omitempty"`
	ShortDescription string    `json:"short_description"`
	Description      string    `json:"description,omitempty"`
	Type             string    `json:"type"`     // standard, normal, emergency
	Priority         string    `json:"priority"` // 1=Critical, 2=High, 3=Moderate, 4=Low
	Risk             string    `json:"risk"`     // high, medium, low
	Impact           string    `json:"impact"`   // 1=High, 2=Medium, 3=Low
	State            string    `json:"state,omitempty"`
	AssignmentGroup  string    `json:"assignment_group,omitempty"`
	AssignedTo       string    `json:"assigned_to,omitempty"`
	RequestedBy      string    `json:"requested_by,omitempty"`
	StartDate        time.Time `json:"start_date,omitempty"`
	EndDate          time.Time `json:"end_date,omitempty"`
	ImplementationPlan string  `json:"implementation_plan,omitempty"`
	BackoutPlan      string    `json:"backout_plan,omitempty"`
	TestPlan         string    `json:"test_plan,omitempty"`
	JustificationReason string `json:"justification,omitempty"`
	ConfigurationItems []string `json:"cmdb_ci,omitempty"`
}

// ConfigurationItem represents a CMDB configuration item
type ConfigurationItem struct {
	SysID            string            `json:"sys_id,omitempty"`
	Name             string            `json:"name"`
	Class            string            `json:"sys_class_name"`
	Category         string            `json:"category,omitempty"`
	Subcategory      string            `json:"subcategory,omitempty"`
	SerialNumber     string            `json:"serial_number,omitempty"`
	AssetTag         string            `json:"asset_tag,omitempty"`
	Model            string            `json:"model_id,omitempty"`
	Manufacturer     string            `json:"manufacturer,omitempty"`
	Location         string            `json:"location,omitempty"`
	AssignedTo       string            `json:"assigned_to,omitempty"`
	ManagedBy        string            `json:"managed_by,omitempty"`
	OwnedBy          string            `json:"owned_by,omitempty"`
	OperationalStatus string           `json:"operational_status,omitempty"` // 1=Operational, 2=Non-Operational
	Environment      string            `json:"environment,omitempty"`
	IPAddress        string            `json:"ip_address,omitempty"`
	Attributes       map[string]string `json:"attributes,omitempty"`
}

// NewServiceNowClient creates a new ServiceNow client
func NewServiceNowClient(cfg ServiceNowConfig) *ServiceNowClient {
	if cfg.Timeout == 0 {
		cfg.Timeout = 30 * time.Second
	}

	return &ServiceNowClient{
		baseURL:  cfg.InstanceURL + "/api/now",
		username: cfg.Username,
		password: cfg.Password,
		client: &http.Client{
			Timeout: cfg.Timeout,
		},
	}
}

// CreateIncident creates a new incident
func (s *ServiceNowClient) CreateIncident(ctx context.Context, incident Incident) (*Incident, error) {
	data, err := json.Marshal(incident)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal incident: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/table/incident", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.SetBasicAuth(s.username, s.password)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Result Incident `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Result, nil
}

// GetIncident retrieves an incident by sys_id
func (s *ServiceNowClient) GetIncident(ctx context.Context, sysID string) (*Incident, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", s.baseURL+"/table/incident/"+sysID, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.SetBasicAuth(s.username, s.password)
	req.Header.Set("Accept", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Result Incident `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Result, nil
}

// UpdateIncident updates an existing incident
func (s *ServiceNowClient) UpdateIncident(ctx context.Context, sysID string, updates map[string]interface{}) (*Incident, error) {
	data, err := json.Marshal(updates)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal updates: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "PUT", s.baseURL+"/table/incident/"+sysID, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.SetBasicAuth(s.username, s.password)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Result Incident `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Result, nil
}

// CreateChangeRequest creates a new change request
func (s *ServiceNowClient) CreateChangeRequest(ctx context.Context, change ChangeRequest) (*ChangeRequest, error) {
	data, err := json.Marshal(change)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal change request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+"/table/change_request", bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.SetBasicAuth(s.username, s.password)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Result ChangeRequest `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Result, nil
}

// CreateConfigurationItem creates a new CMDB CI
func (s *ServiceNowClient) CreateConfigurationItem(ctx context.Context, ci ConfigurationItem) (*ConfigurationItem, error) {
	data, err := json.Marshal(ci)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal CI: %w", err)
	}

	endpoint := fmt.Sprintf("/table/%s", ci.Class)
	if ci.Class == "" {
		endpoint = "/table/cmdb_ci"
	}

	req, err := http.NewRequestWithContext(ctx, "POST", s.baseURL+endpoint, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.SetBasicAuth(s.username, s.password)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Result ConfigurationItem `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &result.Result, nil
}

// QueryIncidents queries incidents with filters
func (s *ServiceNowClient) QueryIncidents(ctx context.Context, query string, limit int) ([]Incident, error) {
	url := fmt.Sprintf("%s/table/incident?sysparm_query=%s&sysparm_limit=%d", s.baseURL, query, limit)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.SetBasicAuth(s.username, s.password)
	req.Header.Set("Accept", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Result []Incident `json:"result"`
	}

	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return result.Result, nil
}

// AutoCreateIncidentFromAlert creates incident from NovaCron alert
func (s *ServiceNowClient) AutoCreateIncidentFromAlert(ctx context.Context, alert map[string]interface{}) (*Incident, error) {
	severity := alert["severity"].(string)
	priority := "3" // Default to Moderate

	switch severity {
	case "critical":
		priority = "1"
	case "high":
		priority = "2"
	case "low":
		priority = "4"
	}

	incident := Incident{
		ShortDescription: fmt.Sprintf("NovaCron Alert: %s", alert["title"]),
		Description:      fmt.Sprintf("%s\n\nSource: %s\nTimestamp: %s", alert["description"], alert["source"], alert["timestamp"]),
		Priority:         priority,
		Urgency:          priority,
		Impact:           priority,
		Category:         "Infrastructure",
		Subcategory:      "Cloud Platform",
		ContactType:      "Monitoring",
		CallerID:         "novacron-system",
	}

	return s.CreateIncident(ctx, incident)
}
