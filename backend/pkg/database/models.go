package database

import (
	"database/sql/driver"
	"encoding/json"
	"fmt"
	"time"
)

// JSONB represents a PostgreSQL JSONB field
type JSONB map[string]interface{}

// Value implements the driver.Valuer interface for JSONB
func (j JSONB) Value() (driver.Value, error) {
	if j == nil {
		return nil, nil
	}
	return json.Marshal(j)
}

// Scan implements the sql.Scanner interface for JSONB
func (j *JSONB) Scan(value interface{}) error {
	if value == nil {
		*j = nil
		return nil
	}

	var bytes []byte
	switch v := value.(type) {
	case []byte:
		bytes = v
	case string:
		bytes = []byte(v)
	default:
		return fmt.Errorf("cannot scan %T into JSONB", value)
	}

	return json.Unmarshal(bytes, j)
}

// User represents a user in the database
type User struct {
	ID         int       `db:"id" json:"id"`
	Username   string    `db:"username" json:"username"`
	Email      string    `db:"email" json:"email"`
	PasswordHash string  `db:"password_hash" json:"-"`
	Role       string    `db:"role" json:"role"`
	TenantID   string    `db:"tenant_id" json:"tenant_id"`
	CreatedAt  time.Time `db:"created_at" json:"created_at"`
	UpdatedAt  time.Time `db:"updated_at" json:"updated_at"`
}

// VM represents a virtual machine in the database
type VM struct {
	ID        string    `db:"id" json:"id"`
	Name      string    `db:"name" json:"name"`
	State     string    `db:"state" json:"state"`
	NodeID    *string   `db:"node_id" json:"node_id,omitempty"`
	OwnerID   *int      `db:"owner_id" json:"owner_id,omitempty"`
	TenantID  string    `db:"tenant_id" json:"tenant_id"`
	Config    JSONB     `db:"config" json:"config,omitempty"`
	CreatedAt time.Time `db:"created_at" json:"created_at"`
	UpdatedAt time.Time `db:"updated_at" json:"updated_at"`
}

// VMMetric represents VM performance metrics in the database
type VMMetric struct {
	ID           int       `db:"id" json:"id"`
	VMID         string    `db:"vm_id" json:"vm_id"`
	CPUUsage     float64   `db:"cpu_usage" json:"cpu_usage"`
	MemoryUsage  float64   `db:"memory_usage" json:"memory_usage"`
	DiskUsage    float64   `db:"disk_usage" json:"disk_usage"`
	NetworkSent  int64     `db:"network_sent" json:"network_sent"`
	NetworkRecv  int64     `db:"network_recv" json:"network_recv"`
	IOPS         int       `db:"iops" json:"iops"`
	Timestamp    time.Time `db:"timestamp" json:"timestamp"`
}

// SystemMetric represents system-wide performance metrics
type SystemMetric struct {
	ID               int       `db:"id" json:"id"`
	NodeID           string    `db:"node_id" json:"node_id"`
	CPUUsage         float64   `db:"cpu_usage" json:"cpu_usage"`
	MemoryUsage      float64   `db:"memory_usage" json:"memory_usage"`
	MemoryTotal      int64     `db:"memory_total" json:"memory_total"`
	MemoryAvailable  int64     `db:"memory_available" json:"memory_available"`
	DiskUsage        float64   `db:"disk_usage" json:"disk_usage"`
	DiskTotal        int64     `db:"disk_total" json:"disk_total"`
	DiskAvailable    int64     `db:"disk_available" json:"disk_available"`
	NetworkSent      int64     `db:"network_sent" json:"network_sent"`
	NetworkRecv      int64     `db:"network_recv" json:"network_recv"`
	LoadAverage1     float64   `db:"load_average_1" json:"load_average_1"`
	LoadAverage5     float64   `db:"load_average_5" json:"load_average_5"`
	LoadAverage15    float64   `db:"load_average_15" json:"load_average_15"`
	Timestamp        time.Time `db:"timestamp" json:"timestamp"`
}

// Alert represents an alert in the database
type Alert struct {
	ID            string    `db:"id" json:"id"`
	Name          string    `db:"name" json:"name"`
	Description   string    `db:"description" json:"description"`
	Severity      string    `db:"severity" json:"severity"`
	Status        string    `db:"status" json:"status"`
	Resource      string    `db:"resource" json:"resource"`
	ResourceID    *string   `db:"resource_id" json:"resource_id,omitempty"`
	MetricName    string    `db:"metric_name" json:"metric_name"`
	Threshold     float64   `db:"threshold" json:"threshold"`
	CurrentValue  float64   `db:"current_value" json:"current_value"`
	Labels        JSONB     `db:"labels" json:"labels,omitempty"`
	StartTime     time.Time `db:"start_time" json:"start_time"`
	EndTime       *time.Time `db:"end_time" json:"end_time,omitempty"`
	AcknowledgedBy *string  `db:"acknowledged_by" json:"acknowledged_by,omitempty"`
	AcknowledgedAt *time.Time `db:"acknowledged_at" json:"acknowledged_at,omitempty"`
	CreatedAt     time.Time `db:"created_at" json:"created_at"`
	UpdatedAt     time.Time `db:"updated_at" json:"updated_at"`
}

// AuditLog represents an audit log entry
type AuditLog struct {
	ID         int       `db:"id" json:"id"`
	UserID     *int      `db:"user_id" json:"user_id,omitempty"`
	Action     string    `db:"action" json:"action"`
	Resource   string    `db:"resource" json:"resource"`
	ResourceID *string   `db:"resource_id" json:"resource_id,omitempty"`
	Details    JSONB     `db:"details" json:"details,omitempty"`
	IPAddress  string    `db:"ip_address" json:"ip_address"`
	UserAgent  string    `db:"user_agent" json:"user_agent"`
	Success    bool      `db:"success" json:"success"`
	ErrorMessage *string `db:"error_message" json:"error_message,omitempty"`
	Timestamp  time.Time `db:"timestamp" json:"timestamp"`
}

// Session represents a user session
type Session struct {
	ID        string    `db:"id" json:"id"`
	UserID    int       `db:"user_id" json:"user_id"`
	Token     string    `db:"token" json:"-"`
	ExpiresAt time.Time `db:"expires_at" json:"expires_at"`
	IPAddress string    `db:"ip_address" json:"ip_address"`
	UserAgent string    `db:"user_agent" json:"user_agent"`
	IsActive  bool      `db:"is_active" json:"is_active"`
	CreatedAt time.Time `db:"created_at" json:"created_at"`
	UpdatedAt time.Time `db:"updated_at" json:"updated_at"`
}

// Migration represents a VM migration record
type Migration struct {
	ID            string    `db:"id" json:"id"`
	VMID          string    `db:"vm_id" json:"vm_id"`
	SourceNodeID  string    `db:"source_node_id" json:"source_node_id"`
	TargetNodeID  string    `db:"target_node_id" json:"target_node_id"`
	Type          string    `db:"type" json:"type"` // cold, warm, live
	Status        string    `db:"status" json:"status"` // pending, running, completed, failed
	Progress      float64   `db:"progress" json:"progress"`
	BytesTotal    int64     `db:"bytes_total" json:"bytes_total"`
	BytesTransferred int64  `db:"bytes_transferred" json:"bytes_transferred"`
	StartedAt     *time.Time `db:"started_at" json:"started_at,omitempty"`
	CompletedAt   *time.Time `db:"completed_at" json:"completed_at,omitempty"`
	ErrorMessage  *string   `db:"error_message" json:"error_message,omitempty"`
	CreatedAt     time.Time `db:"created_at" json:"created_at"`
	UpdatedAt     time.Time `db:"updated_at" json:"updated_at"`
}

// Node represents a hypervisor node in the cluster
type Node struct {
	ID           string    `db:"id" json:"id"`
	Name         string    `db:"name" json:"name"`
	Address      string    `db:"address" json:"address"`
	Status       string    `db:"status" json:"status"` // online, offline, maintenance
	Capabilities JSONB     `db:"capabilities" json:"capabilities,omitempty"`
	Resources    JSONB     `db:"resources" json:"resources,omitempty"`
	Labels       JSONB     `db:"labels" json:"labels,omitempty"`
	LastSeen     time.Time `db:"last_seen" json:"last_seen"`
	CreatedAt    time.Time `db:"created_at" json:"created_at"`
	UpdatedAt    time.Time `db:"updated_at" json:"updated_at"`
}