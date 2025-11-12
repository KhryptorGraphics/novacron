package init

import (
	"context"
	"time"
)

// Component represents an initializable system component
type Component interface {
	// Name returns the component name
	Name() string

	// Dependencies returns list of component names this depends on
	Dependencies() []string

	// Initialize initializes the component with resolved dependencies
	Initialize(ctx context.Context, deps map[string]interface{}) error

	// HealthCheck verifies component health
	HealthCheck() error

	// Shutdown gracefully shuts down the component
	Shutdown(ctx context.Context) error
}

// ConfigurableComponent supports configuration
type ConfigurableComponent interface {
	Component

	// Configure applies configuration to the component
	Configure(config interface{}) error

	// ValidateConfig validates configuration before initialization
	ValidateConfig(config interface{}) error
}

// ObservableComponent emits metrics and logs
type ObservableComponent interface {
	Component

	// Metrics returns current component metrics
	Metrics() map[string]interface{}

	// Status returns current component status
	Status() ComponentStatus
}

// ComponentStatus represents component health status
type ComponentStatus struct {
	State     StatusState            `json:"state"`
	Message   string                 `json:"message"`
	LastCheck time.Time              `json:"last_check"`
	Metrics   map[string]interface{} `json:"metrics"`
	Errors    []error                `json:"errors,omitempty"`
}

// StatusState represents the health state
type StatusState string

const (
	StatusHealthy   StatusState = "healthy"
	StatusDegraded  StatusState = "degraded"
	StatusUnhealthy StatusState = "unhealthy"
	StatusUnknown   StatusState = "unknown"
)

// Configuration represents system configuration
type Configuration struct {
	// Environment
	Environment string `json:"environment" yaml:"environment"` // "datacenter", "internet", "hybrid"

	// Core settings
	LogLevel string `json:"log_level" yaml:"log_level"`
	Debug    bool   `json:"debug" yaml:"debug"`

	// Resource limits
	MinCPU    int   `json:"min_cpu" yaml:"min_cpu"`
	MinMemory int64 `json:"min_memory" yaml:"min_memory"`
	MinDisk   int64 `json:"min_disk" yaml:"min_disk"`

	// Component configurations
	Security   interface{} `json:"security" yaml:"security"`
	Database   interface{} `json:"database" yaml:"database"`
	Cache      interface{} `json:"cache" yaml:"cache"`
	Network    interface{} `json:"network" yaml:"network"`
	DWCP       interface{} `json:"dwcp" yaml:"dwcp"`
	API        interface{} `json:"api" yaml:"api"`
	Monitoring interface{} `json:"monitoring" yaml:"monitoring"`
}

// ConfigurationLoader loads and validates configuration
type ConfigurationLoader interface {
	// Load loads configuration from environment
	Load(env string) (*Configuration, error)

	// Validate validates configuration
	Validate(config *Configuration) error

	// Merge merges configurations (env vars, files, defaults)
	Merge(configs ...*Configuration) (*Configuration, error)
}

// InitError represents an initialization error
type InitError struct {
	Component string
	Phase     string
	Critical  bool
	Retriable bool
	Cause     error
}

func (e *InitError) Error() string {
	return e.Cause.Error()
}

// Error categories
const (
	ErrorCritical = "critical"
	ErrorDegraded = "degraded"
	ErrorWarning  = "warning"
)

// RetryPolicy defines retry behavior
type RetryPolicy struct {
	MaxAttempts int
	Delay       time.Duration
	Backoff     float64 // Exponential backoff multiplier
}

// DefaultRetryPolicy for most components
var DefaultRetryPolicy = RetryPolicy{
	MaxAttempts: 3,
	Delay:       1 * time.Second,
	Backoff:     2.0,
}

// EnvironmentType represents the deployment environment
type EnvironmentType string

const (
	EnvironmentDatacenter EnvironmentType = "datacenter"
	EnvironmentInternet   EnvironmentType = "internet"
	EnvironmentHybrid     EnvironmentType = "hybrid"
)

// EnvironmentDetector detects the current environment
type EnvironmentDetector interface {
	// Detect detects the current environment
	Detect() (EnvironmentType, error)

	// DetectWithHints detects environment with provided hints
	DetectWithHints(hints map[string]string) (EnvironmentType, error)
}

// ResourceValidator validates system resources
type ResourceValidator interface {
	// Validate validates that resources meet minimum requirements
	Validate(minCPU int, minMemory int64, minDisk int64) error

	// GetAvailableResources returns current available resources
	GetAvailableResources() (*ResourceInfo, error)
}

// ResourceInfo contains information about available resources
type ResourceInfo struct {
	CPU    int   `json:"cpu"`
	Memory int64 `json:"memory"`
	Disk   int64 `json:"disk"`
}

// InitPhase represents an initialization phase
type InitPhase interface {
	// Name returns the phase name
	Name() string

	// Execute executes the phase
	Execute(ctx context.Context) error

	// Duration returns the expected duration
	Duration() time.Duration
}

// PreInitResult contains results from pre-initialization phase
type PreInitResult struct {
	Environment EnvironmentType
	Config      *Configuration
	Logger      interface{}
}

// CoreInitResult contains results from core initialization phase
type CoreInitResult struct {
	Registry   *ComponentRegistry
	Components map[string]interface{}
}

// ServiceInitResult contains results from service initialization phase
type ServiceInitResult struct {
	Orchestrator interface{}
	APIServer    interface{}
	Monitoring   interface{}
	MLEngine     interface{}
}
