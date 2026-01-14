package loadbalancer

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// ConfigManager manages dynamic configuration with hot reload capabilities
type ConfigManager struct {
	// Configuration
	config           ConfigManagerConfig
	
	// Current configurations
	loadBalancerConfig *LoadBalancerConfiguration
	configMutex        sync.RWMutex
	
	// Configuration versioning
	configVersion      int64
	configHistory      []*ConfigurationVersion
	historyMutex       sync.RWMutex
	
	// File watching
	fileWatcher        *FileWatcher
	
	// Configuration sources
	sources            map[string]ConfigurationSource
	sourcesMutex       sync.RWMutex
	
	// Configuration validation
	validator          *ConfigValidator
	
	// Change listeners
	listeners          []ConfigChangeListener
	listenersMutex     sync.RWMutex
	
	// Hot reload state
	reloadInProgress   int32
	lastReloadTime     time.Time
	reloadCount        int64
	
	// Rollback capability
	rollbackEnabled    bool
	maxRollbackVersions int
	
	// Configuration templates
	templates          map[string]*ConfigTemplate
	templatesMutex     sync.RWMutex
	
	// Environment-specific configs
	environments       map[string]*EnvironmentConfig
	currentEnvironment string
	
	// Runtime state
	ctx                context.Context
	cancel             context.CancelFunc
	initialized        bool
}

// ConfigManagerConfig holds configuration manager settings
type ConfigManagerConfig struct {
	// File paths
	ConfigFilePath         string            `json:"config_file_path"`
	ConfigDirectory        string            `json:"config_directory"`
	BackupDirectory        string            `json:"backup_directory"`
	TemplateDirectory      string            `json:"template_directory"`
	
	// Watch settings
	EnableFileWatching     bool              `json:"enable_file_watching"`
	WatchInterval          time.Duration     `json:"watch_interval"`
	FileWatchPatterns      []string          `json:"file_watch_patterns"`
	
	// Hot reload settings
	EnableHotReload        bool              `json:"enable_hot_reload"`
	ReloadDelay            time.Duration     `json:"reload_delay"`
	MaxReloadAttempts      int               `json:"max_reload_attempts"`
	ReloadTimeout          time.Duration     `json:"reload_timeout"`
	
	// Validation settings
	EnableValidation       bool              `json:"enable_validation"`
	StrictValidation       bool              `json:"strict_validation"`
	ValidationTimeout      time.Duration     `json:"validation_timeout"`
	
	// Rollback settings
	EnableRollback         bool              `json:"enable_rollback"`
	MaxRollbackVersions    int               `json:"max_rollback_versions"`
	AutoRollbackOnFailure  bool              `json:"auto_rollback_on_failure"`
	RollbackTimeout        time.Duration     `json:"rollback_timeout"`
	
	// Remote configuration
	EnableRemoteConfig     bool              `json:"enable_remote_config"`
	RemoteConfigEndpoints  []RemoteEndpoint  `json:"remote_config_endpoints"`
	RemoteConfigInterval   time.Duration     `json:"remote_config_interval"`
	
	// Environment management
	EnableEnvironments     bool              `json:"enable_environments"`
	DefaultEnvironment     string            `json:"default_environment"`
	EnvironmentConfigPath  string            `json:"environment_config_path"`
	
	// Security settings
	EnableEncryption       bool              `json:"enable_encryption"`
	EncryptionKey          string            `json:"encryption_key,omitempty"`
	EnableSignatureVerification bool        `json:"enable_signature_verification"`
	SignaturePublicKey     string            `json:"signature_public_key,omitempty"`
	
	// Monitoring
	EnableMetrics          bool              `json:"enable_metrics"`
	MetricsInterval        time.Duration     `json:"metrics_interval"`
}

// LoadBalancerConfiguration represents the complete load balancer configuration
type LoadBalancerConfiguration struct {
	// Metadata
	Version             int64                    `json:"version"`
	Name                string                   `json:"name"`
	Description         string                   `json:"description"`
	Environment         string                   `json:"environment"`
	CreatedAt           time.Time                `json:"created_at"`
	UpdatedAt           time.Time                `json:"updated_at"`
	CreatedBy           string                   `json:"created_by"`
	Checksum            string                   `json:"checksum"`
	
	// Core load balancer settings
	LoadBalancer        LoadBalancerConfig       `json:"load_balancer"`
	
	// Services and backends
	Services            []*LoadBalancerService   `json:"services"`
	Backends            []*Backend               `json:"backends"`
	
	// Feature configurations
	HealthChecking      AdvancedHealthConfig     `json:"health_checking"`
	SSLManagement       SSLManagerConfig         `json:"ssl_management"`
	DDoSProtection      DDoSProtectionConfig     `json:"ddos_protection"`
	TrafficShaping      TrafficShapingConfig     `json:"traffic_shaping"`
	SessionPersistence  SessionPersistenceConfig `json:"session_persistence"`
	Metrics             MetricsConfig            `json:"metrics"`
	
	// Advanced features
	GlobalLoadBalancing *GLBConfig               `json:"global_load_balancing,omitempty"`
	MultiTenant         *MultiTenantConfig       `json:"multi_tenant,omitempty"`
	ConnectionPooling   *ConnectionPoolingConfig `json:"connection_pooling,omitempty"`
	
	// Custom extensions
	Extensions          map[string]interface{}   `json:"extensions,omitempty"`
}

// ConfigurationVersion represents a versioned configuration
type ConfigurationVersion struct {
	Version         int64                        `json:"version"`
	Configuration   *LoadBalancerConfiguration   `json:"configuration"`
	ChangeLog       []ConfigChange               `json:"change_log"`
	AppliedAt       time.Time                    `json:"applied_at"`
	AppliedBy       string                       `json:"applied_by"`
	RollbackInfo    *RollbackInfo                `json:"rollback_info,omitempty"`
	ValidationResults *ValidationResults         `json:"validation_results,omitempty"`
}

// ConfigChange represents a configuration change
type ConfigChange struct {
	Type        ChangeType             `json:"type"`
	Path        string                 `json:"path"`
	OldValue    interface{}            `json:"old_value,omitempty"`
	NewValue    interface{}            `json:"new_value,omitempty"`
	Description string                 `json:"description"`
	Timestamp   time.Time              `json:"timestamp"`
}

// RollbackInfo contains information for rollback operations
type RollbackInfo struct {
	PreviousVersion    int64        `json:"previous_version"`
	RollbackReason     string       `json:"rollback_reason"`
	RollbackTimestamp  time.Time    `json:"rollback_timestamp"`
	CanRollback        bool         `json:"can_rollback"`
	RollbackPath       string       `json:"rollback_path"`
}

// ValidationResults contains configuration validation results
type ValidationResults struct {
	Valid           bool                       `json:"valid"`
	Errors          []ValidationError          `json:"errors,omitempty"`
	Warnings        []ValidationWarning        `json:"warnings,omitempty"`
	ValidatedAt     time.Time                  `json:"validated_at"`
	ValidatedBy     string                     `json:"validated_by"`
	ValidationTime  time.Duration              `json:"validation_time"`
}

// ValidationError represents a configuration validation error
type ValidationError struct {
	Path        string                 `json:"path"`
	Message     string                 `json:"message"`
	Severity    ErrorSeverity          `json:"severity"`
	Code        string                 `json:"code"`
	Details     map[string]interface{} `json:"details,omitempty"`
}

// ValidationWarning represents a configuration validation warning
type ValidationWarning struct {
	Path        string                 `json:"path"`
	Message     string                 `json:"message"`
	Code        string                 `json:"code"`
	Details     map[string]interface{} `json:"details,omitempty"`
}

// ConfigurationSource defines interface for configuration sources
type ConfigurationSource interface {
	Load() (*LoadBalancerConfiguration, error)
	Save(config *LoadBalancerConfiguration) error
	Watch(callback func(*LoadBalancerConfiguration)) error
	Close() error
	Name() string
}

// ConfigChangeListener is now defined in types.go

// ConfigValidator validates configuration changes
type ConfigValidator struct {
	rules           []ValidationRule
	strictMode      bool
	timeout         time.Duration
}

// ValidationRule defines a configuration validation rule
type ValidationRule interface {
	Validate(config *LoadBalancerConfiguration) []ValidationError
	Name() string
	Priority() int
}

// FileWatcher watches configuration files for changes
type FileWatcher struct {
	watchPaths      []string
	patterns        []string
	callback        func(string)
	stopCh          chan struct{}
	watchInterval   time.Duration
	lastModTimes    map[string]time.Time
	mutex           sync.RWMutex
}

// ConfigTemplate represents a configuration template
type ConfigTemplate struct {
	ID              string                 `json:"id"`
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Template        string                 `json:"template"`
	Variables       map[string]interface{} `json:"variables"`
	RequiredVars    []string               `json:"required_vars"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// EnvironmentConfig represents environment-specific configuration
type EnvironmentConfig struct {
	Name            string                 `json:"name"`
	Description     string                 `json:"description"`
	Variables       map[string]interface{} `json:"variables"`
	Overrides       map[string]interface{} `json:"overrides"`
	Constraints     map[string]interface{} `json:"constraints"`
	CreatedAt       time.Time              `json:"created_at"`
	UpdatedAt       time.Time              `json:"updated_at"`
}

// RemoteEndpoint represents a remote configuration endpoint
type RemoteEndpoint struct {
	Name        string            `json:"name"`
	URL         string            `json:"url"`
	Method      string            `json:"method"`
	Headers     map[string]string `json:"headers"`
	Timeout     time.Duration     `json:"timeout"`
	RetryCount  int               `json:"retry_count"`
	RetryDelay  time.Duration     `json:"retry_delay"`
	Enabled     bool              `json:"enabled"`
}

// Types and enums
type ChangeType string
type ErrorSeverity string

const (
	ChangeTypeAdd        ChangeType = "add"
	ChangeTypeUpdate     ChangeType = "update"
	ChangeTypeDelete     ChangeType = "delete"
	ChangeTypeMove       ChangeType = "move"
	
	ErrorSeverityError   ErrorSeverity = "error"
	ErrorSeverityWarning ErrorSeverity = "warning"
	ErrorSeverityInfo    ErrorSeverity = "info"
)

// NewConfigManager creates a new configuration manager
func NewConfigManager(config ConfigManagerConfig) *ConfigManager {
	ctx, cancel := context.WithCancel(context.Background())
	
	return &ConfigManager{
		config:              config,
		configHistory:       make([]*ConfigurationVersion, 0),
		sources:             make(map[string]ConfigurationSource),
		listeners:           make([]ConfigChangeListener, 0),
		templates:           make(map[string]*ConfigTemplate),
		environments:        make(map[string]*EnvironmentConfig),
		currentEnvironment:  config.DefaultEnvironment,
		rollbackEnabled:     config.EnableRollback,
		maxRollbackVersions: config.MaxRollbackVersions,
		ctx:                 ctx,
		cancel:              cancel,
	}
}

// Start initializes and starts the configuration manager
func (cm *ConfigManager) Start() error {
	if cm.initialized {
		return fmt.Errorf("configuration manager already started")
	}
	
	// Create directories if they don't exist
	if err := cm.createDirectories(); err != nil {
		return fmt.Errorf("failed to create directories: %w", err)
	}
	
	// Initialize validator
	if cm.config.EnableValidation {
		cm.validator = &ConfigValidator{
			rules:      make([]ValidationRule, 0),
			strictMode: cm.config.StrictValidation,
			timeout:    cm.config.ValidationTimeout,
		}
		cm.registerBuiltinValidationRules()
	}
	
	// Register built-in configuration sources
	cm.registerBuiltinSources()
	
	// Load environment configurations
	if cm.config.EnableEnvironments {
		if err := cm.loadEnvironmentConfigs(); err != nil {
			return fmt.Errorf("failed to load environment configs: %w", err)
		}
	}
	
	// Load configuration templates
	if err := cm.loadConfigTemplates(); err != nil {
		return fmt.Errorf("failed to load config templates: %w", err)
	}
	
	// Load initial configuration
	if err := cm.loadInitialConfiguration(); err != nil {
		return fmt.Errorf("failed to load initial configuration: %w", err)
	}
	
	// Start file watcher if enabled
	if cm.config.EnableFileWatching {
		if err := cm.startFileWatcher(); err != nil {
			return fmt.Errorf("failed to start file watcher: %w", err)
		}
	}
	
	// Start background processes
	if cm.config.EnableRemoteConfig {
		go cm.remoteConfigLoop()
	}
	
	if cm.config.EnableMetrics {
		go cm.metricsLoop()
	}
	
	go cm.maintenanceLoop()
	
	cm.initialized = true
	return nil
}

// Stop stops the configuration manager
func (cm *ConfigManager) Stop() error {
	cm.cancel()
	
	// Stop file watcher
	if cm.fileWatcher != nil {
		close(cm.fileWatcher.stopCh)
	}
	
	// Close all sources
	cm.sourcesMutex.RLock()
	for _, source := range cm.sources {
		source.Close()
	}
	cm.sourcesMutex.RUnlock()
	
	cm.initialized = false
	return nil
}

// GetCurrentConfiguration returns the current configuration
func (cm *ConfigManager) GetCurrentConfiguration() *LoadBalancerConfiguration {
	cm.configMutex.RLock()
	defer cm.configMutex.RUnlock()
	
	if cm.loadBalancerConfig == nil {
		return nil
	}
	
	// Return deep copy
	configBytes, _ := json.Marshal(cm.loadBalancerConfig)
	var configCopy LoadBalancerConfiguration
	json.Unmarshal(configBytes, &configCopy)
	
	return &configCopy
}

// UpdateConfiguration updates the configuration
func (cm *ConfigManager) UpdateConfiguration(newConfig *LoadBalancerConfiguration) error {
	// Validate configuration if validation is enabled
	if cm.config.EnableValidation {
		if err := cm.validateConfiguration(newConfig); err != nil {
			return fmt.Errorf("configuration validation failed: %w", err)
		}
	}
	
	// Get current configuration
	oldConfig := cm.GetCurrentConfiguration()
	
	// Calculate changes
	changes := cm.calculateChanges(oldConfig, newConfig)
	
	// Create configuration version
	version := &ConfigurationVersion{
		Version:       atomic.AddInt64(&cm.configVersion, 1),
		Configuration: newConfig,
		ChangeLog:     changes,
		AppliedAt:     time.Now(),
		AppliedBy:     "system", // Would get from context in real implementation
	}
	
	// Apply configuration
	if err := cm.applyConfiguration(version); err != nil {
		// Rollback if auto-rollback is enabled
		if cm.config.AutoRollbackOnFailure && oldConfig != nil {
			cm.rollbackToVersion(oldConfig.Version)
		}
		return fmt.Errorf("failed to apply configuration: %w", err)
	}
	
	return nil
}

// ReloadConfiguration reloads configuration from sources
func (cm *ConfigManager) ReloadConfiguration() error {
	if !atomic.CompareAndSwapInt32(&cm.reloadInProgress, 0, 1) {
		return fmt.Errorf("reload already in progress")
	}
	defer atomic.StoreInt32(&cm.reloadInProgress, 0)
	
	// Load configuration from primary source
	cm.sourcesMutex.RLock()
	var primarySource ConfigurationSource
	for _, source := range cm.sources {
		primarySource = source
		break // Use first source as primary
	}
	cm.sourcesMutex.RUnlock()
	
	if primarySource == nil {
		return fmt.Errorf("no configuration sources available")
	}
	
	newConfig, err := primarySource.Load()
	if err != nil {
		return fmt.Errorf("failed to load configuration from %s: %w", primarySource.Name(), err)
	}
	
	// Update configuration
	if err := cm.UpdateConfiguration(newConfig); err != nil {
		return err
	}
	
	cm.lastReloadTime = time.Now()
	atomic.AddInt64(&cm.reloadCount, 1)
	
	// Notify listeners
	cm.notifyConfigReload(newConfig)
	
	return nil
}

// RollbackToVersion rolls back to a specific configuration version
func (cm *ConfigManager) RollbackToVersion(version int64) error {
	if !cm.rollbackEnabled {
		return fmt.Errorf("rollback is disabled")
	}
	
	// Find the target version
	cm.historyMutex.RLock()
	var targetVersion *ConfigurationVersion
	for _, v := range cm.configHistory {
		if v.Version == version {
			targetVersion = v
			break
		}
	}
	cm.historyMutex.RUnlock()
	
	if targetVersion == nil {
		return fmt.Errorf("version %d not found in history", version)
	}
	
	return cm.rollbackToVersion(version)
}

// GetConfigurationHistory returns configuration history
func (cm *ConfigManager) GetConfigurationHistory(limit int) []*ConfigurationVersion {
	cm.historyMutex.RLock()
	defer cm.historyMutex.RUnlock()
	
	if limit <= 0 || limit > len(cm.configHistory) {
		limit = len(cm.configHistory)
	}
	
	history := make([]*ConfigurationVersion, limit)
	// Return most recent first
	for i := 0; i < limit; i++ {
		history[i] = cm.configHistory[len(cm.configHistory)-1-i]
	}
	
	return history
}

// RegisterConfigChangeListener registers a configuration change listener
func (cm *ConfigManager) RegisterConfigChangeListener(listener ConfigChangeListener) {
	cm.listenersMutex.Lock()
	defer cm.listenersMutex.Unlock()
	
	cm.listeners = append(cm.listeners, listener)
}

// UnregisterConfigChangeListener unregisters a configuration change listener
func (cm *ConfigManager) UnregisterConfigChangeListener(listenerName string) {
	cm.listenersMutex.Lock()
	defer cm.listenersMutex.Unlock()
	
	for i, listener := range cm.listeners {
		if listener.Name() == listenerName {
			cm.listeners = append(cm.listeners[:i], cm.listeners[i+1:]...)
			break
		}
	}
}

// Configuration management implementation

// createDirectories creates necessary directories
func (cm *ConfigManager) createDirectories() error {
	dirs := []string{
		cm.config.ConfigDirectory,
		cm.config.BackupDirectory,
		cm.config.TemplateDirectory,
	}
	
	for _, dir := range dirs {
		if dir != "" {
			if err := os.MkdirAll(dir, 0755); err != nil {
				return fmt.Errorf("failed to create directory %s: %w", dir, err)
			}
		}
	}
	
	return nil
}

// registerBuiltinSources registers built-in configuration sources
func (cm *ConfigManager) registerBuiltinSources() {
	// File source
	if cm.config.ConfigFilePath != "" {
		fileSource := &FileConfigurationSource{
			filePath: cm.config.ConfigFilePath,
		}
		cm.sources["file"] = fileSource
	}
	
	// Directory source
	if cm.config.ConfigDirectory != "" {
		dirSource := &DirectoryConfigurationSource{
			directory: cm.config.ConfigDirectory,
		}
		cm.sources["directory"] = dirSource
	}
}

// registerBuiltinValidationRules registers built-in validation rules
func (cm *ConfigManager) registerBuiltinValidationRules() {
	rules := []ValidationRule{
		&BasicConfigValidationRule{},
		&ServiceValidationRule{},
		&BackendValidationRule{},
		&SSLValidationRule{},
		&SecurityValidationRule{},
	}
	
	cm.validator.rules = append(cm.validator.rules, rules...)
}

// loadEnvironmentConfigs loads environment-specific configurations
func (cm *ConfigManager) loadEnvironmentConfigs() error {
	if cm.config.EnvironmentConfigPath == "" {
		return nil
	}
	
	files, err := ioutil.ReadDir(cm.config.EnvironmentConfigPath)
	if err != nil {
		return fmt.Errorf("failed to read environment config directory: %w", err)
	}
	
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".json" {
			envName := strings.TrimSuffix(file.Name(), ".json")
			envPath := filepath.Join(cm.config.EnvironmentConfigPath, file.Name())
			
			data, err := ioutil.ReadFile(envPath)
			if err != nil {
				continue
			}
			
			var envConfig EnvironmentConfig
			if err := json.Unmarshal(data, &envConfig); err != nil {
				continue
			}
			
			cm.environments[envName] = &envConfig
		}
	}
	
	return nil
}

// loadConfigTemplates loads configuration templates
func (cm *ConfigManager) loadConfigTemplates() error {
	if cm.config.TemplateDirectory == "" {
		return nil
	}
	
	files, err := ioutil.ReadDir(cm.config.TemplateDirectory)
	if err != nil {
		return fmt.Errorf("failed to read template directory: %w", err)
	}
	
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".json" {
			templatePath := filepath.Join(cm.config.TemplateDirectory, file.Name())
			
			data, err := ioutil.ReadFile(templatePath)
			if err != nil {
				continue
			}
			
			var template ConfigTemplate
			if err := json.Unmarshal(data, &template); err != nil {
				continue
			}
			
			cm.templatesMutex.Lock()
			cm.templates[template.ID] = &template
			cm.templatesMutex.Unlock()
		}
	}
	
	return nil
}

// loadInitialConfiguration loads the initial configuration
func (cm *ConfigManager) loadInitialConfiguration() error {
	cm.sourcesMutex.RLock()
	var primarySource ConfigurationSource
	for _, source := range cm.sources {
		primarySource = source
		break
	}
	cm.sourcesMutex.RUnlock()
	
	if primarySource == nil {
		return fmt.Errorf("no configuration sources available")
	}
	
	config, err := primarySource.Load()
	if err != nil {
		// Create default configuration if none exists
		config = cm.createDefaultConfiguration()
	}
	
	// Apply environment-specific overrides
	if cm.currentEnvironment != "" {
		config = cm.applyEnvironmentOverrides(config, cm.currentEnvironment)
	}
	
	// Set initial version
	config.Version = 1
	config.CreatedAt = time.Now()
	config.UpdatedAt = time.Now()
	config.Checksum = cm.calculateChecksum(config)
	
	cm.configMutex.Lock()
	cm.loadBalancerConfig = config
	atomic.StoreInt64(&cm.configVersion, config.Version)
	cm.configMutex.Unlock()
	
	// Add to history
	version := &ConfigurationVersion{
		Version:       config.Version,
		Configuration: config,
		AppliedAt:     time.Now(),
		AppliedBy:     "system",
	}
	
	cm.historyMutex.Lock()
	cm.configHistory = append(cm.configHistory, version)
	cm.historyMutex.Unlock()
	
	return nil
}

// startFileWatcher starts the file watcher
func (cm *ConfigManager) startFileWatcher() error {
	watchPaths := []string{cm.config.ConfigFilePath}
	if cm.config.ConfigDirectory != "" {
		watchPaths = append(watchPaths, cm.config.ConfigDirectory)
	}
	
	cm.fileWatcher = &FileWatcher{
		watchPaths:    watchPaths,
		patterns:      cm.config.FileWatchPatterns,
		callback:      cm.onFileChange,
		stopCh:        make(chan struct{}),
		watchInterval: cm.config.WatchInterval,
		lastModTimes:  make(map[string]time.Time),
	}
	
	go cm.fileWatcher.watch()
	
	return nil
}

// onFileChange handles file change events
func (cm *ConfigManager) onFileChange(filePath string) {
	if cm.config.EnableHotReload {
		// Add delay to avoid multiple rapid reloads
		time.Sleep(cm.config.ReloadDelay)
		
		if err := cm.ReloadConfiguration(); err != nil {
			fmt.Printf("Failed to reload configuration after file change: %v\n", err)
		}
	}
}

// validateConfiguration validates a configuration
func (cm *ConfigManager) validateConfiguration(config *LoadBalancerConfiguration) error {
	if cm.validator == nil {
		return nil
	}
	
	var allErrors []ValidationError
	// var allWarnings []ValidationWarning // Not currently used
	
	// Run all validation rules
	for _, rule := range cm.validator.rules {
		errors := rule.Validate(config)
		allErrors = append(allErrors, errors...)
	}
	
	// Check for critical errors
	for _, err := range allErrors {
		if err.Severity == ErrorSeverityError {
			if cm.validator.strictMode {
				return fmt.Errorf("validation failed: %s at %s", err.Message, err.Path)
			}
		}
	}
	
	// Validation completed successfully
	
	if len(allErrors) > 0 && cm.validator.strictMode {
		return fmt.Errorf("configuration validation failed with %d errors", len(allErrors))
	}
	
	return nil
}

// calculateChanges calculates changes between two configurations
func (cm *ConfigManager) calculateChanges(oldConfig, newConfig *LoadBalancerConfiguration) []ConfigChange {
	var changes []ConfigChange
	
	// Simple change detection - in practice this would be more sophisticated
	if oldConfig == nil {
		changes = append(changes, ConfigChange{
			Type:        ChangeTypeAdd,
			Path:        "root",
			NewValue:    newConfig,
			Description: "Initial configuration",
			Timestamp:   time.Now(),
		})
		return changes
	}
	
	// Compare versions
	if oldConfig.Version != newConfig.Version {
		changes = append(changes, ConfigChange{
			Type:        ChangeTypeUpdate,
			Path:        "version",
			OldValue:    oldConfig.Version,
			NewValue:    newConfig.Version,
			Description: "Version updated",
			Timestamp:   time.Now(),
		})
	}
	
	// Compare services
	if len(oldConfig.Services) != len(newConfig.Services) {
		changes = append(changes, ConfigChange{
			Type:        ChangeTypeUpdate,
			Path:        "services",
			OldValue:    len(oldConfig.Services),
			NewValue:    len(newConfig.Services),
			Description: "Service count changed",
			Timestamp:   time.Now(),
		})
	}
	
	// Compare backends
	if len(oldConfig.Backends) != len(newConfig.Backends) {
		changes = append(changes, ConfigChange{
			Type:        ChangeTypeUpdate,
			Path:        "backends",
			OldValue:    len(oldConfig.Backends),
			NewValue:    len(newConfig.Backends),
			Description: "Backend count changed",
			Timestamp:   time.Now(),
		})
	}
	
	return changes
}

// applyConfiguration applies a configuration version
func (cm *ConfigManager) applyConfiguration(version *ConfigurationVersion) error {
	// Get current configuration
	oldConfig := cm.GetCurrentConfiguration()
	
	// Update checksum
	version.Configuration.Checksum = cm.calculateChecksum(version.Configuration)
	version.Configuration.UpdatedAt = time.Now()
	
	// Apply configuration atomically
	cm.configMutex.Lock()
	cm.loadBalancerConfig = version.Configuration
	atomic.StoreInt64(&cm.configVersion, version.Version)
	cm.configMutex.Unlock()
	
	// Add to history
	cm.historyMutex.Lock()
	cm.configHistory = append(cm.configHistory, version)
	
	// Limit history size
	if len(cm.configHistory) > cm.maxRollbackVersions {
		cm.configHistory = cm.configHistory[len(cm.configHistory)-cm.maxRollbackVersions:]
	}
	cm.historyMutex.Unlock()
	
	// Save to backup
	if err := cm.saveConfigurationBackup(version.Configuration); err != nil {
		fmt.Printf("Warning: Failed to save configuration backup: %v\n", err)
	}
	
	// Notify listeners
	cm.notifyConfigChange(oldConfig, version.Configuration)
	
	return nil
}

// rollbackToVersion rolls back to a specific version
func (cm *ConfigManager) rollbackToVersion(version int64) error {
	cm.historyMutex.RLock()
	var targetConfig *LoadBalancerConfiguration
	for _, v := range cm.configHistory {
		if v.Version == version {
			targetConfig = v.Configuration
			break
		}
	}
	cm.historyMutex.RUnlock()
	
	if targetConfig == nil {
		return fmt.Errorf("version %d not found", version)
	}
	
	// Create rollback version
	rollbackVersion := &ConfigurationVersion{
		Version:       atomic.AddInt64(&cm.configVersion, 1),
		Configuration: targetConfig,
		AppliedAt:     time.Now(),
		AppliedBy:     "rollback",
		RollbackInfo: &RollbackInfo{
			PreviousVersion:   cm.loadBalancerConfig.Version,
			RollbackReason:    "manual rollback",
			RollbackTimestamp: time.Now(),
			CanRollback:       true,
			RollbackPath:      fmt.Sprintf("rollback to version %d", version),
		},
	}
	
	// Apply rollback configuration
	if err := cm.applyConfiguration(rollbackVersion); err != nil {
		return fmt.Errorf("failed to apply rollback configuration: %w", err)
	}
	
	// Notify listeners
	cm.notifyConfigRollback(targetConfig)
	
	return nil
}

// createDefaultConfiguration creates a default configuration
func (cm *ConfigManager) createDefaultConfiguration() *LoadBalancerConfiguration {
	return &LoadBalancerConfiguration{
		Version:            1,
		Name:               "Default Load Balancer",
		Description:        "Default load balancer configuration",
		Environment:        cm.currentEnvironment,
		CreatedAt:          time.Now(),
		UpdatedAt:          time.Now(),
		CreatedBy:          "system",
		LoadBalancer:       DefaultLoadBalancerConfig(),
		Services:           []*LoadBalancerService{},
		Backends:           []*Backend{},
		HealthChecking:     DefaultAdvancedHealthConfig(),
		SSLManagement:      DefaultSSLManagerConfig(),
		DDoSProtection:     DefaultDDoSProtectionConfig(),
		TrafficShaping:     DefaultTrafficShapingConfig(),
		SessionPersistence: DefaultSessionPersistenceConfig(),
		Metrics:           DefaultMetricsConfig(),
		Extensions:         make(map[string]interface{}),
	}
}

// applyEnvironmentOverrides applies environment-specific overrides
func (cm *ConfigManager) applyEnvironmentOverrides(config *LoadBalancerConfiguration, env string) *LoadBalancerConfiguration {
	envConfig, exists := cm.environments[env]
	if !exists {
		return config
	}
	
	// Apply overrides (simplified)
	config.Environment = env
	
	// Apply variable substitutions
	for key, value := range envConfig.Variables {
		// Would implement template variable substitution here
		config.Extensions[key] = value
	}
	
	return config
}

// calculateChecksum calculates configuration checksum
func (cm *ConfigManager) calculateChecksum(config *LoadBalancerConfiguration) string {
	data, _ := json.Marshal(config)
	hash := sha256.Sum256(data)
	return fmt.Sprintf("%x", hash)
}

// saveConfigurationBackup saves configuration to backup directory
func (cm *ConfigManager) saveConfigurationBackup(config *LoadBalancerConfiguration) error {
	if cm.config.BackupDirectory == "" {
		return nil
	}
	
	filename := fmt.Sprintf("config-v%d-%d.json", config.Version, time.Now().Unix())
	backupPath := filepath.Join(cm.config.BackupDirectory, filename)
	
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	
	return ioutil.WriteFile(backupPath, data, 0644)
}

// Notification methods

// notifyConfigChange notifies listeners of configuration changes
func (cm *ConfigManager) notifyConfigChange(oldConfig, newConfig *LoadBalancerConfiguration) {
	cm.listenersMutex.RLock()
	defer cm.listenersMutex.RUnlock()
	
	for _, listener := range cm.listeners {
		go func(l ConfigChangeListener) {
			if err := l.OnConfigChange(oldConfig, newConfig); err != nil {
				fmt.Printf("Config change listener %s failed: %v\n", l.Name(), err)
			}
		}(listener)
	}
}

// notifyConfigReload notifies listeners of configuration reload
func (cm *ConfigManager) notifyConfigReload(config *LoadBalancerConfiguration) {
	cm.listenersMutex.RLock()
	defer cm.listenersMutex.RUnlock()
	
	for _, listener := range cm.listeners {
		go func(l ConfigChangeListener) {
			if err := l.OnConfigReload(config); err != nil {
				fmt.Printf("Config reload listener %s failed: %v\n", l.Name(), err)
			}
		}(listener)
	}
}

// notifyConfigRollback notifies listeners of configuration rollback
func (cm *ConfigManager) notifyConfigRollback(config *LoadBalancerConfiguration) {
	cm.listenersMutex.RLock()
	defer cm.listenersMutex.RUnlock()
	
	for _, listener := range cm.listeners {
		go func(l ConfigChangeListener) {
			if err := l.OnConfigRollback(config); err != nil {
				fmt.Printf("Config rollback listener %s failed: %v\n", l.Name(), err)
			}
		}(listener)
	}
}

// Background loops

// remoteConfigLoop fetches configuration from remote sources
func (cm *ConfigManager) remoteConfigLoop() {
	ticker := time.NewTicker(cm.config.RemoteConfigInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-cm.ctx.Done():
			return
		case <-ticker.C:
			cm.fetchRemoteConfiguration()
		}
	}
}

// fetchRemoteConfiguration fetches configuration from remote endpoints
func (cm *ConfigManager) fetchRemoteConfiguration() {
	for _, endpoint := range cm.config.RemoteConfigEndpoints {
		if !endpoint.Enabled {
			continue
		}
		
		// Simplified remote config fetch
		// In practice, this would implement HTTP client with retries, authentication, etc.
	}
}

// metricsLoop collects configuration management metrics
func (cm *ConfigManager) metricsLoop() {
	ticker := time.NewTicker(cm.config.MetricsInterval)
	defer ticker.Stop()
	
	for {
		select {
		case <-cm.ctx.Done():
			return
		case <-ticker.C:
			cm.collectMetrics()
		}
	}
}

// collectMetrics collects configuration management metrics
func (cm *ConfigManager) collectMetrics() {
	// Collect metrics about configuration management
	// Number of reloads, rollbacks, validation failures, etc.
}

// maintenanceLoop performs periodic maintenance tasks
func (cm *ConfigManager) maintenanceLoop() {
	ticker := time.NewTicker(time.Hour) // Run maintenance every hour
	defer ticker.Stop()
	
	for {
		select {
		case <-cm.ctx.Done():
			return
		case <-ticker.C:
			cm.performMaintenance()
		}
	}
}

// performMaintenance performs maintenance tasks
func (cm *ConfigManager) performMaintenance() {
	// Clean up old backups
	cm.cleanupOldBackups()
	
	// Compact configuration history
	cm.compactConfigHistory()
}

// cleanupOldBackups removes old configuration backups
func (cm *ConfigManager) cleanupOldBackups() {
	if cm.config.BackupDirectory == "" {
		return
	}
	
	files, err := ioutil.ReadDir(cm.config.BackupDirectory)
	if err != nil {
		return
	}
	
	// Remove backups older than retention period (simplified)
	cutoff := time.Now().Add(-7 * 24 * time.Hour) // 7 days
	
	for _, file := range files {
		if file.ModTime().Before(cutoff) {
			backupPath := filepath.Join(cm.config.BackupDirectory, file.Name())
			os.Remove(backupPath)
		}
	}
}

// compactConfigHistory compacts configuration history
func (cm *ConfigManager) compactConfigHistory() {
	cm.historyMutex.Lock()
	defer cm.historyMutex.Unlock()
	
	// Keep only the most recent versions up to the limit
	if len(cm.configHistory) > cm.maxRollbackVersions {
		cm.configHistory = cm.configHistory[len(cm.configHistory)-cm.maxRollbackVersions:]
	}
}

// File watcher implementation

// watch starts watching files for changes
func (fw *FileWatcher) watch() {
	if fw.watchInterval == 0 {
		fw.watchInterval = 5 * time.Second
	}
	
	ticker := time.NewTicker(fw.watchInterval)
	defer ticker.Stop()
	
	// Initialize modification times
	fw.updateModificationTimes()
	
	for {
		select {
		case <-fw.stopCh:
			return
		case <-ticker.C:
			fw.checkForChanges()
		}
	}
}

// updateModificationTimes updates the modification times for watched files
func (fw *FileWatcher) updateModificationTimes() {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	for _, path := range fw.watchPaths {
		if stat, err := os.Stat(path); err == nil {
			fw.lastModTimes[path] = stat.ModTime()
		}
	}
}

// checkForChanges checks for file changes
func (fw *FileWatcher) checkForChanges() {
	fw.mutex.Lock()
	defer fw.mutex.Unlock()
	
	for _, path := range fw.watchPaths {
		if stat, err := os.Stat(path); err == nil {
			lastMod := fw.lastModTimes[path]
			if stat.ModTime().After(lastMod) {
				fw.lastModTimes[path] = stat.ModTime()
				
				// Notify callback
				if fw.callback != nil {
					go fw.callback(path)
				}
			}
		}
	}
}

// Configuration source implementations

// FileConfigurationSource implements file-based configuration source
type FileConfigurationSource struct {
	filePath string
}

func (fcs *FileConfigurationSource) Load() (*LoadBalancerConfiguration, error) {
	data, err := ioutil.ReadFile(fcs.filePath)
	if err != nil {
		return nil, err
	}
	
	var config LoadBalancerConfiguration
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	
	return &config, nil
}

func (fcs *FileConfigurationSource) Save(config *LoadBalancerConfiguration) error {
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	
	return ioutil.WriteFile(fcs.filePath, data, 0644)
}

func (fcs *FileConfigurationSource) Watch(callback func(*LoadBalancerConfiguration)) error {
	// Would implement file watching here
	return nil
}

func (fcs *FileConfigurationSource) Close() error {
	return nil
}

func (fcs *FileConfigurationSource) Name() string {
	return "file"
}

// DirectoryConfigurationSource implements directory-based configuration source
type DirectoryConfigurationSource struct {
	directory string
}

func (dcs *DirectoryConfigurationSource) Load() (*LoadBalancerConfiguration, error) {
	// Load main config file from directory
	configPath := filepath.Join(dcs.directory, "config.json")
	
	data, err := ioutil.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	
	var config LoadBalancerConfiguration
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, err
	}
	
	return &config, nil
}

func (dcs *DirectoryConfigurationSource) Save(config *LoadBalancerConfiguration) error {
	configPath := filepath.Join(dcs.directory, "config.json")
	
	data, err := json.MarshalIndent(config, "", "  ")
	if err != nil {
		return err
	}
	
	return ioutil.WriteFile(configPath, data, 0644)
}

func (dcs *DirectoryConfigurationSource) Watch(callback func(*LoadBalancerConfiguration)) error {
	return nil
}

func (dcs *DirectoryConfigurationSource) Close() error {
	return nil
}

func (dcs *DirectoryConfigurationSource) Name() string {
	return "directory"
}

// Validation rule implementations

// BasicConfigValidationRule validates basic configuration structure
type BasicConfigValidationRule struct{}

func (bcvr *BasicConfigValidationRule) Validate(config *LoadBalancerConfiguration) []ValidationError {
	var errors []ValidationError
	
	if config.Name == "" {
		errors = append(errors, ValidationError{
			Path:     "name",
			Message:  "Configuration name cannot be empty",
			Severity: ErrorSeverityError,
			Code:     "EMPTY_NAME",
		})
	}
	
	if config.Version <= 0 {
		errors = append(errors, ValidationError{
			Path:     "version",
			Message:  "Configuration version must be positive",
			Severity: ErrorSeverityError,
			Code:     "INVALID_VERSION",
		})
	}
	
	return errors
}

func (bcvr *BasicConfigValidationRule) Name() string {
	return "basic_config"
}

func (bcvr *BasicConfigValidationRule) Priority() int {
	return 100
}

// ServiceValidationRule validates service configuration
type ServiceValidationRule struct{}

func (svr *ServiceValidationRule) Validate(config *LoadBalancerConfiguration) []ValidationError {
	var errors []ValidationError
	
	for i, service := range config.Services {
		path := fmt.Sprintf("services[%d]", i)
		
		if service.Name == "" {
			errors = append(errors, ValidationError{
				Path:     path + ".name",
				Message:  "Service name cannot be empty",
				Severity: ErrorSeverityError,
				Code:     "EMPTY_SERVICE_NAME",
			})
		}
		
		if service.ListenPort <= 0 || service.ListenPort > 65535 {
			errors = append(errors, ValidationError{
				Path:     path + ".listen_port",
				Message:  "Service listen port must be between 1 and 65535",
				Severity: ErrorSeverityError,
				Code:     "INVALID_PORT",
			})
		}
		
		if len(service.Backends) == 0 {
			errors = append(errors, ValidationError{
				Path:     path + ".backends",
				Message:  "Service must have at least one backend",
				Severity: ErrorSeverityError,
				Code:     "NO_BACKENDS",
			})
		}
	}
	
	return errors
}

func (svr *ServiceValidationRule) Name() string {
	return "service_validation"
}

func (svr *ServiceValidationRule) Priority() int {
	return 90
}

// BackendValidationRule validates backend configuration
type BackendValidationRule struct{}

func (bvr *BackendValidationRule) Validate(config *LoadBalancerConfiguration) []ValidationError {
	var errors []ValidationError
	
	for i, backend := range config.Backends {
		path := fmt.Sprintf("backends[%d]", i)
		
		if backend.Address == "" {
			errors = append(errors, ValidationError{
				Path:     path + ".address",
				Message:  "Backend address cannot be empty",
				Severity: ErrorSeverityError,
				Code:     "EMPTY_ADDRESS",
			})
		}
		
		if backend.Port <= 0 || backend.Port > 65535 {
			errors = append(errors, ValidationError{
				Path:     path + ".port",
				Message:  "Backend port must be between 1 and 65535",
				Severity: ErrorSeverityError,
				Code:     "INVALID_PORT",
			})
		}
		
		if backend.Weight < 0 {
			errors = append(errors, ValidationError{
				Path:     path + ".weight",
				Message:  "Backend weight cannot be negative",
				Severity: ErrorSeverityWarning,
				Code:     "NEGATIVE_WEIGHT",
			})
		}
	}
	
	return errors
}

func (bvr *BackendValidationRule) Name() string {
	return "backend_validation"
}

func (bvr *BackendValidationRule) Priority() int {
	return 80
}

// SSLValidationRule validates SSL configuration
type SSLValidationRule struct{}

func (svr *SSLValidationRule) Validate(config *LoadBalancerConfiguration) []ValidationError {
	var errors []ValidationError
	
	if config.SSLManagement.EnableSSL {
		if config.SSLManagement.CertStorePath == "" {
			errors = append(errors, ValidationError{
				Path:     "ssl_management.cert_store_path",
				Message:  "SSL certificate store path cannot be empty when SSL is enabled",
				Severity: ErrorSeverityError,
				Code:     "EMPTY_CERT_PATH",
			})
		}
		
		if config.SSLManagement.PrivateKeyPath == "" {
			errors = append(errors, ValidationError{
				Path:     "ssl_management.private_key_path",
				Message:  "SSL private key path cannot be empty when SSL is enabled",
				Severity: ErrorSeverityError,
				Code:     "EMPTY_KEY_PATH",
			})
		}
	}
	
	return errors
}

func (svr *SSLValidationRule) Name() string {
	return "ssl_validation"
}

func (svr *SSLValidationRule) Priority() int {
	return 70
}

// SecurityValidationRule validates security configuration
type SecurityValidationRule struct{}

func (svr *SecurityValidationRule) Validate(config *LoadBalancerConfiguration) []ValidationError {
	var errors []ValidationError
	
	// Check for insecure configurations
	if !config.DDoSProtection.EnableProtection {
		errors = append(errors, ValidationError{
			Path:     "ddos_protection.enable_protection",
			Message:  "DDoS protection should be enabled for security",
			Severity: ErrorSeverityWarning,
			Code:     "DDOS_DISABLED",
		})
	}
	
	if config.LoadBalancer.MaxConnections <= 0 {
		errors = append(errors, ValidationError{
			Path:     "load_balancer.max_connections",
			Message:  "Maximum connections should be set to prevent resource exhaustion",
			Severity: ErrorSeverityWarning,
			Code:     "UNLIMITED_CONNECTIONS",
		})
	}
	
	return errors
}

func (svr *SecurityValidationRule) Name() string {
	return "security_validation"
}

func (svr *SecurityValidationRule) Priority() int {
	return 60
}

// DefaultConfigManagerConfig returns default configuration manager configuration
func DefaultConfigManagerConfig() ConfigManagerConfig {
	return ConfigManagerConfig{
		ConfigFilePath:           "/etc/novacron/loadbalancer/config.json",
		ConfigDirectory:          "/etc/novacron/loadbalancer/config.d",
		BackupDirectory:          "/var/lib/novacron/loadbalancer/backups",
		TemplateDirectory:        "/etc/novacron/loadbalancer/templates",
		EnableFileWatching:       true,
		WatchInterval:            5 * time.Second,
		FileWatchPatterns:        []string{"*.json", "*.yaml", "*.yml"},
		EnableHotReload:          true,
		ReloadDelay:              2 * time.Second,
		MaxReloadAttempts:        3,
		ReloadTimeout:            30 * time.Second,
		EnableValidation:         true,
		StrictValidation:         true,
		ValidationTimeout:        10 * time.Second,
		EnableRollback:           true,
		MaxRollbackVersions:      10,
		AutoRollbackOnFailure:    true,
		RollbackTimeout:          30 * time.Second,
		EnableRemoteConfig:       false,
		RemoteConfigEndpoints:    []RemoteEndpoint{},
		RemoteConfigInterval:     5 * time.Minute,
		EnableEnvironments:       true,
		DefaultEnvironment:       "production",
		EnvironmentConfigPath:    "/etc/novacron/loadbalancer/environments",
		EnableEncryption:         false,
		EnableSignatureVerification: false,
		EnableMetrics:            true,
		MetricsInterval:          60 * time.Second,
	}
}