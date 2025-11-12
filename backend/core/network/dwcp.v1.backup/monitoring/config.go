package monitoring

import (
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// MonitoringConfig holds complete monitoring system configuration
type MonitoringConfig struct {
	Enabled       bool                `yaml:"enabled"`
	CheckInterval time.Duration       `yaml:"check_interval"`
	BufferSize    int                 `yaml:"buffer_size"`
	Detector      *DetectorConfig     `yaml:"detector"`
	Alert         *AlertConfig        `yaml:"alert"`
	Storage       *StorageConfig      `yaml:"storage"`
}

// StorageConfig configures anomaly storage
type StorageConfig struct {
	Enabled         bool          `yaml:"enabled"`
	Backend         string        `yaml:"backend"` // "memory", "postgres", "elasticsearch"
	RetentionPeriod time.Duration `yaml:"retention_period"`
	MaxAnomalies    int           `yaml:"max_anomalies"`

	// PostgreSQL
	PostgresURL string `yaml:"postgres_url"`

	// Elasticsearch
	ElasticsearchURL string `yaml:"elasticsearch_url"`
	IndexName        string `yaml:"index_name"`
}

// DefaultMonitoringConfig returns default monitoring configuration
func DefaultMonitoringConfig() *MonitoringConfig {
	return &MonitoringConfig{
		Enabled:       true,
		CheckInterval: 10 * time.Second,
		BufferSize:    1000,
		Detector:      DefaultDetectorConfig(),
		Alert:         DefaultAlertConfig(),
		Storage: &StorageConfig{
			Enabled:         true,
			Backend:         "memory",
			RetentionPeriod: 7 * 24 * time.Hour, // 7 days
			MaxAnomalies:    10000,
		},
	}
}

// LoadConfigFromFile loads configuration from YAML file
func LoadConfigFromFile(filepath string) (*MonitoringConfig, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	config := DefaultMonitoringConfig()
	if err := yaml.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("failed to parse config file: %w", err)
	}

	return config, nil
}

// SaveConfigToFile saves configuration to YAML file
func SaveConfigToFile(config *MonitoringConfig, filepath string) error {
	data, err := yaml.Marshal(config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// Validate validates the configuration
func (c *MonitoringConfig) Validate() error {
	if c.CheckInterval <= 0 {
		return fmt.Errorf("check_interval must be positive")
	}

	if c.BufferSize <= 0 {
		return fmt.Errorf("buffer_size must be positive")
	}

	if c.Detector != nil {
		if c.Detector.EnsembleThreshold <= 0 || c.Detector.EnsembleThreshold > 1 {
			return fmt.Errorf("ensemble_threshold must be between 0 and 1")
		}

		if c.Detector.ZScoreWindow <= 0 {
			return fmt.Errorf("zscore_window must be positive")
		}

		if c.Detector.ZScoreThreshold <= 0 {
			return fmt.Errorf("zscore_threshold must be positive")
		}
	}

	if c.Alert != nil {
		if c.Alert.ThrottleDuration < 0 {
			return fmt.Errorf("throttle_duration cannot be negative")
		}
	}

	if c.Storage != nil {
		if c.Storage.RetentionPeriod <= 0 {
			return fmt.Errorf("retention_period must be positive")
		}

		if c.Storage.MaxAnomalies <= 0 {
			return fmt.Errorf("max_anomalies must be positive")
		}
	}

	return nil
}
