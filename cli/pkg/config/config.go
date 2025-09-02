package config

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/mitchellh/go-homedir"
	"gopkg.in/yaml.v3"
)

// Config represents the CLI configuration
type Config struct {
	CurrentCluster string             `json:"currentCluster" yaml:"currentCluster"`
	Clusters       map[string]Cluster `json:"clusters" yaml:"clusters"`
	Preferences    Preferences        `json:"preferences" yaml:"preferences"`
}

// Cluster represents a cluster configuration
type Cluster struct {
	Name      string `json:"name" yaml:"name"`
	Server    string `json:"server" yaml:"server"`
	Insecure  bool   `json:"insecure" yaml:"insecure"`
	Namespace string `json:"namespace" yaml:"namespace"`
	AuthType  string `json:"authType" yaml:"authType"`
	AuthData  string `json:"authData" yaml:"authData"`
}

// Preferences represents user preferences
type Preferences struct {
	Output    string `json:"output" yaml:"output"`
	NoColor   bool   `json:"noColor" yaml:"noColor"`
	Verbose   bool   `json:"verbose" yaml:"verbose"`
	Editor    string `json:"editor" yaml:"editor"`
	PageSize  int    `json:"pageSize" yaml:"pageSize"`
	Timeout   int    `json:"timeout" yaml:"timeout"`
}

// Manager manages CLI configuration
type Manager struct {
	path   string
	config *Config
}

// NewManager creates a new configuration manager
func NewManager(path string) (*Manager, error) {
	if path == "" {
		home, err := homedir.Dir()
		if err != nil {
			return nil, err
		}
		path = filepath.Join(home, ".novacron", "config.yaml")
	}

	m := &Manager{path: path}

	// Load existing config or create default
	if err := m.Load(); err != nil {
		if !os.IsNotExist(err) {
			return nil, err
		}
		// Create default config
		m.config = DefaultConfig()
	}

	return m, nil
}

// DefaultConfig returns the default configuration
func DefaultConfig() *Config {
	return &Config{
		CurrentCluster: "",
		Clusters:       make(map[string]Cluster),
		Preferences: Preferences{
			Output:   "table",
			NoColor:  false,
			Verbose:  false,
			Editor:   os.Getenv("EDITOR"),
			PageSize: 20,
			Timeout:  30,
		},
	}
}

// Load loads the configuration from disk
func (m *Manager) Load() error {
	data, err := ioutil.ReadFile(m.path)
	if err != nil {
		return err
	}

	var config Config
	if err := yaml.Unmarshal(data, &config); err != nil {
		return err
	}

	m.config = &config
	return nil
}

// Save saves the configuration to disk
func (m *Manager) Save() error {
	// Ensure directory exists
	dir := filepath.Dir(m.path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	// Marshal to YAML
	data, err := yaml.Marshal(m.config)
	if err != nil {
		return err
	}

	// Write to file
	return ioutil.WriteFile(m.path, data, 0644)
}

// Get returns the current configuration
func (m *Manager) Get() *Config {
	if m.config == nil {
		m.config = DefaultConfig()
	}
	return m.config
}

// GetCurrentCluster returns the current cluster configuration
func (m *Manager) GetCurrentCluster() (*Cluster, error) {
	config := m.Get()
	if config.CurrentCluster == "" {
		return nil, fmt.Errorf("no current cluster set")
	}

	cluster, ok := config.Clusters[config.CurrentCluster]
	if !ok {
		return nil, fmt.Errorf("cluster %s not found", config.CurrentCluster)
	}

	return &cluster, nil
}

// SetCurrentCluster sets the current cluster
func (m *Manager) SetCurrentCluster(name string) error {
	config := m.Get()
	
	if _, ok := config.Clusters[name]; !ok {
		return fmt.Errorf("cluster %s not found", name)
	}

	config.CurrentCluster = name
	return m.Save()
}

// AddCluster adds a new cluster configuration
func (m *Manager) AddCluster(cluster Cluster) error {
	config := m.Get()
	
	if config.Clusters == nil {
		config.Clusters = make(map[string]Cluster)
	}

	config.Clusters[cluster.Name] = cluster
	
	// Set as current if it's the first cluster
	if config.CurrentCluster == "" {
		config.CurrentCluster = cluster.Name
	}

	return m.Save()
}

// RemoveCluster removes a cluster configuration
func (m *Manager) RemoveCluster(name string) error {
	config := m.Get()
	
	delete(config.Clusters, name)
	
	// Clear current cluster if it was removed
	if config.CurrentCluster == name {
		config.CurrentCluster = ""
		// Set to first available cluster if any
		for k := range config.Clusters {
			config.CurrentCluster = k
			break
		}
	}

	return m.Save()
}

// UpdateCluster updates a cluster configuration
func (m *Manager) UpdateCluster(cluster Cluster) error {
	config := m.Get()
	
	if _, ok := config.Clusters[cluster.Name]; !ok {
		return fmt.Errorf("cluster %s not found", cluster.Name)
	}

	config.Clusters[cluster.Name] = cluster
	return m.Save()
}

// GetPreferences returns user preferences
func (m *Manager) GetPreferences() Preferences {
	config := m.Get()
	return config.Preferences
}

// UpdatePreferences updates user preferences
func (m *Manager) UpdatePreferences(prefs Preferences) error {
	config := m.Get()
	config.Preferences = prefs
	return m.Save()
}

// Export exports configuration to JSON
func (m *Manager) Export() ([]byte, error) {
	config := m.Get()
	return json.MarshalIndent(config, "", "  ")
}

// Import imports configuration from JSON
func (m *Manager) Import(data []byte) error {
	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return err
	}

	m.config = &config
	return m.Save()
}