package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/shirou/gopsutil/v3/disk"
	gopsutilmem "github.com/shirou/gopsutil/v3/mem"
	"gopkg.in/yaml.v2"
	manifestconfig "novacron/backend/core/initialization/config"
)

var (
	configFile    = flag.String("config", "/etc/novacron/config.yaml", "Path to configuration file")
	debugMode     = flag.Bool("debug", false, "Enable debug logging")
	nodeID        = flag.String("node-id", "", "Node ID (defaults to hostname)")
	dataDir       = flag.String("data-dir", "/var/lib/novacron", "Data directory")
	listenAddress = flag.String("listen", "0.0.0.0:8090", "API listen address")
	version       = flag.Bool("version", false, "Print version information and exit")
)

// Version information (set at build time).
var (
	Version   = "dev"
	BuildTime = "unknown"
	GitCommit = "unknown"
)

type runtimeConfig struct {
	Storage    storage.StorageManagerConfig `yaml:"storage"`
	Network    network.NetworkManagerConfig `yaml:"network"`
	Hypervisor hypervisor.Config            `yaml:"hypervisor"`
	VMManager  vm.VMManagerConfig           `yaml:"vm_manager"`
	Scheduler  scheduler.SchedulerConfig    `yaml:"scheduler"`
	Auth       runtimeAuthConfig            `yaml:"auth"`
	Services   runtimeManifestSummary       `yaml:"-"`
}

type runtimeManifestSummary struct {
	Version           string                 `json:"version,omitempty"`
	DeploymentProfile string                 `json:"deployment_profile,omitempty"`
	DiscoveryMode     string                 `json:"discovery_mode,omitempty"`
	FederationMode    string                 `json:"federation_mode,omitempty"`
	MigrationMode     string                 `json:"migration_mode,omitempty"`
	AuthMode          string                 `json:"auth_mode,omitempty"`
	EnabledServices   []string               `json:"enabled_services,omitempty"`
	DiscoverySeeds    []runtimeDiscoverySeed `json:"discovery_seeds,omitempty"`
}

type runtimeDiscoverySeed struct {
	ID        string   `json:"id,omitempty"`
	Address   string   `json:"address"`
	PublicKey string   `json:"public_key,omitempty"`
	Tags      []string `json:"tags,omitempty"`
}

type runtimeServiceStatus struct {
	Name    string `json:"name"`
	Enabled bool   `json:"enabled"`
	State   string `json:"state"`
	Reason  string `json:"reason,omitempty"`
}

type runtimeServiceReport struct {
	Status           string                 `json:"status"`
	Manifest         runtimeManifestSummary `json:"manifest"`
	Services         []runtimeServiceStatus `json:"services"`
	DisabledServices []string               `json:"disabled_services,omitempty"`
}

const (
	runtimeServiceStateRunning     = "running"
	runtimeServiceStateDisabled    = "disabled"
	runtimeServiceStateUnavailable = "unavailable"
)

var runtimeServiceOrder = []string{
	"api",
	"auth",
	"backup",
	"discovery",
	"federation",
	"hypervisor",
	"migration",
	"network",
	"scheduler",
	"storage",
	"vm",
}

type runtimeConfigFile struct {
	Storage    *storageConfigFile    `yaml:"storage"`
	Hypervisor *hypervisorConfigFile `yaml:"hypervisor"`
	VMManager  *vmManagerConfigFile  `yaml:"vm_manager"`
	Scheduler  *schedulerConfigFile  `yaml:"scheduler"`
	Auth       *authConfigFile       `yaml:"auth"`
}

type storageConfigFile struct {
	BasePath string `yaml:"base_path"`
}

type hypervisorConfigFile struct {
	ID       string          `yaml:"id"`
	Name     string          `yaml:"name"`
	Role     hypervisor.Role `yaml:"role"`
	DataDir  string          `yaml:"data_dir"`
	VMConfig *vmConfigFile   `yaml:"vm_config"`
}

type vmConfigFile struct {
	Type      vm.VMType `yaml:"type"`
	CPUShares int       `yaml:"cpu_shares"`
	MemoryMB  int       `yaml:"memory_mb"`
}

type vmManagerConfigFile struct {
	DefaultDriver vm.VMType            `yaml:"default_driver"`
	DefaultVMType vm.VMType            `yaml:"default_vm_type"`
	TenantQuota   vm.TenantQuotaConfig `yaml:"tenant_quota"`
}

type schedulerConfigFile struct {
	MinimumNodeCount int `yaml:"minimum_node_count"`
}

func main() {
	flag.Parse()

	if *version {
		fmt.Printf("NovaCron %s\n", Version)
		fmt.Printf("Build time: %s\n", BuildTime)
		fmt.Printf("Git commit: %s\n", GitCommit)
		fmt.Printf("Go version: %s\n", runtime.Version())
		os.Exit(0)
	}

	log.SetFlags(log.LstdFlags | log.Lshortfile)
	if *debugMode {
		log.Println("Debug mode enabled")
	}

	if *nodeID == "" {
		hostname, err := os.Hostname()
		if err != nil {
			log.Fatalf("Failed to get hostname: %v", err)
		}
		*nodeID = hostname
	}

	if err := os.MkdirAll(*dataDir, 0o755); err != nil {
		log.Fatalf("Failed to create data directory: %v", err)
	}

	config, err := loadConfig(*configFile, *nodeID, *dataDir)
	if err != nil {
		log.Fatalf("Failed to load configuration: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	log.Printf("Starting NovaCron node: %s", *nodeID)

	if !runtimeServiceEnabled(config, "api") {
		log.Fatal("runtime manifest disables api service; backend/core/cmd/novacron requires api to boot")
	}

	var storageManager *storage.StorageManager
	if runtimeServiceEnabled(config, "storage") {
		storageManager, err = initializeStorage(config)
		if err != nil {
			log.Fatalf("Failed to initialize storage: %v", err)
		}
	} else {
		log.Println("Storage service disabled by runtime manifest")
	}

	var vmManager *vm.VMManager
	if runtimeServiceEnabled(config, "vm") {
		vmManager, err = initializeVMManager(config)
		if err != nil {
			log.Fatalf("Failed to initialize VM manager: %v", err)
		}
	} else {
		log.Println("VM service disabled by runtime manifest")
	}

	var schedulerService *scheduler.Scheduler
	if runtimeServiceEnabled(config, "scheduler") {
		schedulerService, err = initializeScheduler(config)
		if err != nil {
			log.Fatalf("Failed to initialize scheduler: %v", err)
		}
	} else {
		log.Println("Scheduler service disabled by runtime manifest")
	}

	var migrationManager *vm.VMMigrationManager
	if runtimeServiceEnabled(config, "migration") {
		migrationManager, err = initializeMigrationManager(*nodeID, *dataDir)
		if err != nil {
			log.Fatalf("Failed to initialize migration manager: %v", err)
		}
	} else {
		log.Println("Migration service disabled by runtime manifest")
	}

	var networkManager *network.NetworkManager
	if runtimeServiceEnabled(config, "network") {
		networkManager, err = initializeNetwork(ctx, config, *nodeID)
		if err != nil {
			log.Fatalf("Failed to initialize network: %v", err)
		}
	} else {
		log.Println("Network service disabled by runtime manifest")
	}

	var hypervisorManager *hypervisor.Hypervisor
	if runtimeServiceEnabled(config, "hypervisor") {
		hypervisorManager, err = initializeHypervisor(ctx, config, *nodeID)
		if err != nil {
			log.Fatalf("Failed to initialize hypervisor: %v", err)
		}
	} else {
		log.Println("Hypervisor service disabled by runtime manifest")
	}

	apiServer, err := initializeAPI(
		ctx,
		config,
		*listenAddress,
		vmManager,
		migrationManager,
		schedulerService,
		networkManager,
		storageManager,
		hypervisorManager,
	)
	if err != nil {
		log.Fatalf("Failed to initialize API server: %v", err)
	}

	signalCh := make(chan os.Signal, 1)
	signal.Notify(signalCh, syscall.SIGINT, syscall.SIGTERM)

	sig := <-signalCh
	log.Printf("Received signal: %v, shutting down...", sig)

	shutdownCtx, shutdownCancel := context.WithCancel(context.Background())
	defer shutdownCancel()

	if err := apiServer.Shutdown(shutdownCtx); err != nil {
		log.Printf("API server shutdown error: %v", err)
	}
	if hypervisorManager != nil {
		if err := hypervisorManager.Stop(); err != nil {
			log.Printf("Hypervisor shutdown error: %v", err)
		}
	}
	if networkManager != nil {
		if err := networkManager.Stop(); err != nil {
			log.Printf("Network manager shutdown error: %v", err)
		}
	}
	if schedulerService != nil {
		if err := schedulerService.Stop(); err != nil {
			log.Printf("Scheduler shutdown error: %v", err)
		}
	}
	if vmManager != nil {
		if err := vmManager.Stop(); err != nil {
			log.Printf("VM manager shutdown error: %v", err)
		}
	}

	log.Println("Shutdown complete")
}

func runtimeServiceReportFromRuntime(
	config runtimeConfig,
	vmManager *vm.VMManager,
	migrationManager *vm.VMMigrationManager,
	schedulerService *scheduler.Scheduler,
	networkManager *network.NetworkManager,
	storageManager *storage.StorageManager,
	hypervisorManager *hypervisor.Hypervisor,
	runtimeAuth *runtimeAuthRuntime,
	discovery *runtimeDiscoveryState,
) runtimeServiceReport {
	disabledServices := make([]string, 0)
	services := make([]runtimeServiceStatus, 0, len(runtimeServiceOrder))
	overallStatus := "healthy"

	authEnabled, authDisabledReason := runtimeAuthServiceEnabled(config)

	for _, serviceName := range runtimeServiceOrder {
		status := runtimeServiceStatus{Name: serviceName}

		switch serviceName {
		case "api":
			status.Enabled = runtimeServiceEnabled(config, serviceName)
			if status.Enabled {
				status.State = runtimeServiceStateRunning
			} else {
				status.State = runtimeServiceStateDisabled
				status.Reason = "disabled by runtime manifest"
			}
		case "auth":
			status.Enabled = authEnabled
			switch {
			case !authEnabled:
				status.State = runtimeServiceStateDisabled
				status.Reason = authDisabledReason
			case runtimeAuth != nil && runtimeAuth.enabled():
				status.State = runtimeServiceStateRunning
			default:
				status.State = runtimeServiceStateUnavailable
				status.Reason = "auth runtime failed to initialize"
			}
		case "backup":
			status = runtimeServiceStatusFromInstance(config, serviceName, false, "backup runtime is gated behind mobility policy integration")
		case "discovery":
			status.Enabled = runtimeDiscoveryEnabled(config)
			switch {
			case !status.Enabled:
				status.State = runtimeServiceStateDisabled
				status.Reason = "disabled by runtime manifest"
			case discovery != nil:
				status.State = runtimeServiceStateRunning
			default:
				status.State = runtimeServiceStateUnavailable
				status.Reason = "discovery runtime failed to initialize"
			}
		case "federation":
			status.Enabled = runtimeFederationEnabled(config)
			if !status.Enabled {
				status.State = runtimeServiceStateDisabled
				status.Reason = "disabled by runtime manifest"
			} else {
				status.State = runtimeServiceStateUnavailable
				status.Reason = "federation runtime is gated behind signed discovery integration"
			}
		case "hypervisor":
			status = runtimeServiceStatusFromInstance(config, serviceName, hypervisorManager != nil, "hypervisor manager unavailable")
		case "migration":
			status = runtimeServiceStatusFromInstance(config, serviceName, migrationManager != nil, "migration manager unavailable")
		case "network":
			status = runtimeServiceStatusFromInstance(config, serviceName, networkManager != nil, "network manager unavailable")
		case "scheduler":
			status = runtimeServiceStatusFromInstance(config, serviceName, schedulerService != nil, "scheduler unavailable")
		case "storage":
			status = runtimeServiceStatusFromInstance(config, serviceName, storageManager != nil, "storage manager unavailable")
		case "vm":
			status = runtimeServiceStatusFromInstance(config, serviceName, vmManager != nil, "vm manager unavailable")
		default:
			status = runtimeServiceStatusFromInstance(config, serviceName, false, "service status unavailable")
		}

		if status.State == runtimeServiceStateDisabled {
			disabledServices = append(disabledServices, status.Name)
		}
		if status.Enabled && status.State != runtimeServiceStateRunning {
			overallStatus = "degraded"
		}
		services = append(services, status)
	}

	sort.Strings(disabledServices)

	return runtimeServiceReport{
		Status:           overallStatus,
		Manifest:         config.Services,
		Services:         services,
		DisabledServices: disabledServices,
	}
}

func runtimeServiceStatusFromInstance(config runtimeConfig, serviceName string, available bool, unavailableReason string) runtimeServiceStatus {
	status := runtimeServiceStatus{
		Name:    serviceName,
		Enabled: runtimeServiceEnabled(config, serviceName),
	}

	if !status.Enabled {
		status.State = runtimeServiceStateDisabled
		status.Reason = "disabled by runtime manifest"
		return status
	}

	if available {
		status.State = runtimeServiceStateRunning
		return status
	}

	status.State = runtimeServiceStateUnavailable
	status.Reason = unavailableReason
	return status
}

func defaultRuntimeConfig(nodeID, dataDir string) runtimeConfig {
	return runtimeConfig{
		Storage: storage.StorageManagerConfig{
			BasePath: filepath.Join(dataDir, "storage"),
		},
		Network: network.DefaultNetworkManagerConfig(),
		Hypervisor: hypervisor.Config{
			ID:      nodeID,
			Name:    nodeID,
			Role:    hypervisor.RoleWorker,
			DataDir: filepath.Join(dataDir, "hypervisor"),
			VMConfig: vm.VMConfig{
				Type:      vm.VMTypeKVM,
				CPUShares: 1024,
				MemoryMB:  1024,
			},
		},
		VMManager: defaultVMManagerRuntimeConfig(nodeID, dataDir),
		Scheduler: scheduler.DefaultSchedulerConfig(),
		Auth:      defaultRuntimeAuthConfig(),
		Services: runtimeManifestSummary{
			Version:           manifestconfig.DefaultRuntimeManifestVersion,
			DeploymentProfile: "legacy-default",
			DiscoveryMode:     "disabled",
			FederationMode:    "disabled",
			MigrationMode:     "disabled",
			AuthMode:          "runtime",
			EnabledServices:   defaultEnabledRuntimeServices(),
		},
	}
}

func defaultEnabledRuntimeServices() []string {
	return []string{"api", "auth", "hypervisor", "migration", "network", "scheduler", "storage", "vm"}
}

func defaultVMManagerRuntimeConfig(nodeID, dataDir string) vm.VMManagerConfig {
	config := vm.DefaultVMManagerConfig()
	applyDefaultVMManagerDriverConfig(&config, nodeID, dataDir)
	return config
}

func loadConfig(path, nodeID, dataDir string) (runtimeConfig, error) {
	config := defaultRuntimeConfig(nodeID, dataDir)

	configBytes, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			log.Printf("Configuration file %s not found, using defaults", path)
			return config, nil
		}
		return runtimeConfig{}, fmt.Errorf("read config %s: %w", path, err)
	}

	if manifestConfig, handled, err := loadSharedRuntimeManifest(path, configBytes, nodeID, dataDir); handled || err != nil {
		return manifestConfig, err
	}

	var fileConfig runtimeConfigFile
	if err := yaml.Unmarshal(configBytes, &fileConfig); err != nil {
		return runtimeConfig{}, fmt.Errorf("parse config %s: %w", path, err)
	}

	mergeRuntimeConfig(&config, fileConfig)
	applyRuntimeConfigDefaults(&config, nodeID, dataDir)
	log.Printf("Loaded configuration from %s", path)
	return config, nil
}

func loadSharedRuntimeManifest(path string, configBytes []byte, nodeID, dataDir string) (runtimeConfig, bool, error) {
	if !looksLikeSharedRuntimeManifest(configBytes) {
		return runtimeConfig{}, false, nil
	}

	loader := manifestconfig.NewLoader(path)
	manifest, err := loader.Load()
	if err != nil {
		return runtimeConfig{}, true, fmt.Errorf("load shared runtime manifest %s: %w", path, err)
	}
	if err := loader.LoadFromEnv(manifest); err != nil {
		return runtimeConfig{}, true, fmt.Errorf("apply shared runtime manifest env overrides: %w", err)
	}

	config := runtimeConfigFromManifest(manifest, nodeID, dataDir)
	log.Printf(
		"Loaded shared runtime manifest from %s (version=%s profile=%s discovery=%s federation=%s migration=%s auth=%s enabled_services=%s)",
		path,
		manifest.Runtime.Version,
		manifest.Runtime.DeploymentProfile,
		manifest.Runtime.DiscoveryMode,
		manifest.Runtime.FederationMode,
		manifest.Runtime.MigrationMode,
		manifest.Runtime.AuthMode,
		strings.Join(manifest.Runtime.EnabledServices, ","),
	)

	return config, true, nil
}

func looksLikeSharedRuntimeManifest(configBytes []byte) bool {
	content := bytes.ToLower(configBytes)
	return bytes.HasPrefix(content, []byte("system:")) ||
		bytes.Contains(content, []byte("\nsystem:")) ||
		bytes.HasPrefix(content, []byte("runtime:")) ||
		bytes.Contains(content, []byte("\nruntime:"))
}

func runtimeConfigFromManifest(manifest *manifestconfig.Config, nodeID, dataDir string) runtimeConfig {
	effectiveNodeID := strings.TrimSpace(nodeID)
	if manifest != nil && strings.TrimSpace(manifest.System.NodeID) != "" {
		effectiveNodeID = strings.TrimSpace(manifest.System.NodeID)
	}

	effectiveDataDir := strings.TrimSpace(dataDir)
	if manifest != nil && strings.TrimSpace(manifest.System.DataDir) != "" {
		effectiveDataDir = strings.TrimSpace(manifest.System.DataDir)
	}

	config := defaultRuntimeConfig(effectiveNodeID, effectiveDataDir)
	if manifest == nil {
		return config
	}

	config.Services = runtimeManifestSummary{
		Version:           manifest.Runtime.Version,
		DeploymentProfile: manifest.Runtime.DeploymentProfile,
		DiscoveryMode:     manifest.Runtime.DiscoveryMode,
		FederationMode:    manifest.Runtime.FederationMode,
		MigrationMode:     manifest.Runtime.MigrationMode,
		AuthMode:          manifest.Runtime.AuthMode,
		EnabledServices:   append([]string(nil), manifest.Runtime.EnabledServices...),
		DiscoverySeeds:    runtimeDiscoverySeedsFromManifest(manifest.Runtime.DiscoverySeeds),
	}

	config.Storage.BasePath = filepath.Join(effectiveDataDir, "storage")
	config.Storage.Encryption = manifest.Security.EnableEncryption
	config.Network.BandwidthMonitoringEnabled = manifest.Monitoring.EnableMetrics
	config.Auth.Enabled = manifest.Security.EnableAuth &&
		manifest.Runtime.AuthMode == "runtime" &&
		serviceEnabled(manifest.Runtime.EnabledServices, "auth")

	return config
}

func runtimeDiscoverySeedsFromManifest(seeds []manifestconfig.RuntimeDiscoverySeed) []runtimeDiscoverySeed {
	if len(seeds) == 0 {
		return nil
	}
	result := make([]runtimeDiscoverySeed, 0, len(seeds))
	for _, seed := range seeds {
		result = append(result, runtimeDiscoverySeed{
			ID:        seed.ID,
			Address:   seed.Address,
			PublicKey: seed.PublicKey,
			Tags:      append([]string(nil), seed.Tags...),
		})
	}
	return result
}

func serviceEnabled(enabledServices []string, service string) bool {
	target := strings.ToLower(strings.TrimSpace(service))
	for _, enabled := range enabledServices {
		if strings.ToLower(strings.TrimSpace(enabled)) == target {
			return true
		}
	}
	return false
}

func runtimeServiceEnabled(config runtimeConfig, service string) bool {
	return serviceEnabled(config.Services.EnabledServices, service)
}

func runtimeAuthServiceEnabled(config runtimeConfig) (bool, string) {
	if !runtimeServiceEnabled(config, "auth") {
		return false, "disabled by runtime manifest"
	}
	if !config.Auth.Enabled {
		switch {
		case strings.TrimSpace(config.Services.AuthMode) != "" && config.Services.AuthMode != "runtime":
			return false, fmt.Sprintf("disabled by auth_mode=%s", config.Services.AuthMode)
		default:
			return false, "disabled by runtime auth configuration"
		}
	}
	return true, ""
}

func mergeRuntimeConfig(config *runtimeConfig, fileConfig runtimeConfigFile) {
	if fileConfig.Storage != nil && fileConfig.Storage.BasePath != "" {
		config.Storage.BasePath = fileConfig.Storage.BasePath
	}

	if fileConfig.Hypervisor != nil {
		if fileConfig.Hypervisor.ID != "" {
			config.Hypervisor.ID = fileConfig.Hypervisor.ID
		}
		if fileConfig.Hypervisor.Name != "" {
			config.Hypervisor.Name = fileConfig.Hypervisor.Name
		}
		if fileConfig.Hypervisor.Role != "" {
			config.Hypervisor.Role = fileConfig.Hypervisor.Role
		}
		if fileConfig.Hypervisor.DataDir != "" {
			config.Hypervisor.DataDir = fileConfig.Hypervisor.DataDir
		}
		if fileConfig.Hypervisor.VMConfig != nil {
			if fileConfig.Hypervisor.VMConfig.Type != "" {
				config.Hypervisor.VMConfig.Type = fileConfig.Hypervisor.VMConfig.Type
			}
			if fileConfig.Hypervisor.VMConfig.CPUShares != 0 {
				config.Hypervisor.VMConfig.CPUShares = fileConfig.Hypervisor.VMConfig.CPUShares
			}
			if fileConfig.Hypervisor.VMConfig.MemoryMB != 0 {
				config.Hypervisor.VMConfig.MemoryMB = fileConfig.Hypervisor.VMConfig.MemoryMB
			}
		}
	}

	if fileConfig.VMManager != nil {
		if fileConfig.VMManager.DefaultDriver != "" {
			config.VMManager.DefaultDriver = fileConfig.VMManager.DefaultDriver
		}
		if fileConfig.VMManager.DefaultVMType != "" {
			config.VMManager.DefaultVMType = fileConfig.VMManager.DefaultVMType
		}
		if fileConfig.VMManager.TenantQuota.Default.MaxVMs != 0 {
			config.VMManager.TenantQuota.Default.MaxVMs = fileConfig.VMManager.TenantQuota.Default.MaxVMs
		}
		if fileConfig.VMManager.TenantQuota.Default.MaxCPUUnits != 0 {
			config.VMManager.TenantQuota.Default.MaxCPUUnits = fileConfig.VMManager.TenantQuota.Default.MaxCPUUnits
		}
		if fileConfig.VMManager.TenantQuota.Default.MaxMemoryMB != 0 {
			config.VMManager.TenantQuota.Default.MaxMemoryMB = fileConfig.VMManager.TenantQuota.Default.MaxMemoryMB
		}
		if len(fileConfig.VMManager.TenantQuota.Overrides) > 0 {
			if config.VMManager.TenantQuota.Overrides == nil {
				config.VMManager.TenantQuota.Overrides = make(map[string]vm.TenantQuotaLimits)
			}
			for tenantID, limits := range fileConfig.VMManager.TenantQuota.Overrides {
				config.VMManager.TenantQuota.Overrides[tenantID] = limits
			}
		}
	}

	if fileConfig.Scheduler != nil && fileConfig.Scheduler.MinimumNodeCount != 0 {
		config.Scheduler.MinimumNodeCount = fileConfig.Scheduler.MinimumNodeCount
	}

	mergeRuntimeAuthConfig(&config.Auth, fileConfig.Auth)
}

func applyRuntimeConfigDefaults(config *runtimeConfig, nodeID, dataDir string) {
	if config.Storage.BasePath == "" {
		config.Storage.BasePath = filepath.Join(dataDir, "storage")
	}

	if config.Hypervisor.ID == "" {
		config.Hypervisor.ID = nodeID
	}
	if config.Hypervisor.Name == "" {
		config.Hypervisor.Name = nodeID
	}
	if config.Hypervisor.Role == "" {
		config.Hypervisor.Role = hypervisor.RoleWorker
	}
	if config.Hypervisor.DataDir == "" {
		config.Hypervisor.DataDir = filepath.Join(dataDir, "hypervisor")
	}
	if config.Hypervisor.VMConfig.Type == "" {
		config.Hypervisor.VMConfig.Type = vm.VMTypeKVM
	}
	if config.Hypervisor.VMConfig.CPUShares == 0 {
		config.Hypervisor.VMConfig.CPUShares = 1024
	}
	if config.Hypervisor.VMConfig.MemoryMB == 0 {
		config.Hypervisor.VMConfig.MemoryMB = 1024
	}

	if config.VMManager.DefaultDriver == "" {
		config.VMManager.DefaultDriver = vm.VMTypeKVM
	}
	if config.VMManager.DefaultVMType == "" {
		config.VMManager.DefaultVMType = vm.VMTypeKVM
	}
	if config.VMManager.Drivers == nil {
		config.VMManager.Drivers = make(map[vm.VMType]vm.VMDriverConfigManager)
	}
	if config.VMManager.TenantQuota.Overrides == nil {
		config.VMManager.TenantQuota.Overrides = make(map[string]vm.TenantQuotaLimits)
	}
	applyDefaultVMManagerDriverConfig(&config.VMManager, nodeID, dataDir)
	applyRuntimeAuthDefaults(&config.Auth)
	applyRuntimeServiceDefaults(config)
}

func applyRuntimeServiceDefaults(config *runtimeConfig) {
	if config == nil {
		return
	}

	if len(config.Services.EnabledServices) == 0 {
		config.Services.EnabledServices = defaultEnabledRuntimeServices()
	}
	if config.Services.Version == "" {
		config.Services.Version = manifestconfig.DefaultRuntimeManifestVersion
	}
	if config.Services.DeploymentProfile == "" {
		config.Services.DeploymentProfile = "legacy-default"
	}
	if config.Services.DiscoveryMode == "" {
		config.Services.DiscoveryMode = "disabled"
	}
	if config.Services.FederationMode == "" {
		config.Services.FederationMode = "disabled"
	}
	if config.Services.MigrationMode == "" {
		config.Services.MigrationMode = "disabled"
	}
	if config.Services.AuthMode == "" {
		if config.Auth.Enabled {
			config.Services.AuthMode = "runtime"
		} else {
			config.Services.AuthMode = "disabled"
		}
	}
}

func applyDefaultVMManagerDriverConfig(config *vm.VMManagerConfig, nodeID, dataDir string) {
	if config == nil {
		return
	}
	if config.Drivers == nil {
		config.Drivers = make(map[vm.VMType]vm.VMDriverConfigManager)
	}

	if driverConfig, exists := config.Drivers[vm.VMTypeKVM]; exists {
		if driverConfig.Config == nil {
			driverConfig.Config = make(map[string]interface{})
		}
		if nodeIDValue, ok := driverConfig.Config["node_id"].(string); !ok || nodeIDValue == "" {
			driverConfig.Config["node_id"] = nodeID
		}
		if qemuPathValue, ok := driverConfig.Config["qemu_path"].(string); !ok || qemuPathValue == "" {
			driverConfig.Config["qemu_path"] = "qemu-system-x86_64"
		}
		if vmPathValue, ok := driverConfig.Config["vm_path"].(string); !ok || vmPathValue == "" {
			driverConfig.Config["vm_path"] = filepath.Join(dataDir, "vms")
		}
		driverConfig.Enabled = true
		config.Drivers[vm.VMTypeKVM] = driverConfig
		return
	}

	config.Drivers[vm.VMTypeKVM] = vm.VMDriverConfigManager{
		Enabled: true,
		Config: map[string]interface{}{
			"node_id":   nodeID,
			"qemu_path": "qemu-system-x86_64",
			"vm_path":   filepath.Join(dataDir, "vms"),
		},
	}
}

func initializeStorage(config runtimeConfig) (*storage.StorageManager, error) {
	log.Println("Initializing storage manager")

	manager, err := storage.NewStorageManager(config.Storage)
	if err != nil {
		return nil, fmt.Errorf("create storage manager: %w", err)
	}

	return manager, nil
}

func initializeNetwork(
	ctx context.Context,
	config runtimeConfig,
	nodeID string,
) (*network.NetworkManager, error) {
	_ = ctx

	log.Println("Initializing network manager")

	manager := network.NewNetworkManager(config.Network, nodeID, nil)
	if err := manager.Start(); err != nil {
		return nil, fmt.Errorf("start network manager: %w", err)
	}

	return manager, nil
}

func initializeHypervisor(
	ctx context.Context,
	config runtimeConfig,
	nodeID string,
) (*hypervisor.Hypervisor, error) {
	log.Println("Initializing hypervisor")

	hypervisorConfig := config.Hypervisor
	if hypervisorConfig.ID == "" {
		hypervisorConfig.ID = nodeID
	}
	if hypervisorConfig.Name == "" {
		hypervisorConfig.Name = nodeID
	}

	manager, err := hypervisor.NewHypervisor(hypervisorConfig)
	if err != nil {
		return nil, fmt.Errorf("create hypervisor: %w", err)
	}
	if err := manager.Start(ctx); err != nil {
		return nil, fmt.Errorf("start hypervisor: %w", err)
	}

	return manager, nil
}

func initializeVMManager(config runtimeConfig) (*vm.VMManager, error) {
	log.Println("Initializing VM manager")

	manager, err := vm.NewVMManager(config.VMManager)
	if err != nil {
		return nil, fmt.Errorf("create VM manager: %w", err)
	}

	if err := ensureNonStubKVMRuntime(manager); err != nil {
		return nil, err
	}

	if err := registerLocalSchedulerNode(manager, config.Hypervisor.ID, config.Storage.BasePath); err != nil {
		return nil, fmt.Errorf("register local scheduler node: %w", err)
	}

	if err := manager.Start(); err != nil {
		return nil, fmt.Errorf("start VM manager: %w", err)
	}

	return manager, nil
}

func registerLocalSchedulerNode(manager *vm.VMManager, nodeID, storagePath string) error {
	if manager == nil {
		return fmt.Errorf("vm manager is required")
	}
	if nodeID == "" {
		return fmt.Errorf("node id is required")
	}

	totalMemoryMB := 1024
	usedMemoryMB := 0
	if memoryStats, err := gopsutilmem.VirtualMemory(); err == nil {
		totalMemoryMB = int(memoryStats.Total / (1024 * 1024))
		usedMemoryMB = int((memoryStats.Total - memoryStats.Available) / (1024 * 1024))
	}

	totalDiskGB := 1
	usedDiskGB := 0
	if storagePath != "" {
		if diskUsage, err := disk.Usage(storagePath); err == nil {
			totalDiskGB = int(diskUsage.Total / (1024 * 1024 * 1024))
			usedDiskGB = int((diskUsage.Total - diskUsage.Free) / (1024 * 1024 * 1024))
		}
	}

	totalCPU := runtime.NumCPU()
	if totalCPU < 1 {
		totalCPU = 1
	}
	if totalMemoryMB < 1 {
		totalMemoryMB = 1
	}
	if totalDiskGB < 1 {
		totalDiskGB = 1
	}

	nodeInfo := &vm.NodeResourceInfo{
		NodeID:             nodeID,
		TotalCPU:           totalCPU,
		UsedCPU:            0,
		TotalMemoryMB:      totalMemoryMB,
		UsedMemoryMB:       clampInt(usedMemoryMB, 0, totalMemoryMB),
		TotalDiskGB:        totalDiskGB,
		UsedDiskGB:         clampInt(usedDiskGB, 0, totalDiskGB),
		CPUUsagePercent:    0,
		MemoryUsagePercent: percent(clampInt(usedMemoryMB, 0, totalMemoryMB), totalMemoryMB),
		DiskUsagePercent:   percent(clampInt(usedDiskGB, 0, totalDiskGB), totalDiskGB),
		Status:             "available",
		Labels:             map[string]string{"runtime": "novacron", "hypervisor": string(vm.VMTypeKVM)},
	}

	return manager.RegisterSchedulerNode(nodeInfo)
}

func percent(used, total int) float64 {
	if total <= 0 {
		return 0
	}
	return float64(used) / float64(total) * 100
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}

func ensureNonStubKVMRuntime(manager *vm.VMManager) error {
	driver, err := manager.GetDriverForConfig(vm.VMConfig{
		Type: vm.VMTypeKVM,
		Tags: map[string]string{
			"vm_type": string(vm.VMTypeKVM),
		},
	})
	if err != nil {
		return fmt.Errorf("resolve default KVM driver: %w", err)
	}

	if _, ok := driver.(*vm.CoreStubDriver); ok {
		return fmt.Errorf(
			"default KVM runtime resolves to vm.CoreStubDriver; cmd/novacron refuses to boot as a hypervisor until a real KVM driver is wired into the default build",
		)
	}

	return nil
}

func initializeScheduler(config runtimeConfig) (*scheduler.Scheduler, error) {
	log.Println("Initializing scheduler")

	service := scheduler.NewScheduler(config.Scheduler)
	if err := service.Start(); err != nil {
		return nil, fmt.Errorf("start scheduler: %w", err)
	}

	return service, nil
}

func initializeMigrationManager(nodeID, dataDir string) (*vm.VMMigrationManager, error) {
	log.Println("Initializing migration manager")

	migrationDir := filepath.Join(dataDir, "migrations")
	if err := os.MkdirAll(migrationDir, 0o755); err != nil {
		return nil, fmt.Errorf("create migration directory: %w", err)
	}

	return vm.NewVMMigrationManager(nodeID, migrationDir), nil
}

func initializeAPI(
	ctx context.Context,
	config runtimeConfig,
	listenAddress string,
	vmManager *vm.VMManager,
	migrationManager *vm.VMMigrationManager,
	schedulerService *scheduler.Scheduler,
	networkManager *network.NetworkManager,
	storageManager *storage.StorageManager,
	hypervisorManager *hypervisor.Hypervisor,
) (*APIServer, error) {
	applyRuntimeServiceDefaults(&config)

	log.Printf("Starting API server on %s", listenAddress)

	inventoryStore, err := newRuntimeInventoryStoreFromEnv()
	if err != nil {
		return nil, fmt.Errorf("initialize runtime inventory store: %w", err)
	}

	runtimeAuth, err := initializeRuntimeAuth(config.Auth)
	if err != nil {
		if inventoryStore != nil {
			_ = inventoryStore.Close()
		}
		return nil, fmt.Errorf("initialize runtime auth: %w", err)
	}

	listener, err := net.Listen("tcp", listenAddress)
	if err != nil {
		return nil, fmt.Errorf("listen on %s: %w", listenAddress, err)
	}

	discoveryState, err := newRuntimeDiscoveryState(config, listener.Addr().String())
	if err != nil {
		_ = listener.Close()
		if inventoryStore != nil {
			_ = inventoryStore.Close()
		}
		if runtimeAuth != nil && runtimeAuth.persistence != nil {
			_ = runtimeAuth.persistence.Close()
		}
		return nil, fmt.Errorf("initialize runtime discovery: %w", err)
	}

	router := newRuntimeRouter(config, vmManager, migrationManager, schedulerService, networkManager, storageManager, hypervisorManager, inventoryStore, runtimeAuth, discoveryState)
	server := &http.Server{
		Handler: router,
		BaseContext: func(net.Listener) context.Context {
			return ctx
		},
	}

	apiServer := &APIServer{
		server:      server,
		listener:    listener,
		address:     listener.Addr().String(),
		runtimeAuth: runtimeAuth,
		inventory:   inventoryStore,
	}
	if err := apiServer.Start(); err != nil {
		return nil, fmt.Errorf("start API server: %w", err)
	}

	return apiServer, nil
}

type APIServer struct {
	server      *http.Server
	listener    net.Listener
	address     string
	runtimeAuth *runtimeAuthRuntime
	inventory   *runtimeInventoryStore
}

func (s *APIServer) Start() error {
	if s == nil || s.server == nil || s.listener == nil {
		return fmt.Errorf("api server is not configured")
	}

	go func() {
		if err := s.server.Serve(s.listener); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Printf("API server stopped with error: %v", err)
		}
	}()

	return nil
}

func (s *APIServer) Shutdown(ctx context.Context) error {
	if s == nil {
		return nil
	}
	var shutdownErr error
	if s.server != nil {
		shutdownErr = errors.Join(shutdownErr, s.server.Shutdown(ctx))
	}
	if s.inventory != nil {
		shutdownErr = errors.Join(shutdownErr, s.inventory.Close())
	}
	if s.runtimeAuth != nil && s.runtimeAuth.persistence != nil {
		shutdownErr = errors.Join(shutdownErr, s.runtimeAuth.persistence.Close())
	}
	return shutdownErr
}

func newRuntimeRouter(
	config runtimeConfig,
	vmManager *vm.VMManager,
	migrationManager *vm.VMMigrationManager,
	schedulerService *scheduler.Scheduler,
	networkManager *network.NetworkManager,
	storageManager *storage.StorageManager,
	hypervisorManager *hypervisor.Hypervisor,
	inventoryStore *runtimeInventoryStore,
	runtimeAuth *runtimeAuthRuntime,
	discovery *runtimeDiscoveryState,
) *mux.Router {
	router := mux.NewRouter()
	router.Use(runtimeCORSMiddleware(config.Auth))
	router.HandleFunc("/healthz", handleHealthz).Methods(http.MethodGet)
	registerRuntimeAuthRoutes(router, runtimeAuth)
	router.HandleFunc("/internal/runtime/v1/monitoring/metrics", runtimeGetMonitoringMetricsHandler(vmManager)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/vms", runtimeGetVMsHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/vms/{id}", runtimeGetVMHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/vms/{id}/metrics", runtimeGetVMMetricsHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/monitoring/vms", runtimeGetMonitoringVMsHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/networks", runtimeGetNetworksHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/networks/{id}", runtimeGetNetworkHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/vms/{vm_id}/interfaces", runtimeGetVMInterfacesHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/vms/{vm_id}/interfaces/{id}", runtimeGetVMInterfaceHandler(inventoryStore)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/services", runtimeGetServicesHandler(config, vmManager, migrationManager, schedulerService, networkManager, storageManager, hypervisorManager, runtimeAuth, discovery)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/discovery/inventory", runtimeGetDiscoveryInventoryHandler(discovery)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/discovery/seeds", runtimeGetDiscoverySeedsHandler(discovery)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/discovery/seeds/{id}/verify", runtimeVerifyDiscoverySeedInventoryHandler(discovery)).Methods(http.MethodPost)
	router.HandleFunc("/internal/runtime/v1/mobility/policy", runtimeGetMobilityPolicyHandler(config)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/mobility/operations", runtimeListMobilityOperationsHandler(config, migrationManager)).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/mobility/cold-migrations", runtimeStartColdMigrationHandler(config, migrationManager)).Methods(http.MethodPost)
	router.HandleFunc("/internal/runtime/v1/mobility/checkpoint-restores", runtimeStartCheckpointRestoreHandler(config, migrationManager)).Methods(http.MethodPost)
	clusterHandler := runtimeProtectClusterRoute(runtimeAuth)
	router.Handle("/api/cluster/nodes", clusterHandler(runtimeListNodesHandler(vmManager))).Methods(http.MethodGet)
	router.Handle("/api/cluster/nodes/{id}", clusterHandler(runtimeGetNodeHandler(vmManager))).Methods(http.MethodGet)
	router.Handle("/api/cluster/health", clusterHandler(runtimeGetClusterHealthHandler(vmManager, runtimeAuth))).Methods(http.MethodGet)
	router.Handle("/api/cluster/leader", clusterHandler(runtimeGetLeaderHandler(vmManager))).Methods(http.MethodGet)
	router.Handle("/api/cluster/federation", clusterHandler(runtimeGetFederationHandler(runtimeAuth))).Methods(http.MethodGet)
	return router
}

type runtimeNode struct {
	ID                 string            `json:"id"`
	Address            string            `json:"address,omitempty"`
	Status             string            `json:"status"`
	CPU                int               `json:"cpu,omitempty"`
	Memory             int64             `json:"memory,omitempty"`
	Disk               int64             `json:"disk,omitempty"`
	UsedCPU            int               `json:"used_cpu,omitempty"`
	RemainingCPU       int               `json:"remaining_cpu,omitempty"`
	UsedMemoryMB       int64             `json:"used_memory_mb,omitempty"`
	RemainingMemoryMB  int64             `json:"remaining_memory_mb,omitempty"`
	UsedDiskGB         int64             `json:"used_disk_gb,omitempty"`
	RemainingDiskGB    int64             `json:"remaining_disk_gb,omitempty"`
	CPUUsagePercent    float64           `json:"cpu_usage_percent,omitempty"`
	MemoryUsagePercent float64           `json:"memory_usage_percent,omitempty"`
	DiskUsagePercent   float64           `json:"disk_usage_percent,omitempty"`
	VMCount            int               `json:"vm_count,omitempty"`
	Schedulable        bool              `json:"schedulable"`
	Labels             map[string]string `json:"labels,omitempty"`
}

type runtimeClusterHealth struct {
	Status       string                   `json:"status"`
	TotalNodes   int                      `json:"total_nodes"`
	HealthyNodes int                      `json:"healthy_nodes"`
	HasQuorum    bool                     `json:"has_quorum"`
	Leader       string                   `json:"leader"`
	LastUpdated  time.Time                `json:"last_updated"`
	Auth         *runtimeAuthHealth       `json:"auth,omitempty"`
	Federation   *runtimeFederationHealth `json:"federation,omitempty"`
}

type runtimeAuthHealth struct {
	Enabled                bool   `json:"enabled"`
	SessionTransport       string `json:"sessionTransport,omitempty"`
	AutoAdmit              bool   `json:"autoAdmit,omitempty"`
	DefaultMembershipState string `json:"defaultMembershipState,omitempty"`
}

type runtimeFederationHealth struct {
	Enabled            bool                            `json:"enabled"`
	TotalClusters      int                             `json:"totalClusters,omitempty"`
	ActiveMemberships  int                             `json:"activeMemberships,omitempty"`
	PendingMemberships int                             `json:"pendingMemberships,omitempty"`
	RevokedMemberships int                             `json:"revokedMemberships,omitempty"`
	SelectedClusterID  string                          `json:"selectedClusterId,omitempty"`
	HighestTier        string                          `json:"highestTier,omitempty"`
	Clusters           []runtimeClusterSummaryResponse `json:"clusters,omitempty"`
}

type runtimeFederationResponse struct {
	Auth            runtimeAuthHealth              `json:"auth"`
	Federation      runtimeFederationHealth        `json:"federation"`
	Memberships     []runtimeAdmissionResponse     `json:"memberships,omitempty"`
	SelectedCluster *runtimeClusterSummaryResponse `json:"selectedCluster,omitempty"`
}

type runtimeMonitoringSummary struct {
	CurrentCpuUsage         float64  `json:"currentCpuUsage"`
	CurrentMemoryUsage      float64  `json:"currentMemoryUsage"`
	CurrentDiskUsage        float64  `json:"currentDiskUsage"`
	CurrentNetworkUsage     float64  `json:"currentNetworkUsage"`
	CpuChangePercentage     float64  `json:"cpuChangePercentage"`
	MemoryChangePercentage  float64  `json:"memoryChangePercentage"`
	DiskChangePercentage    float64  `json:"diskChangePercentage"`
	NetworkChangePercentage float64  `json:"networkChangePercentage"`
	TimeLabels              []string `json:"timeLabels"`
	CpuAnalysis             string   `json:"cpuAnalysis"`
	MemoryAnalysis          string   `json:"memoryAnalysis"`
}

func handleHealthz(w http.ResponseWriter, _ *http.Request) {
	respondRuntimeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func runtimeGetServicesHandler(
	config runtimeConfig,
	vmManager *vm.VMManager,
	migrationManager *vm.VMMigrationManager,
	schedulerService *scheduler.Scheduler,
	networkManager *network.NetworkManager,
	storageManager *storage.StorageManager,
	hypervisorManager *hypervisor.Hypervisor,
	runtimeAuth *runtimeAuthRuntime,
	discovery *runtimeDiscoveryState,
) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		report := runtimeServiceReportFromRuntime(
			config,
			vmManager,
			migrationManager,
			schedulerService,
			networkManager,
			storageManager,
			hypervisorManager,
			runtimeAuth,
			discovery,
		)
		respondRuntimeJSON(w, http.StatusOK, report)
	}
}

func runtimeGetMonitoringMetricsHandler(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		var inventory []*vm.NodeResourceInfo
		if vmManager != nil {
			inventory = vmManager.ListSchedulerNodes()
		}
		respondRuntimeJSON(w, http.StatusOK, runtimeMonitoringSummaryFromInventory(inventory))
	}
}

func runtimeMonitoringSummaryFromInventory(inventory []*vm.NodeResourceInfo) runtimeMonitoringSummary {
	summary := runtimeMonitoringSummary{
		TimeLabels: runtimeMonitoringTimeLabels(),
	}

	totalCPU := 0
	usedCPU := 0
	totalMemoryMB := 0
	usedMemoryMB := 0
	totalDiskGB := 0
	usedDiskGB := 0
	nodeCount := 0

	for _, nodeInfo := range inventory {
		if nodeInfo == nil {
			continue
		}

		nodeCount++

		nodeTotalCPU := maxInt(0, nodeInfo.TotalCPU)
		nodeTotalMemoryMB := maxInt(0, nodeInfo.TotalMemoryMB)
		nodeTotalDiskGB := maxInt(0, nodeInfo.TotalDiskGB)

		totalCPU += nodeTotalCPU
		usedCPU += clampInt(nodeInfo.UsedCPU, 0, nodeTotalCPU)
		totalMemoryMB += nodeTotalMemoryMB
		usedMemoryMB += clampInt(nodeInfo.UsedMemoryMB, 0, nodeTotalMemoryMB)
		totalDiskGB += nodeTotalDiskGB
		usedDiskGB += clampInt(nodeInfo.UsedDiskGB, 0, nodeTotalDiskGB)
	}

	summary.CurrentCpuUsage = percent(usedCPU, totalCPU)
	summary.CurrentMemoryUsage = percent(usedMemoryMB, totalMemoryMB)
	summary.CurrentDiskUsage = percent(usedDiskGB, totalDiskGB)
	summary.CpuAnalysis = runtimeUsageAnalysis("CPU", summary.CurrentCpuUsage, nodeCount)
	summary.MemoryAnalysis = runtimeUsageAnalysis("Memory", summary.CurrentMemoryUsage, nodeCount)

	return summary
}

func runtimeMonitoringTimeLabels() []string {
	return []string{"00:00", "00:05", "00:10", "00:15", "00:20"}
}

func runtimeUsageAnalysis(resource string, usage float64, nodeCount int) string {
	if nodeCount == 0 {
		return fmt.Sprintf("%s usage is unavailable because no scheduler nodes are registered.", resource)
	}

	switch {
	case usage >= 90:
		return fmt.Sprintf("%s usage is critical across registered scheduler nodes.", resource)
	case usage >= 75:
		return fmt.Sprintf("%s usage is elevated across registered scheduler nodes.", resource)
	case usage > 0:
		return fmt.Sprintf("%s usage reflects current scheduler node allocations.", resource)
	default:
		return fmt.Sprintf("%s usage is idle across registered scheduler nodes.", resource)
	}
}

func runtimeListNodesHandler(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if vmManager == nil {
			respondRuntimeJSON(w, http.StatusOK, []runtimeNode{})
			return
		}
		inventory := vmManager.ListSchedulerNodes()
		nodes := make([]runtimeNode, 0, len(inventory))
		for _, nodeInfo := range inventory {
			nodes = append(nodes, runtimeSchedulerNodeToAPI(nodeInfo))
		}
		sort.Slice(nodes, func(i, j int) bool {
			return nodes[i].ID < nodes[j].ID
		})
		respondRuntimeJSON(w, http.StatusOK, nodes)
	}
}

func runtimeGetNodeHandler(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if vmManager == nil {
			http.Error(w, "vm service is disabled", http.StatusServiceUnavailable)
			return
		}
		nodeID := mux.Vars(r)["id"]
		for _, nodeInfo := range vmManager.ListSchedulerNodes() {
			if nodeInfo.NodeID == nodeID {
				respondRuntimeJSON(w, http.StatusOK, runtimeSchedulerNodeToAPI(nodeInfo))
				return
			}
		}
		http.Error(w, "node not found", http.StatusNotFound)
	}
}

func runtimeGetClusterHealthHandler(vmManager *vm.VMManager, runtimeAuth *runtimeAuthRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		inventory := []*vm.NodeResourceInfo(nil)
		if vmManager != nil {
			inventory = vmManager.ListSchedulerNodes()
		}
		totalNodes := len(inventory)
		healthyNodes := 0
		leaderID := ""
		for _, nodeInfo := range inventory {
			if runtimeIsSchedulableNode(nodeInfo) {
				healthyNodes++
				if leaderID == "" {
					leaderID = nodeInfo.NodeID
				}
			}
		}

		status := "unavailable"
		if healthyNodes > 0 && healthyNodes == totalNodes {
			status = "healthy"
		} else if healthyNodes > 0 {
			status = "degraded"
		}

		authHealth, federationHealth, _, selectedCluster := runtimeRuntimeHealth(runtimeAuth, req)
		if selectedCluster != nil && leaderID == "" {
			leaderID = selectedCluster.ID
		}

		respondRuntimeJSON(w, http.StatusOK, runtimeClusterHealth{
			Status:       status,
			TotalNodes:   totalNodes,
			HealthyNodes: healthyNodes,
			HasQuorum:    healthyNodes > 0,
			Leader:       leaderID,
			LastUpdated:  time.Now().UTC(),
			Auth:         authHealth,
			Federation:   federationHealth,
		})
	}
}

func runtimeGetFederationHandler(runtimeAuth *runtimeAuthRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		authHealth, federationHealth, memberships, selectedCluster := runtimeRuntimeHealth(runtimeAuth, req)
		if authHealth == nil || federationHealth == nil {
			respondRuntimeJSON(w, http.StatusOK, runtimeFederationResponse{
				Auth:       runtimeAuthHealth{Enabled: false},
				Federation: runtimeFederationHealth{Enabled: false},
			})
			return
		}
		respondRuntimeJSON(w, http.StatusOK, runtimeFederationResponse{
			Auth:            *authHealth,
			Federation:      *federationHealth,
			Memberships:     memberships,
			SelectedCluster: selectedCluster,
		})
	}
}

func runtimeProtectClusterRoute(runtimeAuth *runtimeAuthRuntime) func(http.HandlerFunc) http.Handler {
	return func(next http.HandlerFunc) http.Handler {
		if runtimeAuth != nil && runtimeAuth.enabled() {
			return runtimeAuth.requireAuthenticated(http.HandlerFunc(next))
		}
		return http.HandlerFunc(next)
	}
}

func runtimeRuntimeHealth(runtimeAuth *runtimeAuthRuntime, req *http.Request) (*runtimeAuthHealth, *runtimeFederationHealth, []runtimeAdmissionResponse, *runtimeClusterSummaryResponse) {
	if runtimeAuth == nil || !runtimeAuth.enabled() {
		return &runtimeAuthHealth{Enabled: false}, &runtimeFederationHealth{Enabled: false}, nil, nil
	}

	authHealth := &runtimeAuthHealth{
		Enabled:                true,
		SessionTransport:       runtimeAuth.config.Session.Transport,
		AutoAdmit:              runtimeAuth.config.Membership.AutoAdmit,
		DefaultMembershipState: runtimeAuth.config.Membership.DefaultState,
	}

	clusters, err := runtimeAuth.persistence.clusters.List()
	if err != nil {
		return authHealth, &runtimeFederationHealth{Enabled: true}, nil, nil
	}

	summaries := make([]runtimeClusterSummaryResponse, 0, len(clusters))
	highestTier := ""
	for _, cluster := range clusters {
		summary := runtimeClusterSummaryResponseFromRecord(&cluster)
		summaries = append(summaries, summary)
		if highestTier == "" {
			highestTier = summary.Tier
		}
	}
	sort.SliceStable(summaries, func(i, j int) bool {
		if summaries[i].PerformanceScore == summaries[j].PerformanceScore {
			return summaries[i].ID < summaries[j].ID
		}
		return summaries[i].PerformanceScore > summaries[j].PerformanceScore
	})

	var memberships []runtimeAdmissionResponse
	var selectedCluster *runtimeClusterSummaryResponse
	active := 0
	pending := 0
	revoked := 0
	if principal, ok := runtimePrincipalFromContext(req.Context()); ok && principal != nil {
		memberships, selectedCluster = runtimeAuth.membershipResponses(principal.User.ID, principal.Session.SelectedClusterID)
		for _, membership := range memberships {
			switch membership.State {
			case "active":
				active++
			case "pending":
				pending++
			case "revoked":
				revoked++
			}
		}
	}

	return authHealth, &runtimeFederationHealth{
		Enabled:            true,
		TotalClusters:      len(summaries),
		ActiveMemberships:  active,
		PendingMemberships: pending,
		RevokedMemberships: revoked,
		SelectedClusterID: func() string {
			if selectedCluster == nil {
				return ""
			}
			return selectedCluster.ID
		}(),
		HighestTier: highestTier,
		Clusters:    summaries,
	}, memberships, selectedCluster
}

func runtimeGetLeaderHandler(vmManager *vm.VMManager) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if vmManager == nil {
			http.Error(w, "vm service is disabled", http.StatusServiceUnavailable)
			return
		}
		for _, nodeInfo := range vmManager.ListSchedulerNodes() {
			if runtimeIsSchedulableNode(nodeInfo) {
				respondRuntimeJSON(w, http.StatusOK, map[string]string{
					"id":     nodeInfo.NodeID,
					"status": nodeInfo.Status,
					"scope":  "cluster-local",
				})
				return
			}
		}
		http.Error(w, "no schedulable leader available", http.StatusNotFound)
	}
}

func runtimeSchedulerNodeToAPI(nodeInfo *vm.NodeResourceInfo) runtimeNode {
	if nodeInfo == nil {
		return runtimeNode{}
	}

	totalMemoryMB := int64(nodeInfo.TotalMemoryMB)
	usedMemoryMB := int64(nodeInfo.UsedMemoryMB)
	totalDiskGB := int64(nodeInfo.TotalDiskGB)
	usedDiskGB := int64(nodeInfo.UsedDiskGB)
	totalCPU := nodeInfo.TotalCPU
	usedCPU := nodeInfo.UsedCPU

	return runtimeNode{
		ID:                 nodeInfo.NodeID,
		Status:             nodeInfo.Status,
		CPU:                totalCPU,
		Memory:             totalMemoryMB,
		Disk:               totalDiskGB,
		UsedCPU:            usedCPU,
		RemainingCPU:       maxInt(0, totalCPU-usedCPU),
		UsedMemoryMB:       usedMemoryMB,
		RemainingMemoryMB:  maxInt64(0, totalMemoryMB-usedMemoryMB),
		UsedDiskGB:         usedDiskGB,
		RemainingDiskGB:    maxInt64(0, totalDiskGB-usedDiskGB),
		CPUUsagePercent:    nodeInfo.CPUUsagePercent,
		MemoryUsagePercent: nodeInfo.MemoryUsagePercent,
		DiskUsagePercent:   nodeInfo.DiskUsagePercent,
		VMCount:            nodeInfo.VMCount,
		Schedulable:        runtimeIsSchedulableNode(nodeInfo),
		Labels:             cloneStringMap(nodeInfo.Labels),
	}
}

func runtimeIsSchedulableNode(nodeInfo *vm.NodeResourceInfo) bool {
	return nodeInfo != nil && nodeInfo.Status == "available"
}

func respondRuntimeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		log.Printf("failed to encode runtime API response: %v", err)
	}
}

func cloneStringMap(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}

	dst := make(map[string]string, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func maxInt64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}
