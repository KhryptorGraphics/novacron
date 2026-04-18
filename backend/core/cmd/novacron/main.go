package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"runtime"
	"syscall"

	"github.com/khryptorgraphics/novacron/backend/core/hypervisor"
	"github.com/khryptorgraphics/novacron/backend/core/network"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/shirou/gopsutil/v3/disk"
	gopsutilmem "github.com/shirou/gopsutil/v3/mem"
	"gopkg.in/yaml.v2"
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
}

type runtimeConfigFile struct {
	Storage    *storageConfigFile    `yaml:"storage"`
	Hypervisor *hypervisorConfigFile `yaml:"hypervisor"`
	VMManager  *vmManagerConfigFile  `yaml:"vm_manager"`
	Scheduler  *schedulerConfigFile  `yaml:"scheduler"`
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

	storageManager, err := initializeStorage(config)
	if err != nil {
		log.Fatalf("Failed to initialize storage: %v", err)
	}

	vmManager, err := initializeVMManager(config)
	if err != nil {
		log.Fatalf("Failed to initialize VM manager: %v", err)
	}

	schedulerService, err := initializeScheduler(config)
	if err != nil {
		log.Fatalf("Failed to initialize scheduler: %v", err)
	}

	migrationManager, err := initializeMigrationManager(*nodeID, *dataDir)
	if err != nil {
		log.Fatalf("Failed to initialize migration manager: %v", err)
	}

	networkManager, err := initializeNetwork(ctx, config, *nodeID)
	if err != nil {
		log.Fatalf("Failed to initialize network: %v", err)
	}

	hypervisorManager, err := initializeHypervisor(ctx, config, *nodeID)
	if err != nil {
		log.Fatalf("Failed to initialize hypervisor: %v", err)
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
	if err := hypervisorManager.Stop(); err != nil {
		log.Printf("Hypervisor shutdown error: %v", err)
	}
	if err := networkManager.Stop(); err != nil {
		log.Printf("Network manager shutdown error: %v", err)
	}
	if err := schedulerService.Stop(); err != nil {
		log.Printf("Scheduler shutdown error: %v", err)
	}
	if err := vmManager.Stop(); err != nil {
		log.Printf("VM manager shutdown error: %v", err)
	}

	log.Println("Shutdown complete")
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
	}
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

	var fileConfig runtimeConfigFile
	if err := yaml.Unmarshal(configBytes, &fileConfig); err != nil {
		return runtimeConfig{}, fmt.Errorf("parse config %s: %w", path, err)
	}

	mergeRuntimeConfig(&config, fileConfig)
	applyRuntimeConfigDefaults(&config, nodeID, dataDir)
	log.Printf("Loaded configuration from %s", path)
	return config, nil
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
) (*APIServer, error) {
	_ = ctx
	_ = config
	_ = vmManager
	_ = migrationManager
	_ = schedulerService
	_ = networkManager
	_ = storageManager

	log.Printf("Starting API server on %s", listenAddress)

	apiServer := &APIServer{}
	if err := apiServer.Start(); err != nil {
		return nil, fmt.Errorf("start API server: %w", err)
	}

	return apiServer, nil
}

type APIServer struct{}

func (s *APIServer) Start() error {
	return nil
}

func (s *APIServer) Shutdown(ctx context.Context) error {
	_ = ctx
	return nil
}
