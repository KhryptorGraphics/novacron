package storageplugins

import (
	"fmt"
	"log"
	"sync"

	"github.com/khryptorgraphics/novacron/backend/core/plugins/storage/ceph"
	"github.com/khryptorgraphics/novacron/backend/core/plugins/storage/netfs"
	"github.com/khryptorgraphics/novacron/backend/core/plugins/storage/objectstorage"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
)

// StoragePluginRegistry maintains the registry of available storage plugins
type StoragePluginRegistry struct {
	// Map of plugin type -> map of plugin name -> plugin info
	plugins     map[string]map[string]interface{}
	initialized bool
	mu          sync.RWMutex
}

// Global registry instance
var Registry = &StoragePluginRegistry{
	plugins:     make(map[string]map[string]interface{}),
	initialized: false,
}

// InitializeRegistry initializes the storage plugin registry
func InitializeRegistry() error {
	Registry.mu.Lock()
	defer Registry.mu.Unlock()

	if Registry.initialized {
		return nil
	}

	// Create plugin type map if it doesn't exist
	if _, exists := Registry.plugins["StorageDriver"]; !exists {
		Registry.plugins["StorageDriver"] = make(map[string]interface{})
	}

	// Register Ceph storage driver
	Registry.plugins["StorageDriver"][ceph.CephPluginInfo.Name] = ceph.CephPluginInfo

	// Register Network File System storage driver
	Registry.plugins["StorageDriver"][netfs.NetworkFSPluginInfo.Name] = netfs.NetworkFSPluginInfo

	// Register Object Storage driver
	Registry.plugins["StorageDriver"][objectstorage.ObjectStoragePluginInfo.Name] = objectstorage.ObjectStoragePluginInfo

	// Register factories with the core storage system
	registerStorageDriverFactories()

	Registry.initialized = true
	log.Println("Storage plugin registry initialized with", len(Registry.plugins["StorageDriver"]), "storage drivers")
	return nil
}

// GetStoragePlugin returns a storage plugin by name
func (r *StoragePluginRegistry) GetStoragePlugin(name string) (interface{}, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return nil, fmt.Errorf("plugin registry not initialized")
	}

	if plugin, exists := r.plugins["StorageDriver"][name]; exists {
		return plugin, nil
	}

	return nil, fmt.Errorf("storage plugin '%s' not found", name)
}

// ListStoragePlugins returns a list of all registered storage plugins
func (r *StoragePluginRegistry) ListStoragePlugins() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if !r.initialized {
		return []string{}
	}

	plugins := make([]string, 0, len(r.plugins["StorageDriver"]))
	for name := range r.plugins["StorageDriver"] {
		plugins = append(plugins, name)
	}
	return plugins
}

// registerStorageDriverFactories registers the storage driver factories with the core storage system
func registerStorageDriverFactories() {
	// Register Ceph storage driver factory
	storage.RegisterDriver(ceph.CephPluginInfo.Name, func(config map[string]interface{}) (storage.StorageDriver, error) {
		// Convert generic config to Ceph-specific config
		cephConfig := ceph.DefaultCephConfig()

		// Apply configuration overrides if provided
		if monHosts, ok := config["mon_hosts"].([]string); ok {
			cephConfig.MonHosts = monHosts
		}
		if pool, ok := config["pool"].(string); ok {
			cephConfig.Pool = pool
		}
		if user, ok := config["user"].(string); ok {
			cephConfig.User = user
		}
		// Additional config parameters would be handled similarly

		// Create and return the driver
		return ceph.NewCephStorageDriver(cephConfig), nil
	})

	// Register Network File System storage driver factory
	storage.RegisterDriver(netfs.NetworkFSPluginInfo.Name, func(config map[string]interface{}) (storage.StorageDriver, error) {
		// Convert generic config to NetFS-specific config
		netFSConfig := netfs.DefaultNetworkFileConfig()

		// Apply configuration overrides if provided
		if defaultProtocol, ok := config["default_protocol"].(string); ok {
			netFSConfig.DefaultProtocol = defaultProtocol
		}
		if mountBasePath, ok := config["mount_base_path"].(string); ok {
			netFSConfig.MountBasePath = mountBasePath
		}
		// Additional config parameters would be handled similarly

		// Create and return the driver
		return netfs.NewNetworkFileStorageDriver(netFSConfig), nil
	})

	// Register Object Storage driver factory
	storage.RegisterDriver(objectstorage.ObjectStoragePluginInfo.Name, func(config map[string]interface{}) (storage.StorageDriver, error) {
		// Convert generic config to Object Storage-specific config
		objConfig := objectstorage.DefaultObjectStorageConfig()

		// Apply configuration overrides if provided
		if provider, ok := config["provider"].(string); ok {
			objConfig.Provider = provider
		}
		if endpoint, ok := config["endpoint"].(string); ok {
			objConfig.Endpoint = endpoint
		}
		if accessKey, ok := config["access_key"].(string); ok {
			objConfig.AccessKey = accessKey
		}
		if secretKey, ok := config["secret_key"].(string); ok {
			objConfig.SecretKey = secretKey
		}
		// Additional config parameters would be handled similarly

		// Create and return the driver
		return objectstorage.NewObjectStorageDriver(objConfig), nil
	})
}
