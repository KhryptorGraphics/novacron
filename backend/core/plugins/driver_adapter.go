package plugins

import (
	"fmt"
)

// VMDriverAdapter provides an adapter between the plugin system and the VM drivers
// This allows VM drivers to be loaded dynamically as plugins
type VMDriverAdapter struct {
	// The plugin manager used to discover and load VM driver plugins
	pluginManager *PluginManager

	// Cache of VM drivers loaded from plugins
	driverCache map[string]interface{}
}

// NewVMDriverAdapter creates a new VM driver adapter
func NewVMDriverAdapter(pluginManager *PluginManager) *VMDriverAdapter {
	return &VMDriverAdapter{
		pluginManager: pluginManager,
		driverCache:   make(map[string]interface{}),
	}
}

// DiscoverDrivers discovers all VM driver plugins
func (a *VMDriverAdapter) DiscoverDrivers() ([]string, error) {
	// Load all plugins
	if err := a.pluginManager.LoadAllPlugins(); err != nil {
		return nil, fmt.Errorf("failed to load plugins: %v", err)
	}

	// Get all plugins
	plugins := a.pluginManager.ListPlugins()

	// Enable all plugins
	for _, plugin := range plugins {
		if !plugin.Enabled {
			if err := a.pluginManager.EnablePlugin(plugin.Info.ID); err != nil {
				// Log the error but continue
				fmt.Printf("Failed to enable plugin %s: %v\n", plugin.Info.ID, err)
			}
		}
	}

	// Get all VM driver implementations
	drivers := a.pluginManager.GetInterfaceImpl("VMDriver")

	driverNames := make([]string, 0, len(drivers))
	for i, driver := range drivers {
		// Store in cache
		driverName := fmt.Sprintf("plugin-driver-%d", i+1)
		if driver != nil {
			if named, ok := driver.(interface{ Name() string }); ok {
				driverName = named.Name()
			}
		}
		a.driverCache[driverName] = driver
		driverNames = append(driverNames, driverName)
	}

	return driverNames, nil
}

// GetDriver returns a VM driver by name
func (a *VMDriverAdapter) GetDriver(name string) (interface{}, error) {
	driver, exists := a.driverCache[name]
	if !exists {
		return nil, fmt.Errorf("driver %s not found", name)
	}
	return driver, nil
}

// StorageDriverAdapter provides an adapter between the plugin system and the storage drivers
type StorageDriverAdapter struct {
	// The plugin manager used to discover and load storage driver plugins
	pluginManager *PluginManager

	// Cache of storage drivers loaded from plugins
	driverCache map[string]interface{}
}

// NewStorageDriverAdapter creates a new storage driver adapter
func NewStorageDriverAdapter(pluginManager *PluginManager) *StorageDriverAdapter {
	return &StorageDriverAdapter{
		pluginManager: pluginManager,
		driverCache:   make(map[string]interface{}),
	}
}

// DiscoverDrivers discovers all storage driver plugins
func (a *StorageDriverAdapter) DiscoverDrivers() ([]string, error) {
	// Load all plugins
	if err := a.pluginManager.LoadAllPlugins(); err != nil {
		return nil, fmt.Errorf("failed to load plugins: %v", err)
	}

	// Get all plugins
	plugins := a.pluginManager.ListPlugins()

	// Enable all plugins
	for _, plugin := range plugins {
		if !plugin.Enabled {
			if err := a.pluginManager.EnablePlugin(plugin.Info.ID); err != nil {
				// Log the error but continue
				fmt.Printf("Failed to enable plugin %s: %v\n", plugin.Info.ID, err)
			}
		}
	}

	// Get all storage driver implementations
	drivers := a.pluginManager.GetInterfaceImpl("StorageDriver")

	driverNames := make([]string, 0, len(drivers))
	for i, driver := range drivers {
		// Store in cache
		driverName := fmt.Sprintf("plugin-driver-%d", i+1)
		if driver != nil {
			if named, ok := driver.(interface{ Name() string }); ok {
				driverName = named.Name()
			}
		}
		a.driverCache[driverName] = driver
		driverNames = append(driverNames, driverName)
	}

	return driverNames, nil
}

// GetDriver returns a storage driver by name
func (a *StorageDriverAdapter) GetDriver(name string) (interface{}, error) {
	driver, exists := a.driverCache[name]
	if !exists {
		return nil, fmt.Errorf("driver %s not found", name)
	}
	return driver, nil
}

// NetworkDriverAdapter provides an adapter between the plugin system and the network drivers
type NetworkDriverAdapter struct {
	// The plugin manager used to discover and load network driver plugins
	pluginManager *PluginManager

	// Cache of network drivers loaded from plugins
	driverCache map[string]interface{}
}

// NewNetworkDriverAdapter creates a new network driver adapter
func NewNetworkDriverAdapter(pluginManager *PluginManager) *NetworkDriverAdapter {
	return &NetworkDriverAdapter{
		pluginManager: pluginManager,
		driverCache:   make(map[string]interface{}),
	}
}

// DiscoverDrivers discovers all network driver plugins
func (a *NetworkDriverAdapter) DiscoverDrivers() ([]string, error) {
	// Load all plugins
	if err := a.pluginManager.LoadAllPlugins(); err != nil {
		return nil, fmt.Errorf("failed to load plugins: %v", err)
	}

	// Get all plugins
	plugins := a.pluginManager.ListPlugins()

	// Enable all plugins
	for _, plugin := range plugins {
		if !plugin.Enabled {
			if err := a.pluginManager.EnablePlugin(plugin.Info.ID); err != nil {
				// Log the error but continue
				fmt.Printf("Failed to enable plugin %s: %v\n", plugin.Info.ID, err)
			}
		}
	}

	// Get all network driver implementations
	drivers := a.pluginManager.GetInterfaceImpl("NetworkDriver")

	driverNames := make([]string, 0, len(drivers))
	for i, driver := range drivers {
		// Store in cache
		driverName := fmt.Sprintf("plugin-driver-%d", i+1)
		if driver != nil {
			if named, ok := driver.(interface{ Name() string }); ok {
				driverName = named.Name()
			}
		}
		a.driverCache[driverName] = driver
		driverNames = append(driverNames, driverName)
	}

	return driverNames, nil
}

// GetDriver returns a network driver by name
func (a *NetworkDriverAdapter) GetDriver(name string) (interface{}, error) {
	driver, exists := a.driverCache[name]
	if !exists {
		return nil, fmt.Errorf("driver %s not found", name)
	}
	return driver, nil
}

// DriverRegistry provides a central registry for all driver adapters
type DriverRegistry struct {
	VMDriverAdapter      *VMDriverAdapter
	StorageDriverAdapter *StorageDriverAdapter
	NetworkDriverAdapter *NetworkDriverAdapter
}

// NewDriverRegistry creates a new driver registry with all driver adapters
func NewDriverRegistry(pluginManager *PluginManager) *DriverRegistry {
	return &DriverRegistry{
		VMDriverAdapter:      NewVMDriverAdapter(pluginManager),
		StorageDriverAdapter: NewStorageDriverAdapter(pluginManager),
		NetworkDriverAdapter: NewNetworkDriverAdapter(pluginManager),
	}
}

// DiscoverAllDrivers discovers all drivers of all types
func (r *DriverRegistry) DiscoverAllDrivers() (map[string][]string, error) {
	result := make(map[string][]string)

	// Discover VM drivers
	vmDrivers, err := r.VMDriverAdapter.DiscoverDrivers()
	if err != nil {
		return nil, fmt.Errorf("failed to discover VM drivers: %v", err)
	}
	result["vm"] = vmDrivers

	// Discover storage drivers
	storageDrivers, err := r.StorageDriverAdapter.DiscoverDrivers()
	if err != nil {
		return nil, fmt.Errorf("failed to discover storage drivers: %v", err)
	}
	result["storage"] = storageDrivers

	// Discover network drivers
	networkDrivers, err := r.NetworkDriverAdapter.DiscoverDrivers()
	if err != nil {
		return nil, fmt.Errorf("failed to discover network drivers: %v", err)
	}
	result["network"] = networkDrivers

	return result, nil
}
