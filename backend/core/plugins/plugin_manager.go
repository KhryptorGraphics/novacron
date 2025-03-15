package plugins

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"plugin"
	"reflect"
	"sync"
)

// PluginInfo contains metadata about a plugin
type PluginInfo struct {
	// Unique identifier for the plugin
	ID string

	// Human-readable name
	Name string

	// Version of the plugin
	Version string

	// Author information
	Author string

	// Description of what the plugin does
	Description string

	// License information
	License string

	// Compatibility information (e.g., "v1.0+")
	Compatibility string

	// Tags for categorization and search
	Tags []string

	// Dependencies on other plugins
	Dependencies []string

	// Path to the plugin on disk
	Path string
}

// PluginInstance represents a loaded plugin
type PluginInstance struct {
	// Plugin metadata
	Info PluginInfo

	// The loaded plugin
	Plugin *plugin.Plugin

	// Extracted symbols (function pointers, etc.)
	Symbols map[string]plugin.Symbol

	// Instance-specific state
	State map[string]interface{}

	// Whether the plugin is enabled
	Enabled bool

	// Error encountered during loading, if any
	LoadError error
}

// PluginManager handles the discovery, loading, and lifecycle of plugins
type PluginManager struct {
	// Map of plugin ID to plugin instances
	plugins map[string]*PluginInstance

	// Plugin discovery paths
	paths []string

	// Lock for concurrent access
	lock sync.RWMutex

	// Hooks that plugins can register for
	hooks map[string][]plugin.Symbol

	// Plugin load order based on dependencies
	loadOrder []string

	// Map of interface types to handlers
	interfaceHandlers map[string][]plugin.Symbol
}

// PluginManagerConfig contains configuration for the plugin manager
type PluginManagerConfig struct {
	// Paths to search for plugins
	Paths []string

	// Whether to validate dependencies
	ValidateDependencies bool

	// Whether to auto-enable plugins after loading
	AutoEnable bool

	// Plugin load timeout in seconds
	LoadTimeoutSeconds int

	// Whether to skip plugins with errors
	SkipErrorPlugins bool
}

// DefaultPluginManagerConfig returns a default configuration
func DefaultPluginManagerConfig() PluginManagerConfig {
	return PluginManagerConfig{
		Paths:                []string{"./plugins"},
		ValidateDependencies: true,
		AutoEnable:           true,
		LoadTimeoutSeconds:   10,
		SkipErrorPlugins:     true,
	}
}

// NewPluginManager creates a new plugin manager
func NewPluginManager(config PluginManagerConfig) *PluginManager {
	return &PluginManager{
		plugins:           make(map[string]*PluginInstance),
		paths:             config.Paths,
		hooks:             make(map[string][]plugin.Symbol),
		interfaceHandlers: make(map[string][]plugin.Symbol),
	}
}

// DiscoverPlugins searches for plugins in the configured paths
func (pm *PluginManager) DiscoverPlugins() ([]PluginInfo, error) {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	var plugins []PluginInfo

	for _, path := range pm.paths {
		err := filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}

			// Only consider .so files as potential plugins
			if !info.IsDir() && filepath.Ext(p) == ".so" {
				pluginInfo, err := pm.inspectPlugin(p)
				if err != nil {
					log.Printf("Failed to inspect plugin %s: %v", p, err)
					return nil // Continue to the next file
				}

				plugins = append(plugins, pluginInfo)
			}

			return nil
		})

		if err != nil {
			return nil, fmt.Errorf("failed to discover plugins in %s: %v", path, err)
		}
	}

	return plugins, nil
}

// inspectPlugin extracts metadata from a plugin file
func (pm *PluginManager) inspectPlugin(path string) (PluginInfo, error) {
	// Open the plugin
	p, err := plugin.Open(path)
	if err != nil {
		return PluginInfo{}, fmt.Errorf("failed to open plugin: %v", err)
	}

	// Look up the plugin info symbol
	infoSymbol, err := p.Lookup("PluginInfo")
	if err != nil {
		return PluginInfo{}, fmt.Errorf("plugin does not export PluginInfo: %v", err)
	}

	// Assert that it's a pointer to PluginInfo
	infoPtr, ok := infoSymbol.(*PluginInfo)
	if !ok {
		return PluginInfo{}, fmt.Errorf("plugin's PluginInfo is not of type *PluginInfo")
	}

	// Make a copy of the info
	info := *infoPtr
	info.Path = path

	return info, nil
}

// LoadPlugin loads a specific plugin by path
func (pm *PluginManager) LoadPlugin(path string) (*PluginInstance, error) {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	// Inspect the plugin first
	info, err := pm.inspectPlugin(path)
	if err != nil {
		return nil, err
	}

	// Check if plugin is already loaded
	if instance, exists := pm.plugins[info.ID]; exists {
		return instance, nil
	}

	// Open the plugin
	p, err := plugin.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open plugin: %v", err)
	}

	// Create a new plugin instance
	instance := &PluginInstance{
		Info:      info,
		Plugin:    p,
		Symbols:   make(map[string]plugin.Symbol),
		State:     make(map[string]interface{}),
		Enabled:   false,
		LoadError: nil,
	}

	// Look up common symbols
	for _, symbolName := range []string{"Initialize", "Start", "Stop", "GetInterface"} {
		symbol, err := p.Lookup(symbolName)
		if err == nil {
			instance.Symbols[symbolName] = symbol
		}
	}

	// Store the plugin
	pm.plugins[info.ID] = instance

	// Add to load order
	pm.loadOrder = append(pm.loadOrder, info.ID)

	return instance, nil
}

// LoadAllPlugins loads all plugins from the configured paths
func (pm *PluginManager) LoadAllPlugins() error {
	// Discover plugins first
	pluginInfos, err := pm.DiscoverPlugins()
	if err != nil {
		return err
	}

	// Load each plugin
	for _, info := range pluginInfos {
		_, err := pm.LoadPlugin(info.Path)
		if err != nil {
			log.Printf("Failed to load plugin %s: %v", info.ID, err)
		}
	}

	return nil
}

// GetPlugin returns a plugin by ID
func (pm *PluginManager) GetPlugin(id string) (*PluginInstance, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	plugin, exists := pm.plugins[id]
	return plugin, exists
}

// ListPlugins returns a list of all loaded plugins
func (pm *PluginManager) ListPlugins() []*PluginInstance {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	plugins := make([]*PluginInstance, 0, len(pm.plugins))
	for _, instance := range pm.plugins {
		plugins = append(plugins, instance)
	}

	return plugins
}

// EnablePlugin enables a specific plugin
func (pm *PluginManager) EnablePlugin(id string) error {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	instance, exists := pm.plugins[id]
	if !exists {
		return fmt.Errorf("plugin %s not found", id)
	}

	if instance.Enabled {
		return nil // Already enabled
	}

	// Call the Initialize function if it exists
	if initFunc, exists := instance.Symbols["Initialize"]; exists {
		if initializer, ok := initFunc.(func() error); ok {
			if err := initializer(); err != nil {
				return fmt.Errorf("failed to initialize plugin %s: %v", id, err)
			}
		}
	}

	// Call the Start function if it exists
	if startFunc, exists := instance.Symbols["Start"]; exists {
		if starter, ok := startFunc.(func() error); ok {
			if err := starter(); err != nil {
				return fmt.Errorf("failed to start plugin %s: %v", id, err)
			}
		}
	}

	// Register hooks
	if getHooksFunc, exists := instance.Symbols["GetHooks"]; exists {
		if hookGetter, ok := getHooksFunc.(func() map[string]plugin.Symbol); ok {
			hooks := hookGetter()
			for hookName, hookFunc := range hooks {
				pm.hooks[hookName] = append(pm.hooks[hookName], hookFunc)
			}
		}
	}

	// Register interface handlers
	if getInterfaceFunc, exists := instance.Symbols["GetInterface"]; exists {
		if interfaceGetter, ok := getInterfaceFunc.(func(string) interface{}); ok {
			// We can't know all interfaces in advance, so this is just a placeholder
			// In a real implementation, this would be more sophisticated

			// Register common interfaces
			commonInterfaces := []string{"VMDriver", "StorageDriver", "NetworkDriver"}
			for _, iface := range commonInterfaces {
				if impl := interfaceGetter(iface); impl != nil {
					pm.interfaceHandlers[iface] = append(pm.interfaceHandlers[iface],
						func() interface{} { return impl })
				}
			}
		}
	}

	instance.Enabled = true
	return nil
}

// DisablePlugin disables a specific plugin
func (pm *PluginManager) DisablePlugin(id string) error {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	instance, exists := pm.plugins[id]
	if !exists {
		return fmt.Errorf("plugin %s not found", id)
	}

	if !instance.Enabled {
		return nil // Already disabled
	}

	// Call the Stop function if it exists
	if stopFunc, exists := instance.Symbols["Stop"]; exists {
		if stopper, ok := stopFunc.(func() error); ok {
			if err := stopper(); err != nil {
				return fmt.Errorf("failed to stop plugin %s: %v", id, err)
			}
		}
	}

	// Unregister hooks (a more sophisticated implementation would track which hooks came from which plugin)
	// Unregister interface handlers

	instance.Enabled = false
	return nil
}

// ExecuteHook executes all registered handlers for a specific hook
func (pm *PluginManager) ExecuteHook(hookName string, args ...interface{}) []interface{} {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	results := make([]interface{}, 0)

	// Get handlers for this hook
	handlers, exists := pm.hooks[hookName]
	if !exists {
		return results
	}

	// Prepare arguments for reflection
	reflectArgs := make([]reflect.Value, len(args))
	for i, arg := range args {
		reflectArgs[i] = reflect.ValueOf(arg)
	}

	// Call each handler
	for _, handler := range handlers {
		handlerValue := reflect.ValueOf(handler)
		resultValues := handlerValue.Call(reflectArgs)

		// Collect results
		for _, resultValue := range resultValues {
			results = append(results, resultValue.Interface())
		}
	}

	return results
}

// GetInterfaceImpl returns all implementations of a specific interface
func (pm *PluginManager) GetInterfaceImpl(interfaceName string) []interface{} {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	results := make([]interface{}, 0)

	// Get handlers for this interface
	handlers, exists := pm.interfaceHandlers[interfaceName]
	if !exists {
		return results
	}

	// Call each handler to get the interface implementation
	for _, handler := range handlers {
		if getter, ok := handler.(func() interface{}); ok {
			impl := getter()
			results = append(results, impl)
		}
	}

	return results
}

// SortPluginsByDependencies sorts plugins based on their dependencies
func (pm *PluginManager) SortPluginsByDependencies() ([]string, error) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	// Build a dependency graph
	graph := make(map[string][]string)
	for id, instance := range pm.plugins {
		graph[id] = instance.Info.Dependencies
	}

	// Perform topological sort
	visited := make(map[string]bool)
	temp := make(map[string]bool)
	order := make([]string, 0)

	var visit func(string) error
	visit = func(id string) error {
		if temp[id] {
			return fmt.Errorf("cyclic dependency detected involving plugin %s", id)
		}
		if visited[id] {
			return nil
		}
		temp[id] = true

		for _, depID := range graph[id] {
			if err := visit(depID); err != nil {
				return err
			}
		}

		visited[id] = true
		temp[id] = false
		order = append(order, id)
		return nil
	}

	for id := range pm.plugins {
		if !visited[id] {
			if err := visit(id); err != nil {
				return nil, err
			}
		}
	}

	// Reverse the order to get the correct load sequence
	for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}

	return order, nil
}

// Shutdown gracefully stops all plugins
func (pm *PluginManager) Shutdown() error {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	// Get plugins in reverse dependency order
	order, err := pm.SortPluginsByDependencies()
	if err != nil {
		return err
	}

	// Reverse the order for shutdown
	for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}

	// Shutdown each plugin
	for _, id := range order {
		instance := pm.plugins[id]
		if instance.Enabled {
			if err := pm.DisablePlugin(id); err != nil {
				log.Printf("Error disabling plugin %s: %v", id, err)
			}
		}
	}

	return nil
}
