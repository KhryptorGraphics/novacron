// Package di provides dependency injection for NovaCron components
package di

import (
	"fmt"
	"reflect"
	"sync"
)

// Container manages service registration and resolution
type Container struct {
	mu        sync.RWMutex
	services  map[string]*serviceEntry
	singletons map[string]interface{}
}

// serviceEntry represents a registered service
type serviceEntry struct {
	factory   interface{}
	singleton bool
	instance  interface{}
}

// NewContainer creates a new DI container
func NewContainer() *Container {
	return &Container{
		services:   make(map[string]*serviceEntry),
		singletons: make(map[string]interface{}),
	}
}

// Register registers a service factory function
// Factory signature: func(container *Container) (T, error)
func (c *Container) Register(name string, factory interface{}, singleton bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.services[name]; exists {
		return fmt.Errorf("service already registered: %s", name)
	}

	// Validate factory function
	factoryType := reflect.TypeOf(factory)
	if factoryType.Kind() != reflect.Func {
		return fmt.Errorf("factory must be a function")
	}

	if factoryType.NumIn() != 1 || factoryType.In(0) != reflect.TypeOf(c) {
		return fmt.Errorf("factory must accept *Container as first argument")
	}

	if factoryType.NumOut() != 2 {
		return fmt.Errorf("factory must return (T, error)")
	}

	if !factoryType.Out(1).Implements(reflect.TypeOf((*error)(nil)).Elem()) {
		return fmt.Errorf("factory second return value must be error")
	}

	c.services[name] = &serviceEntry{
		factory:   factory,
		singleton: singleton,
	}

	return nil
}

// RegisterSingleton registers a singleton service factory
func (c *Container) RegisterSingleton(name string, factory interface{}) error {
	return c.Register(name, factory, true)
}

// RegisterTransient registers a transient service factory (new instance each time)
func (c *Container) RegisterTransient(name string, factory interface{}) error {
	return c.Register(name, factory, false)
}

// RegisterInstance registers a pre-created instance as a singleton
func (c *Container) RegisterInstance(name string, instance interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.services[name]; exists {
		return fmt.Errorf("service already registered: %s", name)
	}

	c.services[name] = &serviceEntry{
		singleton: true,
		instance:  instance,
	}
	c.singletons[name] = instance

	return nil
}

// Resolve resolves a service by name
func (c *Container) Resolve(name string) (interface{}, error) {
	c.mu.RLock()
	entry, exists := c.services[name]
	c.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("service not registered: %s", name)
	}

	// Return existing singleton instance
	if entry.singleton && entry.instance != nil {
		return entry.instance, nil
	}

	// Create new instance
	if entry.factory == nil {
		return nil, fmt.Errorf("no factory registered for: %s", name)
	}

	instance, err := c.invokeFactory(entry.factory)
	if err != nil {
		return nil, fmt.Errorf("failed to create instance for %s: %w", name, err)
	}

	// Store singleton instance
	if entry.singleton {
		c.mu.Lock()
		entry.instance = instance
		c.singletons[name] = instance
		c.mu.Unlock()
	}

	return instance, nil
}

// MustResolve resolves a service and panics on error
func (c *Container) MustResolve(name string) interface{} {
	instance, err := c.Resolve(name)
	if err != nil {
		panic(fmt.Sprintf("failed to resolve %s: %v", name, err))
	}
	return instance
}

// ResolveAs resolves a service and type-asserts to T
func ResolveAs[T any](c *Container, name string) (T, error) {
	var zero T

	instance, err := c.Resolve(name)
	if err != nil {
		return zero, err
	}

	typed, ok := instance.(T)
	if !ok {
		return zero, fmt.Errorf("service %s is not of type %T", name, zero)
	}

	return typed, nil
}

// MustResolveAs resolves a service and type-asserts to T, panics on error
func MustResolveAs[T any](c *Container, name string) T {
	instance, err := ResolveAs[T](c, name)
	if err != nil {
		panic(fmt.Sprintf("failed to resolve %s: %v", name, err))
	}
	return instance
}

// invokeFactory invokes a factory function
func (c *Container) invokeFactory(factory interface{}) (interface{}, error) {
	factoryValue := reflect.ValueOf(factory)

	// Call factory with container as argument
	results := factoryValue.Call([]reflect.Value{reflect.ValueOf(c)})

	// Check for error
	if !results[1].IsNil() {
		return nil, results[1].Interface().(error)
	}

	return results[0].Interface(), nil
}

// Has checks if a service is registered
func (c *Container) Has(name string) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	_, exists := c.services[name]
	return exists
}

// Clear removes all registered services
func (c *Container) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.services = make(map[string]*serviceEntry)
	c.singletons = make(map[string]interface{})
}

// GetRegisteredServices returns list of all registered service names
func (c *Container) GetRegisteredServices() []string {
	c.mu.RLock()
	defer c.mu.RUnlock()

	names := make([]string, 0, len(c.services))
	for name := range c.services {
		names = append(names, name)
	}
	return names
}

// GetSingletonInstance returns singleton instance if exists
func (c *Container) GetSingletonInstance(name string) (interface{}, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	instance, exists := c.singletons[name]
	return instance, exists
}

// Inject performs constructor injection using reflection
func (c *Container) Inject(target interface{}) error {
	targetValue := reflect.ValueOf(target)
	if targetValue.Kind() != reflect.Ptr {
		return fmt.Errorf("target must be a pointer")
	}

	targetValue = targetValue.Elem()
	targetType := targetValue.Type()

	if targetType.Kind() != reflect.Struct {
		return fmt.Errorf("target must be a struct pointer")
	}

	// Iterate through struct fields
	for i := 0; i < targetType.NumField(); i++ {
		field := targetType.Field(i)
		fieldValue := targetValue.Field(i)

		// Check for "inject" tag
		injectTag := field.Tag.Get("inject")
		if injectTag == "" {
			continue
		}

		if !fieldValue.CanSet() {
			return fmt.Errorf("cannot set field %s", field.Name)
		}

		// Resolve dependency
		instance, err := c.Resolve(injectTag)
		if err != nil {
			return fmt.Errorf("failed to inject %s: %w", field.Name, err)
		}

		// Set field value
		instanceValue := reflect.ValueOf(instance)
		if !instanceValue.Type().AssignableTo(fieldValue.Type()) {
			return fmt.Errorf("cannot assign %s to field %s", instanceValue.Type(), field.Name)
		}

		fieldValue.Set(instanceValue)
	}

	return nil
}

// Build builds an instance using constructor injection
func (c *Container) Build(constructor interface{}) (interface{}, error) {
	constructorValue := reflect.ValueOf(constructor)
	constructorType := constructorValue.Type()

	if constructorType.Kind() != reflect.Func {
		return nil, fmt.Errorf("constructor must be a function")
	}

	// Resolve dependencies for each parameter
	numParams := constructorType.NumIn()
	params := make([]reflect.Value, numParams)

	for i := 0; i < numParams; i++ {
		paramType := constructorType.In(i)

		// Try to resolve by type name
		typeName := paramType.String()
		instance, err := c.Resolve(typeName)
		if err != nil {
			return nil, fmt.Errorf("failed to resolve parameter %d (%s): %w", i, typeName, err)
		}

		params[i] = reflect.ValueOf(instance)
	}

	// Call constructor
	results := constructorValue.Call(params)

	// Check for error return
	if len(results) > 1 && results[len(results)-1].Type().Implements(reflect.TypeOf((*error)(nil)).Elem()) {
		if !results[len(results)-1].IsNil() {
			return nil, results[len(results)-1].Interface().(error)
		}
	}

	return results[0].Interface(), nil
}
