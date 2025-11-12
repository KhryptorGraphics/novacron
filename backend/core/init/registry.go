package init

import (
	"context"
	"fmt"
	"sync"
)

// ComponentRegistry manages component lifecycle
type ComponentRegistry struct {
	components map[string]Component
	order      []string
	mu         sync.RWMutex
}

// NewComponentRegistry creates a new component registry
func NewComponentRegistry() *ComponentRegistry {
	return &ComponentRegistry{
		components: make(map[string]Component),
		order:      make([]string, 0),
	}
}

// Register registers a component
func (r *ComponentRegistry) Register(name string, component Component) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if _, exists := r.components[name]; exists {
		return fmt.Errorf("component already registered: %s", name)
	}

	r.components[name] = component
	r.order = append(r.order, name)
	return nil
}

// Get retrieves a component by name
func (r *ComponentRegistry) Get(name string) (Component, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	component, exists := r.components[name]
	if !exists {
		return nil, fmt.Errorf("component not found: %s", name)
	}

	return component, nil
}

// GetAll returns all registered components
func (r *ComponentRegistry) GetAll() map[string]Component {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make(map[string]Component, len(r.components))
	for name, component := range r.components {
		result[name] = component
	}

	return result
}

// GetOrder returns the registration order of components
func (r *ComponentRegistry) GetOrder() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	result := make([]string, len(r.order))
	copy(result, r.order)
	return result
}

// Shutdown shuts down all components in reverse order
func (r *ComponentRegistry) Shutdown(ctx context.Context) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	var lastErr error

	// Shutdown in reverse order
	for i := len(r.order) - 1; i >= 0; i-- {
		name := r.order[i]
		component := r.components[name]

		if err := component.Shutdown(ctx); err != nil {
			lastErr = err
			// Log error but continue with other components
			fmt.Printf("Failed to shutdown %s: %v\n", name, err)
		}
	}

	return lastErr
}

// HealthCheck runs health checks on all components
func (r *ComponentRegistry) HealthCheck() map[string]error {
	r.mu.RLock()
	defer r.mu.RUnlock()

	results := make(map[string]error)

	for name, component := range r.components {
		if err := component.HealthCheck(); err != nil {
			results[name] = err
		}
	}

	return results
}

// DependencyResolver resolves component initialization order
type DependencyResolver struct {
	components map[string]Component
}

// NewDependencyResolver creates a new dependency resolver
func NewDependencyResolver(components map[string]Component) *DependencyResolver {
	return &DependencyResolver{
		components: components,
	}
}

// Resolve returns initialization order using topological sort
func (r *DependencyResolver) Resolve() ([]string, error) {
	// Build adjacency list and in-degree map
	graph := make(map[string][]string)
	inDegree := make(map[string]int)

	// Initialize all components with 0 in-degree
	for name := range r.components {
		inDegree[name] = 0
	}

	// Build graph and calculate in-degrees
	for name, component := range r.components {
		deps := component.Dependencies()
		graph[name] = deps

		// Verify all dependencies exist
		for _, dep := range deps {
			if _, exists := r.components[dep]; !exists {
				return nil, fmt.Errorf("missing dependency: %s for component %s", dep, name)
			}
			inDegree[dep]++
		}
	}

	// Topological sort using Kahn's algorithm
	queue := []string{}
	for name, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, name)
		}
	}

	order := []string{}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		order = append(order, current)

		for _, dep := range graph[current] {
			inDegree[dep]--
			if inDegree[dep] == 0 {
				queue = append(queue, dep)
			}
		}
	}

	// Check for cycles
	if len(order) != len(r.components) {
		return nil, fmt.Errorf("circular dependency detected")
	}

	return order, nil
}
