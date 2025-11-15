package lifecycle

import (
	"fmt"
	"sort"
)

// DependencyGraph manages component dependencies
type DependencyGraph struct {
	// nodes maps component name to its dependencies
	nodes map[string][]string

	// dependents maps component to components that depend on it
	dependents map[string][]string
}

// NewDependencyGraph creates a new dependency graph
func NewDependencyGraph() *DependencyGraph {
	return &DependencyGraph{
		nodes:      make(map[string][]string),
		dependents: make(map[string][]string),
	}
}

// AddNode adds a component and its dependencies to the graph
func (g *DependencyGraph) AddNode(component string, dependencies []string) error {
	// Check for self-dependency
	for _, dep := range dependencies {
		if dep == component {
			return fmt.Errorf("component %s cannot depend on itself", component)
		}
	}

	// Add node
	g.nodes[component] = dependencies

	// Update dependents
	for _, dep := range dependencies {
		g.dependents[dep] = append(g.dependents[dep], component)
	}

	// Check for cycles
	if g.hasCycle() {
		// Rollback if cycle detected
		delete(g.nodes, component)
		for _, dep := range dependencies {
			g.removeDependentRelation(dep, component)
		}
		return fmt.Errorf("adding component %s would create a dependency cycle", component)
	}

	return nil
}

// RemoveNode removes a component from the graph
func (g *DependencyGraph) RemoveNode(component string) {
	// Get dependencies of this component
	dependencies := g.nodes[component]

	// Remove from nodes
	delete(g.nodes, component)

	// Remove from dependents of its dependencies
	for _, dep := range dependencies {
		g.removeDependentRelation(dep, component)
	}

	// Remove dependents of this component
	delete(g.dependents, component)
}

// removeDependentRelation removes a dependent relationship
func (g *DependencyGraph) removeDependentRelation(dependency, dependent string) {
	deps := g.dependents[dependency]
	for i, dep := range deps {
		if dep == dependent {
			g.dependents[dependency] = append(deps[:i], deps[i+1:]...)
			break
		}
	}
}

// GetDependencies returns direct dependencies of a component
func (g *DependencyGraph) GetDependencies(component string) []string {
	deps := g.nodes[component]
	result := make([]string, len(deps))
	copy(result, deps)
	return result
}

// GetDependents returns components that depend on the given component
func (g *DependencyGraph) GetDependents(component string) []string {
	deps := g.dependents[component]
	result := make([]string, len(deps))
	copy(result, deps)
	return result
}

// GetStartOrder returns components in order they should be started
// Components with no dependencies come first
func (g *DependencyGraph) GetStartOrder() ([]string, error) {
	// Use topological sort
	return g.topologicalSort()
}

// GetStopOrder returns components in order they should be stopped
// Reverse of start order
func (g *DependencyGraph) GetStopOrder() ([]string, error) {
	startOrder, err := g.GetStartOrder()
	if err != nil {
		return nil, err
	}

	// Reverse the order
	stopOrder := make([]string, len(startOrder))
	for i, j := 0, len(startOrder)-1; i < len(startOrder); i, j = i+1, j-1 {
		stopOrder[i] = startOrder[j]
	}

	return stopOrder, nil
}

// topologicalSort performs topological sorting using Kahn's algorithm
func (g *DependencyGraph) topologicalSort() ([]string, error) {
	// Calculate in-degree for each node (number of nodes that depend on it)
	inDegree := make(map[string]int)
	for node := range g.nodes {
		inDegree[node] = len(g.nodes[node]) // Count dependencies (inward edges)
	}

	// Queue of nodes with no incoming edges
	queue := make([]string, 0)
	for node, degree := range inDegree {
		if degree == 0 {
			queue = append(queue, node)
		}
	}

	// Sort queue for deterministic order
	sort.Strings(queue)

	result := make([]string, 0, len(g.nodes))

	for len(queue) > 0 {
		// Pop from queue
		current := queue[0]
		queue = queue[1:]

		result = append(result, current)

		// Reduce in-degree for dependents
		for _, dependent := range g.dependents[current] {
			inDegree[dependent]--
			if inDegree[dependent] == 0 {
				queue = append(queue, dependent)
				sort.Strings(queue) // Keep sorted for deterministic order
			}
		}
	}

	// Check if all nodes were processed (no cycles)
	if len(result) != len(g.nodes) {
		return nil, fmt.Errorf("dependency cycle detected")
	}

	return result, nil
}

// hasCycle detects if there's a cycle in the dependency graph
func (g *DependencyGraph) hasCycle() bool {
	visited := make(map[string]bool)
	recStack := make(map[string]bool)

	for node := range g.nodes {
		if !visited[node] {
			if g.hasCycleUtil(node, visited, recStack) {
				return true
			}
		}
	}

	return false
}

// hasCycleUtil is a recursive helper for cycle detection
func (g *DependencyGraph) hasCycleUtil(node string, visited, recStack map[string]bool) bool {
	visited[node] = true
	recStack[node] = true

	for _, dep := range g.nodes[node] {
		if !visited[dep] {
			if g.hasCycleUtil(dep, visited, recStack) {
				return true
			}
		} else if recStack[dep] {
			return true
		}
	}

	recStack[node] = false
	return false
}

// GetDependencyLevels returns components grouped by dependency level
// Level 0 has no dependencies, Level 1 depends only on Level 0, etc.
func (g *DependencyGraph) GetDependencyLevels() [][]string {
	levels := make([][]string, 0)
	processed := make(map[string]bool)
	remaining := make(map[string]bool)

	for node := range g.nodes {
		remaining[node] = true
	}

	for len(remaining) > 0 {
		currentLevel := make([]string, 0)

		for node := range remaining {
			// Check if all dependencies are processed
			allDepsProcessed := true
			for _, dep := range g.nodes[node] {
				if !processed[dep] {
					allDepsProcessed = false
					break
				}
			}

			if allDepsProcessed {
				currentLevel = append(currentLevel, node)
			}
		}

		if len(currentLevel) == 0 {
			// No progress made - there's a cycle
			break
		}

		// Sort for deterministic order
		sort.Strings(currentLevel)
		levels = append(levels, currentLevel)

		// Mark as processed and remove from remaining
		for _, node := range currentLevel {
			processed[node] = true
			delete(remaining, node)
		}
	}

	return levels
}

// Validate validates the dependency graph
func (g *DependencyGraph) Validate() error {
	// Check for cycles
	if g.hasCycle() {
		return fmt.Errorf("dependency graph contains cycles")
	}

	// Check for missing dependencies
	for component, dependencies := range g.nodes {
		for _, dep := range dependencies {
			if _, exists := g.nodes[dep]; !exists {
				return fmt.Errorf("component %s depends on non-existent component %s",
					component, dep)
			}
		}
	}

	return nil
}

// Size returns number of components in the graph
func (g *DependencyGraph) Size() int {
	return len(g.nodes)
}

// Clear removes all nodes from the graph
func (g *DependencyGraph) Clear() {
	g.nodes = make(map[string][]string)
	g.dependents = make(map[string][]string)
}

// Clone creates a deep copy of the dependency graph
func (g *DependencyGraph) Clone() *DependencyGraph {
	clone := NewDependencyGraph()

	for node, deps := range g.nodes {
		depsCopy := make([]string, len(deps))
		copy(depsCopy, deps)
		clone.nodes[node] = depsCopy
	}

	for node, deps := range g.dependents {
		depsCopy := make([]string, len(deps))
		copy(depsCopy, deps)
		clone.dependents[node] = depsCopy
	}

	return clone
}
