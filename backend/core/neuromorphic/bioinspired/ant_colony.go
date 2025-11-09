package bioinspired

import (
	"context"
	"math"
	"math/rand"
	"sync"
)

// AntColonyOptimizer implements Ant Colony Optimization for routing
type AntColonyOptimizer struct {
	mu              sync.RWMutex
	numAnts         int
	numNodes        int
	alpha           float64 // pheromone importance
	beta            float64 // heuristic importance
	evaporationRate float64
	pheromone       [][]float64
	distance        [][]float64
	bestPath        []int
	bestDistance    float64
}

// NewAntColonyOptimizer creates a new ACO instance
func NewAntColonyOptimizer(numAnts, numNodes int) *AntColonyOptimizer {
	aco := &AntColonyOptimizer{
		numAnts:         numAnts,
		numNodes:        numNodes,
		alpha:           1.0,
		beta:            2.0,
		evaporationRate: 0.5,
		pheromone:       make([][]float64, numNodes),
		distance:        make([][]float64, numNodes),
		bestDistance:    math.MaxFloat64,
	}

	// Initialize pheromone matrix
	for i := 0; i < numNodes; i++ {
		aco.pheromone[i] = make([]float64, numNodes)
		aco.distance[i] = make([]float64, numNodes)
		for j := 0; j < numNodes; j++ {
			aco.pheromone[i][j] = 1.0
		}
	}

	return aco
}

// SetDistanceMatrix sets the distance matrix
func (aco *AntColonyOptimizer) SetDistanceMatrix(distances [][]float64) {
	aco.mu.Lock()
	defer aco.mu.Unlock()
	aco.distance = distances
}

// Optimize runs the ACO algorithm
func (aco *AntColonyOptimizer) Optimize(ctx context.Context, maxIterations int) ([]int, float64, error) {
	for iter := 0; iter < maxIterations; iter++ {
		// Create ants and find paths
		paths := make([][]int, aco.numAnts)
		distances := make([]float64, aco.numAnts)

		for ant := 0; ant < aco.numAnts; ant++ {
			path, dist := aco.constructPath()
			paths[ant] = path
			distances[ant] = dist

			// Update best solution
			aco.mu.Lock()
			if dist < aco.bestDistance {
				aco.bestDistance = dist
				aco.bestPath = make([]int, len(path))
				copy(aco.bestPath, path)
			}
			aco.mu.Unlock()
		}

		// Update pheromones
		aco.updatePheromones(paths, distances)

		// Check context
		select {
		case <-ctx.Done():
			return aco.bestPath, aco.bestDistance, ctx.Err()
		default:
		}
	}

	return aco.bestPath, aco.bestDistance, nil
}

// constructPath builds a path using probabilistic rules
func (aco *AntColonyOptimizer) constructPath() ([]int, float64) {
	aco.mu.RLock()
	defer aco.mu.RUnlock()

	visited := make(map[int]bool)
	path := make([]int, 0, aco.numNodes)
	current := rand.Intn(aco.numNodes)
	path = append(path, current)
	visited[current] = true
	totalDistance := 0.0

	for len(visited) < aco.numNodes {
		next := aco.selectNext(current, visited)
		totalDistance += aco.distance[current][next]
		path = append(path, next)
		visited[next] = true
		current = next
	}

	// Return to start
	totalDistance += aco.distance[current][path[0]]

	return path, totalDistance
}

// selectNext selects next node probabilistically
func (aco *AntColonyOptimizer) selectNext(current int, visited map[int]bool) int {
	probabilities := make([]float64, aco.numNodes)
	sum := 0.0

	for i := 0; i < aco.numNodes; i++ {
		if visited[i] || i == current {
			probabilities[i] = 0
			continue
		}

		pheromone := math.Pow(aco.pheromone[current][i], aco.alpha)
		heuristic := math.Pow(1.0/aco.distance[current][i], aco.beta)
		probabilities[i] = pheromone * heuristic
		sum += probabilities[i]
	}

	// Normalize
	for i := range probabilities {
		probabilities[i] /= sum
	}

	// Roulette wheel selection
	r := rand.Float64()
	cumulative := 0.0
	for i, prob := range probabilities {
		cumulative += prob
		if r <= cumulative {
			return i
		}
	}

	// Fallback: return first unvisited
	for i := 0; i < aco.numNodes; i++ {
		if !visited[i] && i != current {
			return i
		}
	}

	return 0
}

// updatePheromones updates pheromone levels
func (aco *AntColonyOptimizer) updatePheromones(paths [][]int, distances []float64) {
	aco.mu.Lock()
	defer aco.mu.Unlock()

	// Evaporation
	for i := 0; i < aco.numNodes; i++ {
		for j := 0; j < aco.numNodes; j++ {
			aco.pheromone[i][j] *= (1.0 - aco.evaporationRate)
		}
	}

	// Deposit new pheromones
	for ant := 0; ant < aco.numAnts; ant++ {
		path := paths[ant]
		deposit := 1.0 / distances[ant]

		for i := 0; i < len(path)-1; i++ {
			from := path[i]
			to := path[i+1]
			aco.pheromone[from][to] += deposit
			aco.pheromone[to][from] += deposit
		}

		// Close the loop
		from := path[len(path)-1]
		to := path[0]
		aco.pheromone[from][to] += deposit
		aco.pheromone[to][from] += deposit
	}
}

// GetBestPath returns the best path found
func (aco *AntColonyOptimizer) GetBestPath() ([]int, float64) {
	aco.mu.RLock()
	defer aco.mu.RUnlock()
	return aco.bestPath, aco.bestDistance
}
