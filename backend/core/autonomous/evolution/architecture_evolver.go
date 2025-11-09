package evolution

import (
	"context"
	"math"
	"math/rand"
	"sync"
	"time"

	"go.uber.org/zap"
)

// ArchitectureEvolver evolves system architecture using genetic algorithms
type ArchitectureEvolver struct {
	logger         *zap.Logger
	population     []*Architecture
	generation     int
	maxGenerations int
	populationSize int
	mutationRate   float64
	crossoverRate  float64
	eliteRatio     float64
	fitness        *FitnessEvaluator
	history        []*GenerationStats
	bestEver       *Architecture
	mu             sync.RWMutex
}

// Architecture represents a system architecture
type Architecture struct {
	ID          string
	Topology    *Topology
	Fitness     float64
	Performance float64
	Cost        float64
	Reliability float64
	Generation  int
	Parents     []string
	Mutations   []string
}

// Topology represents the system topology
type Topology struct {
	Nodes       []*Node
	Connections []*Connection
	Layers      int
	Redundancy  int
	LoadBalance string
}

// Node represents a system node
type Node struct {
	ID         string
	Type       NodeType
	Capacity   float64
	Cost       float64
	Reliability float64
	Location   string
	Resources  *Resources
}

// NodeType defines types of nodes
type NodeType string

const (
	ComputeNode  NodeType = "compute"
	StorageNode  NodeType = "storage"
	NetworkNode  NodeType = "network"
	DatabaseNode NodeType = "database"
	CacheNode    NodeType = "cache"
	LoadBalancer NodeType = "loadbalancer"
)

// Connection represents a connection between nodes
type Connection struct {
	From      string
	To        string
	Bandwidth float64
	Latency   float64
	Cost      float64
}

// Resources represents node resources
type Resources struct {
	CPU    float64
	Memory float64
	Disk   float64
	Network float64
}

// FitnessEvaluator evaluates architecture fitness
type FitnessEvaluator struct {
	logger           *zap.Logger
	performanceWeight float64
	costWeight       float64
	reliabilityWeight float64
	simulator        *ArchitectureSimulator
}

// ArchitectureSimulator simulates architecture performance
type ArchitectureSimulator struct {
	logger       *zap.Logger
	workload     *Workload
	metrics      *SimulationMetrics
}

// Workload represents system workload
type Workload struct {
	RequestRate float64
	DataSize    float64
	ReadRatio   float64
	WriteRatio  float64
	PeakFactor  float64
}

// SimulationMetrics contains simulation results
type SimulationMetrics struct {
	Throughput   float64
	Latency      float64
	Availability float64
	CostPerHour  float64
	ErrorRate    float64
}

// GenerationStats tracks generation statistics
type GenerationStats struct {
	Generation    int
	BestFitness   float64
	AvgFitness    float64
	WorstFitness  float64
	Diversity     float64
	Improvements  int
	Timestamp     time.Time
}

// MutationOperator defines mutation operations
type MutationOperator interface {
	Mutate(arch *Architecture) *Architecture
	GetType() string
}

// CrossoverOperator defines crossover operations
type CrossoverOperator interface {
	Crossover(parent1, parent2 *Architecture) *Architecture
	GetType() string
}

// NewArchitectureEvolver creates a new architecture evolver
func NewArchitectureEvolver(logger *zap.Logger) *ArchitectureEvolver {
	return &ArchitectureEvolver{
		logger:         logger,
		maxGenerations: 500,
		populationSize: 100,
		mutationRate:   0.1,
		crossoverRate:  0.7,
		eliteRatio:     0.1,
		fitness:        NewFitnessEvaluator(logger),
		history:        make([]*GenerationStats, 0),
	}
}

// NewFitnessEvaluator creates a new fitness evaluator
func NewFitnessEvaluator(logger *zap.Logger) *FitnessEvaluator {
	return &FitnessEvaluator{
		logger:            logger,
		performanceWeight: 0.4,
		costWeight:        0.3,
		reliabilityWeight: 0.3,
		simulator:         NewArchitectureSimulator(logger),
	}
}

// NewArchitectureSimulator creates a new architecture simulator
func NewArchitectureSimulator(logger *zap.Logger) *ArchitectureSimulator {
	return &ArchitectureSimulator{
		logger: logger,
		workload: &Workload{
			RequestRate: 10000,
			DataSize:    1024,
			ReadRatio:   0.8,
			WriteRatio:  0.2,
			PeakFactor:  2.0,
		},
		metrics: &SimulationMetrics{},
	}
}

// Evolve runs the genetic algorithm to evolve architecture
func (ae *ArchitectureEvolver) Evolve(ctx context.Context) (*Architecture, error) {
	ae.logger.Info("Starting architecture evolution",
		zap.Int("generations", ae.maxGenerations),
		zap.Int("population", ae.populationSize))

	// Initialize population
	ae.initializePopulation()

	// Evolution loop
	for ae.generation = 0; ae.generation < ae.maxGenerations; ae.generation++ {
		select {
		case <-ctx.Done():
			return ae.bestEver, ctx.Err()
		default:
		}

		// Evaluate fitness
		ae.evaluateFitness()

		// Record statistics
		stats := ae.recordGenerationStats()
		ae.history = append(ae.history, stats)

		// Check for convergence
		if ae.hasConverged() {
			ae.logger.Info("Evolution converged",
				zap.Int("generation", ae.generation),
				zap.Float64("best_fitness", ae.bestEver.Fitness))
			break
		}

		// Selection
		parents := ae.selection()

		// Crossover and mutation
		offspring := ae.reproduce(parents)

		// Replacement
		ae.population = ae.replacement(ae.population, offspring)

		// Log progress
		if ae.generation%10 == 0 {
			ae.logger.Info("Evolution progress",
				zap.Int("generation", ae.generation),
				zap.Float64("best_fitness", stats.BestFitness),
				zap.Float64("avg_fitness", stats.AvgFitness))
		}
	}

	ae.logger.Info("Evolution completed",
		zap.Int("final_generation", ae.generation),
		zap.Float64("best_fitness", ae.bestEver.Fitness),
		zap.Float64("improvement", ae.calculateImprovement()))

	return ae.bestEver, nil
}

// initializePopulation creates initial population
func (ae *ArchitectureEvolver) initializePopulation() {
	ae.population = make([]*Architecture, ae.populationSize)

	for i := 0; i < ae.populationSize; i++ {
		ae.population[i] = ae.createRandomArchitecture()
	}

	ae.logger.Info("Population initialized", zap.Int("size", ae.populationSize))
}

// createRandomArchitecture creates a random architecture
func (ae *ArchitectureEvolver) createRandomArchitecture() *Architecture {
	topology := &Topology{
		Nodes:       ae.createRandomNodes(),
		Connections: []*Connection{},
		Layers:      rand.Intn(5) + 2,
		Redundancy:  rand.Intn(3) + 1,
		LoadBalance: ae.randomLoadBalance(),
	}

	// Create connections between nodes
	topology.Connections = ae.createConnections(topology.Nodes)

	return &Architecture{
		ID:         generateArchID(),
		Topology:   topology,
		Generation: ae.generation,
	}
}

// createRandomNodes creates random nodes
func (ae *ArchitectureEvolver) createRandomNodes() []*Node {
	nodeCount := rand.Intn(20) + 10
	nodes := make([]*Node, nodeCount)

	for i := 0; i < nodeCount; i++ {
		nodes[i] = &Node{
			ID:          generateNodeID(),
			Type:        ae.randomNodeType(),
			Capacity:    rand.Float64() * 100,
			Cost:        rand.Float64() * 10,
			Reliability: 0.9 + rand.Float64()*0.099,
			Location:    ae.randomLocation(),
			Resources: &Resources{
				CPU:     rand.Float64() * 32,
				Memory:  rand.Float64() * 128,
				Disk:    rand.Float64() * 1000,
				Network: rand.Float64() * 10,
			},
		}
	}

	return nodes
}

// createConnections creates connections between nodes
func (ae *ArchitectureEvolver) createConnections(nodes []*Node) []*Connection {
	connections := make([]*Connection, 0)

	// Create mesh or hierarchical connections
	for i := 0; i < len(nodes); i++ {
		for j := i + 1; j < len(nodes); j++ {
			if rand.Float64() < 0.3 { // 30% chance of connection
				conn := &Connection{
					From:      nodes[i].ID,
					To:        nodes[j].ID,
					Bandwidth: rand.Float64() * 10,
					Latency:   rand.Float64() * 100,
					Cost:      rand.Float64() * 1,
				}
				connections = append(connections, conn)
			}
		}
	}

	return connections
}

// evaluateFitness evaluates fitness of all architectures
func (ae *ArchitectureEvolver) evaluateFitness() {
	var wg sync.WaitGroup
	fitnessChan := make(chan struct {
		index   int
		fitness float64
	}, ae.populationSize)

	for i, arch := range ae.population {
		wg.Add(1)
		go func(idx int, a *Architecture) {
			defer wg.Done()
			fitness := ae.fitness.Evaluate(a)
			fitnessChan <- struct {
				index   int
				fitness float64
			}{idx, fitness}
		}(i, arch)
	}

	go func() {
		wg.Wait()
		close(fitnessChan)
	}()

	// Update fitness values
	for result := range fitnessChan {
		ae.population[result.index].Fitness = result.fitness

		// Update best ever
		ae.mu.Lock()
		if ae.bestEver == nil || result.fitness > ae.bestEver.Fitness {
			ae.bestEver = ae.population[result.index]
		}
		ae.mu.Unlock()
	}
}

// Evaluate calculates architecture fitness
func (fe *FitnessEvaluator) Evaluate(arch *Architecture) float64 {
	// Simulate architecture
	metrics := fe.simulator.Simulate(arch)

	// Calculate performance score (higher is better)
	performanceScore := metrics.Throughput / 10000 * (1 - metrics.Latency/1000)

	// Calculate cost score (lower is better, so invert)
	costScore := 1.0 / (1.0 + metrics.CostPerHour)

	// Calculate reliability score
	reliabilityScore := metrics.Availability * (1 - metrics.ErrorRate)

	// Weighted fitness
	fitness := fe.performanceWeight*performanceScore +
		fe.costWeight*costScore +
		fe.reliabilityWeight*reliabilityScore

	// Store individual scores
	arch.Performance = performanceScore
	arch.Cost = metrics.CostPerHour
	arch.Reliability = reliabilityScore

	return fitness
}

// Simulate simulates architecture performance
func (as *ArchitectureSimulator) Simulate(arch *Architecture) *SimulationMetrics {
	metrics := &SimulationMetrics{}

	// Calculate throughput based on node capacity
	totalCapacity := 0.0
	for _, node := range arch.Topology.Nodes {
		if node.Type == ComputeNode {
			totalCapacity += node.Capacity
		}
	}
	metrics.Throughput = totalCapacity * 100 * as.workload.RequestRate / 10000

	// Calculate latency based on connections
	avgLatency := 0.0
	if len(arch.Topology.Connections) > 0 {
		for _, conn := range arch.Topology.Connections {
			avgLatency += conn.Latency
		}
		avgLatency /= float64(len(arch.Topology.Connections))
	}
	metrics.Latency = avgLatency + 10 // Base latency

	// Calculate availability based on redundancy and reliability
	minReliability := 1.0
	for _, node := range arch.Topology.Nodes {
		if node.Reliability < minReliability {
			minReliability = node.Reliability
		}
	}
	metrics.Availability = minReliability * (1 + float64(arch.Topology.Redundancy)*0.05)
	if metrics.Availability > 0.9999 {
		metrics.Availability = 0.9999
	}

	// Calculate cost
	totalCost := 0.0
	for _, node := range arch.Topology.Nodes {
		totalCost += node.Cost
	}
	for _, conn := range arch.Topology.Connections {
		totalCost += conn.Cost
	}
	metrics.CostPerHour = totalCost

	// Calculate error rate
	metrics.ErrorRate = (1 - metrics.Availability) * 0.01

	return metrics
}

// selection performs selection of parents
func (ae *ArchitectureEvolver) selection() []*Architecture {
	// Tournament selection
	tournamentSize := 5
	parents := make([]*Architecture, ae.populationSize)

	for i := 0; i < ae.populationSize; i++ {
		parents[i] = ae.tournamentSelect(tournamentSize)
	}

	return parents
}

// tournamentSelect performs tournament selection
func (ae *ArchitectureEvolver) tournamentSelect(size int) *Architecture {
	tournament := make([]*Architecture, size)

	for i := 0; i < size; i++ {
		tournament[i] = ae.population[rand.Intn(ae.populationSize)]
	}

	// Find best in tournament
	best := tournament[0]
	for _, arch := range tournament[1:] {
		if arch.Fitness > best.Fitness {
			best = arch
		}
	}

	return best
}

// reproduce performs crossover and mutation
func (ae *ArchitectureEvolver) reproduce(parents []*Architecture) []*Architecture {
	offspring := make([]*Architecture, 0)

	for i := 0; i < len(parents)-1; i += 2 {
		if rand.Float64() < ae.crossoverRate {
			// Crossover
			child1, child2 := ae.crossover(parents[i], parents[i+1])
			offspring = append(offspring, child1, child2)
		} else {
			// Clone parents
			offspring = append(offspring, ae.clone(parents[i]), ae.clone(parents[i+1]))
		}
	}

	// Apply mutation
	for _, child := range offspring {
		if rand.Float64() < ae.mutationRate {
			ae.mutate(child)
		}
	}

	return offspring
}

// crossover performs crossover between two architectures
func (ae *ArchitectureEvolver) crossover(parent1, parent2 *Architecture) (*Architecture, *Architecture) {
	child1 := ae.clone(parent1)
	child2 := ae.clone(parent2)

	// Swap random nodes
	if len(parent1.Topology.Nodes) > 0 && len(parent2.Topology.Nodes) > 0 {
		point := rand.Intn(min(len(parent1.Topology.Nodes), len(parent2.Topology.Nodes)))

		// Swap nodes after crossover point
		child1.Topology.Nodes = append(parent1.Topology.Nodes[:point], parent2.Topology.Nodes[point:]...)
		child2.Topology.Nodes = append(parent2.Topology.Nodes[:point], parent1.Topology.Nodes[point:]...)
	}

	child1.Generation = ae.generation
	child2.Generation = ae.generation
	child1.Parents = []string{parent1.ID, parent2.ID}
	child2.Parents = []string{parent2.ID, parent1.ID}

	return child1, child2
}

// mutate applies mutation to architecture
func (ae *ArchitectureEvolver) mutate(arch *Architecture) {
	mutationType := rand.Intn(4)

	switch mutationType {
	case 0:
		// Add node
		newNode := ae.createRandomNode()
		arch.Topology.Nodes = append(arch.Topology.Nodes, newNode)
		arch.Mutations = append(arch.Mutations, "add_node")

	case 1:
		// Remove node
		if len(arch.Topology.Nodes) > 5 {
			idx := rand.Intn(len(arch.Topology.Nodes))
			arch.Topology.Nodes = append(arch.Topology.Nodes[:idx], arch.Topology.Nodes[idx+1:]...)
			arch.Mutations = append(arch.Mutations, "remove_node")
		}

	case 2:
		// Modify node
		if len(arch.Topology.Nodes) > 0 {
			idx := rand.Intn(len(arch.Topology.Nodes))
			arch.Topology.Nodes[idx].Capacity *= (0.5 + rand.Float64())
			arch.Topology.Nodes[idx].Cost *= (0.8 + rand.Float64()*0.4)
			arch.Mutations = append(arch.Mutations, "modify_node")
		}

	case 3:
		// Change connections
		arch.Topology.Connections = ae.createConnections(arch.Topology.Nodes)
		arch.Mutations = append(arch.Mutations, "change_connections")
	}
}

// replacement performs generational replacement with elitism
func (ae *ArchitectureEvolver) replacement(population, offspring []*Architecture) []*Architecture {
	// Sort population by fitness
	ae.sortByFitness(population)

	// Keep elite
	eliteSize := int(float64(ae.populationSize) * ae.eliteRatio)
	newPopulation := make([]*Architecture, 0, ae.populationSize)
	newPopulation = append(newPopulation, population[:eliteSize]...)

	// Fill rest with offspring
	ae.sortByFitness(offspring)
	remaining := ae.populationSize - eliteSize
	if len(offspring) >= remaining {
		newPopulation = append(newPopulation, offspring[:remaining]...)
	} else {
		newPopulation = append(newPopulation, offspring...)
		// Fill remaining with random
		for len(newPopulation) < ae.populationSize {
			newPopulation = append(newPopulation, ae.createRandomArchitecture())
		}
	}

	return newPopulation
}

// hasConverged checks for convergence
func (ae *ArchitectureEvolver) hasConverged() bool {
	if len(ae.history) < 10 {
		return false
	}

	// Check if fitness hasn't improved in last 10 generations
	recent := ae.history[len(ae.history)-10:]
	maxFitness := recent[0].BestFitness
	minFitness := recent[0].BestFitness

	for _, stats := range recent {
		if stats.BestFitness > maxFitness {
			maxFitness = stats.BestFitness
		}
		if stats.BestFitness < minFitness {
			minFitness = stats.BestFitness
		}
	}

	// Converged if improvement is less than 1%
	return (maxFitness-minFitness)/minFitness < 0.01
}

// recordGenerationStats records statistics for current generation
func (ae *ArchitectureEvolver) recordGenerationStats() *GenerationStats {
	stats := &GenerationStats{
		Generation:  ae.generation,
		Timestamp:   time.Now(),
	}

	// Calculate fitness statistics
	totalFitness := 0.0
	stats.BestFitness = 0
	stats.WorstFitness = math.MaxFloat64

	for _, arch := range ae.population {
		totalFitness += arch.Fitness
		if arch.Fitness > stats.BestFitness {
			stats.BestFitness = arch.Fitness
		}
		if arch.Fitness < stats.WorstFitness {
			stats.WorstFitness = arch.Fitness
		}
	}

	stats.AvgFitness = totalFitness / float64(ae.populationSize)

	// Calculate diversity
	stats.Diversity = ae.calculateDiversity()

	return stats
}

// calculateDiversity calculates population diversity
func (ae *ArchitectureEvolver) calculateDiversity() float64 {
	// Simple diversity metric based on fitness variance
	mean := 0.0
	for _, arch := range ae.population {
		mean += arch.Fitness
	}
	mean /= float64(len(ae.population))

	variance := 0.0
	for _, arch := range ae.population {
		diff := arch.Fitness - mean
		variance += diff * diff
	}
	variance /= float64(len(ae.population))

	return math.Sqrt(variance)
}

// calculateImprovement calculates total improvement
func (ae *ArchitectureEvolver) calculateImprovement() float64 {
	if len(ae.history) == 0 {
		return 0
	}

	initial := ae.history[0].BestFitness
	final := ae.bestEver.Fitness

	if initial == 0 {
		return 0
	}

	return ((final - initial) / initial) * 100
}

// GetBestArchitecture returns the best architecture found
func (ae *ArchitectureEvolver) GetBestArchitecture() *Architecture {
	ae.mu.RLock()
	defer ae.mu.RUnlock()
	return ae.bestEver
}

// GetHistory returns evolution history
func (ae *ArchitectureEvolver) GetHistory() []*GenerationStats {
	ae.mu.RLock()
	defer ae.mu.RUnlock()
	return ae.history
}

// Helper functions

func (ae *ArchitectureEvolver) randomNodeType() NodeType {
	types := []NodeType{ComputeNode, StorageNode, NetworkNode, DatabaseNode, CacheNode, LoadBalancer}
	return types[rand.Intn(len(types))]
}

func (ae *ArchitectureEvolver) randomLocation() string {
	locations := []string{"us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"}
	return locations[rand.Intn(len(locations))]
}

func (ae *ArchitectureEvolver) randomLoadBalance() string {
	strategies := []string{"round-robin", "least-connections", "ip-hash", "weighted"}
	return strategies[rand.Intn(len(strategies))]
}

func (ae *ArchitectureEvolver) createRandomNode() *Node {
	return &Node{
		ID:          generateNodeID(),
		Type:        ae.randomNodeType(),
		Capacity:    rand.Float64() * 100,
		Cost:        rand.Float64() * 10,
		Reliability: 0.9 + rand.Float64()*0.099,
		Location:    ae.randomLocation(),
		Resources: &Resources{
			CPU:     rand.Float64() * 32,
			Memory:  rand.Float64() * 128,
			Disk:    rand.Float64() * 1000,
			Network: rand.Float64() * 10,
		},
	}
}

func (ae *ArchitectureEvolver) clone(arch *Architecture) *Architecture {
	// Deep clone architecture
	clone := &Architecture{
		ID:          generateArchID(),
		Fitness:     arch.Fitness,
		Performance: arch.Performance,
		Cost:        arch.Cost,
		Reliability: arch.Reliability,
		Generation:  arch.Generation,
		Parents:     make([]string, len(arch.Parents)),
		Mutations:   make([]string, len(arch.Mutations)),
	}

	copy(clone.Parents, arch.Parents)
	copy(clone.Mutations, arch.Mutations)

	// Clone topology
	clone.Topology = &Topology{
		Nodes:       make([]*Node, len(arch.Topology.Nodes)),
		Connections: make([]*Connection, len(arch.Topology.Connections)),
		Layers:      arch.Topology.Layers,
		Redundancy:  arch.Topology.Redundancy,
		LoadBalance: arch.Topology.LoadBalance,
	}

	for i, node := range arch.Topology.Nodes {
		clone.Topology.Nodes[i] = &Node{
			ID:          node.ID,
			Type:        node.Type,
			Capacity:    node.Capacity,
			Cost:        node.Cost,
			Reliability: node.Reliability,
			Location:    node.Location,
			Resources: &Resources{
				CPU:     node.Resources.CPU,
				Memory:  node.Resources.Memory,
				Disk:    node.Resources.Disk,
				Network: node.Resources.Network,
			},
		}
	}

	for i, conn := range arch.Topology.Connections {
		clone.Topology.Connections[i] = &Connection{
			From:      conn.From,
			To:        conn.To,
			Bandwidth: conn.Bandwidth,
			Latency:   conn.Latency,
			Cost:      conn.Cost,
		}
	}

	return clone
}

func (ae *ArchitectureEvolver) sortByFitness(population []*Architecture) {
	// Simple bubble sort (replace with better sorting for production)
	for i := 0; i < len(population); i++ {
		for j := i + 1; j < len(population); j++ {
			if population[i].Fitness < population[j].Fitness {
				population[i], population[j] = population[j], population[i]
			}
		}
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func generateArchID() string {
	return "arch-" + generateID()
}

func generateNodeID() string {
	return "node-" + generateID()
}