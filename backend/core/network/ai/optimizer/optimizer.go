package optimizer

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// Objective represents an optimization objective
type Objective struct {
	Name      string
	Type      string // "minimize" or "maximize"
	Weight    float64
	Function  ObjectiveFunc
}

// ObjectiveFunc evaluates an objective
type ObjectiveFunc func(solution Solution) float64

// Solution represents a network configuration
type Solution struct {
	ID           string
	Configuration map[string]interface{}
	Objectives   map[string]float64
	Constraints  map[string]bool
	Fitness      float64
}

// NetworkOptimizer performs multi-objective optimization
type NetworkOptimizer struct {
	mu sync.RWMutex

	// Optimization algorithms
	genetic      *GeneticAlgorithm
	pso          *ParticleSwarmOptimizer
	simAnnealing *SimulatedAnnealing

	// Objectives and constraints
	objectives  []Objective
	constraints []Constraint

	// Pareto analysis
	paretoFront []Solution
	archive     []Solution

	// Performance metrics
	iterations    int64
	bestFitness   float64
	convergenceRate float64
}

// GeneticAlgorithm implements GA optimization
type GeneticAlgorithm struct {
	populationSize int
	generations    int
	crossoverRate  float64
	mutationRate   float64
	eliteSize      int

	population     []Individual
	bestIndividual Individual
}

// Individual in genetic algorithm
type Individual struct {
	Genes   []float64
	Fitness float64
}

// ParticleSwarmOptimizer implements PSO
type ParticleSwarmOptimizer struct {
	swarmSize    int
	iterations   int
	inertia      float64
	cognitive    float64
	social       float64

	particles    []Particle
	globalBest   Position
}

// Particle in PSO
type Particle struct {
	Position     Position
	Velocity     []float64
	PersonalBest Position
}

// Position in search space
type Position struct {
	Values  []float64
	Fitness float64
}

// SimulatedAnnealing implements SA optimization
type SimulatedAnnealing struct {
	temperature     float64
	coolingRate     float64
	minTemperature  float64
	iterations      int

	currentSolution Solution
	bestSolution    Solution
}

// Constraint for optimization
type Constraint struct {
	Name     string
	Function ConstraintFunc
}

// ConstraintFunc evaluates a constraint
type ConstraintFunc func(solution Solution) bool

// NewNetworkOptimizer creates a new optimizer
func NewNetworkOptimizer() *NetworkOptimizer {
	return &NetworkOptimizer{
		archive: make([]Solution, 0, 1000),
	}
}

// Initialize initializes the optimizer
func (o *NetworkOptimizer) Initialize(ctx context.Context) error {
	o.mu.Lock()
	defer o.mu.Unlock()

	// Initialize genetic algorithm
	o.genetic = &GeneticAlgorithm{
		populationSize: 100,
		generations:    50,
		crossoverRate:  0.8,
		mutationRate:   0.1,
		eliteSize:      10,
	}

	// Initialize PSO
	o.pso = &ParticleSwarmOptimizer{
		swarmSize:  50,
		iterations: 100,
		inertia:    0.7,
		cognitive:  1.5,
		social:     1.5,
	}

	// Initialize simulated annealing
	o.simAnnealing = &SimulatedAnnealing{
		temperature:    1000,
		coolingRate:    0.95,
		minTemperature: 1,
		iterations:     1000,
	}

	// Define default objectives
	o.defineObjectives()

	// Define default constraints
	o.defineConstraints()

	return nil
}

// Optimize performs multi-objective optimization
func (o *NetworkOptimizer) Optimize(ctx context.Context) ([]Solution, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	// Run genetic algorithm
	gaResults := o.genetic.optimize(o.objectives, o.constraints)

	// Run PSO
	psoResults := o.pso.optimize(o.objectives, o.constraints)

	// Run simulated annealing
	saResults := o.simAnnealing.optimize(o.objectives, o.constraints)

	// Combine results
	allResults := append(gaResults, psoResults...)
	allResults = append(allResults, saResults...)

	// Find Pareto front
	o.paretoFront = o.findParetoFront(allResults)

	// Update archive
	o.updateArchive(o.paretoFront)

	// Update metrics
	o.iterations++
	if len(o.paretoFront) > 0 {
		o.bestFitness = o.paretoFront[0].Fitness
	}

	return o.paretoFront, nil
}

// defineObjectives defines optimization objectives
func (o *NetworkOptimizer) defineObjectives() {
	// Minimize latency
	o.objectives = append(o.objectives, Objective{
		Name:   "latency",
		Type:   "minimize",
		Weight: 0.3,
		Function: func(s Solution) float64 {
			if latency, ok := s.Configuration["avg_latency"].(float64); ok {
				return latency
			}
			return math.Inf(1)
		},
	})

	// Maximize throughput
	o.objectives = append(o.objectives, Objective{
		Name:   "throughput",
		Type:   "maximize",
		Weight: 0.3,
		Function: func(s Solution) float64 {
			if throughput, ok := s.Configuration["throughput"].(float64); ok {
				return throughput
			}
			return 0
		},
	})

	// Minimize cost
	o.objectives = append(o.objectives, Objective{
		Name:   "cost",
		Type:   "minimize",
		Weight: 0.2,
		Function: func(s Solution) float64 {
			if cost, ok := s.Configuration["cost"].(float64); ok {
				return cost
			}
			return math.Inf(1)
		},
	})

	// Maximize reliability
	o.objectives = append(o.objectives, Objective{
		Name:   "reliability",
		Type:   "maximize",
		Weight: 0.2,
		Function: func(s Solution) float64 {
			if reliability, ok := s.Configuration["reliability"].(float64); ok {
				return reliability
			}
			return 0
		},
	})
}

// defineConstraints defines optimization constraints
func (o *NetworkOptimizer) defineConstraints() {
	// SLA constraint
	o.constraints = append(o.constraints, Constraint{
		Name: "sla_latency",
		Function: func(s Solution) bool {
			if latency, ok := s.Configuration["avg_latency"].(float64); ok {
				return latency <= 100 // Max 100ms
			}
			return false
		},
	})

	// Bandwidth constraint
	o.constraints = append(o.constraints, Constraint{
		Name: "bandwidth",
		Function: func(s Solution) bool {
			if util, ok := s.Configuration["link_utilization"].(float64); ok {
				return util <= 0.95 // Max 95% utilization
			}
			return false
		},
	})

	// Budget constraint
	o.constraints = append(o.constraints, Constraint{
		Name: "budget",
		Function: func(s Solution) bool {
			if cost, ok := s.Configuration["cost"].(float64); ok {
				return cost <= 10000 // Max budget
			}
			return false
		},
	})
}

// GeneticAlgorithm methods
func (ga *GeneticAlgorithm) optimize(objectives []Objective, constraints []Constraint) []Solution {
	// Initialize population
	ga.initializePopulation()

	// Evolution loop
	for gen := 0; gen < ga.generations; gen++ {
		// Evaluate fitness
		ga.evaluateFitness(objectives, constraints)

		// Selection
		parents := ga.selection()

		// Crossover
		offspring := ga.crossover(parents)

		// Mutation
		ga.mutation(offspring)

		// Replacement
		ga.replacement(offspring)
	}

	// Convert to solutions
	return ga.toSolutions()
}

func (ga *GeneticAlgorithm) initializePopulation() {
	ga.population = make([]Individual, ga.populationSize)

	for i := range ga.population {
		ga.population[i] = Individual{
			Genes: ga.randomGenes(),
		}
	}
}

func (ga *GeneticAlgorithm) randomGenes() []float64 {
	genes := make([]float64, 10) // 10 parameters
	for i := range genes {
		genes[i] = rand.Float64()
	}
	return genes
}

func (ga *GeneticAlgorithm) evaluateFitness(objectives []Objective, constraints []Constraint) {
	for i := range ga.population {
		solution := ga.genesToSolution(ga.population[i].Genes)

		// Evaluate objectives
		fitness := 0.0
		for _, obj := range objectives {
			value := obj.Function(solution)
			if obj.Type == "minimize" {
				fitness -= value * obj.Weight
			} else {
				fitness += value * obj.Weight
			}
		}

		// Apply constraint penalties
		for _, constraint := range constraints {
			if !constraint.Function(solution) {
				fitness -= 1000 // Penalty
			}
		}

		ga.population[i].Fitness = fitness
	}

	// Track best individual
	sort.Slice(ga.population, func(i, j int) bool {
		return ga.population[i].Fitness > ga.population[j].Fitness
	})

	if ga.population[0].Fitness > ga.bestIndividual.Fitness {
		ga.bestIndividual = ga.population[0]
	}
}

func (ga *GeneticAlgorithm) genesToSolution(genes []float64) Solution {
	return Solution{
		ID: fmt.Sprintf("ga-%d", time.Now().UnixNano()),
		Configuration: map[string]interface{}{
			"avg_latency":       genes[0] * 200,    // 0-200ms
			"throughput":        genes[1] * 10000,  // 0-10Gbps
			"cost":              genes[2] * 20000,  // 0-20k
			"reliability":       genes[3],          // 0-1
			"link_utilization":  genes[4],          // 0-1
		},
		Objectives: make(map[string]float64),
	}
}

func (ga *GeneticAlgorithm) selection() []Individual {
	// Tournament selection
	parents := make([]Individual, ga.populationSize/2)

	for i := range parents {
		tournament := make([]Individual, 3)
		for j := range tournament {
			tournament[j] = ga.population[rand.Intn(ga.populationSize)]
		}

		// Select best from tournament
		sort.Slice(tournament, func(i, j int) bool {
			return tournament[i].Fitness > tournament[j].Fitness
		})
		parents[i] = tournament[0]
	}

	return parents
}

func (ga *GeneticAlgorithm) crossover(parents []Individual) []Individual {
	offspring := make([]Individual, len(parents))

	for i := 0; i < len(parents)-1; i += 2 {
		if rand.Float64() < ga.crossoverRate {
			// Single-point crossover
			point := rand.Intn(len(parents[i].Genes))

			child1 := Individual{Genes: make([]float64, len(parents[i].Genes))}
			child2 := Individual{Genes: make([]float64, len(parents[i].Genes))}

			copy(child1.Genes[:point], parents[i].Genes[:point])
			copy(child1.Genes[point:], parents[i+1].Genes[point:])

			copy(child2.Genes[:point], parents[i+1].Genes[:point])
			copy(child2.Genes[point:], parents[i].Genes[point:])

			offspring[i] = child1
			offspring[i+1] = child2
		} else {
			offspring[i] = parents[i]
			offspring[i+1] = parents[i+1]
		}
	}

	return offspring
}

func (ga *GeneticAlgorithm) mutation(offspring []Individual) {
	for i := range offspring {
		for j := range offspring[i].Genes {
			if rand.Float64() < ga.mutationRate {
				// Gaussian mutation
				offspring[i].Genes[j] += rand.NormFloat64() * 0.1
				offspring[i].Genes[j] = math.Max(0, math.Min(1, offspring[i].Genes[j]))
			}
		}
	}
}

func (ga *GeneticAlgorithm) replacement(offspring []Individual) {
	// Elitism + offspring
	newPopulation := make([]Individual, ga.populationSize)

	// Keep elite
	copy(newPopulation[:ga.eliteSize], ga.population[:ga.eliteSize])

	// Add offspring
	copy(newPopulation[ga.eliteSize:], offspring[:ga.populationSize-ga.eliteSize])

	ga.population = newPopulation
}

func (ga *GeneticAlgorithm) toSolutions() []Solution {
	solutions := make([]Solution, len(ga.population))
	for i, ind := range ga.population {
		solutions[i] = ga.genesToSolution(ind.Genes)
		solutions[i].Fitness = ind.Fitness
	}
	return solutions
}

// ParticleSwarmOptimizer methods
func (pso *ParticleSwarmOptimizer) optimize(objectives []Objective, constraints []Constraint) []Solution {
	// Initialize swarm
	pso.initializeSwarm()

	// Optimization loop
	for iter := 0; iter < pso.iterations; iter++ {
		// Update particles
		pso.updateParticles(objectives, constraints)
	}

	// Convert to solutions
	return pso.toSolutions()
}

func (pso *ParticleSwarmOptimizer) initializeSwarm() {
	pso.particles = make([]Particle, pso.swarmSize)

	for i := range pso.particles {
		position := Position{
			Values: pso.randomPosition(),
		}

		pso.particles[i] = Particle{
			Position:     position,
			Velocity:     pso.randomVelocity(),
			PersonalBest: position,
		}
	}
}

func (pso *ParticleSwarmOptimizer) randomPosition() []float64 {
	pos := make([]float64, 10)
	for i := range pos {
		pos[i] = rand.Float64()
	}
	return pos
}

func (pso *ParticleSwarmOptimizer) randomVelocity() []float64 {
	vel := make([]float64, 10)
	for i := range vel {
		vel[i] = (rand.Float64() - 0.5) * 0.1
	}
	return vel
}

func (pso *ParticleSwarmOptimizer) updateParticles(objectives []Objective, constraints []Constraint) {
	// Evaluate fitness
	for i := range pso.particles {
		solution := pso.positionToSolution(pso.particles[i].Position)
		fitness := pso.evaluateFitness(solution, objectives, constraints)
		pso.particles[i].Position.Fitness = fitness

		// Update personal best
		if fitness > pso.particles[i].PersonalBest.Fitness {
			pso.particles[i].PersonalBest = pso.particles[i].Position
		}

		// Update global best
		if fitness > pso.globalBest.Fitness {
			pso.globalBest = pso.particles[i].Position
		}
	}

	// Update velocity and position
	for i := range pso.particles {
		for j := range pso.particles[i].Velocity {
			// Update velocity
			cognitive := pso.cognitive * rand.Float64() * (pso.particles[i].PersonalBest.Values[j] - pso.particles[i].Position.Values[j])
			social := pso.social * rand.Float64() * (pso.globalBest.Values[j] - pso.particles[i].Position.Values[j])

			pso.particles[i].Velocity[j] = pso.inertia*pso.particles[i].Velocity[j] + cognitive + social

			// Update position
			pso.particles[i].Position.Values[j] += pso.particles[i].Velocity[j]
			pso.particles[i].Position.Values[j] = math.Max(0, math.Min(1, pso.particles[i].Position.Values[j]))
		}
	}
}

func (pso *ParticleSwarmOptimizer) positionToSolution(pos Position) Solution {
	return Solution{
		ID: fmt.Sprintf("pso-%d", time.Now().UnixNano()),
		Configuration: map[string]interface{}{
			"avg_latency":      pos.Values[0] * 200,
			"throughput":       pos.Values[1] * 10000,
			"cost":             pos.Values[2] * 20000,
			"reliability":      pos.Values[3],
			"link_utilization": pos.Values[4],
		},
	}
}

func (pso *ParticleSwarmOptimizer) evaluateFitness(solution Solution, objectives []Objective, constraints []Constraint) float64 {
	fitness := 0.0

	for _, obj := range objectives {
		value := obj.Function(solution)
		if obj.Type == "minimize" {
			fitness -= value * obj.Weight
		} else {
			fitness += value * obj.Weight
		}
	}

	for _, constraint := range constraints {
		if !constraint.Function(solution) {
			fitness -= 1000
		}
	}

	return fitness
}

func (pso *ParticleSwarmOptimizer) toSolutions() []Solution {
	solutions := make([]Solution, pso.swarmSize)
	for i, particle := range pso.particles {
		solutions[i] = pso.positionToSolution(particle.Position)
		solutions[i].Fitness = particle.Position.Fitness
	}
	return solutions
}

// SimulatedAnnealing methods
func (sa *SimulatedAnnealing) optimize(objectives []Objective, constraints []Constraint) []Solution {
	// Initialize
	sa.currentSolution = sa.randomSolution()
	sa.bestSolution = sa.currentSolution

	// Annealing loop
	for iter := 0; iter < sa.iterations && sa.temperature > sa.minTemperature; iter++ {
		// Generate neighbor
		neighbor := sa.generateNeighbor(sa.currentSolution)

		// Evaluate fitness
		currentFitness := sa.evaluateFitness(sa.currentSolution, objectives, constraints)
		neighborFitness := sa.evaluateFitness(neighbor, objectives, constraints)

		// Accept or reject
		delta := neighborFitness - currentFitness
		if delta > 0 || rand.Float64() < math.Exp(delta/sa.temperature) {
			sa.currentSolution = neighbor

			if neighborFitness > sa.evaluateFitness(sa.bestSolution, objectives, constraints) {
				sa.bestSolution = neighbor
			}
		}

		// Cool down
		sa.temperature *= sa.coolingRate
	}

	return []Solution{sa.bestSolution}
}

func (sa *SimulatedAnnealing) randomSolution() Solution {
	return Solution{
		ID: fmt.Sprintf("sa-%d", time.Now().UnixNano()),
		Configuration: map[string]interface{}{
			"avg_latency":      rand.Float64() * 200,
			"throughput":       rand.Float64() * 10000,
			"cost":             rand.Float64() * 20000,
			"reliability":      rand.Float64(),
			"link_utilization": rand.Float64(),
		},
	}
}

func (sa *SimulatedAnnealing) generateNeighbor(current Solution) Solution {
	neighbor := current
	neighbor.ID = fmt.Sprintf("sa-%d", time.Now().UnixNano())

	// Perturb one parameter
	params := []string{"avg_latency", "throughput", "cost", "reliability", "link_utilization"}
	param := params[rand.Intn(len(params))]

	if val, ok := neighbor.Configuration[param].(float64); ok {
		neighbor.Configuration[param] = val + (rand.Float64()-0.5)*0.1*val
	}

	return neighbor
}

func (sa *SimulatedAnnealing) evaluateFitness(solution Solution, objectives []Objective, constraints []Constraint) float64 {
	fitness := 0.0

	for _, obj := range objectives {
		value := obj.Function(solution)
		if obj.Type == "minimize" {
			fitness -= value * obj.Weight
		} else {
			fitness += value * obj.Weight
		}
	}

	for _, constraint := range constraints {
		if !constraint.Function(solution) {
			fitness -= 1000
		}
	}

	return fitness
}

// findParetoFront finds Pareto optimal solutions
func (o *NetworkOptimizer) findParetoFront(solutions []Solution) []Solution {
	pareto := []Solution{}

	for i, sol1 := range solutions {
		dominated := false

		for j, sol2 := range solutions {
			if i == j {
				continue
			}

			if o.dominates(sol2, sol1) {
				dominated = true
				break
			}
		}

		if !dominated {
			pareto = append(pareto, sol1)
		}
	}

	return pareto
}

func (o *NetworkOptimizer) dominates(s1, s2 Solution) bool {
	betterInOne := false
	worseInOne := false

	for _, obj := range o.objectives {
		v1 := obj.Function(s1)
		v2 := obj.Function(s2)

		if obj.Type == "minimize" {
			if v1 < v2 {
				betterInOne = true
			} else if v1 > v2 {
				worseInOne = true
			}
		} else {
			if v1 > v2 {
				betterInOne = true
			} else if v1 < v2 {
				worseInOne = true
			}
		}
	}

	return betterInOne && !worseInOne
}

func (o *NetworkOptimizer) updateArchive(solutions []Solution) {
	o.archive = append(o.archive, solutions...)

	// Keep best solutions
	if len(o.archive) > 1000 {
		sort.Slice(o.archive, func(i, j int) bool {
			return o.archive[i].Fitness > o.archive[j].Fitness
		})
		o.archive = o.archive[:1000]
	}
}

// GetMetrics returns optimizer metrics
func (o *NetworkOptimizer) GetMetrics() map[string]interface{} {
	o.mu.RLock()
	defer o.mu.RUnlock()

	return map[string]interface{}{
		"iterations":       o.iterations,
		"best_fitness":     o.bestFitness,
		"pareto_size":      len(o.paretoFront),
		"archive_size":     len(o.archive),
		"convergence_rate": o.convergenceRate,
	}
}

// GetParetoFront returns the current Pareto front
func (o *NetworkOptimizer) GetParetoFront() []Solution {
	o.mu.RLock()
	defer o.mu.RUnlock()

	return o.paretoFront
}