package bioinspired

import (
	"context"
	"math"
	"math/rand"
	"sync"
)

// ParticleSwarmOptimizer implements Particle Swarm Optimization
type ParticleSwarmOptimizer struct {
	mu              sync.RWMutex
	numParticles    int
	dimensions      int
	particles       []*Particle
	globalBest      []float64
	globalBestScore float64
	bounds          [][2]float64
	inertia         float64
	cognitive       float64
	social          float64
}

// Particle represents a particle in PSO
type Particle struct {
	Position      []float64
	Velocity      []float64
	BestPosition  []float64
	BestScore     float64
}

// ObjectiveFunc is a function to minimize
type ObjectiveFunc func([]float64) float64

// NewParticleSwarmOptimizer creates a new PSO instance
func NewParticleSwarmOptimizer(numParticles, dimensions int, bounds [][2]float64) *ParticleSwarmOptimizer {
	pso := &ParticleSwarmOptimizer{
		numParticles:    numParticles,
		dimensions:      dimensions,
		particles:       make([]*Particle, numParticles),
		globalBestScore: math.MaxFloat64,
		bounds:          bounds,
		inertia:         0.7,
		cognitive:       1.5,
		social:          1.5,
	}

	// Initialize particles
	for i := 0; i < numParticles; i++ {
		pso.particles[i] = pso.createParticle()
	}

	return pso
}

// createParticle creates a new particle with random position
func (pso *ParticleSwarmOptimizer) createParticle() *Particle {
	particle := &Particle{
		Position:     make([]float64, pso.dimensions),
		Velocity:     make([]float64, pso.dimensions),
		BestPosition: make([]float64, pso.dimensions),
		BestScore:    math.MaxFloat64,
	}

	// Random initialization
	for d := 0; d < pso.dimensions; d++ {
		min := pso.bounds[d][0]
		max := pso.bounds[d][1]
		particle.Position[d] = min + rand.Float64()*(max-min)
		particle.Velocity[d] = (rand.Float64()*2 - 1) * (max - min) * 0.1
		particle.BestPosition[d] = particle.Position[d]
	}

	return particle
}

// Optimize runs PSO optimization
func (pso *ParticleSwarmOptimizer) Optimize(ctx context.Context, objective ObjectiveFunc, maxIterations int) ([]float64, float64, error) {
	for iter := 0; iter < maxIterations; iter++ {
		// Evaluate all particles
		for _, particle := range pso.particles {
			score := objective(particle.Position)

			// Update personal best
			if score < particle.BestScore {
				particle.BestScore = score
				copy(particle.BestPosition, particle.Position)
			}

			// Update global best
			pso.mu.Lock()
			if score < pso.globalBestScore {
				pso.globalBestScore = score
				if pso.globalBest == nil {
					pso.globalBest = make([]float64, pso.dimensions)
				}
				copy(pso.globalBest, particle.Position)
			}
			pso.mu.Unlock()
		}

		// Update particle velocities and positions
		for _, particle := range pso.particles {
			pso.updateParticle(particle)
		}

		// Check context
		select {
		case <-ctx.Done():
			return pso.globalBest, pso.globalBestScore, ctx.Err()
		default:
		}
	}

	return pso.globalBest, pso.globalBestScore, nil
}

// updateParticle updates particle velocity and position
func (pso *ParticleSwarmOptimizer) updateParticle(particle *Particle) {
	pso.mu.RLock()
	defer pso.mu.RUnlock()

	for d := 0; d < pso.dimensions; d++ {
		r1 := rand.Float64()
		r2 := rand.Float64()

		// Velocity update
		cognitive := pso.cognitive * r1 * (particle.BestPosition[d] - particle.Position[d])
		social := pso.social * r2 * (pso.globalBest[d] - particle.Position[d])
		particle.Velocity[d] = pso.inertia*particle.Velocity[d] + cognitive + social

		// Position update
		particle.Position[d] += particle.Velocity[d]

		// Boundary handling
		min := pso.bounds[d][0]
		max := pso.bounds[d][1]
		if particle.Position[d] < min {
			particle.Position[d] = min
			particle.Velocity[d] *= -0.5
		} else if particle.Position[d] > max {
			particle.Position[d] = max
			particle.Velocity[d] *= -0.5
		}
	}
}

// GetBest returns the best solution found
func (pso *ParticleSwarmOptimizer) GetBest() ([]float64, float64) {
	pso.mu.RLock()
	defer pso.mu.RUnlock()
	return pso.globalBest, pso.globalBestScore
}
