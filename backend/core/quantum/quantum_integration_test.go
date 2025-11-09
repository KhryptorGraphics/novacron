package quantum

import (
	"context"
	"testing"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/quantum/algorithms"
	"github.com/khryptorgraphics/novacron/backend/core/quantum/compiler"
	"github.com/khryptorgraphics/novacron/backend/core/quantum/ecc"
	"github.com/khryptorgraphics/novacron/backend/core/quantum/executor"
	"github.com/khryptorgraphics/novacron/backend/core/quantum/qkd"
	"github.com/khryptorgraphics/novacron/backend/core/quantum/simulator"
)

// Test Quantum Circuit Compiler
func TestCircuitCompiler(t *testing.T) {
	t.Run("BasicCompilation", func(t *testing.T) {
		cc := compiler.NewCircuitCompiler(2)

		circuit := &compiler.Circuit{
			ID:            "test-1",
			Name:          "Test Circuit",
			Qubits:        4,
			ClassicalBits: 4,
			Gates: []compiler.Gate{
				{Type: "H", Qubits: []int{0}},
				{Type: "CNOT", Qubits: []int{0, 1}},
				{Type: "X", Qubits: []int{2}},
				{Type: "X", Qubits: []int{2}}, // Should be removed
				{Type: "H", Qubits: []int{3}},
			},
		}

		compiled, err := cc.Compile(context.Background(), circuit)
		if err != nil {
			t.Fatalf("Compilation failed: %v", err)
		}

		if compiled.OptimizedGates >= compiled.OriginalGates {
			t.Error("Expected gate reduction")
		}

		if compiled.CompilationTime == 0 {
			t.Error("Expected non-zero compilation time")
		}

		t.Logf("Original gates: %d, Optimized gates: %d, Reduction: %.1f%%",
			compiled.OriginalGates, compiled.OptimizedGates,
			float64(compiled.OriginalGates-compiled.OptimizedGates)/float64(compiled.OriginalGates)*100)
	})

	t.Run("QiskitTranspilation", func(t *testing.T) {
		cc := compiler.NewCircuitCompiler(1)

		circuit := &compiler.Circuit{
			ID:     "test-qiskit",
			Qubits: 2,
			Gates: []compiler.Gate{
				{Type: "H", Qubits: []int{0}},
				{Type: "CNOT", Qubits: []int{0, 1}},
			},
			Measurements: []compiler.Measurement{
				{Qubit: 0, ClassicalBit: 0, Basis: "Z"},
				{Qubit: 1, ClassicalBit: 1, Basis: "Z"},
			},
		}

		qasm, err := cc.TranspileToQiskit(circuit)
		if err != nil {
			t.Fatalf("Qiskit transpilation failed: %v", err)
		}

		if len(qasm) == 0 {
			t.Error("Expected non-empty QASM output")
		}

		t.Logf("Generated QASM:\n%s", qasm)
	})
}

// Test Hybrid Executor
func TestHybridExecutor(t *testing.T) {
	t.Run("SequentialExecution", func(t *testing.T) {
		executor := executor.NewHybridExecutor(4, 10, true)

		classicalPart := &executor.ClassicalPart{
			Type:           "preprocessing",
			RequiredCores:  2,
			RequiredMemory: 1024 * 1024 * 1024, // 1GB
		}

		quantumPart := &executor.QuantumPart{
			Circuit: &compiler.Circuit{
				ID:     "test-quantum",
				Qubits: 4,
				Gates:  []compiler.Gate{{Type: "H", Qubits: []int{0}}},
			},
			RequiredQubits: 4,
			RequiredDepth:  10,
		}

		workload := &executor.HybridWorkload{
			ID:               "test-workload",
			Name:             "Test Sequential",
			ClassicalPart:    classicalPart,
			QuantumPart:      quantumPart,
			CoordinationType: executor.CoordinationSequential,
			Metadata:         make(map[string]interface{}),
		}

		result, err := executor.Execute(context.Background(), workload)
		if err != nil {
			t.Fatalf("Execution failed: %v", err)
		}

		if result.TotalExecutionTime == 0 {
			t.Error("Expected non-zero execution time")
		}

		if result.Overhead > 0.5 {
			t.Errorf("Overhead too high: %.2f%%", result.Overhead*100)
		}

		t.Logf("Execution time: %v, Overhead: %.2f%%",
			result.TotalExecutionTime, result.Overhead*100)
	})

	t.Run("ParallelExecution", func(t *testing.T) {
		executor := executor.NewHybridExecutor(4, 10, true)

		workload := &executor.HybridWorkload{
			ID:   "test-parallel",
			Name: "Test Parallel",
			ClassicalPart: &executor.ClassicalPart{
				Type:           "preprocessing",
				RequiredCores:  2,
				RequiredMemory: 1024 * 1024 * 1024,
			},
			QuantumPart: &executor.QuantumPart{
				Circuit: &compiler.Circuit{
					ID:     "parallel-quantum",
					Qubits: 4,
					Gates:  []compiler.Gate{{Type: "H", Qubits: []int{0}}},
				},
				RequiredQubits: 4,
			},
			CoordinationType: executor.CoordinationParallel,
			Metadata:         make(map[string]interface{}),
		}

		result, err := executor.Execute(context.Background(), workload)
		if err != nil {
			t.Fatalf("Parallel execution failed: %v", err)
		}

		if result == nil {
			t.Fatal("Expected non-nil result")
		}
	})
}

// Test Quantum Algorithms
func TestQuantumAlgorithms(t *testing.T) {
	t.Run("ShorAlgorithm", func(t *testing.T) {
		result, err := algorithms.FactorSmallNumber(15)
		if err != nil {
			t.Fatalf("Shor's algorithm failed: %v", err)
		}

		if len(result.Factors) != 2 {
			t.Errorf("Expected 2 factors, got %d", len(result.Factors))
		}

		product := result.Factors[0].Int64() * result.Factors[1].Int64()
		if product != 15 {
			t.Errorf("Factors don't multiply to 15: %d * %d = %d",
				result.Factors[0].Int64(), result.Factors[1].Int64(), product)
		}

		t.Logf("Factored 15 = %d × %d, Quantum speedup: %.1fx, Qubits: %d",
			result.Factors[0].Int64(), result.Factors[1].Int64(),
			result.QuantumSpeedup, result.CircuitQubits)
	})

	t.Run("GroverSearch", func(t *testing.T) {
		result, err := algorithms.DemoGroverSearch()
		if err != nil {
			t.Fatalf("Grover's search failed: %v", err)
		}

		if len(result.FoundStates) == 0 {
			t.Error("Expected to find target states")
		}

		if result.QuantumSpeedup < 2.0 {
			t.Errorf("Expected speedup >= 2.0, got %.2f", result.QuantumSpeedup)
		}

		t.Logf("Searched database of %d items, found %d targets, Speedup: %.1fx",
			result.DatabaseSize, len(result.FoundStates), result.QuantumSpeedup)
	})

	t.Run("VQE", func(t *testing.T) {
		result, err := algorithms.SolveH2Molecule()
		if err != nil {
			t.Fatalf("VQE failed: %v", err)
		}

		// H2 ground state energy should be around -1.137 Hartree
		expectedEnergy := -1.137
		tolerance := 0.2

		if result.GroundStateEnergy < expectedEnergy-tolerance ||
			result.GroundStateEnergy > expectedEnergy+tolerance {
			t.Logf("Warning: Ground state energy %.3f outside expected range [%.3f, %.3f]",
				result.GroundStateEnergy, expectedEnergy-tolerance, expectedEnergy+tolerance)
		}

		t.Logf("VQE: Ground state energy = %.4f Ha, Iterations = %d, Converged = %v",
			result.GroundStateEnergy, result.Iterations, result.Converged)
	})

	t.Run("QAOA", func(t *testing.T) {
		result, err := algorithms.DemoMaxCut()
		if err != nil {
			t.Fatalf("QAOA failed: %v", err)
		}

		if len(result.OptimalSolution) == 0 {
			t.Error("Expected non-empty optimal solution")
		}

		if result.OptimalCost < 0 {
			t.Error("Expected non-negative cost")
		}

		t.Logf("QAOA: Max-Cut cost = %.2f, Solution = %v, Layers = %d",
			result.OptimalCost, result.OptimalSolution, result.NumLayers)
	})
}

// Test Quantum Key Distribution
func TestQKD(t *testing.T) {
	t.Run("BB84Protocol", func(t *testing.T) {
		config := qkd.DefaultQKDConfig()
		config.Protocol = qkd.ProtocolBB84
		config.QBER = 0.05 // 5% error rate

		manager := qkd.NewQKDManager(config)

		// Establish channel
		channel, err := manager.EstablishChannel(context.Background(), "alice", "bob")
		if err != nil {
			t.Fatalf("Failed to establish channel: %v", err)
		}

		// Generate key
		keyLength := 256 // 256 bytes = 2048 bits
		key, err := manager.GenerateKey(context.Background(), channel.ID, keyLength)
		if err != nil {
			t.Fatalf("Key generation failed: %v", err)
		}

		if len(key) != keyLength {
			t.Errorf("Expected key length %d, got %d", keyLength, len(key))
		}

		if channel.Metrics.QBER > 0.11 {
			t.Error("QBER above security threshold")
		}

		t.Logf("Generated %d-bit key, QBER: %.2f%%, Key rate: %.0f bits/sec",
			len(key)*8, channel.Metrics.QBER*100, channel.Metrics.KeyGenerationRate)
	})

	t.Run("KeyRateEstimation", func(t *testing.T) {
		// Estimate key rate for different distances
		distances := []float64{10, 50, 100, 200}

		for _, distance := range distances {
			keyRate := qkd.EstimateKeyRate(distance, 0.2, 0.01)
			t.Logf("Distance: %d km, Key rate: %.0f bits/sec", int(distance), keyRate)

			if distance > 100 && keyRate <= 0 {
				t.Logf("Note: At %d km, key rate is very low or zero", int(distance))
			}
		}
	})
}

// Test Quantum Simulator
func TestQuantumSimulator(t *testing.T) {
	t.Run("StateVectorSimulation", func(t *testing.T) {
		sim := simulator.NewQuantumSimulator(simulator.SimTypeStateVector, 10, false)

		circuit := &compiler.Circuit{
			ID:     "bell-state",
			Qubits: 2,
			Gates: []compiler.Gate{
				{Type: "H", Qubits: []int{0}},
				{Type: "CNOT", Qubits: []int{0, 1}},
			},
			Measurements: []compiler.Measurement{
				{Qubit: 0, ClassicalBit: 0, Basis: "Z"},
				{Qubit: 1, ClassicalBit: 1, Basis: "Z"},
			},
		}

		result, err := sim.Simulate(context.Background(), circuit, 1024)
		if err != nil {
			t.Fatalf("Simulation failed: %v", err)
		}

		// Bell state should give |00⟩ and |11⟩ with equal probability
		prob00 := result.Probabilities["00"]
		prob11 := result.Probabilities["11"]

		if prob00 < 0.3 || prob00 > 0.7 {
			t.Logf("Warning: P(00) = %.3f, expected ~0.5", prob00)
		}

		if prob11 < 0.3 || prob11 > 0.7 {
			t.Logf("Warning: P(11) = %.3f, expected ~0.5", prob11)
		}

		t.Logf("Bell state probabilities: |00⟩=%.3f, |11⟩=%.3f", prob00, prob11)
	})

	t.Run("NoisySimulation", func(t *testing.T) {
		sim := simulator.NewQuantumSimulator(simulator.SimTypeDensityMatrix, 5, false)

		noiseModel := &simulator.NoiseModel{
			Enabled:          true,
			Type:             "depolarizing",
			SingleQubitError: 0.001,
			TwoQubitError:    0.01,
			MeasurementError: 0.01,
		}
		sim.SetNoiseModel(noiseModel)

		circuit := &compiler.Circuit{
			ID:     "noisy-test",
			Qubits: 3,
			Gates: []compiler.Gate{
				{Type: "H", Qubits: []int{0}},
				{Type: "CNOT", Qubits: []int{0, 1}},
				{Type: "CNOT", Qubits: []int{1, 2}},
			},
		}

		result, err := sim.Simulate(context.Background(), circuit, 1000)
		if err != nil {
			t.Fatalf("Noisy simulation failed: %v", err)
		}

		if result.Fidelity > 0.99 {
			t.Log("Warning: Fidelity unexpectedly high for noisy simulation")
		}

		t.Logf("Noisy simulation fidelity: %.3f", result.Fidelity)
	})
}

// Test Error Correction
func TestErrorCorrection(t *testing.T) {
	t.Run("Shor9Code", func(t *testing.T) {
		ec := ecc.NewErrorCorrector(ecc.CodeShor9, 1)

		circuit := &compiler.Circuit{
			ID:     "error-corrected",
			Qubits: 9,
		}

		err := ec.EncodeLogicalQubit(circuit, 0)
		if err != nil {
			t.Fatalf("Encoding failed: %v", err)
		}

		// Test error correction
		physicalErrorRate := 0.001
		result, err := ec.DetectAndCorrect(circuit, physicalErrorRate)
		if err != nil {
			t.Fatalf("Error correction failed: %v", err)
		}

		if result.LogicalErrorRate >= physicalErrorRate {
			t.Error("Expected logical error rate < physical error rate")
		}

		t.Logf("Error correction: Detected=%d, Corrected=%d, Logical error rate=%.4f",
			result.ErrorsDetected, result.ErrorsCorrected, result.LogicalErrorRate)
	})

	t.Run("OverheadEstimation", func(t *testing.T) {
		targetError := 1e-6
		physicalError := 0.001

		overhead, logicalError := ecc.EstimateOverhead(ecc.CodeSurface, targetError, physicalError)

		t.Logf("Surface code overhead: %dx physical qubits per logical qubit", overhead)
		t.Logf("Achieved logical error rate: %.2e", logicalError)

		if logicalError > targetError*10 {
			t.Logf("Warning: Logical error rate higher than target")
		}
	})
}

// Test Full Quantum Stack Integration
func TestFullQuantumStack(t *testing.T) {
	t.Run("EndToEndWorkflow", func(t *testing.T) {
		// 1. Create and compile circuit
		cc := compiler.NewCircuitCompiler(2)
		circuit := &compiler.Circuit{
			ID:     "e2e-test",
			Qubits: 4,
			Gates: []compiler.Gate{
				{Type: "H", Qubits: []int{0}},
				{Type: "CNOT", Qubits: []int{0, 1}},
				{Type: "CNOT", Qubits: []int{1, 2}},
				{Type: "H", Qubits: []int{3}},
			},
		}

		compiled, err := cc.Compile(context.Background(), circuit)
		if err != nil {
			t.Fatalf("Compilation failed: %v", err)
		}

		// 2. Simulate circuit
		sim := simulator.NewQuantumSimulator(simulator.SimTypeStateVector, 10, false)
		simResult, err := sim.Simulate(context.Background(), compiled.OptimizedCircuit, 1024)
		if err != nil {
			t.Fatalf("Simulation failed: %v", err)
		}

		// 3. Execute hybrid workload
		exec := executor.NewHybridExecutor(4, 10, true)
		workload := &executor.HybridWorkload{
			ID:   "e2e-hybrid",
			Name: "End-to-End Test",
			QuantumPart: &executor.QuantumPart{
				Circuit:        compiled.OptimizedCircuit,
				RequiredQubits: 4,
			},
			CoordinationType: executor.CoordinationSequential,
			Metadata:         make(map[string]interface{}),
		}

		execResult, err := exec.Execute(context.Background(), workload)
		if err != nil {
			t.Fatalf("Hybrid execution failed: %v", err)
		}

		t.Logf("End-to-end test completed:")
		t.Logf("  - Compilation: %d → %d gates (%.1f%% reduction)",
			compiled.OriginalGates, compiled.OptimizedGates,
			float64(compiled.OriginalGates-compiled.OptimizedGates)/float64(compiled.OriginalGates)*100)
		t.Logf("  - Simulation: %d shots, fidelity=%.3f",
			simResult.Shots, simResult.Fidelity)
		t.Logf("  - Execution: %v, overhead=%.2f%%",
			execResult.TotalExecutionTime, execResult.Overhead*100)
	})
}

// Benchmark tests
func BenchmarkCircuitCompilation(b *testing.B) {
	cc := compiler.NewCircuitCompiler(2)
	circuit := &compiler.Circuit{
		ID:     "bench",
		Qubits: 10,
		Gates:  make([]compiler.Gate, 100),
	}

	for i := range circuit.Gates {
		circuit.Gates[i] = compiler.Gate{
			Type:   "H",
			Qubits: []int{i % circuit.Qubits},
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cc.Compile(context.Background(), circuit)
	}
}

func BenchmarkQuantumSimulation(b *testing.B) {
	sim := simulator.NewQuantumSimulator(simulator.SimTypeStateVector, 15, false)
	circuit := &compiler.Circuit{
		ID:     "bench-sim",
		Qubits: 10,
		Gates: []compiler.Gate{
			{Type: "H", Qubits: []int{0}},
			{Type: "CNOT", Qubits: []int{0, 1}},
		},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = sim.Simulate(context.Background(), circuit, 100)
	}
}

func TestQuantumConfig(t *testing.T) {
	t.Run("DefaultConfig", func(t *testing.T) {
		config := DefaultQuantumConfig()

		if err := config.ValidateConfig(); err != nil {
			t.Fatalf("Default config validation failed: %v", err)
		}

		if !config.EnableQuantum {
			t.Error("Quantum should be enabled by default")
		}

		if config.PreferredProvider != "simulator" {
			t.Error("Default provider should be simulator")
		}
	})

	t.Run("ProductionConfig", func(t *testing.T) {
		config := ProductionQuantumConfig()

		if config.PreferredProvider == "simulator" {
			t.Error("Production should use real quantum hardware")
		}

		if config.MaxQubits < 100 {
			t.Error("Production should support >100 qubits")
		}
	})
}
