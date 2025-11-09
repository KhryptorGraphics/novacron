package compiler

import (
	"context"
	"fmt"
	"math"
	"time"
)

// CircuitCompiler transpiles and optimizes quantum circuits
type CircuitCompiler struct {
	optimizationLevel int
	targetGateSet     []string
	gateReduction     bool
	decomposition     bool
	metrics           *CompilationMetrics
}

// CompilationMetrics tracks compiler performance
type CompilationMetrics struct {
	TotalCompilations    int64         `json:"total_compilations"`
	SuccessfulCompilations int64       `json:"successful_compilations"`
	FailedCompilations   int64         `json:"failed_compilations"`
	AverageCompileTime   time.Duration `json:"average_compile_time"`
	TotalGatesReduced    int64         `json:"total_gates_reduced"`
	AverageDepthReduction float64      `json:"average_depth_reduction"`
}

// Circuit represents a quantum circuit
type Circuit struct {
	ID              string              `json:"id"`
	Name            string              `json:"name"`
	Qubits          int                 `json:"qubits"`
	ClassicalBits   int                 `json:"classical_bits"`
	Gates           []Gate              `json:"gates"`
	Measurements    []Measurement       `json:"measurements"`
	Metadata        map[string]interface{} `json:"metadata"`
}

// Gate represents a quantum gate
type Gate struct {
	Type          string    `json:"type"`
	Qubits        []int     `json:"qubits"`
	Parameters    []float64 `json:"parameters,omitempty"`
	ControlQubits []int     `json:"control_qubits,omitempty"`
	Label         string    `json:"label,omitempty"`
}

// Measurement represents a quantum measurement
type Measurement struct {
	Qubit        int    `json:"qubit"`
	ClassicalBit int    `json:"classical_bit"`
	Basis        string `json:"basis"` // "Z", "X", "Y"
}

// CompiledCircuit represents a compiled quantum circuit
type CompiledCircuit struct {
	OriginalCircuit  *Circuit          `json:"original_circuit"`
	OptimizedCircuit *Circuit          `json:"optimized_circuit"`
	Transpiled       bool              `json:"transpiled"`
	OriginalDepth    int               `json:"original_depth"`
	OptimizedDepth   int               `json:"optimized_depth"`
	OriginalGates    int               `json:"original_gates"`
	OptimizedGates   int               `json:"optimized_gates"`
	CompilationTime  time.Duration     `json:"compilation_time"`
	Format           string            `json:"format"` // "qiskit", "cirq", "qsharp", "native"
	Warnings         []string          `json:"warnings,omitempty"`
}

// NewCircuitCompiler creates a new circuit compiler
func NewCircuitCompiler(optimizationLevel int) *CircuitCompiler {
	return &CircuitCompiler{
		optimizationLevel: optimizationLevel,
		targetGateSet:     []string{"X", "Y", "Z", "H", "S", "T", "CNOT", "CZ", "RX", "RY", "RZ"},
		gateReduction:     true,
		decomposition:     true,
		metrics:           &CompilationMetrics{},
	}
}

// Compile compiles a quantum circuit with optimization
func (cc *CircuitCompiler) Compile(ctx context.Context, circuit *Circuit) (*CompiledCircuit, error) {
	startTime := time.Now()

	// Validate circuit
	if err := cc.validateCircuit(circuit); err != nil {
		cc.metrics.FailedCompilations++
		return nil, fmt.Errorf("circuit validation failed: %w", err)
	}

	// Calculate original metrics
	originalDepth := cc.calculateCircuitDepth(circuit)
	originalGates := len(circuit.Gates)

	// Create optimized copy
	optimized := cc.copyCircuit(circuit)

	// Apply optimization passes based on level
	warnings := []string{}

	if cc.optimizationLevel >= 1 {
		// Level 1: Basic optimization
		optimized = cc.removeIdentityGates(optimized)
		optimized = cc.mergeSingleQubitGates(optimized)
	}

	if cc.optimizationLevel >= 2 {
		// Level 2: Advanced optimization
		optimized = cc.cancelAdjacentGates(optimized)
		optimized = cc.commuteThroughCNOTs(optimized)
		warnings = append(warnings, cc.checkTwoQubitGateDepth(optimized)...)
	}

	if cc.optimizationLevel >= 3 {
		// Level 3: Aggressive optimization
		optimized = cc.optimizeTwoQubitGates(optimized)
		optimized = cc.applyTemplateMatching(optimized)
	}

	// Gate decomposition to target gate set
	if cc.decomposition {
		optimized = cc.decomposeToTargetGates(optimized)
	}

	// Calculate optimized metrics
	optimizedDepth := cc.calculateCircuitDepth(optimized)
	optimizedGates := len(optimized.Gates)
	compilationTime := time.Since(startTime)

	// Update compiler metrics
	cc.metrics.TotalCompilations++
	cc.metrics.SuccessfulCompilations++
	cc.metrics.AverageCompileTime = (cc.metrics.AverageCompileTime + compilationTime) / 2
	cc.metrics.TotalGatesReduced += int64(originalGates - optimizedGates)

	compiled := &CompiledCircuit{
		OriginalCircuit:  circuit,
		OptimizedCircuit: optimized,
		Transpiled:       true,
		OriginalDepth:    originalDepth,
		OptimizedDepth:   optimizedDepth,
		OriginalGates:    originalGates,
		OptimizedGates:   optimizedGates,
		CompilationTime:  compilationTime,
		Format:           "native",
		Warnings:         warnings,
	}

	return compiled, nil
}

// TranspileToQiskit transpiles circuit to Qiskit format
func (cc *CircuitCompiler) TranspileToQiskit(circuit *Circuit) (string, error) {
	qasm := fmt.Sprintf("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n")
	qasm += fmt.Sprintf("qreg q[%d];\n", circuit.Qubits)

	if circuit.ClassicalBits > 0 {
		qasm += fmt.Sprintf("creg c[%d];\n", circuit.ClassicalBits)
	}

	for _, gate := range circuit.Gates {
		qasm += cc.gateToQASM(gate) + "\n"
	}

	for _, meas := range circuit.Measurements {
		qasm += fmt.Sprintf("measure q[%d] -> c[%d];\n", meas.Qubit, meas.ClassicalBit)
	}

	return qasm, nil
}

// TranspileToCirq transpiles circuit to Cirq format (Python code)
func (cc *CircuitCompiler) TranspileToCirq(circuit *Circuit) (string, error) {
	cirq := "import cirq\n\n"
	cirq += fmt.Sprintf("# Create qubits\nqubits = [cirq.LineQubit(i) for i in range(%d)]\n", circuit.Qubits)
	cirq += "circuit = cirq.Circuit()\n\n"

	for _, gate := range circuit.Gates {
		cirq += cc.gateToCirq(gate) + "\n"
	}

	cirq += "\n# Measurements\n"
	for _, meas := range circuit.Measurements {
		cirq += fmt.Sprintf("circuit.append(cirq.measure(qubits[%d], key='m%d'))\n",
			meas.Qubit, meas.ClassicalBit)
	}

	return cirq, nil
}

// TranspileToQSharp transpiles circuit to Q# format
func (cc *CircuitCompiler) TranspileToQSharp(circuit *Circuit) (string, error) {
	qsharp := "namespace QuantumCircuit {\n"
	qsharp += "    open Microsoft.Quantum.Intrinsic;\n"
	qsharp += "    open Microsoft.Quantum.Measurement;\n\n"
	qsharp += fmt.Sprintf("    operation RunCircuit() : Result[] {\n")
	qsharp += fmt.Sprintf("        use qubits = Qubit[%d];\n", circuit.Qubits)
	qsharp += "        mutable results = [];\n\n"

	for _, gate := range circuit.Gates {
		qsharp += "        " + cc.gateToQSharp(gate) + "\n"
	}

	qsharp += "\n        // Measurements\n"
	for _, meas := range circuit.Measurements {
		qsharp += fmt.Sprintf("        set results += [M(qubits[%d])];\n", meas.Qubit)
	}

	qsharp += "\n        ResetAll(qubits);\n"
	qsharp += "        return results;\n"
	qsharp += "    }\n"
	qsharp += "}\n"

	return qsharp, nil
}

// Optimization passes

func (cc *CircuitCompiler) removeIdentityGates(circuit *Circuit) *Circuit {
	optimized := cc.copyCircuit(circuit)
	filteredGates := []Gate{}

	for _, gate := range optimized.Gates {
		// Remove identity gates (e.g., X followed by X, or gates with zero rotation)
		if !cc.isIdentityGate(gate) {
			filteredGates = append(filteredGates, gate)
		}
	}

	optimized.Gates = filteredGates
	return optimized
}

func (cc *CircuitCompiler) mergeSingleQubitGates(circuit *Circuit) *Circuit {
	optimized := cc.copyCircuit(circuit)
	merged := []Gate{}

	for i := 0; i < len(optimized.Gates); i++ {
		gate := optimized.Gates[i]

		// Try to merge with next gate if both are single-qubit on same qubit
		if i < len(optimized.Gates)-1 {
			nextGate := optimized.Gates[i+1]
			if cc.canMergeSingleQubitGates(gate, nextGate) {
				mergedGate := cc.mergeTwoSingleQubitGates(gate, nextGate)
				merged = append(merged, mergedGate)
				i++ // Skip next gate
				continue
			}
		}

		merged = append(merged, gate)
	}

	optimized.Gates = merged
	return optimized
}

func (cc *CircuitCompiler) cancelAdjacentGates(circuit *Circuit) *Circuit {
	optimized := cc.copyCircuit(circuit)
	filtered := []Gate{}

	for i := 0; i < len(optimized.Gates); i++ {
		gate := optimized.Gates[i]

		// Check if next gate cancels this one
		if i < len(optimized.Gates)-1 {
			nextGate := optimized.Gates[i+1]
			if cc.gatesCancelOut(gate, nextGate) {
				i++ // Skip both gates
				continue
			}
		}

		filtered = append(filtered, gate)
	}

	optimized.Gates = filtered
	return optimized
}

func (cc *CircuitCompiler) commuteThroughCNOTs(circuit *Circuit) *Circuit {
	// Commute single-qubit gates through CNOTs where possible
	// This is a simplified version - full implementation would be more complex
	optimized := cc.copyCircuit(circuit)

	for i := 0; i < len(optimized.Gates)-1; i++ {
		gate := optimized.Gates[i]
		nextGate := optimized.Gates[i+1]

		if cc.canCommute(gate, nextGate) {
			// Swap gates
			optimized.Gates[i] = nextGate
			optimized.Gates[i+1] = gate
		}
	}

	return optimized
}

func (cc *CircuitCompiler) optimizeTwoQubitGates(circuit *Circuit) *Circuit {
	optimized := cc.copyCircuit(circuit)

	// Replace multiple CNOTs with more efficient two-qubit gates where possible
	filtered := []Gate{}

	for i := 0; i < len(optimized.Gates); i++ {
		gate := optimized.Gates[i]

		// Check for CNOT chains that can be optimized
		if gate.Type == "CNOT" && i < len(optimized.Gates)-2 {
			chain := []Gate{gate}
			j := i + 1

			for j < len(optimized.Gates) && optimized.Gates[j].Type == "CNOT" {
				chain = append(chain, optimized.Gates[j])
				j++
			}

			if len(chain) >= 3 {
				// Optimize CNOT chain
				optimizedChain := cc.optimizeCNOTChain(chain)
				filtered = append(filtered, optimizedChain...)
				i = j - 1
				continue
			}
		}

		filtered = append(filtered, gate)
	}

	optimized.Gates = filtered
	return optimized
}

func (cc *CircuitCompiler) applyTemplateMatching(circuit *Circuit) *Circuit {
	// Match common gate patterns and replace with optimized equivalents
	optimized := cc.copyCircuit(circuit)

	// Example: Replace H-CNOT-H with CZ
	for i := 0; i < len(optimized.Gates)-2; i++ {
		if optimized.Gates[i].Type == "H" &&
		   optimized.Gates[i+1].Type == "CNOT" &&
		   optimized.Gates[i+2].Type == "H" {
			// Check if pattern matches
			if cc.matchesHCNOTHPattern(optimized.Gates[i:i+3]) {
				// Replace with CZ
				czGate := Gate{
					Type:   "CZ",
					Qubits: optimized.Gates[i+1].Qubits,
				}
				optimized.Gates = append(optimized.Gates[:i],
					append([]Gate{czGate}, optimized.Gates[i+3:]...)...)
			}
		}
	}

	return optimized
}

func (cc *CircuitCompiler) decomposeToTargetGates(circuit *Circuit) *Circuit {
	optimized := cc.copyCircuit(circuit)
	decomposed := []Gate{}

	for _, gate := range optimized.Gates {
		if cc.isInTargetGateSet(gate.Type) {
			decomposed = append(decomposed, gate)
		} else {
			// Decompose gate into target gate set
			decomposedGates := cc.decomposeGate(gate)
			decomposed = append(decomposed, decomposedGates...)
		}
	}

	optimized.Gates = decomposed
	return optimized
}

// Helper functions

func (cc *CircuitCompiler) validateCircuit(circuit *Circuit) error {
	if circuit.Qubits < 1 {
		return fmt.Errorf("circuit must have at least 1 qubit")
	}

	if circuit.Qubits > 1000 {
		return fmt.Errorf("circuit has too many qubits: %d", circuit.Qubits)
	}

	// Validate all gates
	for i, gate := range circuit.Gates {
		if err := cc.validateGate(gate, circuit.Qubits); err != nil {
			return fmt.Errorf("invalid gate at position %d: %w", i, err)
		}
	}

	return nil
}

func (cc *CircuitCompiler) validateGate(gate Gate, numQubits int) error {
	// Check qubit indices
	for _, q := range gate.Qubits {
		if q < 0 || q >= numQubits {
			return fmt.Errorf("qubit index %d out of range [0, %d)", q, numQubits)
		}
	}

	// Check control qubit indices
	for _, q := range gate.ControlQubits {
		if q < 0 || q >= numQubits {
			return fmt.Errorf("control qubit index %d out of range [0, %d)", q, numQubits)
		}
	}

	// Validate gate-specific requirements
	switch gate.Type {
	case "RX", "RY", "RZ", "U1":
		if len(gate.Parameters) != 1 {
			return fmt.Errorf("gate %s requires 1 parameter", gate.Type)
		}
	case "U2":
		if len(gate.Parameters) != 2 {
			return fmt.Errorf("gate U2 requires 2 parameters")
		}
	case "U3":
		if len(gate.Parameters) != 3 {
			return fmt.Errorf("gate U3 requires 3 parameters")
		}
	}

	return nil
}

func (cc *CircuitCompiler) calculateCircuitDepth(circuit *Circuit) int {
	// Calculate circuit depth (critical path length)
	qubitDepth := make([]int, circuit.Qubits)

	for _, gate := range circuit.Gates {
		maxDepth := 0
		for _, q := range gate.Qubits {
			if qubitDepth[q] > maxDepth {
				maxDepth = qubitDepth[q]
			}
		}
		for _, q := range gate.ControlQubits {
			if qubitDepth[q] > maxDepth {
				maxDepth = qubitDepth[q]
			}
		}

		// Update depths
		allQubits := append(gate.Qubits, gate.ControlQubits...)
		for _, q := range allQubits {
			qubitDepth[q] = maxDepth + 1
		}
	}

	// Find maximum depth
	depth := 0
	for _, d := range qubitDepth {
		if d > depth {
			depth = d
		}
	}

	return depth
}

func (cc *CircuitCompiler) copyCircuit(circuit *Circuit) *Circuit {
	copied := &Circuit{
		ID:            circuit.ID,
		Name:          circuit.Name,
		Qubits:        circuit.Qubits,
		ClassicalBits: circuit.ClassicalBits,
		Gates:         make([]Gate, len(circuit.Gates)),
		Measurements:  make([]Measurement, len(circuit.Measurements)),
		Metadata:      make(map[string]interface{}),
	}

	copy(copied.Gates, circuit.Gates)
	copy(copied.Measurements, circuit.Measurements)

	for k, v := range circuit.Metadata {
		copied.Metadata[k] = v
	}

	return copied
}

func (cc *CircuitCompiler) isIdentityGate(gate Gate) bool {
	// Check if gate is identity (e.g., rotation by 0)
	if (gate.Type == "RX" || gate.Type == "RY" || gate.Type == "RZ") && len(gate.Parameters) > 0 {
		angle := gate.Parameters[0]
		return math.Abs(angle) < 1e-10
	}
	return false
}

func (cc *CircuitCompiler) canMergeSingleQubitGates(g1, g2 Gate) bool {
	// Can merge if both are single-qubit gates on the same qubit
	return len(g1.Qubits) == 1 && len(g2.Qubits) == 1 &&
	       g1.Qubits[0] == g2.Qubits[0] &&
	       len(g1.ControlQubits) == 0 && len(g2.ControlQubits) == 0
}

func (cc *CircuitCompiler) mergeTwoSingleQubitGates(g1, g2 Gate) Gate {
	// Simplified: Just combine rotations
	if g1.Type == "RZ" && g2.Type == "RZ" {
		return Gate{
			Type:       "RZ",
			Qubits:     g1.Qubits,
			Parameters: []float64{g1.Parameters[0] + g2.Parameters[0]},
		}
	}

	// For complex merges, would compute matrix product and decompose
	// For now, return U3 gate as placeholder
	return Gate{
		Type:       "U3",
		Qubits:     g1.Qubits,
		Parameters: []float64{0, 0, 0},
	}
}

func (cc *CircuitCompiler) gatesCancelOut(g1, g2 Gate) bool {
	// Check if gates cancel (e.g., X followed by X)
	if g1.Type == g2.Type && len(g1.Qubits) == len(g2.Qubits) {
		for i := range g1.Qubits {
			if g1.Qubits[i] != g2.Qubits[i] {
				return false
			}
		}

		// Self-inverse gates
		switch g1.Type {
		case "X", "Y", "Z", "H", "CNOT", "CZ", "SWAP":
			return true
		}
	}

	return false
}

func (cc *CircuitCompiler) canCommute(g1, g2 Gate) bool {
	// Simplified commutation check
	// Gates commute if they act on different qubits
	qubits1 := make(map[int]bool)
	for _, q := range append(g1.Qubits, g1.ControlQubits...) {
		qubits1[q] = true
	}

	for _, q := range append(g2.Qubits, g2.ControlQubits...) {
		if qubits1[q] {
			return false // Share a qubit
		}
	}

	return true
}

func (cc *CircuitCompiler) checkTwoQubitGateDepth(circuit *Circuit) []string {
	warnings := []string{}
	twoQubitGates := 0

	for _, gate := range circuit.Gates {
		if len(gate.Qubits) == 2 || len(gate.ControlQubits) > 0 {
			twoQubitGates++
		}
	}

	if twoQubitGates > circuit.Qubits*10 {
		warnings = append(warnings, fmt.Sprintf(
			"High two-qubit gate count (%d) may impact fidelity", twoQubitGates))
	}

	return warnings
}

func (cc *CircuitCompiler) optimizeCNOTChain(chain []Gate) []Gate {
	// Optimize chain of CNOTs
	// Simplified: Remove redundant CNOTs
	optimized := []Gate{}
	used := make(map[string]bool)

	for _, gate := range chain {
		key := fmt.Sprintf("%d-%d", gate.Qubits[0], gate.Qubits[1])
		if !used[key] {
			optimized = append(optimized, gate)
			used[key] = true
		}
	}

	return optimized
}

func (cc *CircuitCompiler) matchesHCNOTHPattern(gates []Gate) bool {
	// Check if H-CNOT-H pattern applies to correct qubits
	return gates[0].Qubits[0] == gates[1].Qubits[1] &&
	       gates[2].Qubits[0] == gates[1].Qubits[1]
}

func (cc *CircuitCompiler) isInTargetGateSet(gateType string) bool {
	for _, target := range cc.targetGateSet {
		if gateType == target {
			return true
		}
	}
	return false
}

func (cc *CircuitCompiler) decomposeGate(gate Gate) []Gate {
	// Decompose non-target gates into target gate set
	switch gate.Type {
	case "TOFFOLI":
		// Decompose Toffoli into CNOTs and single-qubit gates
		return cc.decomposeToffoli(gate)
	case "FREDKIN":
		// Decompose Fredkin (CSWAP)
		return cc.decomposeFredkin(gate)
	default:
		// Unknown gate, return as-is with warning
		return []Gate{gate}
	}
}

func (cc *CircuitCompiler) decomposeToffoli(gate Gate) []Gate {
	// Toffoli decomposition using 6 CNOTs
	if len(gate.Qubits) != 3 {
		return []Gate{gate}
	}

	c1, c2, t := gate.Qubits[0], gate.Qubits[1], gate.Qubits[2]

	return []Gate{
		{Type: "H", Qubits: []int{t}},
		{Type: "CNOT", Qubits: []int{c2, t}},
		{Type: "T", Qubits: []int{t}},
		{Type: "CNOT", Qubits: []int{c1, t}},
		{Type: "T", Qubits: []int{t}},
		{Type: "CNOT", Qubits: []int{c2, t}},
		{Type: "T", Qubits: []int{t}},
		{Type: "CNOT", Qubits: []int{c1, t}},
		{Type: "H", Qubits: []int{t}},
	}
}

func (cc *CircuitCompiler) decomposeFredkin(gate Gate) []Gate {
	// Simplified Fredkin decomposition
	return []Gate{gate} // Placeholder
}

// Transpilation helpers

func (cc *CircuitCompiler) gateToQASM(gate Gate) string {
	switch gate.Type {
	case "X":
		return fmt.Sprintf("x q[%d];", gate.Qubits[0])
	case "Y":
		return fmt.Sprintf("y q[%d];", gate.Qubits[0])
	case "Z":
		return fmt.Sprintf("z q[%d];", gate.Qubits[0])
	case "H":
		return fmt.Sprintf("h q[%d];", gate.Qubits[0])
	case "S":
		return fmt.Sprintf("s q[%d];", gate.Qubits[0])
	case "T":
		return fmt.Sprintf("t q[%d];", gate.Qubits[0])
	case "CNOT":
		return fmt.Sprintf("cx q[%d],q[%d];", gate.Qubits[0], gate.Qubits[1])
	case "CZ":
		return fmt.Sprintf("cz q[%d],q[%d];", gate.Qubits[0], gate.Qubits[1])
	case "RX":
		return fmt.Sprintf("rx(%.6f) q[%d];", gate.Parameters[0], gate.Qubits[0])
	case "RY":
		return fmt.Sprintf("ry(%.6f) q[%d];", gate.Parameters[0], gate.Qubits[0])
	case "RZ":
		return fmt.Sprintf("rz(%.6f) q[%d];", gate.Parameters[0], gate.Qubits[0])
	default:
		return fmt.Sprintf("// Unknown gate: %s", gate.Type)
	}
}

func (cc *CircuitCompiler) gateToCirq(gate Gate) string {
	switch gate.Type {
	case "X":
		return fmt.Sprintf("circuit.append(cirq.X(qubits[%d]))", gate.Qubits[0])
	case "Y":
		return fmt.Sprintf("circuit.append(cirq.Y(qubits[%d]))", gate.Qubits[0])
	case "Z":
		return fmt.Sprintf("circuit.append(cirq.Z(qubits[%d]))", gate.Qubits[0])
	case "H":
		return fmt.Sprintf("circuit.append(cirq.H(qubits[%d]))", gate.Qubits[0])
	case "CNOT":
		return fmt.Sprintf("circuit.append(cirq.CNOT(qubits[%d], qubits[%d]))",
			gate.Qubits[0], gate.Qubits[1])
	case "CZ":
		return fmt.Sprintf("circuit.append(cirq.CZ(qubits[%d], qubits[%d]))",
			gate.Qubits[0], gate.Qubits[1])
	default:
		return fmt.Sprintf("# Unknown gate: %s", gate.Type)
	}
}

func (cc *CircuitCompiler) gateToQSharp(gate Gate) string {
	switch gate.Type {
	case "X":
		return fmt.Sprintf("X(qubits[%d]);", gate.Qubits[0])
	case "Y":
		return fmt.Sprintf("Y(qubits[%d]);", gate.Qubits[0])
	case "Z":
		return fmt.Sprintf("Z(qubits[%d]);", gate.Qubits[0])
	case "H":
		return fmt.Sprintf("H(qubits[%d]);", gate.Qubits[0])
	case "CNOT":
		return fmt.Sprintf("CNOT(qubits[%d], qubits[%d]);",
			gate.Qubits[0], gate.Qubits[1])
	default:
		return fmt.Sprintf("// Unknown gate: %s", gate.Type)
	}
}

// GetMetrics returns compilation metrics
func (cc *CircuitCompiler) GetMetrics() *CompilationMetrics {
	return cc.metrics
}
