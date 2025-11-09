package quantum

import (
	"errors"
	"fmt"
)

// Common quantum errors
var (
	// Configuration errors
	ErrInvalidQubitCount          = errors.New("invalid qubit count")
	ErrTooManyQubits              = errors.New("too many qubits requested")
	ErrInvalidOptimizationLevel   = errors.New("invalid optimization level")
	ErrInvalidShotCount           = errors.New("invalid shot count")
	ErrMissingProviderCredentials = errors.New("missing provider credentials")
	ErrUnsupportedProvider        = errors.New("unsupported quantum provider")

	// Circuit errors
	ErrInvalidCircuit             = errors.New("invalid quantum circuit")
	ErrCircuitTooDeep             = errors.New("circuit depth exceeds maximum")
	ErrUnsupportedGate            = errors.New("unsupported quantum gate")
	ErrInvalidGateParameters      = errors.New("invalid gate parameters")
	ErrQubitIndexOutOfRange       = errors.New("qubit index out of range")
	ErrCircuitCompilationFailed   = errors.New("circuit compilation failed")

	// Execution errors
	ErrSimulatorNotAvailable      = errors.New("quantum simulator not available")
	ErrProviderNotAvailable       = errors.New("quantum provider not available")
	ErrExecutionTimeout           = errors.New("quantum execution timeout")
	ErrExecutionFailed            = errors.New("quantum execution failed")
	ErrInsufficientQubits         = errors.New("insufficient qubits available")

	// Algorithm errors
	ErrAlgorithmNotSupported      = errors.New("quantum algorithm not supported")
	ErrInvalidAlgorithmParameters = errors.New("invalid algorithm parameters")
	ErrAlgorithmFailed            = errors.New("algorithm execution failed")

	// QKD errors
	ErrQKDNotEnabled              = errors.New("QKD not enabled")
	ErrQKDProtocolFailed          = errors.New("QKD protocol failed")
	ErrInsufficientKeyMaterial    = errors.New("insufficient key material")
	ErrQKDAuthenticationFailed    = errors.New("QKD authentication failed")

	// Error correction errors
	ErrErrorCorrectionFailed      = errors.New("error correction failed")
	ErrSyndromeDecodingFailed     = errors.New("syndrome decoding failed")
	ErrTooManyErrors              = errors.New("too many errors to correct")

	// Cost errors
	ErrCostLimitExceeded          = errors.New("cost limit exceeded")
	ErrCostEstimationFailed       = errors.New("cost estimation failed")
)

// QuantumError represents a quantum-specific error with context
type QuantumError struct {
	Operation string
	Provider  string
	CircuitID string
	Cause     error
	Details   map[string]interface{}
}

func (qe *QuantumError) Error() string {
	if qe.Cause != nil {
		return fmt.Sprintf("quantum error in %s (provider: %s, circuit: %s): %v",
			qe.Operation, qe.Provider, qe.CircuitID, qe.Cause)
	}
	return fmt.Sprintf("quantum error in %s (provider: %s, circuit: %s)",
		qe.Operation, qe.Provider, qe.CircuitID)
}

func (qe *QuantumError) Unwrap() error {
	return qe.Cause
}

// NewQuantumError creates a new quantum error
func NewQuantumError(operation, provider, circuitID string, cause error) *QuantumError {
	return &QuantumError{
		Operation: operation,
		Provider:  provider,
		CircuitID: circuitID,
		Cause:     cause,
		Details:   make(map[string]interface{}),
	}
}

// WithDetail adds a detail to the quantum error
func (qe *QuantumError) WithDetail(key string, value interface{}) *QuantumError {
	qe.Details[key] = value
	return qe
}
