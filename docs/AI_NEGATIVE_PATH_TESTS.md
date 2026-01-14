# AI Optimization Negative Path Tests

## Overview

This document describes the comprehensive negative path tests added to `ai_optimization_test.go` for schema mismatch handling and error recovery.

## Test Coverage

### 1. Missing Required Fields (`MissingRequiredFields`)
- **Purpose**: Tests handling of AI responses missing critical fields like `predictions`, `confidence`
- **Behavior**: Returns proper error message about parsing failure
- **Circuit Breaker**: Verifies failure count increments
- **Expected Error**: `"failed to parse prediction response"`

### 2. Wrong Data Types (`WrongDataTypes`)
- **Purpose**: Tests handling of responses with incorrect data types
- **Scenarios**:
  - `predictions` as string instead of `[]float64`
  - `confidence` as string instead of `float64`
  - `accuracy` as string instead of `float64`
- **Expected Error**: `"failed to parse prediction response"`

### 3. Invalid JSON Response (`InvalidJSONResponse`)
- **Purpose**: Tests handling of malformed JSON responses
- **Scenarios**: Missing braces, unquoted keys, incomplete structures
- **Expected Error**: `"failed to parse AI response"`

### 4. Empty Response (`EmptyResponse`)
- **Purpose**: Tests handling of completely empty response bodies
- **Expected Error**: `"failed to parse AI response"`

### 5. Timeout Scenarios (`TimeoutScenario`)
- **Purpose**: Tests timeout handling and retry behavior
- **Configuration**: Short timeout (500ms), reduced retries
- **Expected Error**: `"HTTP request failed"`

### 6. AI Processing Failure (`AIProcessingFailure`)
- **Purpose**: Tests handling when AI service returns `success: false`
- **Scenario**: Model training failures, insufficient data quality
- **Expected Error**: `"AI processing failed"` with specific error message

### 7. HTTP Error Status Codes (`HTTPErrorStatusCodes`)
- **Test Cases**:
  - 400 Bad Request: Invalid request parameters
  - 401 Unauthorized: Authentication failed
  - 500 Internal Server Error: Server errors
  - 503 Service Unavailable: Temporary unavailability
- **Expected**: Status-specific error messages and circuit breaker activation

### 8. Circuit Breaker Behavior (`CircuitBreakerBehavior`)
- **Purpose**: Tests circuit breaker pattern implementation
- **Behavior**:
  - Tracks failure count across requests
  - Opens circuit after threshold exceeded
  - Returns `"circuit breaker is open"` error when open
- **Metrics**: Verifies state tracking and trip counts

### 9. Fallback Paths for Different Services (`FallbackPathsForDifferentServices`)
- **Services Tested**:
  - `PredictResourceDemand`
  - `OptimizePerformance`
  - `DetectAnomalies`
  - `AnalyzeWorkloadPattern`
- **Purpose**: Ensures all AI service types handle schema mismatches consistently

### 10. Context Cancellation (`ContextCancellation`)
- **Purpose**: Tests proper handling of cancelled contexts
- **Scenario**: Context timeout before server response
- **Expected**: Context cancellation error

### 11. Malformed Nested Structures (`MalformedNestedStructures`)
- **Purpose**: Tests deeply nested malformed data structures
- **Complex Scenarios**:
  - Wrong types in `model_info` nested object
  - Malformed `recommendations` array structures
  - Invalid `risk_assessment` object types
- **Target**: Performance optimization responses with complex nesting

## Key Assertions

### Error Handling
```go
assert.Error(t, err)
assert.Contains(t, err.Error(), "expected_error_substring")
```

### Circuit Breaker Validation
```go
metrics := aiLayer.GetMetrics()
assert.Greater(t, metrics["failed_requests"].(int64), int64(0))
assert.Equal(t, "open", metrics["circuit_breaker_state"].(string))
```

### Fallback Verification
- Tests verify that all AI service methods properly handle malformed responses
- Ensures consistent error handling across service types
- Validates metrics tracking for all failure scenarios

## Configuration Settings

### Circuit Breaker Configuration
```go
aiConfig.CircuitBreakerThreshold = 2-3  // Low threshold for testing
aiConfig.Retries = 1                    // Reduced for faster tests
```

### Timeout Configuration
```go
aiConfig.Timeout = 500 * time.Millisecond  // Short timeout for tests
```

## Benefits

1. **Robust Error Handling**: Ensures system gracefully handles all types of malformed AI responses
2. **Circuit Breaker Validation**: Prevents cascade failures when AI service is unhealthy
3. **Consistent Behavior**: All AI service methods handle errors uniformly
4. **Comprehensive Coverage**: Tests edge cases, timeouts, network errors, and malformed data
5. **Production Readiness**: Validates fallback paths work correctly under adverse conditions

## Usage

Run the tests with:
```bash
cd tests/integration
go test -run TestAINegativePathsAndSchemaMismatches -v
```

The tests provide comprehensive validation that the AI integration layer properly handles all failure scenarios and maintains system stability when the AI service returns malformed or invalid data.