#!/bin/bash
set -e

echo "Running AI Negative Path and Schema Mismatch Tests..."
echo "=================================================="

# Navigate to integration tests directory
cd "$(dirname "$0")/../tests/integration"

# Download missing dependencies if needed
echo "Downloading dependencies..."
go mod download

# Run the specific negative path tests
echo "Running TestAINegativePathsAndSchemaMismatches..."
go test -run TestAINegativePathsAndSchemaMismatches -v -timeout 300s

echo ""
echo "✅ AI negative path tests completed successfully!"
echo ""
echo "Test Coverage Summary:"
echo "- Missing required fields"
echo "- Wrong data types"
echo "- Invalid JSON responses"
echo "- Empty responses"
echo "- Timeout scenarios"
echo "- AI processing failures"
echo "- HTTP error status codes"
echo "- Circuit breaker behavior"
echo "- Fallback paths for all AI services"
echo "- Context cancellation"
echo "- Malformed nested structures"
echo ""
echo "All error handling and schema validation tests passed!"