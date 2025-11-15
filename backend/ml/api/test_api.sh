#!/bin/bash
# Test script for Compression Selector API

BASE_URL="http://localhost:5000"

echo "=== Compression Selector API Tests ==="
echo ""

# Health check
echo "1. Health Check:"
curl -s "${BASE_URL}/health" | jq .
echo ""

# List algorithms
echo "2. List Algorithms:"
curl -s "${BASE_URL}/api/v1/compression/algorithms" | jq .
echo ""

# Model info
echo "3. Model Info:"
curl -s "${BASE_URL}/api/v1/model/info" | jq .
echo ""

# Single prediction - Large text file
echo "4. Predict: Large text file (50MB, 5s latency, 50Mbps)"
curl -s -X POST "${BASE_URL}/api/v1/compression/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "text",
    "size": 52428800,
    "latency": 5000,
    "bandwidth": 50
  }' | jq .
echo ""

# Single prediction - Real-time JSON
echo "5. Predict: Real-time JSON (10KB, 10ms latency, 1Gbps)"
curl -s -X POST "${BASE_URL}/api/v1/compression/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data_type": "json",
    "size": 10240,
    "latency": 10,
    "bandwidth": 1000
  }' | jq .
echo ""

# Batch prediction
echo "6. Batch Predictions:"
curl -s -X POST "${BASE_URL}/api/v1/compression/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {"data_type": "text", "size": 1048576, "latency": 100, "bandwidth": 100},
      {"data_type": "binary", "size": 10485760, "latency": 500, "bandwidth": 50},
      {"data_type": "json", "size": 102400, "latency": 10, "bandwidth": 1000}
    ]
  }' | jq .
echo ""

echo "=== Tests Complete ==="
