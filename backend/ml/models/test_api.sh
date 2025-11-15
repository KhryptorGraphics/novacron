#!/bin/bash

# Test Node Reliability Predictor API

echo "🧪 Testing Reliability Predictor API..."
echo ""

BASE_URL="http://localhost:5002"

# Health check
echo "1️⃣ Health Check:"
curl -s "${BASE_URL}/health" | python3 -m json.tool
echo ""

# Single prediction
echo "2️⃣ Single Prediction (High Reliability Node):"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "uptime": 99.9,
    "failure_rate": 0.01,
    "network_quality": 0.95,
    "distance": 50
  }' | python3 -m json.tool
echo ""

# Single prediction - Low reliability
echo "3️⃣ Single Prediction (Low Reliability Node):"
curl -s -X POST "${BASE_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "uptime": 60.0,
    "failure_rate": 5.0,
    "network_quality": 0.30,
    "distance": 8000
  }' | python3 -m json.tool
echo ""

# Batch prediction
echo "4️⃣ Batch Prediction (3 nodes):"
curl -s -X POST "${BASE_URL}/predict-batch" \
  -H "Content-Type: application/json" \
  -d '{
    "nodes": [
      {
        "uptime": 99.9,
        "failure_rate": 0.01,
        "network_quality": 0.95,
        "distance": 50
      },
      {
        "uptime": 85.0,
        "failure_rate": 0.5,
        "network_quality": 0.70,
        "distance": 2000
      },
      {
        "uptime": 60.0,
        "failure_rate": 5.0,
        "network_quality": 0.30,
        "distance": 8000
      }
    ]
  }' | python3 -m json.tool
echo ""

# Model info
echo "5️⃣ Model Information:"
curl -s "${BASE_URL}/model/info" | python3 -m json.tool | head -20
echo ""

echo "✅ API tests complete!"
