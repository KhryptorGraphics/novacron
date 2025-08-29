#!/bin/bash

# Test script for NovaCron API Server

API_BASE="http://localhost:8090"
TOKEN=""

echo "Testing NovaCron API Server..."
echo "================================"

# Test health check
echo "1. Testing health check..."
curl -s "$API_BASE/health" | jq '.'
echo ""

# Test API info
echo "2. Testing API info..."
curl -s "$API_BASE/api/info" | jq '.'
echo ""

# Test user registration
echo "3. Testing user registration..."
REGISTER_RESPONSE=$(curl -s -X POST "$API_BASE/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com", 
    "password": "testpass123"
  }')
echo "$REGISTER_RESPONSE" | jq '.'

# Check if registration was successful
if echo "$REGISTER_RESPONSE" | jq -e '.user.id' > /dev/null; then
  echo "✓ User registration successful"
else
  echo "✗ User registration failed"
fi
echo ""

# Test user login
echo "4. Testing user login..."
LOGIN_RESPONSE=$(curl -s -X POST "$API_BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }')
echo "$LOGIN_RESPONSE" | jq '.'

# Extract token for authenticated requests
TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.token // empty')

if [ ! -z "$TOKEN" ] && [ "$TOKEN" != "null" ]; then
  echo "✓ Login successful, token acquired"
  
  # Test authenticated endpoints
  echo ""
  echo "5. Testing authenticated VM list endpoint..."
  curl -s "$API_BASE/api/vms" \
    -H "Authorization: Bearer $TOKEN" | jq '.'
  
  echo ""
  echo "6. Testing VM creation..."
  CREATE_VM_RESPONSE=$(curl -s -X POST "$API_BASE/api/vms" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{
      "name": "test-vm-001",
      "node_id": "node-01"
    }')
  echo "$CREATE_VM_RESPONSE" | jq '.'
  
  echo ""
  echo "7. Testing monitoring metrics..."
  curl -s "$API_BASE/api/monitoring/metrics" \
    -H "Authorization: Bearer $TOKEN" | jq '.currentCpuUsage, .currentMemoryUsage'
  
  echo ""
  echo "8. Testing VM metrics..."
  curl -s "$API_BASE/api/monitoring/vms" \
    -H "Authorization: Bearer $TOKEN" | jq '.[0] // "No VMs found"'
  
else
  echo "✗ Login failed, cannot test authenticated endpoints"
fi

echo ""
echo "API testing completed!"