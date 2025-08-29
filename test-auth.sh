#!/bin/bash

# Test authentication endpoints for NovaCron

set -e

API_BASE="http://localhost:8090"
echo "Testing NovaCron Authentication System"
echo "======================================"

# Test 1: Register a new user
echo "1. Testing user registration..."
REGISTER_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser", 
    "email": "test@example.com", 
    "password": "testpass123",
    "tenant_id": "default"
  }' \
  ${API_BASE}/auth/register)

echo "Registration response: $REGISTER_RESPONSE"

# Test 2: Login with the user
echo -e "\n2. Testing user login..."
LOGIN_RESPONSE=$(curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser", 
    "password": "testpass123"
  }' \
  ${API_BASE}/auth/login)

echo "Login response: $LOGIN_RESPONSE"

# Extract token from login response
TOKEN=$(echo $LOGIN_RESPONSE | grep -o '"token":"[^"]*' | cut -d'"' -f4)
echo "Extracted token: ${TOKEN:0:20}..."

if [ -z "$TOKEN" ]; then
  echo "ERROR: No token received from login"
  exit 1
fi

# Test 3: Validate token
echo -e "\n3. Testing token validation..."
VALIDATE_RESPONSE=$(curl -s -X GET \
  -H "Authorization: Bearer $TOKEN" \
  ${API_BASE}/auth/validate)

echo "Token validation response: $VALIDATE_RESPONSE"

# Test 4: Access protected endpoint
echo -e "\n4. Testing protected endpoint access..."
PROTECTED_RESPONSE=$(curl -s -X GET \
  -H "Authorization: Bearer $TOKEN" \
  ${API_BASE}/api/info)

echo "Protected endpoint response: $PROTECTED_RESPONSE"

# Test 5: Test without token (should fail)
echo -e "\n5. Testing protected endpoint without token..."
UNAUTH_RESPONSE=$(curl -s -X GET ${API_BASE}/api/info)
echo "Unauthorized response: $UNAUTH_RESPONSE"

# Test 6: Logout
echo -e "\n6. Testing logout..."
LOGOUT_RESPONSE=$(curl -s -X POST \
  -H "Authorization: Bearer $TOKEN" \
  ${API_BASE}/auth/logout)

echo "Logout response: $LOGOUT_RESPONSE"

echo -e "\n======================================"
echo "Authentication test completed!"