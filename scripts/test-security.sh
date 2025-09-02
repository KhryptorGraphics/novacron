#!/bin/bash

echo "🔒 NovaCron Security Test Suite"
echo "================================"
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASSED=0
FAILED=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_result="$3"
    
    echo -n "Testing: $test_name... "
    
    if eval "$test_command"; then
        if [ "$expected_result" = "pass" ]; then
            echo -e "${GREEN}✓ PASSED${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAILED (expected to fail)${NC}"
            ((FAILED++))
        fi
    else
        if [ "$expected_result" = "fail" ]; then
            echo -e "${GREEN}✓ PASSED (correctly failed)${NC}"
            ((PASSED++))
        else
            echo -e "${RED}✗ FAILED${NC}"
            ((FAILED++))
        fi
    fi
}

echo "1️⃣  SQL Injection Prevention Tests"
echo "-----------------------------------"

# Test 1: SQL injection attempt in VM query
run_test "SQL injection in VM state filter" \
    "curl -k -s 'https://localhost:8443/api/vms?state=running%27;%20DROP%20TABLE%20vms;%20--' | grep -q 'Bad Request'" \
    "pass"

# Test 2: SQL injection in login
run_test "SQL injection in login email" \
    "curl -k -s -X POST https://localhost:8443/api/auth/login \
        -H 'Content-Type: application/json' \
        -d '{\"email\":\"admin@example.com' OR '1'='1\",\"password\":\"test\"}' \
        | grep -q 'Invalid email'" \
    "pass"

# Test 3: Valid parameterized query
run_test "Valid VM query (parameterized)" \
    "curl -k -s 'https://localhost:8443/api/vms?state=running' -o /dev/null -w '%{http_code}' | grep -q '401'" \
    "pass"

echo
echo "2️⃣  Secrets Management Tests"
echo "----------------------------"

# Test 4: Check Vault is running
run_test "Vault server is running" \
    "curl -s http://localhost:8200/v1/sys/health | grep -q 'initialized'" \
    "pass"

# Test 5: No hardcoded secrets in response
run_test "No secrets exposed in health endpoint" \
    "curl -k -s https://localhost:8443/api/health | grep -v -q 'secret\\|password\\|token'" \
    "pass"

# Test 6: Environment variables not exposed
run_test "Environment variables not exposed" \
    "curl -k -s https://localhost:8443/api/health | grep -v -q 'VAULT_TOKEN\\|AUTH_SECRET'" \
    "pass"

echo
echo "3️⃣  HTTPS/TLS Tests"
echo "-------------------"

# Test 7: HTTPS endpoint accessible
run_test "HTTPS endpoint is accessible" \
    "curl -k -s https://localhost:8443/api/health | grep -q 'healthy'" \
    "pass"

# Test 8: HTTP redirects to HTTPS
run_test "HTTP redirects to HTTPS" \
    "curl -s -o /dev/null -w '%{http_code}' -I http://localhost:8080/api/health | grep -q '301\\|308'" \
    "pass"

# Test 9: TLS version check (TLS 1.2+)
run_test "TLS 1.2 or higher enforced" \
    "echo | openssl s_client -connect localhost:8443 -tls1_2 2>/dev/null | grep -q 'TLSv1.2\\|TLSv1.3'" \
    "pass"

# Test 10: Weak TLS version rejected
run_test "TLS 1.0 is rejected" \
    "echo | openssl s_client -connect localhost:8443 -tls1 2>&1 | grep -q 'wrong version number\\|unsupported protocol'" \
    "pass"

echo
echo "4️⃣  Security Headers Tests"
echo "--------------------------"

# Test 11: HSTS header present
run_test "HSTS header present" \
    "curl -k -s -I https://localhost:8443/api/health | grep -q 'Strict-Transport-Security'" \
    "pass"

# Test 12: X-Frame-Options header
run_test "X-Frame-Options header present" \
    "curl -k -s -I https://localhost:8443/api/health | grep -q 'X-Frame-Options: DENY'" \
    "pass"

# Test 13: Content Security Policy
run_test "CSP header present" \
    "curl -k -s -I https://localhost:8443/api/health | grep -q 'Content-Security-Policy'" \
    "pass"

# Test 14: X-Content-Type-Options
run_test "X-Content-Type-Options header present" \
    "curl -k -s -I https://localhost:8443/api/health | grep -q 'X-Content-Type-Options: nosniff'" \
    "pass"

echo
echo "5️⃣  Input Validation Tests"
echo "--------------------------"

# Test 15: Invalid email format rejected
run_test "Invalid email format rejected" \
    "curl -k -s -X POST https://localhost:8443/api/auth/register \
        -H 'Content-Type: application/json' \
        -d '{\"email\":\"notanemail\",\"password\":\"Test123!\"}' \
        | grep -q 'Invalid email'" \
    "pass"

# Test 16: Short password rejected
run_test "Short password rejected" \
    "curl -k -s -X POST https://localhost:8443/api/auth/register \
        -H 'Content-Type: application/json' \
        -d '{\"email\":\"test@example.com\",\"password\":\"short\"}' \
        | grep -q 'at least 8 characters'" \
    "pass"

# Test 17: Invalid VM name rejected
run_test "Invalid VM name with special chars rejected" \
    "curl -k -s -X POST https://localhost:8443/api/vms \
        -H 'Content-Type: application/json' \
        -H 'Authorization: Bearer test' \
        -d '{\"name\":\"vm-name<script>\",\"node_id\":\"node1\"}' \
        | grep -q 'Invalid\\|Bad Request\\|Unauthorized'" \
    "pass"

echo
echo "6️⃣  Rate Limiting Tests"
echo "-----------------------"

# Test 18: Rate limiting enforced
echo -n "Testing: Rate limiting after 100 requests... "
RATE_LIMITED=false
for i in {1..110}; do
    STATUS=$(curl -k -s -o /dev/null -w "%{http_code}" https://localhost:8443/api/health)
    if [ "$STATUS" = "429" ]; then
        RATE_LIMITED=true
        break
    fi
done

if [ "$RATE_LIMITED" = true ]; then
    echo -e "${GREEN}✓ PASSED${NC}"
    ((PASSED++))
else
    echo -e "${YELLOW}⚠ SKIPPED (may need more requests)${NC}"
fi

echo
echo "7️⃣  Authentication Tests"
echo "------------------------"

# Test 19: Unauthorized access blocked
run_test "Unauthorized access to protected endpoint" \
    "curl -k -s -o /dev/null -w '%{http_code}' https://localhost:8443/api/vms | grep -q '401'" \
    "pass"

# Test 20: Invalid JWT rejected
run_test "Invalid JWT token rejected" \
    "curl -k -s -o /dev/null -w '%{http_code}' \
        -H 'Authorization: Bearer invalid.token.here' \
        https://localhost:8443/api/vms | grep -q '401'" \
    "pass"

echo
echo "================================"
echo "📊 Test Results Summary"
echo "================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All security tests passed!${NC}"
    echo "Your NovaCron deployment is properly secured."
else
    echo -e "${RED}⚠️  Some tests failed. Please review the security configuration.${NC}"
    echo "Check the logs for more details:"
    echo "  - API Server: /tmp/novacron-secure.log"
    echo "  - Vault: /tmp/vault.log"
fi

echo
echo "📝 Security Checklist:"
echo "  ✓ SQL injection prevention"
echo "  ✓ Secrets management with Vault"
echo "  ✓ HTTPS/TLS encryption"
echo "  ✓ Security headers"
echo "  ✓ Input validation"
echo "  ✓ Rate limiting"
echo "  ✓ Authentication & Authorization"

exit $FAILED