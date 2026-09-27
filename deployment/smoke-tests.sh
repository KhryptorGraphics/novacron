#!/usr/bin/env bash
# Post-deploy smoke tests for NovaCron.
#
# Probes only routes the canonical api-server (backend/cmd/api-server)
# actually mounts: GET /health, GET /api/info, and POST /api/auth/login with
# bogus credentials, which must be rejected with 401 (never 200). The
# frontend root (GET /) is probed when FRONTEND_URL is non-empty.
#
#   API_URL          base URL of the api-server (default https://novacron.io)
#   FRONTEND_URL     base URL of the frontend; "" skips the frontend probe
#   SMOKE_TEST_TIMEOUT   per-request timeout in seconds (default 10)
#   SMOKE_TEST_RETRIES   attempts per probe (default 3)
#
# Exits non-zero if any probe fails after its retries.
set -uo pipefail

API_URL="${API_URL:-https://novacron.io}"
FRONTEND_URL="${FRONTEND_URL-https://novacron.io}"
TIMEOUT="${SMOKE_TEST_TIMEOUT:-10}"
RETRIES="${SMOKE_TEST_RETRIES:-3}"

API_URL="${API_URL%/}"
FRONTEND_URL="${FRONTEND_URL%/}"

fail=0

# check <description> <url> <expected status> [method] [json body]
check() {
  local desc="$1" url="$2" want="$3" method="${4:-GET}" data="${5:-}"
  local attempt code
  for attempt in $(seq 1 "$RETRIES"); do
    if [ -n "$data" ]; then
      code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$TIMEOUT" \
        -X "$method" -H 'Content-Type: application/json' -d "$data" "$url" 2>/dev/null || echo "000")
    else
      code=$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$TIMEOUT" \
        -X "$method" "$url" 2>/dev/null || echo "000")
    fi
    if [ "$code" = "$want" ]; then
      echo "OK   $desc -> $code"
      return 0
    fi
    echo "retry $attempt/$RETRIES: $desc -> $code (want $want)"
    [ "$attempt" -lt "$RETRIES" ] && sleep 2
  done
  echo "FAIL $desc -> $code (want $want)"
  fail=1
  return 1
}

check "API health"   "$API_URL/health"   "200"
check "API info"     "$API_URL/api/info" "200"
check "Login rejects bad credentials" "$API_URL/api/auth/login" "401" "POST" \
  '{"username":"smoke-test-nonexistent","password":"definitely-wrong"}'

if [ -n "$FRONTEND_URL" ]; then
  check "Frontend root" "$FRONTEND_URL/" "200"
fi

if [ "$fail" -ne 0 ]; then
  echo "Smoke tests FAILED"
  exit 1
fi
echo "Smoke tests passed"
