package main

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// TestLoginRateLimiterClientIPFallsBackToUnknownForUnparsableAddress covers
// the one case netutil's own suite does not: clientIP's rate-limiter-only
// "unknown" bucket fallback (netutil.ClientIPString returns "" for this
// input, not "unknown" — that translation is loginRateLimiter's).
func TestLoginRateLimiterClientIPFallsBackToUnknownForUnparsableAddress(t *testing.T) {
	limiter := newLoginRateLimiter(1, time.Minute)
	req := httptest.NewRequest(http.MethodPost, "/api/auth/login", nil)
	req.RemoteAddr = "not-an-address"
	if got := limiter.clientIP(req); got != "unknown" {
		t.Fatalf("clientIP = %q, want \"unknown\" fallback", got)
	}
}
