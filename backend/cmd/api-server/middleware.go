//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"net/http"
	"os"
	"runtime/debug"
	"strconv"

	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// defaultMaxBodyBytes caps request bodies at 1 MiB unless NOVACRON_MAX_BODY_BYTES
// overrides it. No canonical handler wrapped r.Body in a MaxBytesReader, so a
// single oversized POST could buffer unbounded memory; this bounds it centrally.
// ponytail: no >1MB body route exists today (create/restore use JSON+path params);
// bump the env var if a large-upload route is added.
const defaultMaxBodyBytes int64 = 1 << 20

// recoverMiddleware turns a handler panic into a clean HTTP 500 plus one
// centralized, stack-bearing log line. net/http already recovers per-connection,
// so the server does NOT crash without this; the narrow value is converting a
// silently-dropped connection into an observable 500 (for the client) and a
// single actionable log entry (for the operator).
func recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			rec := recover()
			if rec == nil {
				return
			}
			// http.ErrAbortHandler is net/http's intentional abort sentinel;
			// re-panic so the server handles it as designed (silent abort),
			// not as a logged 500.
			if rec == http.ErrAbortHandler {
				panic(rec)
			}
			logger.Error("panic recovered in HTTP handler",
				"method", r.Method,
				"path", r.URL.Path,
				"panic", rec,
				"stack", string(debug.Stack()),
			)
			writeJSONError(w, http.StatusInternalServerError, "internal server error")
		}()
		next.ServeHTTP(w, r)
	})
}

// maxBodyBytesMiddleware wraps each request body in an http.MaxBytesReader so a
// read past the limit fails cleanly (handlers surface it as 400/413) instead of
// buffering unbounded memory. Returned as the unnamed func type so it satisfies
// mux.MiddlewareFunc without importing mux here.
func maxBodyBytesMiddleware(limit int64) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Body != nil {
				r.Body = http.MaxBytesReader(w, r.Body, limit)
			}
			next.ServeHTTP(w, r)
		})
	}
}

// maxBodyBytes resolves the configured request-body cap, honoring
// NOVACRON_MAX_BODY_BYTES (bytes) when set to a positive integer.
func maxBodyBytes() int64 {
	if v := os.Getenv("NOVACRON_MAX_BODY_BYTES"); v != "" {
		if n, err := strconv.ParseInt(v, 10, 64); err == nil && n > 0 {
			return n
		}
	}
	return defaultMaxBodyBytes
}
