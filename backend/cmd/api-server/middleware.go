//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"net"
	"net/http"
	"net/netip"
	"os"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"time"

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

// Login attempt limiting -----------------------------------------------------
//
// POST /api/auth/login has no other guard: each request costs a bcrypt
// comparison and a database round trip, and an unauthenticated caller can keep
// guessing passwords for as long as they like. The limiter below caps attempts
// per client IP before the handler runs.

const (
	// 10 attempts per 5 minutes caps one IP at ~2880 guesses/day — far below
	// what an offline cracking run needs, and far above what a user who
	// mistypes a password twice needs.
	defaultLoginRateLimit  = 10
	defaultLoginRateWindow = 5 * time.Minute
	// loginRateMaxClients bounds the limiter's memory: it is reachable by
	// unauthenticated callers, so it must not grow with the number of source
	// IPs an attacker can present.
	loginRateMaxClients = 4096
)

// loginRateLimiter is a per-client-IP sliding-window attempt limiter: it keeps
// the timestamp of every attempt still inside the window and rejects the next
// one once the window is full. Every attempt counts, successful logins
// included — the limiter sits in front of the handler and never sees the
// outcome. A rejected attempt is deliberately NOT recorded, so a client that
// keeps hammering cannot extend its own block beyond one window (which a
// counter-with-lockout would let it do indefinitely).
type loginRateLimiter struct {
	limit  int
	window time.Duration
	// trustedProxies are the peers whose X-Forwarded-For/X-Real-IP is believed.
	trustedProxies []netip.Prefix
	// now is a field so tests can drive the window without sleeping.
	now func() time.Time

	mu   sync.Mutex
	hits map[string][]time.Time
}

// newLoginRateLimiter returns nil — the limiter disabled — when limit or window
// is non-positive (NOVACRON_LOGIN_RATE_LIMIT=0).
func newLoginRateLimiter(limit int, window time.Duration) *loginRateLimiter {
	if limit <= 0 || window <= 0 {
		return nil
	}
	return &loginRateLimiter{
		limit:  limit,
		window: window,
		now:    time.Now,
		hits:   make(map[string][]time.Time),
	}
}

// newLoginRateLimiterFromEnv resolves NOVACRON_LOGIN_RATE_LIMIT (attempts per
// window, 0 disables) and NOVACRON_LOGIN_RATE_WINDOW_S (seconds), falling back
// to the defaults above for unset or unparsable values, plus
// NOVACRON_TRUSTED_PROXIES (see parseTrustedProxies).
func newLoginRateLimiterFromEnv() *loginRateLimiter {
	limit := defaultLoginRateLimit
	if v := os.Getenv("NOVACRON_LOGIN_RATE_LIMIT"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			limit = n
		}
	}

	window := defaultLoginRateWindow
	if v := os.Getenv("NOVACRON_LOGIN_RATE_WINDOW_S"); v != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && n > 0 {
			window = time.Duration(n) * time.Second
		}
	}

	limiter := newLoginRateLimiter(limit, window)
	if limiter != nil {
		limiter.trustedProxies = parseTrustedProxies(os.Getenv("NOVACRON_TRUSTED_PROXIES"))
	}
	return limiter
}

// loginRateLimitMiddleware enforces the limiter in front of the login handler,
// so an over-limit attempt never reaches the bcrypt comparison or the database.
// A nil limiter is a no-op.
func loginRateLimitMiddleware(limiter *loginRateLimiter) func(http.Handler) http.Handler {
	if limiter == nil {
		return func(next http.Handler) http.Handler { return next }
	}
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			allowed, retryAfter := limiter.allow(limiter.clientIP(r))
			if !allowed {
				// Retry-After is the documented 429 signal; rounding up keeps a
				// client from retrying inside the window it was rejected in.
				w.Header().Set("Retry-After", strconv.Itoa(int((retryAfter+time.Second-1)/time.Second)))
				writeJSONError(w, http.StatusTooManyRequests, "too many login attempts; try again later")
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// allow records one attempt for key and reports whether it is within the limit;
// when it is not, retryAfter is how long until the oldest recorded attempt
// leaves the window (at least one second, the granularity of Retry-After).
func (l *loginRateLimiter) allow(key string) (bool, time.Duration) {
	now := l.now()

	l.mu.Lock()
	defer l.mu.Unlock()

	kept := pruneOlderThan(l.hits[key], now.Add(-l.window))
	if len(kept) >= l.limit {
		l.hits[key] = kept
		if retryAfter := kept[0].Add(l.window).Sub(now); retryAfter > time.Second {
			return false, retryAfter
		}
		return false, time.Second
	}

	if len(kept) == 0 {
		l.makeRoomLocked(now)
	}
	l.hits[key] = append(kept, now)

	return true, 0
}

// makeRoomLocked keeps the tracked-IP map bounded at loginRateMaxClients: once
// the cap is reached it drops every entry whose attempts have all expired, and
// if that freed nothing it drops the least recently active entry, so a caller
// with many source addresses cannot grow this map without bound.
func (l *loginRateLimiter) makeRoomLocked(now time.Time) {
	if len(l.hits) < loginRateMaxClients {
		return
	}

	cutoff := now.Add(-l.window)
	oldestKey := ""
	var oldest time.Time
	for key, hits := range l.hits {
		kept := pruneOlderThan(hits, cutoff)
		if len(kept) == 0 {
			delete(l.hits, key)
			continue
		}
		l.hits[key] = kept
		if last := kept[len(kept)-1]; oldestKey == "" || last.Before(oldest) {
			oldestKey, oldest = key, last
		}
	}

	if len(l.hits) >= loginRateMaxClients && oldestKey != "" {
		delete(l.hits, oldestKey)
	}
}

// clientIP is the limiter's bucket key. X-Forwarded-For/X-Real-IP are honored
// only when the immediate peer is listed in NOVACRON_TRUSTED_PROXIES: those
// headers are client-supplied, so believing them by default would let one
// caller mint a new bucket per request and bypass the limit entirely.
func (l *loginRateLimiter) clientIP(r *http.Request) string {
	remote := hostOf(r.RemoteAddr)
	if remote != "" && l.isTrustedProxy(remote) {
		if forwarded := r.Header.Get("X-Forwarded-For"); forwarded != "" {
			if first := strings.TrimSpace(strings.Split(forwarded, ",")[0]); first != "" {
				return first
			}
		}
		if realIP := strings.TrimSpace(r.Header.Get("X-Real-IP")); realIP != "" {
			return realIP
		}
	}
	if remote != "" {
		return remote
	}
	return "unknown"
}

func (l *loginRateLimiter) isTrustedProxy(remote string) bool {
	addr, err := netip.ParseAddr(remote)
	if err != nil {
		return false
	}
	for _, prefix := range l.trustedProxies {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}

// parseTrustedProxies parses NOVACRON_TRUSTED_PROXIES — a comma-separated list
// of IPs or CIDR blocks. Entries that do not parse are ignored, so a typo
// narrows trust rather than widening it.
func parseTrustedProxies(raw string) []netip.Prefix {
	var proxies []netip.Prefix
	for _, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		if prefix, err := netip.ParsePrefix(entry); err == nil {
			proxies = append(proxies, prefix.Masked())
			continue
		}
		if addr, err := netip.ParseAddr(entry); err == nil {
			proxies = append(proxies, netip.PrefixFrom(addr, addr.BitLen()))
		}
	}
	return proxies
}

// hostOf strips the port from a RemoteAddr, tolerating the portless forms
// (bare IP) that tests and unix-socket peers produce.
func hostOf(remoteAddr string) string {
	if host, _, err := net.SplitHostPort(remoteAddr); err == nil {
		return host
	}
	return strings.TrimSpace(remoteAddr)
}

// pruneOlderThan drops attempts at or before cutoff. hits is append-ordered, so
// the survivors are a suffix and the backing array can be reused.
func pruneOlderThan(hits []time.Time, cutoff time.Time) []time.Time {
	i := 0
	for i < len(hits) && !hits[i].After(cutoff) {
		i++
	}
	return hits[i:]
}
