package main

import (
	"database/sql"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/gorilla/mux"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
)

// buildHardenedTestRouter mirrors buildCanonicalServer's cross-cutting wiring:
// recover + body-cap on the ROOT router, and a POST route under an /api
// SUBROUTER so we prove the cap propagates to subrouter routes (the real POST
// attack surface). Uses a small cap so tiny bodies exercise the limit without
// the "server responds mid-upload -> client EPIPE" race a 1 MiB body can hit.
func buildHardenedTestRouter(limit int64) http.Handler {
	router := mux.NewRouter()
	router.Use(recoverMiddleware, maxBodyBytesMiddleware(limit))

	router.HandleFunc("/boom", func(w http.ResponseWriter, r *http.Request) {
		panic("intentional test panic")
	}).Methods(http.MethodGet)

	api := router.PathPrefix("/api").Subrouter()
	api.HandleFunc("/echo", func(w http.ResponseWriter, r *http.Request) {
		_, err := io.ReadAll(r.Body)
		var maxErr *http.MaxBytesError
		if errors.As(err, &maxErr) {
			writeJSONError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		w.WriteHeader(http.StatusOK)
	}).Methods(http.MethodPost)

	return router
}

// TestRecoverMiddlewareReturns500 discriminates on "client receives 500": with
// no recovery middleware, net/http's per-conn recover drops the connection and
// http.Get returns an error (not a 500). Only the middleware yields a real 500.
func TestRecoverMiddlewareReturns500(t *testing.T) {
	ts := httptest.NewServer(buildHardenedTestRouter(64))
	defer ts.Close()

	resp, err := http.Get(ts.URL + "/boom")
	if err != nil {
		t.Fatalf("panic route dropped the connection (recovery missing?): %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusInternalServerError {
		t.Fatalf("panic route: got HTTP %d, want 500", resp.StatusCode)
	}
}

// TestMaxBodyBytesMiddleware discriminates the cap through a subrouter: under
// cap -> 200, over cap -> 413. Remove the middleware and the over-cap case
// returns 200 instead, failing this test.
func TestMaxBodyBytesMiddleware(t *testing.T) {
	const limit = 64
	ts := httptest.NewServer(buildHardenedTestRouter(limit))
	defer ts.Close()

	under, err := http.Post(ts.URL+"/api/echo", "application/octet-stream", strings.NewReader(strings.Repeat("a", 10)))
	if err != nil {
		t.Fatalf("under-cap POST failed: %v", err)
	}
	under.Body.Close()
	if under.StatusCode != http.StatusOK {
		t.Fatalf("under-cap body: got HTTP %d, want 200", under.StatusCode)
	}

	over, err := http.Post(ts.URL+"/api/echo", "application/octet-stream", strings.NewReader(strings.Repeat("a", limit*4)))
	if err != nil {
		t.Fatalf("over-cap POST failed: %v", err)
	}
	over.Body.Close()
	if over.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("over-cap body: got HTTP %d, want 413", over.StatusCode)
	}
}

// TestHardeningViaCurl exercises the shipped middleware with the literal curl
// binary over real TCP, so the report carries real curl status codes.
func TestHardeningViaCurl(t *testing.T) {
	if _, err := exec.LookPath("curl"); err != nil {
		t.Skip("curl not available")
	}
	const limit = 64
	ts := httptest.NewServer(buildHardenedTestRouter(limit))
	defer ts.Close()

	code := curlCode(t, ts.URL+"/boom")
	t.Logf("curl GET /boom -> HTTP %s", code)
	if code != "500" {
		t.Fatalf("curl /boom: got %s, want 500", code)
	}

	payload := filepath.Join(t.TempDir(), "over.bin")
	if err := os.WriteFile(payload, make([]byte, limit*4), 0o600); err != nil {
		t.Fatal(err)
	}
	code = curlCode(t, "-X", "POST", "--data-binary", "@"+payload, ts.URL+"/api/echo")
	t.Logf("curl POST /api/echo (over cap) -> HTTP %s", code)
	if code != "413" {
		t.Fatalf("curl over-cap POST: got %s, want 413", code)
	}
}

// TestBuildCanonicalServerWiring guards the actual deliverable: the hardening
// wired INSIDE buildCanonicalServer (not a mirror of it). It builds the real
// server with minimal, DB-free deps and asserts on its output. Fails if the
// MaxHeaderBytes field or the body-cap middleware wiring is removed.
func TestBuildCanonicalServerWiring(t *testing.T) {
	// Tiny cap so a small valid-JSON body trips it deterministically (no
	// mid-upload close race), while still exercising the real wiring.
	t.Setenv("NOVACRON_MAX_BODY_BYTES", "64")

	cfg := &config.Config{}
	cfg.VM.StoragePath = t.TempDir()
	cfg.Server.APIPort = "0"

	// Unconnected handle: never pinged; the login DB lookup fails fast (-> 401).
	db, err := sql.Open("postgres", "postgres://127.0.0.1:1/none?sslmode=disable")
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager("test-secret", db)
	services, err := initializeCanonicalServices(cfg, db, authManager)
	if err != nil {
		t.Fatalf("initializeCanonicalServices: %v", err)
	}
	defer services.shutdown()

	srv := buildCanonicalServer(cfg, db, authManager, services, nil)

	// Bar 3 on the real *http.Server: fails if the MaxHeaderBytes line is removed.
	if srv.MaxHeaderBytes != 64<<10 {
		t.Fatalf("buildCanonicalServer MaxHeaderBytes: got %d, want %d", srv.MaxHeaderBytes, 64<<10)
	}

	// Bar 2 on the real router: oversized VALID JSON to /auth/login. With the cap
	// wired, MaxBytesReader trips json.Decode -> 400; without it, Decode succeeds
	// and the (failing) DB lookup -> 401. The 400-vs-401 delta discriminates the
	// cap wiring inside buildCanonicalServer.
	ts := httptest.NewServer(srv.Handler)
	defer ts.Close()

	body := `{"username":"` + strings.Repeat("A", 100) + `","password":"x"}`
	resp, err := http.Post(ts.URL+"/auth/login", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("login POST: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("oversized /auth/login body: got HTTP %d, want 400 (body cap not wired?)", resp.StatusCode)
	}
}

func curlCode(t *testing.T, args ...string) string {
	t.Helper()
	full := append([]string{"-s", "-o", "/dev/null", "-w", "%{http_code}"}, args...)
	out, err := exec.Command("curl", full...).Output()
	if err != nil {
		t.Fatalf("curl failed: %v", err)
	}
	return strings.TrimSpace(string(out))
}
