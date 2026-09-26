//go:build !novacron_enhanced && !novacron_improved && !novacron_multicloud && !novacron_production && !novacron_real_backend && !novacron_secure && !novacron_working && !novacron_simple_api

package main

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	"github.com/sirupsen/logrus"

	"github.com/golang-jwt/jwt/v5"
	"github.com/google/uuid"
	"github.com/jmoiron/sqlx"
	graphqlapi "github.com/khryptorgraphics/novacron/backend/api/graphql"
	orchestrationapi "github.com/khryptorgraphics/novacron/backend/api/orchestration"
	securityapi "github.com/khryptorgraphics/novacron/backend/api/security"
	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/autoscaling"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/healing"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/placement"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/policy"
	"github.com/khryptorgraphics/novacron/backend/core/storage"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	"github.com/khryptorgraphics/novacron/backend/pkg/database"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
	monitoring_svc "github.com/khryptorgraphics/novacron/backend/pkg/services"
	_ "github.com/lib/pq"
	"golang.org/x/crypto/bcrypt"
)

type canonicalServices struct {
	twoFactorService    *auth.TwoFactorService
	securityHandlers    *securityapi.SecurityHandlers
	websocketHandler    *websocketapi.WebSocketHandler
	graphqlHandler      http.Handler
	orchestrationAPI    *orchestrationapi.OrchestrationAPI
	orchestrationEngine *orchestration.DefaultOrchestrationEngine
	shutdown            func()
}

// restartSupervisor is package-level because route registration happens inside
// registerSecureAPIRoutes (a separate function called by main) — handlers
// check this pointer (never nil from within a running server).
var restartSupervisor *core_vm.RestartSupervisor

func main() {
	cfg, err := config.Load()
	if err != nil {
		logger.Fatal("Failed to load configuration", "error", err)
	}

	if err := cfg.Validate(); err != nil {
		logger.Fatal("Invalid configuration", "error", err)
	}

	appLogger := logger.NewFromConfig(
		cfg.Logging.Level,
		cfg.Logging.Format,
		cfg.Logging.Output,
		cfg.Logging.Structured,
	)
	logger.SetGlobalLogger(appLogger)

	appLogger.Info("Starting NovaCron API Server...",
		"version", "1.0.0",
		"api_port", cfg.Server.APIPort,
		"ws_port", cfg.Server.WSPort,
	)

	db, err := initDatabase(cfg)
	if err != nil {
		appLogger.Fatal("Failed to initialize database", "error", err)
	}
	defer db.Close()

	authManager := auth.NewSimpleAuthManager(cfg.Auth.Secret, db)
	services, err := initializeCanonicalServices(cfg, db, authManager)
	if err != nil {
		appLogger.Fatal("Failed to initialize canonical backend services", "error", err)
	}
	defer services.shutdown()

	vmManager := newVMManager(cfg)

	// Start the VM manager loops so updateLoop/cleanupLoop run
	if vmManager != nil {
		if err := vmManager.Start(); err != nil {
			appLogger.Warn("Failed to start VM manager", "error", err)
		} else {
			defer func() { _ = vmManager.Stop() }()
		}
	}

	// Wire the VM metrics collector against the canonical manager: samples are
	// inserted into vm_metrics on a 30s tick so /vms/{id}/metrics and
	// /monitoring/vms stop reporting empty results. DeferStop until server
	// shutdown. The canonical path uses *sql.DB; MonitoringService needs the
	// sqlx-backed wrapper, so bridge via sqlx.NewDb on the existing handle.
	sqlxDB := sqlx.NewDb(db, "postgres")
	dbWrapper := &database.DB{DB: sqlxDB}
	metricsSvc := monitoring_svc.NewMonitoringService(dbWrapper, nil, vmManager)
	if err := metricsSvc.Start(); err != nil {
		appLogger.Warn("Failed to start metrics collector", "error", err)
	} else {
		defer func() {
			_ = metricsSvc.Stop()
		}()
	}

	// PATH-3: watch for actions that crash the process and restart them per
	// tenant policy (default on_failure). Supervisor polls at 5s; a failed VM
	// is restared from its recorded config (see 000012_vm_restart_state).
	restartSupervisor = core_vm.NewRestartSupervisor(vmManager, db, 5*time.Second)
	restartSupervisor.Start(context.Background())

	// GATE-1: reconcile persisted VM rows against actually-running qemu processes
	// (pidfile rediscovery) so a restart reflects reality, not just stale rows.
	reconcileVMState(db, vmBasePath(cfg), vmManager)
	// GATE-1b: a migration destination whose registerMigratedDest goroutine
	// died with a prior api-server process (killed/restarted before the
	// guest resumed, or between resuming and that goroutine running) has a
	// live qemu and a config.json but NO vms row at all -- reconcileVMState
	// above never sees it (its query only covers rows that already exist).
	// Must run after reconcileVMState, whose adoptManagerVM calls already
	// cover every id that DOES have a row (novacron-05h).
	reconcileOrphanedMigrationDests(db, vmBasePath(cfg), vmManager)

	// Register migration peers from NOVACRON_PEERS so a migrate request can resolve
	// a bare target_node to its address without the caller passing target_addr.
	registerConfiguredPeers(vmManager)

	// Fabric membership (P1/G1): persisted peers (signed-join protocol) load
	// after the static env map, and a heartbeat loop keeps link profiles
	// fresh until shutdown.
	loadPersistedPeers(vmManager, db)
	fabricCtx, fabricCancel := context.WithCancel(context.Background())
	defer fabricCancel()
	go clusterHeartbeatLoop(fabricCtx, db, vmManager)

	server := buildCanonicalServer(cfg, db, authManager, services, vmManager)

	go func() {
		appLogger.Info("API Server starting", "port", cfg.Server.APIPort)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			appLogger.Fatal("Server failed to start", "error", err)
		}
	}()

	// Join AFTER the listener is up: the seed's mandatory reachability probe
	// calls this node back at NOVACRON_JOIN_ADDR, so a pre-listen join is
	// always rejected 403 (connection refused on the callback). A dial probe
	// of our own addr is belt-and-braces for slow listener starts.
	joinDeadline := time.Now().Add(10 * time.Second)
	for {
		if probe, err := net.DialTimeout("tcp", selfJoinAddr(), time.Second); err == nil {
			probe.Close()
			break
		} else if time.Now().After(joinDeadline) {
			appLogger.Warn("fabric join skipped: own listener not ready", "addr", selfJoinAddr())
			break
		}
		time.Sleep(200 * time.Millisecond)
	}
	joinAtBoot(fabricCtx, selfNodeID(), vmManager, db)

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	if restartSupervisor != nil {
		restartSupervisor.Stop() // stop watcher cleanly before the manager itself goes down
	}

	appLogger.Info("Shutting down server...")

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Server.ShutdownTimeout)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		appLogger.Fatal("Server forced to shutdown", "error", err)
	}

	appLogger.Info("Server exited gracefully")
}

func buildCanonicalServer(cfg *config.Config, db *sql.DB, authManager *auth.SimpleAuthManager, services *canonicalServices, vmManager *core_vm.VMManager) *http.Server {
	router := mux.NewRouter()
	router.StrictSlash(true)

	// Cross-cutting hardening applied on the root router (gorilla/mux propagates
	// parent middleware to subrouter routes, parent-outermost): recover panics
	// into a clean 500 first, then cap request-body size. Auth/RBAC live on the
	// /api subrouters and run inside these.
	router.Use(recoverMiddleware, maxBodyBytesMiddleware(maxBodyBytes()))

	// Email delivery is optional: cfg.Email.SMTPHost == "" disables it and the
	// auth token routes fail closed with 503 before touching the database.
	var emailService *auth.EmailService
	if cfg.Email.SMTPHost != "" {
		emailService = auth.NewEmailService(auth.EmailConfig{
			SMTPHost:    cfg.Email.SMTPHost,
			SMTPPort:    cfg.Email.SMTPPort,
			Username:    cfg.Email.SMTPUsername,
			Password:    cfg.Email.SMTPPassword,
			FromAddress: cfg.Email.SMTPFromAddress,
			FromName:    cfg.Email.SMTPFromName,
			UseTLS:      cfg.Email.SMTPUseTLS,
			UseSSL:      cfg.Email.SMTPUseSSL,
			FrontendURL: cfg.Email.FrontendURL,
		})
	}

	corsHandler := buildCORSHandler(cfg)

	registerPublicRoutes(router, authManager, db, services.twoFactorService, emailService)

	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.Use(requireAuth(authManager, db))
	registerSecureAPIRoutes(apiRouter, db, vmManager, vmBasePath(cfg))

	apiV1Router := router.PathPrefix("/api/v1").Subrouter()
	apiV1Router.Use(requireAuth(authManager, db))
	registerSecureAPIRoutes(apiV1Router, db, vmManager, vmBasePath(cfg))
	// Signed fabric join/leave RPCs (node-to-node) and the authed
	// /api/cluster/nodes inventory with live link profiles.
	registerClusterJoinRoutes(router, apiRouter, db, vmManager, vmBasePath(cfg))

	// Fabric compute jobs (P2/G2): submit/status/logs/cancel on the authed
	// router; the start/stop/delete/log node helpers on the internal one.
	registerFabricJobRoutes(apiRouter, db, vmManager, vmBasePath(cfg))
	registerFabricNodeRPCs(router, db, vmManager, vmBasePath(cfg))

	// Fabric transfers (P3/G3): admission-controlled migrations with the
	// measured link budget + compression decision recorded per transfer.
	registerFabricTransferRoutes(apiRouter, db, vmManager, vmBasePath(cfg))

	// Node lifecycle (drain): per-node detail + POST drain, which moves every
	// drainable VM off the node through the transfer machinery above.
	registerNodeAdminRoutes(apiRouter, db, vmManager, vmBasePath(cfg))

	// Usage-metered billing read API (PR-1): measured egress, migrations, job
	// seconds, vCPU-seconds per org; metering data comes from the fabric and
	// is persisted by migration 000010.
	registerBillingUsageRoutes(apiRouter, db)

	// Node-to-node migration RPC: intentionally OFF the JWT router (the peer is a
	// node, not a user). Gated by an optional shared secret; see the handler.
	registerInternalMigrationRoutes(router, db, vmManager, vmBasePath(cfg))

	// Cross-node cluster: aggregate inventory (/api/cluster) + placement/dispatch
	// RPCs (/internal/cluster/capacity, /internal/vms/create).
	registerClusterRoutes(router, apiRouter, db, vmManager, vmBasePath(cfg))

	registerCanonicalSecurityRoutes(router, authManager, db, services.securityHandlers)
	registerSecurityWebSocketAliases(router, authManager, db, services.securityHandlers)
	registerCanonicalAdminRoutes(router, authManager, db)
	registerCanonicalGraphQLRoute(router, authManager, db, services.graphqlHandler)
	services.websocketHandler.RegisterWebSocketRoutes(router, func(required string, next http.HandlerFunc) http.Handler {
		return requireAuth(authManager, db)(requireRoleHandler(required, next))
	})
	// Register orchestration API routes directly on /api — the handlers already
	// register /orchestration/status, /orchestration/policies, etc.
	if services.orchestrationAPI != nil {
		services.orchestrationAPI.RegisterRoutes(apiRouter)
	}

	// GET /api/auth/me - returns current user with memberships
	meHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		userID, ok := r.Context().Value("user_id").(string)
		if !ok || userID == "" {
			writeJSONError(w, http.StatusUnauthorized, "authentication required")
			return
		}

		user, err := authManager.GetUser(userID)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "user not found")
			return
		}

		// Get user's cluster memberships from runtime tables (created by runtime_auth persistence)
		rows, err := db.Query(`
			SELECT m.cluster_id, m.state, m.role, m.source, m.created_at, m.tenant_id,
			       c.id, c.name, c.tier, c.performance_score, c.interconnect_latency_ms,
			       c.interconnect_bandwidth_mbps, c.current_node_count, c.max_supported_node_count,
			       c.growth_state, c.federation_state, c.degraded, c.last_evaluated_at
			FROM runtime_cluster_memberships m
			JOIN runtime_clusters c ON m.cluster_id = c.id
			WHERE m.user_id = $1
			ORDER BY m.created_at DESC
		`, userID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query cluster memberships")
			return
		}
		defer rows.Close()

		type AdmissionResponse struct {
			Admitted     bool   `json:"admitted"`
			State        string `json:"state,omitempty"`
			ClusterID    string `json:"cluster_id"`
			Role         string `json:"role,omitempty"`
			Source       string `json:"source,omitempty"`
			AdmittedAt   string `json:"admitted_at,omitempty"`
			TenantID     string `json:"tenant_id,omitempty"`
			Selected     bool   `json:"selected,omitempty"`
			Cluster      *ClusterSummaryResponse `json:"cluster,omitempty"`
		}

		memberships := make([]AdmissionResponse, 0)
		var selectedCluster *ClusterSummaryResponse
		// Track best cluster for selectedCluster (highest performance_score)
		var bestCluster *ClusterSummaryResponse
		bestScore := -1.0

		for rows.Next() {
			var adm AdmissionResponse
			var cluster ClusterSummaryResponse
			err := rows.Scan(
				&adm.ClusterID, &adm.State, &adm.Role, &adm.Source, &adm.AdmittedAt, &adm.TenantID,
				&cluster.ID, &cluster.Name, &cluster.Tier, &cluster.PerformanceScore,
				&cluster.InterconnectLatencyMs, &cluster.InterconnectBandwidthMbps,
				&cluster.CurrentNodeCount, &cluster.MaxSupportedNodeCount,
				&cluster.GrowthState, &cluster.FederationState, &cluster.Degraded,
				&cluster.LastEvaluatedAt,
			)
			if err != nil {
				writeJSONError(w, http.StatusInternalServerError, "failed to scan cluster membership")
				return
			}
			// Derive admitted from state
			adm.Admitted = (adm.State == "active")
			adm.Selected = false
			memberships = append(memberships, adm)

			// Track best cluster for selectedCluster (highest performance_score)
			if cluster.PerformanceScore > bestScore {
				bestScore = cluster.PerformanceScore
				bestCluster = &cluster
			}
		}

		// Determine selected cluster (highest performance score among memberships)
		selectedCluster = bestCluster

		// Get session info
		var sessionID, sessionExpiresAt string
		db.QueryRow(`
			SELECT id, expires_at FROM sessions WHERE user_id = $1 AND revoked_at IS NULL
			ORDER BY created_at DESC LIMIT 1
		`, userID).Scan(&sessionID, &sessionExpiresAt)

		// Build response
		userResp := frontendUser(user)
		admission := AdmissionResponse{}
		if selectedCluster != nil {
			admission = AdmissionResponse{
				Admitted:  true,
				ClusterID: selectedCluster.ID,
				Cluster:   selectedCluster,
				Selected:  true,
			}
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"user":              userResp,
			"admission":         admission,
			"memberships":       memberships,
			"selectedCluster":   selectedCluster,
			"session": map[string]interface{}{
				"id":             sessionID,
				"expiresAt":      sessionExpiresAt,
				"createdAt":      time.Now().UTC().Format(time.RFC3339),
				"lastAccessedAt": time.Now().UTC().Format(time.RFC3339),
			},
		})
	})
	router.HandleFunc("/api/auth/me", meHandler).Methods(http.MethodGet)

	router.HandleFunc("/health", healthCheckHandler(cfg, db)).Methods(http.MethodGet)

	router.HandleFunc("/api/info", apiInfoHandler()).Methods(http.MethodGet)

	return &http.Server{
		Addr:         ":" + cfg.Server.APIPort,
		Handler:      corsHandler(router),
		ReadTimeout:  cfg.Server.ReadTimeout,
		WriteTimeout: cfg.Server.WriteTimeout,
		IdleTimeout:  cfg.Server.IdleTimeout,
		// 64 KiB: a real 16x reduction from net/http's 1 MiB default (used when 0),
		// comfortably above any legit JWT+cookie header set; caps slowloris-style
		// header floods.
		MaxHeaderBytes: 64 << 10,
	}
}


// ClusterSummaryResponse represents cluster summary information for API responses
type ClusterSummaryResponse struct {
	ID                         string    `json:"id"`
	Name                       string    `json:"name"`
	Tier                       string    `json:"tier"`
	PerformanceScore           float64   `json:"performanceScore"`
	InterconnectLatencyMs      float64   `json:"interconnectLatencyMs"`
	InterconnectBandwidthMbps  float64   `json:"interconnectBandwidthMbps"`
	CurrentNodeCount           int       `json:"currentNodeCount"`
	MaxSupportedNodeCount      int       `json:"maxSupportedNodeCount"`
	GrowthState                string    `json:"growthState"`
	FederationState            string    `json:"federationState"`
	Degraded                   bool      `json:"degraded"`
	LastEvaluatedAt            time.Time `json:"lastEvaluatedAt"`
	EdgeLatencyMs              float64   `json:"edgeLatencyMs,omitempty"`
	EdgeBandwidthMbps          float64   `json:"edgeBandwidthMbps,omitempty"`
}


const incomingMigrationMarker = ".novacron-incoming-migration"

func prepareIncomingMigrationDir(destDir string) error {
	if err := os.Mkdir(destDir, 0o700); err != nil {
		return err
	}
	marker, err := os.OpenFile(filepath.Join(destDir, incomingMigrationMarker), os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		_ = os.Remove(destDir)
		return err
	}
	if err := marker.Close(); err != nil {
		_ = os.Remove(filepath.Join(destDir, incomingMigrationMarker))
		_ = os.Remove(destDir)
		return err
	}
	return nil
}

func incomingMigrationMarkerExists(destDir string) bool {
	info, err := os.Lstat(filepath.Join(destDir, incomingMigrationMarker))
	return err == nil && info.Mode().IsRegular()
}

func clearIncomingMigrationMarker(destDir string) {
	if err := os.Remove(filepath.Join(destDir, incomingMigrationMarker)); err != nil && !errors.Is(err, os.ErrNotExist) {
		log.Printf("incoming migration: remove marker %s: %v", destDir, err)
	}
}

// cleanupOrphanedIncoming stops and removes a half-started migration
// destination whose guest never resumed (source died or was interrupted).
// Without it the dest keeps a paused qemu holding the disk lock, the VM id is
// unusable for a retry, and nothing in the DB references it — an orphan that
// only manual cleanup finds (observed live 2026-09-20: an interrupted transfer
// left a locked dest disk that failed the next attempt with "Failed to get
// write lock").
func cleanupOrphanedIncoming(kd *core_vm.KVMDriverEnhanced, vmID, destDir string) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	if err := kd.Stop(ctx, vmID); err != nil {
		log.Printf("orphan cleanup: stop %s: %v", vmID, err)
	}
	if err := kd.Delete(ctx, vmID); err != nil {
		log.Printf("orphan cleanup: delete %s: %v", vmID, err)
	}
	// Defensive: Delete removes the driver's runtime dir; RemoveAll covers a
	// dest dir created before the driver registered the VM.
	if err := os.RemoveAll(destDir); err != nil {
		log.Printf("orphan cleanup: remove %s: %v", destDir, err)
	}
	log.Printf("orphan cleanup: removed half-started migration dest %s", vmID)
}

func buildCORSHandler(cfg *config.Config) mux.MiddlewareFunc {
	allowedOrigins := cfg.CORS.AllowedOrigins
	if len(allowedOrigins) == 0 {
		allowedOrigins = []string{
			"http://localhost:8092",
			"http://localhost:3001",
			"http://localhost:3000",
		}
	}

	allowedMethods := cfg.CORS.AllowedMethods
	if len(allowedMethods) == 0 {
		allowedMethods = []string{http.MethodGet, http.MethodPost, http.MethodPut, http.MethodDelete, http.MethodPatch, http.MethodOptions}
	}

	allowedHeaders := cfg.CORS.AllowedHeaders
	if len(allowedHeaders) == 0 {
		allowedHeaders = []string{"Content-Type", "Authorization", "X-User-Email"}
	}

	return handlers.CORS(
		handlers.AllowedOrigins(allowedOrigins),
		handlers.AllowedMethods(allowedMethods),
		handlers.AllowedHeaders(allowedHeaders),
		handlers.AllowCredentials(),
	)
}

func initDatabase(cfg *config.Config) (*sql.DB, error) {
	db, err := sql.Open("postgres", cfg.Database.URL)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	db.SetMaxOpenConns(cfg.Database.MaxConnections)
	db.SetMaxIdleConns(cfg.Database.MaxConnections / 2)
	db.SetConnMaxLifetime(cfg.Database.ConnMaxLifetime)
	db.SetConnMaxIdleTime(cfg.Database.ConnMaxIdleTime)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	if err := requireMigratedSchema(db); err != nil {
		db.Close()
		return nil, err
	}

	return db, nil
}

// requireMigratedSchema verifies the canonical golang-migrate-managed schema is
// present (database/migrations, applied by `make db-migrate`, the docker-compose
// `migrate` init service, or the k8s migrate Job — see docker/api-entrypoint.sh).
// The server deliberately does NOT create tables: an earlier embedded DDL
// drifted from the canonical schema (VARCHAR ids, a vm_interfaces table) and
// could not even boot against it (REFERENCES on the UUID vms.id cannot be
// implemented for a VARCHAR column), so it was removed rather than forked.
// One statement probes the two load-bearing canonical tables; anything but a
// clean zero-row result means the schema is absent or partial.
func requireMigratedSchema(db *sql.DB) error {
	var present bool
	if err := db.QueryRow(
		`SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'vms')
		 AND EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'users')`,
	).Scan(&present); err != nil {
		return fmt.Errorf("failed to inspect database schema: %w", err)
	}
	if !present {
		return fmt.Errorf("canonical database schema not present: apply database/migrations first (make db-migrate, the docker-compose 'migrate' service, or the k8s migrate Job)")
	}
	return nil
}

func requireAuth(authManager *auth.SimpleAuthManager, db *sql.DB) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			tokenString, err := extractBearerToken(r.Header.Get("Authorization"))
			if err != nil {
				writeJSONError(w, http.StatusUnauthorized, err.Error())
				return
			}

			claims, err := validateJWT(tokenString, authManager.GetJWTSecret())
			if err != nil {
				writeJSONError(w, http.StatusUnauthorized, "invalid or expired token")
				return
			}
			if stringClaim(claims, "purpose") == "pending_2fa" {
				writeJSONError(w, http.StatusUnauthorized, "two-factor authentication is not complete")
				return
			}

			userID := stringClaim(claims, "user_id", "sub")
			if userID == "" {
				writeJSONError(w, http.StatusUnauthorized, "token missing user identity")
				return
			}

			// Session revocation: if the user's record has been bumped since this
			// token was issued (password reset, role change, admin edit), the
			// token is no longer valid. This is a stateless-JWT pattern — the DB
			// check is a sub-millisecond indexed lookup, so the extra round trip is
			// free against reality.
			if db != nil {
				var updatedAt time.Time
				if err := db.QueryRowContext(r.Context(),
					`SELECT updated_at FROM users WHERE id = $1`, userID).Scan(&updatedAt); err == nil {
					if iat, err := claims.GetIssuedAt(); err == nil && iat != nil {
						if iat.Before(updatedAt.UTC()) {
							writeJSONError(w, http.StatusUnauthorized, "token revoked (session invalidated by profile change)")
							return
						}
					}
				}
			}

			ctx := context.WithValue(r.Context(), "user_id", userID)
			ctx = context.WithValue(ctx, "tenant_id", stringClaim(claims, "tenant_id"))
			ctx = context.WithValue(ctx, "organization_id", stringClaim(claims, "tenant_id"))
			ctx = context.WithValue(ctx, "role", stringClaim(claims, "role"))
			ctx = context.WithValue(ctx, "roles", stringSliceClaim(claims, "roles"))

			next.ServeHTTP(w, r.WithContext(ctx))
		})
	}
}

func requireAnyRoleMiddleware(requiredRoles ...string) mux.MiddlewareFunc {
	return func(next http.Handler) http.Handler {
		return requireAnyRole(requiredRoles...)(next)
	}
}

func requireAnyRole(requiredRoles ...string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if hasAnyRole(r.Context(), requiredRoles...) {
				next.ServeHTTP(w, r)
				return
			}

			writeJSONError(w, http.StatusForbidden, "insufficient permissions")
		})
	}
}

func requireRoleHandler(requiredRole string, next http.HandlerFunc) http.Handler {
	return requireAnyRole(requiredRole)(next)
}

func registerPublicRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB, twoFactorService *auth.TwoFactorService, emailService *auth.EmailService) {
	// emailService carries optional SMTP delivery for the auth token routes;
	// nil means email is unconfigured and those routes fail closed with 503.
	loginHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var loginReq struct {
			Username string `json:"username"`
			Email    string `json:"email"`
			Password string `json:"password"`
		}

		if err := json.NewDecoder(r.Body).Decode(&loginReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		identifier := strings.TrimSpace(loginReq.Username)
		if identifier == "" {
			identifier = strings.TrimSpace(loginReq.Email)
		}
		if identifier == "" || strings.TrimSpace(loginReq.Password) == "" {
			writeJSONError(w, http.StatusBadRequest, "email or username and password are required")
			return
		}

		username, err := resolveLoginUsername(db, identifier)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "invalid credentials")
			return
		}

		user, token, err := authManager.Authenticate(username, loginReq.Password)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "invalid credentials")
			return
		}

		userPayload := frontendUser(user)
		twoFactorEnabled := userHasEnabledTwoFactor(twoFactorService, user.ID)
		userPayload["two_factor_enabled"] = twoFactorEnabled

		if twoFactorEnabled {
			tempToken, err := issuePending2FAToken(authManager.GetJWTSecret(), user)
			if err != nil {
				writeJSONError(w, http.StatusInternalServerError, "failed to create 2FA challenge")
				return
			}

			writeJSON(w, http.StatusOK, map[string]interface{}{
				"requires_2fa": true,
				"temp_token":   tempToken,
				"expiresAt":    time.Now().UTC().Add(10 * time.Minute).Format(time.RFC3339),
				"user":         userPayload,
			})
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"token":     token,
			"expiresAt": time.Now().UTC().Add(24 * time.Hour).Format(time.RFC3339),
			"user":      userPayload,
		})
	})

	registerHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var registerReq struct {
			Username  string `json:"username"`
			Email     string `json:"email"`
			Password  string `json:"password"`
			TenantID  string `json:"tenant_id,omitempty"`
			TenantID2 string `json:"tenantId,omitempty"`
			FirstName string `json:"firstName,omitempty"`
			LastName  string `json:"lastName,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&registerReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		email := strings.TrimSpace(registerReq.Email)
		password := strings.TrimSpace(registerReq.Password)
		if email == "" || password == "" {
			writeJSONError(w, http.StatusBadRequest, "email and password are required")
			return
		}

		tenantID := strings.TrimSpace(registerReq.TenantID)
		if tenantID == "" {
			tenantID = strings.TrimSpace(registerReq.TenantID2)
		}
		if tenantID == "" {
			tenantID = "default"
		}

		username := strings.TrimSpace(registerReq.Username)
		if username == "" {
			username = defaultUsernameFromEmail(email)
		}

		user, err := authManager.CreateUser(username, email, password, "user", tenantID)
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("failed to create user: %v", err))
			return
		}

		resp := frontendUser(user)
		if registerReq.FirstName != "" {
			resp["firstName"] = registerReq.FirstName
		}
		if registerReq.LastName != "" {
			resp["lastName"] = registerReq.LastName
		}

		writeJSON(w, http.StatusCreated, resp)

		if emailService != nil && email != "" {
			// Best-effort verification email: never fail the registration over it.
			if rawToken, err := insertAuthToken(db, user.ID, "email_verification", 24*time.Hour); err != nil {
				log.Printf("registration: verification token for %s not created: %v", email, err)
			} else if err := emailService.SendAccountVerification(email, user.Username, rawToken, 24); err != nil {
				log.Printf("registration: verification email to %s failed: %v", email, err)
			}
		}
	})

	checkEmailHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		email := strings.TrimSpace(r.URL.Query().Get("email"))
		if email == "" {
			writeJSONError(w, http.StatusBadRequest, "email is required")
			return
		}

		var exists bool
		err := db.QueryRow(`SELECT EXISTS(SELECT 1 FROM users WHERE email = $1)`, email).Scan(&exists)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to check email availability")
			return
		}

		writeJSON(w, http.StatusOK, map[string]bool{"available": !exists})
	})

	// Login is the one unauthenticated route that spends a bcrypt comparison and
	// a database round trip per request, so the per-IP limiter wraps the handler
	// itself: both mount points (/auth/login and its /api alias) then share one
	// bucket per client, and an over-limit attempt never reaches Authenticate.
	rateLimitedLogin := loginRateLimitMiddleware(newLoginRateLimiterFromEnv())(loginHandler)
	for _, path := range []string{"/auth/login", "/api/auth/login"} {
		router.Handle(path, rateLimitedLogin).Methods(http.MethodPost)
	}
	for _, path := range []string{"/auth/register", "/api/auth/register"} {
		router.Handle(path, registerHandler).Methods(http.MethodPost)
	}
	router.Handle("/api/auth/check-email", checkEmailHandler).Methods(http.MethodGet)

	forgotPasswordHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Email string `json:"email"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		email := strings.TrimSpace(req.Email)
		if email == "" {
			writeJSONError(w, http.StatusBadRequest, "email is required")
			return
		}

		if emailService == nil {
			writeJSONError(w, http.StatusServiceUnavailable, "email delivery is not configured (set SMTP_HOST)")
			return
		}

		var userID, username string
		err := db.QueryRow(`SELECT id, username FROM users WHERE email = $1`, email).Scan(&userID, &username)
		if err == nil && userID != "" {
			// One live reset token per user: invalidate any outstanding one first.
			if rawToken, terr := insertAuthToken(db, userID, "password_reset", 60*time.Minute); terr != nil {
				log.Printf("forgot-password: reset token for %s not created: %v", email, terr)
			} else if serr := emailService.SendPasswordReset(email, username, rawToken, 60); serr != nil {
				// Still 200: the response must not leak whether the account exists.
				log.Printf("forgot-password: reset email to %s failed: %v", email, serr)
			}
		}

		writeJSON(w, http.StatusOK, map[string]string{
			"message": "If an account exists for that email, a reset link has been sent",
		})
	})

	resetPasswordHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Token    string `json:"token"`
			Password string `json:"password"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		token := strings.TrimSpace(req.Token)
		password := strings.TrimSpace(req.Password)
		if token == "" {
			writeJSONError(w, http.StatusBadRequest, "token is required")
			return
		}
		if err := validateCanonicalPassword(password); err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}

		tokenHash := authTokenSHA256Hex(token)
		var userID string
		err := db.QueryRow(`
			SELECT user_id FROM auth_tokens
			WHERE token_hash = $1 AND purpose = 'password_reset' AND used_at IS NULL AND expires_at > NOW()
		`, tokenHash).Scan(&userID)
		if err != nil {
			if err != sql.ErrNoRows {
				writeJSONError(w, http.StatusInternalServerError, "failed to look up reset token")
				return
			}
			writeJSONError(w, http.StatusBadRequest, "invalid or expired token")
			return
		}

		passwordHash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to hash password")
			return
		}

		tx, err := db.Begin()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to begin password reset transaction")
			return
		}
		defer tx.Rollback()

		if _, err := tx.Exec(`UPDATE users SET password_hash = $1, updated_at = NOW() WHERE id = $2`, string(passwordHash), userID); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to update password")
			return
		}
		if _, err := tx.Exec(`UPDATE auth_tokens SET used_at = NOW() WHERE token_hash = $1 AND purpose = 'password_reset'`, tokenHash); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to mark token used")
			return
		}
		if _, err := tx.Exec(`DELETE FROM sessions WHERE user_id = $1`, userID); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to revoke sessions")
			return
		}

		if err := tx.Commit(); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to commit password reset")
			return
		}

		writeJSON(w, http.StatusOK, map[string]string{"message": "Password reset successfully"})
	})

	verifyEmailHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Token string `json:"token"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		token := strings.TrimSpace(req.Token)
		if token == "" {
			writeJSONError(w, http.StatusBadRequest, "token is required")
			return
		}

		tokenHash := authTokenSHA256Hex(token)
		var userID string
		err := db.QueryRow(`
			SELECT user_id FROM auth_tokens
			WHERE token_hash = $1 AND purpose = 'email_verification' AND used_at IS NULL AND expires_at > NOW()
		`, tokenHash).Scan(&userID)
		if err != nil {
			if err != sql.ErrNoRows {
				writeJSONError(w, http.StatusInternalServerError, "failed to look up verification token")
				return
			}
			writeJSONError(w, http.StatusBadRequest, "invalid or expired token")
			return
		}

		tx, err := db.Begin()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to begin email verification transaction")
			return
		}
		defer tx.Rollback()

		if _, err := tx.Exec(`
			UPDATE users
			SET email_verified = TRUE,
			    status = CASE WHEN status = 'pending' THEN 'active'::user_status ELSE status END,
			    updated_at = NOW()
			WHERE id = $1
		`, userID); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to verify email")
			return
		}
		if _, err := tx.Exec(`UPDATE auth_tokens SET used_at = NOW() WHERE token_hash = $1 AND purpose = 'email_verification'`, tokenHash); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to mark token used")
			return
		}

		if err := tx.Commit(); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to commit email verification")
			return
		}

		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	resendVerificationHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Email string `json:"email"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		email := strings.TrimSpace(req.Email)
		if email == "" {
			writeJSONError(w, http.StatusBadRequest, "email is required")
			return
		}

		if emailService == nil {
			writeJSONError(w, http.StatusServiceUnavailable, "email delivery is not configured (set SMTP_HOST)")
			return
		}

		var userID, username string
		var emailVerified bool
		err := db.QueryRow(`SELECT id, username, email_verified FROM users WHERE email = $1`, email).Scan(&userID, &username, &emailVerified)
		if err == nil && userID != "" && !emailVerified {
			if rawToken, terr := insertAuthToken(db, userID, "email_verification", 24*time.Hour); terr != nil {
				log.Printf("resend-verification: token for %s not created: %v", email, terr)
			} else if serr := emailService.SendAccountVerification(email, username, rawToken, 24); serr != nil {
				log.Printf("resend-verification: email to %s failed: %v", email, serr)
			}
		}

		writeJSON(w, http.StatusOK, map[string]bool{"success": true})
	})

	router.Handle("/api/auth/forgot-password", forgotPasswordHandler).Methods(http.MethodPost)
	router.Handle("/api/auth/reset-password", resetPasswordHandler).Methods(http.MethodPost)
	router.Handle("/api/auth/verify-email", verifyEmailHandler).Methods(http.MethodPost)
	router.Handle("/api/auth/resend-verification", resendVerificationHandler).Methods(http.MethodPost)

	router.HandleFunc("/api/auth/2fa/verify-login", func(w http.ResponseWriter, r *http.Request) {
		if twoFactorService == nil {
			writeJSON(w, http.StatusNotImplemented, map[string]interface{}{
				"error":   "not_supported",
				"message": "two-factor verification is not configured in the canonical server",
			})
			return
		}

		var verifyReq struct {
			UserID       string `json:"user_id"`
			Code         string `json:"code"`
			IsBackupCode bool   `json:"is_backup_code"`
			TempToken    string `json:"temp_token"`
		}
		if err := json.NewDecoder(r.Body).Decode(&verifyReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		claims, err := validatePending2FAToken(verifyReq.TempToken, authManager.GetJWTSecret())
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "invalid or expired temporary token")
			return
		}

		userID := stringClaim(claims, "user_id", "sub")
		if verifyReq.UserID != "" && verifyReq.UserID != userID {
			writeJSONError(w, http.StatusUnauthorized, "temporary token does not match requested user")
			return
		}

		verifyResponse, err := twoFactorService.VerifyCode(auth.TwoFactorVerifyRequest{
			UserID:       userID,
			Code:         verifyReq.Code,
			IsBackupCode: verifyReq.IsBackupCode,
		})
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to verify two-factor code")
			return
		}
		if !verifyResponse.Valid {
			writeJSONError(w, http.StatusUnauthorized, "invalid two-factor code")
			return
		}

		user, err := authManager.GetUser(userID)
		if err != nil {
			writeJSONError(w, http.StatusUnauthorized, "user not found for temporary token")
			return
		}

		sessionToken, err := issueSessionToken(authManager.GetJWTSecret(), user)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to create session token")
			return
		}

		userPayload := frontendUser(user)
		userPayload["two_factor_enabled"] = true
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"token":                  sessionToken,
			"expiresAt":              time.Now().UTC().Add(24 * time.Hour).Format(time.RFC3339),
			"user":                   userPayload,
			"remaining_backup_codes": verifyResponse.RemainingCodes,
		})
	}).Methods(http.MethodPost)
}

// requireOrgScope resolves the tenancy scope of the authenticated caller from
// the verified JWT claims requireAuth put on the request context and -- when
// db and vmID are both given -- authorizes that one VM against it.
//
// SEMANTIC: the org filter HIDES EXISTENCE. A VM (or fabric job) outside the
// caller's scope does not exist for them: single-resource reads answer 404
// (never 403), lists omit it entirely, and writes act as if the row were
// absent -- the API must never confirm that an id belongs to another org.
// Callers therefore treat visible=false as "respond 404", indistinguishable
// from the row genuinely not existing.
//
//   - admin/super-admin roles pass unscoped (isAdmin=true, orgID=""): they see
//     and act on every org, and NO query is issued for them -- row-level
//     visibility for an admin falls out of the caller's own lookup, keeping
//     the unscoped statements byte-identical to the pre-hardening queries.
//   - every other caller is scoped: orgID is the normalized uuid
//     organization_id claim (orgLabelForVM). A legacy/non-uuid tenant label
//     (or no claims at all, e.g. a handler invoked without the middleware)
//     normalizes to "" and scopes to NULL-organization rows -- exactly the
//     rows such a caller's own creates stamp, and never another org's rows.
//   - with db+vmID set, a scoped caller's check is one indexed existence
//     probe (`SELECT EXISTS ... AND <org predicate>`), used where a decision
//     must be made BEFORE touching the real VM (delete) or where no row
//     payload is needed. Statement-level callers instead fold the same
//     predicate into their own query and pass (nil, "").
func requireOrgScope(ctx context.Context, db *sql.DB, vmID string) (orgID string, isAdmin bool, visible bool) {
	role, _ := ctx.Value("role").(string)
	if role == "admin" || role == "super-admin" {
		return "", true, true
	}
	claim, _ := ctx.Value("organization_id").(string)
	orgID = orgLabelForVM(claim)
	if db == nil || vmID == "" {
		return orgID, false, true // scope resolution only
	}
	var exists bool
	var err error
	if orgID != "" {
		err = db.QueryRowContext(ctx,
			`SELECT EXISTS (SELECT 1 FROM vms WHERE id = $1 AND organization_id = $2)`,
			vmID, orgID).Scan(&exists)
	} else {
		err = db.QueryRowContext(ctx,
			`SELECT EXISTS (SELECT 1 FROM vms WHERE id = $1 AND organization_id IS NULL)`,
			vmID).Scan(&exists)
	}
	if err != nil {
		return orgID, false, false // a failed probe must fail closed (404)
	}
	return orgID, false, exists
}

// orgVisible reports whether a vms row carrying rowOrg falls inside the
// caller's scope: a uuid scope matches rows stamped with that same org, and
// the legacy/NULL scope matches rows whose organization_id is NULL.
func orgVisible(scopeOrg string, rowOrg sql.NullString) bool {
	if scopeOrg == "" {
		return !rowOrg.Valid || rowOrg.String == ""
	}
	return rowOrg.Valid && rowOrg.String == scopeOrg
}

func registerSecureAPIRoutes(router *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, storagePath string) {
	router.HandleFunc("/vms", func(w http.ResponseWriter, r *http.Request) {
		// Org filtering: admin/super-admin roles see everything; everyone else
		// is scoped by requireOrgScope. VMs outside the caller's scope are
		// INVISIBLE, not denied (see requireOrgScope's 404-hides-existence
		// semantics); defense-in-depth RLS remains a separate bead (novacron-ok7).
		scopeOrg, isAdmin, _ := requireOrgScope(r.Context(), nil, "")

		// Unfiltered (admin) and uuid-org statements stay byte-identical to
		// the pre-org queries so existing callers (and mocks) are unaffected.
		// The NULL-org variant is the legacy/claim-less scope: those callers
		// see only unattributed rows, never another org's. (This also fixes a
		// live bug: the old code passed a non-uuid tenant label like
		// "default" into `organization_id = $1`, which Postgres rejects --
		// uuid has no text equality operator -- 500ing every legacy list.)
		var rows *sql.Rows
		var err error
		switch {
		case isAdmin:
			rows, err = db.Query(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at FROM vms ORDER BY created_at DESC`)
		case scopeOrg != "":
			rows, err = db.Query(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at FROM vms WHERE organization_id = $1 ORDER BY created_at DESC`, scopeOrg)
		default:
			rows, err = db.Query(`SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at FROM vms WHERE organization_id IS NULL ORDER BY created_at DESC`)
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VMs")
			return
		}
		defer rows.Close()

		vms := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name, state string
			var orgID sql.NullString
			var nodeID sql.NullString
			var cpuCores, memoryMB, diskGB int
			var createdAt, updatedAt time.Time

			if err := rows.Scan(&id, &name, &state, &nodeID, &orgID, &cpuCores, &memoryMB, &diskGB, &createdAt, &updatedAt); err != nil {
				continue
			}

			// Reconcile DB metadata with the manager's live runtime state.
			state = liveVMState(vmManager, id, state)

			vms = append(vms, map[string]interface{}{
				"id":              id,
				"name":            name,
				"state":           state,
				"status":          state,
				"vcpus":           cpuCores,
				"memory_mb":       memoryMB,
				"disk_gb":         diskGB,
				"node_id":         nullableString(nodeID),
				"organization_id": nullableString(orgID),
				"created_at":      createdAt.Format(time.RFC3339),
				"updated_at":      updatedAt.Format(time.RFC3339),
			})
		}

		writeJSON(w, http.StatusOK, vms)
	}).Methods(http.MethodGet)

	// Cross-node cluster placement (see cluster.go): node_id ""/"auto"/"cluster"
	// best-fits across all nodes and dispatches to the chosen one; an explicit peer
	// id dispatches there; this node's own id or an unknown label creates locally.
	router.HandleFunc("/vms", clusteredCreateHandler(db, vmManager, storagePath)).Methods(http.MethodPost)

	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]

		var id, name, state string
		var orgID sql.NullString
		var nodeID sql.NullString
		var cpuCores, memoryMB, diskGB int
		var createdAt, updatedAt time.Time
		err := db.QueryRow(`
			SELECT id, name, state, node_id, organization_id, cpu_cores, memory_mb, disk_gb, created_at, updated_at
			FROM vms WHERE id = $1
		`, vmID).Scan(&id, &name, &state, &nodeID, &orgID, &cpuCores, &memoryMB, &diskGB, &createdAt, &updatedAt)
		if err != nil {
			if err == sql.ErrNoRows {
				writeJSONError(w, http.StatusNotFound, "vm not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM")
			return
		}

		// Tenancy: a VM outside the caller's org scope is invisible -- 404,
		// exactly as if the id did not exist (see requireOrgScope). The row
		// was already loaded, so the decision is made against rowOrg instead
		// of a second probe.
		scopeOrg, isAdmin, _ := requireOrgScope(r.Context(), nil, "")
		if !isAdmin && !orgVisible(scopeOrg, orgID) {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		state = liveVMState(vmManager, id, state)

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":              id,
			"name":            name,
			"state":           state,
			"status":          state,
			"vcpus":           cpuCores,
			"memory_mb":       memoryMB,
			"disk_gb":         diskGB,
			"node_id":         nullableString(nodeID),
			"organization_id": nullableString(orgID),
			"created_at":      createdAt.Format(time.RFC3339),
			"updated_at":      updatedAt.Format(time.RFC3339),
		})
	}).Methods(http.MethodGet)

	registerVMDeleteRoute(router, db, vmManager)

	registerVMPowerRoute(router, db, vmManager, "start")
	registerVMPowerRoute(router, db, vmManager, "stop")
	registerVMPowerRoute(router, db, vmManager, "pause")
	registerVMPowerRoute(router, db, vmManager, "resume")
	registerVMPowerRoute(router, db, vmManager, "restart")
	registerVMMigrateRoute(router, db, vmManager)
	// Async variant of the migrate route (202 + job id, migration runs in the
	// background) so large/slow migrations can't trip the request WRITE_TIMEOUT.
	registerVMMigrateAsyncRoutes(router, db, vmManager)

	router.HandleFunc("/vms/{id}/metrics", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]
		_, _, visible := requireOrgScope(r.Context(), db, vmID)
		if !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		var cpuUsage, memoryUsage float64
		err := db.QueryRow(`
			SELECT COALESCE(cpu_usage, 0), COALESCE(memory_usage, 0)
			FROM vm_metrics WHERE vm_id = $1
			ORDER BY timestamp DESC
			LIMIT 1
		`, vmID).Scan(&cpuUsage, &memoryUsage)
		if errors.Is(err, sql.ErrNoRows) {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM metrics")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":           vmID,
			"cpu_usage":    cpuUsage,
			"memory_usage": memoryUsage,
		})
	}).Methods(http.MethodGet)

	router.HandleFunc("/monitoring/metrics", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, hostMetrics(storagePath))
	}).Methods(http.MethodGet)

	router.HandleFunc("/monitoring/vms", func(w http.ResponseWriter, r *http.Request) {
		scopeOrg, isAdmin, _ := requireOrgScope(r.Context(), nil, "")
		var rows *sql.Rows
		var err error
		switch {
		case isAdmin:
			rows, err = db.Query(`SELECT id, name, state FROM vms ORDER BY created_at DESC`)
		case scopeOrg != "":
			rows, err = db.Query(`SELECT id, name, state, organization_id FROM vms WHERE organization_id = $1 ORDER BY created_at DESC`, scopeOrg)
		default:
			rows, err = db.Query(`SELECT id, name, state, organization_id FROM vms WHERE organization_id IS NULL ORDER BY created_at DESC`)
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VMs")
			return
		}
		defer rows.Close()

		type vmRow struct{ id, name, state string }
		var base []vmRow
		for rows.Next() {
			var v vmRow
			if isAdmin {
				if err := rows.Scan(&v.id, &v.name, &v.state); err != nil {
					continue
				}
			} else {
				var rowOrg sql.NullString
				if err := rows.Scan(&v.id, &v.name, &v.state, &rowOrg); err != nil || !orgVisible(scopeOrg, rowOrg) {
					continue
				}
			}
			base = append(base, v)
		}

		// Real per-VM metrics from the vm_metrics table (same source as
		// /vms/{id}/metrics). Fields are omitted when no sample has been
		// recorded, rather than serving a fabricated placeholder value.
		vmMetrics := make([]map[string]interface{}, 0, len(base))
		for _, v := range base {
			entry := map[string]interface{}{
				"vmId":   v.id,
				"name":   v.name,
				"status": v.state,
			}
			var cpu, mem sql.NullFloat64
			if err := db.QueryRow(`
				SELECT cpu_usage, memory_usage FROM vm_metrics
				WHERE vm_id = $1 ORDER BY timestamp DESC LIMIT 1
			`, v.id).Scan(&cpu, &mem); err == nil {
				if cpu.Valid {
					entry["cpuUsage"] = cpu.Float64
				}
				if mem.Valid {
					entry["memoryUsage"] = mem.Float64
				}
			}
			vmMetrics = append(vmMetrics, entry)
		}

		writeJSON(w, http.StatusOK, vmMetrics)
	}).Methods(http.MethodGet)

	router.HandleFunc("/monitoring/alerts", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, []map[string]interface{}{})
	}).Methods(http.MethodGet)

	// Networks: the canonical schema has NO networks catalog table (only the
	// per-VM network_interfaces table). The catalog routes report the empty
	// catalog honestly instead of 500ing on a table that does not exist.
	router.HandleFunc("/networks", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, []map[string]interface{}{})
	}).Methods(http.MethodGet)

	router.HandleFunc("/networks", func(w http.ResponseWriter, r *http.Request) {
		writeJSONError(w, http.StatusNotImplemented, "no networks catalog in the canonical schema; per-VM interfaces live at /vms/{vm_id}/interfaces")
	}).Methods(http.MethodPost)

	router.HandleFunc("/networks/{id}", func(w http.ResponseWriter, r *http.Request) {
		writeJSONError(w, http.StatusNotFound, "network not found")
	}).Methods(http.MethodGet)

	router.HandleFunc("/networks/{id}", func(w http.ResponseWriter, r *http.Request) {
		writeJSONError(w, http.StatusNotFound, "network not found")
	}).Methods(http.MethodDelete)
	// VM network interfaces: the canonical schema's network_interfaces table is
	// the equivalent of the legacy vm_interfaces table (vm_id, name, mac_address,
	// ip_address). A bridge/gateway can be attached per-interface; there is no
	// canonical networks catalog table, so interface rows stand alone.
	router.HandleFunc("/vms/{vm_id}/interfaces", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		_, _, visible := requireOrgScope(r.Context(), db, vmID)
		if !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		rows, err := db.Query(`
			SELECT id, vm_id, name, mac_address, ip_address, created_at
			FROM network_interfaces
			WHERE vm_id::text = $1
			ORDER BY created_at DESC
		`, vmID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM interfaces")
			return
		}
		defer rows.Close()

		interfaces := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name string
			var currentVMID sql.NullString
			var macAddress, ipAddress sql.NullString
			var createdAt time.Time

			if err := rows.Scan(&id, &currentVMID, &name, &macAddress, &ipAddress, &createdAt); err != nil {
				continue
			}

			entry := map[string]interface{}{
				"id":          id,
				"vm_id":       nullableString(currentVMID),
				"name":        name,
				"mac_address": nullableString(macAddress),
				"ip_address":  nullableString(ipAddress),
				"status":      "attached",
				"created_at":  createdAt.Format(time.RFC3339),
				"updated_at":  createdAt.Format(time.RFC3339),
			}
			if r.URL.Query().Get("include") == "network" {
				entry["network_id"] = nil
			}
			interfaces = append(interfaces, entry)
		}

		writeJSON(w, http.StatusOK, interfaces)
	}).Methods(http.MethodGet)

	router.HandleFunc("/vms/{vm_id}/interfaces", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		_, _, visible := requireOrgScope(r.Context(), db, vmID)
		if !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		var createReq struct {
			NetworkID  string `json:"network_id"`
			Name       string `json:"name"`
			MACAddress string `json:"mac_address"`
			IPAddress  string `json:"ip_address"`
		}

		if err := json.NewDecoder(r.Body).Decode(&createReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		if strings.TrimSpace(createReq.Name) == "" || strings.TrimSpace(createReq.MACAddress) == "" {
			writeJSONError(w, http.StatusBadRequest, "name and mac_address are required")
			return
		}

		var bridgeName interface{}
		if createReq.NetworkID != "" {
			bridgeName = createReq.NetworkID // repurposed: legacy network ids were bridge labels
		}
		interfaceID := uuid.NewString()
		if _, err := db.Exec(`
			INSERT INTO network_interfaces (id, vm_id, name, mac_address, ip_address, bridge_name, created_at)
			VALUES ($1, $2, $3, $4, NULLIF($5, '')::inet, $6, NOW())
		`, interfaceID, vmID, createReq.Name, createReq.MACAddress, createReq.IPAddress, bridgeName); err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to attach interface")
			return
		}

		writeJSON(w, http.StatusCreated, map[string]interface{}{
			"id":          interfaceID,
			"vm_id":       vmID,
			"network_id":  emptyStringToNil(createReq.NetworkID),
			"name":        createReq.Name,
			"mac_address": createReq.MACAddress,
			"ip_address":  emptyStringToNil(createReq.IPAddress),
			"status":      "attached",
			"created_at":  time.Now().UTC().Format(time.RFC3339),
			"updated_at":  time.Now().UTC().Format(time.RFC3339),
		})
	}).Methods(http.MethodPost)

	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		interfaceID := mux.Vars(r)["id"]
		if _, _, visible := requireOrgScope(r.Context(), db, vmID); !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		var id, name string
		var currentVMID, macAddress, ipAddress, bridgeName sql.NullString
		var createdAt time.Time
		err := db.QueryRow(`
			SELECT id, vm_id, name, mac_address, ip_address, bridge_name, created_at
			FROM network_interfaces
			WHERE vm_id::text = $1 AND id = $2
		`, vmID, interfaceID).Scan(&id, &currentVMID, &name, &macAddress, &ipAddress, &bridgeName, &createdAt)
		if err != nil {
			if err == sql.ErrNoRows {
				writeJSONError(w, http.StatusNotFound, "vm interface not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, "failed to query VM interface")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":          id,
			"vm_id":       nullableString(currentVMID),
			"network_id":  nullableString(bridgeName),
			"name":        name,
			"mac_address": nullableString(macAddress),
			"ip_address":  nullableString(ipAddress),
			"status":      "attached",
			"created_at":  createdAt.Format(time.RFC3339),
			"updated_at":  createdAt.Format(time.RFC3339),
		})
	}).Methods(http.MethodGet)

	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		interfaceID := mux.Vars(r)["id"]
		if _, _, visible := requireOrgScope(r.Context(), db, vmID); !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		var updateReq struct {
			NetworkID string `json:"network_id"`
			Name      string `json:"name"`
			IPAddress string `json:"ip_address"`
			Status    string `json:"status"`
		}

		if err := json.NewDecoder(r.Body).Decode(&updateReq); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		result, err := db.Exec(`
			UPDATE network_interfaces
			SET name = COALESCE(NULLIF($3, ''), name),
				ip_address = COALESCE(NULLIF($4, '')::inet, ip_address),
				bridge_name = COALESCE(NULLIF($5, ''), bridge_name)
			WHERE vm_id::text = $1 AND id = $2
		`, vmID, interfaceID, updateReq.Name, updateReq.IPAddress, updateReq.NetworkID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to update VM interface")
			return
		}

		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "vm interface not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":         interfaceID,
			"vm_id":      vmID,
			"network_id": emptyStringToNil(updateReq.NetworkID),
			"name":       updateReq.Name,
			"ip_address": emptyStringToNil(updateReq.IPAddress),
			"status":     "attached",
		})
	}).Methods(http.MethodPut)

	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		interfaceID := mux.Vars(r)["id"]
		if _, _, visible := requireOrgScope(r.Context(), db, vmID); !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		result, err := db.Exec(`DELETE FROM network_interfaces WHERE vm_id::text = $1 AND id = $2`, vmID, interfaceID)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to detach VM interface")
			return
		}

		rowsAffected, _ := result.RowsAffected()
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "vm interface not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":     interfaceID,
			"vm_id":  vmID,
			"status": "detached",
		})
	}).Methods(http.MethodDelete)
}

// vmBasePath is where the KVM driver stores per-VM dirs (disk + qemu.pid).
func vmBasePath(cfg *config.Config) string {
	return filepath.Join(cfg.VM.StoragePath, "vms")
}

// newVMManager builds the real VM manager. No qemu_path is set, so the KVM
// driver resolves the host-arch binary (qemu-system-aarch64 on arm64) via the
// arch fix. Returns nil (runtime ops disabled) if the manager can't start.
func newVMManager(cfg *config.Config) *core_vm.VMManager {
	vmCfg := core_vm.DefaultVMManagerConfig()
	vmCfg.Drivers[core_vm.VMTypeKVM] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config:  map[string]interface{}{"vm_path": vmBasePath(cfg)},
	}
	// Fabric jobs run as Process VMs. Configure the driver explicitly (rather
	// than relying on the factory default /var/lib/novacron/processes) so each
	// node's processes live under its own storage root — otherwise a second
	// node with a different STORAGE_PATH would run its jobs out of the first
	// node's directory. Path: <StoragePath>/vms/processes/<vmID>/.
	vmCfg.Drivers[core_vm.VMTypeProcess] = core_vm.VMDriverConfigManager{
		Enabled: true,
		Config:  map[string]interface{}{"base_path": filepath.Join(vmBasePath(cfg), "processes")},
	}
	m, err := core_vm.NewVMManager(vmCfg)
	if err != nil {
		logger.Warn("VM manager init failed; VM runtime operations disabled", "error", err)
		return nil
	}
	return m
}

// liveVMState overlays the manager's live runtime state onto the DB metadata
// state when the manager knows the VM; otherwise the stored state stands.
func liveVMState(vmManager *core_vm.VMManager, vmID, dbState string) string {
	if vmManager == nil {
		return dbState
	}
	if vm, err := vmManager.GetVM(vmID); err == nil {
		return string(vm.State())
	}
	return dbState
}

func vmActionStatus(err error) int {
	if errors.Is(err, core_vm.ErrVMNotFound) {
		return http.StatusNotFound
	}
	return http.StatusInternalServerError
}

// registerVMDeleteRoute drops source metadata only after the real driver has
// deleted a managed VM; a driver failure leaves the database row intact.
func registerVMDeleteRoute(router *mux.Router, db *sql.DB, vmManager *core_vm.VMManager) {
	var deleteVM func(context.Context, string) error
	if vmManager != nil {
		deleteVM = func(ctx context.Context, vmID string) error {
			if _, err := vmManager.GetVM(vmID); err != nil {
				return nil
			}
			return vmManager.DeleteVM(ctx, vmID)
		}
	}
	registerVMDeleteRouteWithDeleter(router, db, deleteVM)
}

func registerVMDeleteRouteWithDeleter(router *mux.Router, db *sql.DB, deleteVM func(context.Context, string) error) {
	router.HandleFunc("/vms/{id}", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]
		scopeOrg, isAdmin, visible := requireOrgScope(r.Context(), db, vmID)
		if !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		if deleteVM != nil {
			ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
			defer cancel()
			if err := deleteVM(ctx, vmID); err != nil {
				writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to delete VM: %v", err))
				return
			}
		}

		var result sql.Result
		var err error
		switch {
		case isAdmin:
			result, err = db.Exec(`DELETE FROM vms WHERE id = $1`, vmID)
		case scopeOrg != "":
			result, err = db.Exec(`DELETE FROM vms WHERE id = $1 AND organization_id = $2`, vmID, scopeOrg)
		default:
			result, err = db.Exec(`DELETE FROM vms WHERE id = $1 AND organization_id IS NULL`, vmID)
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to delete VM")
			return
		}
		rowsAffected, err := result.RowsAffected()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to confirm VM deletion")
			return
		}
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{"id": vmID, "status": "deleted"})
	}).Methods(http.MethodDelete)
}

func registerVMPowerRoute(router *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, action string) {
	router.HandleFunc("/vms/{id}/"+action, func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]
		scopeOrg, isAdmin, visible := requireOrgScope(r.Context(), db, vmID)
		if !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}
		if vmManager == nil {
			writeJSONError(w, http.StatusServiceUnavailable, "vm manager unavailable")
			return
		}

		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
		defer cancel()

		var err error
		if action == "start" {
			err = vmManager.StartVM(ctx, vmID)
			if err == nil && restartSupervisor != nil {
				restartSupervisor.RecordStart(ctx, vmID) // declare to the supervisor: this VM runs now
			}
		} else if action == "stop" {
			err = vmManager.StopVM(ctx, vmID)
			if err == nil && restartSupervisor != nil {
				restartSupervisor.RecordStop(ctx, vmID) // not crashed — an intentional stop (never restarted under policy=always)
			}
		} else if action == "pause" {
			err = vmManager.PauseVM(ctx, vmID)
		} else if action == "resume" {
			err = vmManager.ResumeVM(ctx, vmID)
		} else if action == "restart" {
			// Restart is implemented as stop then start
			err = vmManager.StopVM(ctx, vmID)
			if err == nil {
				err = vmManager.StartVM(ctx, vmID)
				if err == nil && restartSupervisor != nil {
					restartSupervisor.RecordStart(ctx, vmID)
				}
			}
		} else {
			err = fmt.Errorf("unsupported action: %s", action)
		}
		if err != nil {
			writeJSONError(w, vmActionStatus(err), fmt.Sprintf("failed to %s VM: %v", action, err))
			return
		}

		vm, gerr := vmManager.GetVM(vmID)
		if gerr != nil {
			writeJSONError(w, http.StatusInternalServerError, "vm state unavailable after "+action)
			return
		}
		state := string(vm.State())

		var result sql.Result
		if isAdmin {
			result, err = db.Exec(`UPDATE vms SET state = $2, updated_at = NOW() WHERE id = $1`, vmID, state)
		} else if scopeOrg != "" {
			result, err = db.Exec(`UPDATE vms SET state = $2, updated_at = NOW() WHERE id = $1 AND organization_id = $3`, vmID, state, scopeOrg)
		} else {
			result, err = db.Exec(`UPDATE vms SET state = $2, updated_at = NOW() WHERE id = $1 AND organization_id IS NULL`, vmID, state)
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, "failed to update VM state")
			return
		}
		if rowsAffected, _ := result.RowsAffected(); rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":     vmID,
			"state":  state,
			"status": state,
		})
	}).Methods(http.MethodPost)
}

// registerInternalMigrationRoutes wires POST /internal/migrate/incoming: the
// target side of a cross-node migration. It receives the source VM's spec +
// absolute disk path, launches a destination qemu waiting on a freshly-picked
// port over shared storage, and returns that port. This is node-to-node (no JWT)
// and gated by a required shared secret from NOVACRON_MIGRATION_SECRET. Fails
// closed: an unset/empty secret means no incoming migration is accepted; see
// migrationAuthOK in migration_auth.go for the constant-time comparison.
// ponytail: static shared-secret header; swap for mTLS if peers aren't trusted.
func registerInternalMigrationRoutes(router *mux.Router, db *sql.DB, vmManager *core_vm.VMManager, vmBase string) {
	router.HandleFunc("/internal/migrate/incoming", func(w http.ResponseWriter, r *http.Request) {
		if !migrationAuthOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		if vmManager == nil {
			writeJSONError(w, http.StatusServiceUnavailable, "vm manager unavailable")
			return
		}

		var req core_vm.IncomingMigrationRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if req.VMID == "" {
			writeJSONError(w, http.StatusBadRequest, "vm_id is required")
			return
		}
		if !req.Block && req.DiskPath == "" {
			writeJSONError(w, http.StatusBadRequest, "disk_path is required for shared-storage migration")
			return
		}
		if req.Block && req.DiskSizeBytes <= 0 {
			writeJSONError(w, http.StatusBadRequest, "disk_size_bytes is required for block migration")
			return
		}

		driver, err := vmManager.GetDriverForConfig(core_vm.VMConfig{Type: core_vm.VMTypeKVM})
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("resolve KVM driver: %v", err))
			return
		}
		kd, ok := driver.(*core_vm.KVMDriverEnhanced)
		if !ok {
			writeJSONError(w, http.StatusInternalServerError, "KVM driver unavailable on this node")
			return
		}

		port, err := freeMigrationPort()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("allocate migration port: %v", err))
			return
		}
		// Canonical <vmBase>/<id> dir (NOT <id>-incoming): the source is on another
		// node/vmBase, so there is no collision, and reconcile/adoptManagerVM/driver
		// adoption all key off <id> -- an -incoming suffix would orphan the migrated
		// qemu on a dest restart (reconcile looks in <id> and marks it stopped).
		destDir := filepath.Join(vmBase, req.VMID)
		uri := fmt.Sprintf("tcp:0.0.0.0:%d", port)

		ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
		defer cancel()
		if err := prepareIncomingMigrationDir(destDir); err != nil {
			writeJSONError(w, http.StatusConflict, fmt.Sprintf("incoming VM destination is not available: %v", err))
			return
		}

		if req.Block {
			host := req.AdvertiseHost
			if host == "" {
				host = "127.0.0.1"
			}
			_, nbdURI, berr := kd.StartIncomingBlock(ctx, req.VMID, destDir, uri, host, req.DiskSizeBytes, req.Config)
			if berr != nil {
				cleanupOrphanedIncoming(kd, req.VMID, destDir)
				writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("start incoming block destination: %v", berr))
				return
			}
			// Once the guest resumes here, tear down the NBD export AND register the
			// migrated VM in this node's manager + DB. Background: not tied to the
			// request ctx, which is cancelled the moment we respond below.
			go func() {
				if err := kd.AwaitFinishIncomingBlock(req.VMID, 10*time.Minute); err != nil {
					logger.Warn("block incoming did not resume; cleaning up orphaned dest", "vm", req.VMID, "error", err)
					cleanupOrphanedIncoming(kd, req.VMID, destDir)
					return
				}
				clearIncomingMigrationMarker(destDir)
				registerMigratedDest(db, vmManager, req.VMID, req.Config, req.TargetNodeID, req.OrganizationID)
			}()
			writeJSON(w, http.StatusOK, core_vm.IncomingMigrationResponse{Port: port, NBDURI: nbdURI})
			return
		}
		if _, err := kd.StartIncomingWithDisk(ctx, req.VMID, destDir, uri, req.DiskPath, req.Config); err != nil {
			cleanupOrphanedIncoming(kd, req.VMID, destDir)
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("start incoming destination: %v", err))
			return
		}
		// Register the migrated VM in this node's manager + DB once its guest
		// resumes. Background: not tied to the request ctx (see block branch).
		go func() {
			if err := kd.WaitResumed(req.VMID, 10*time.Minute); err != nil {
				logger.Warn("shared incoming did not resume; cleaning up orphaned dest", "vm", req.VMID, "error", err)
				cleanupOrphanedIncoming(kd, req.VMID, destDir)
				return
			}
			clearIncomingMigrationMarker(destDir)
			registerMigratedDest(db, vmManager, req.VMID, req.Config, req.TargetNodeID, req.OrganizationID)
		}()

		writeJSON(w, http.StatusOK, core_vm.IncomingMigrationResponse{Port: port})
	}).Methods(http.MethodPost)

	// POST /internal/migrate/abort -- the source tells this dest to release a
	// half-started incoming VM after a source-side failure (e.g. unsupported
	// compression parameters). Without it the dest keeps a paused qemu holding
	// the disk lock until the no-resume watchdog fires.
	router.HandleFunc("/internal/migrate/abort", func(w http.ResponseWriter, r *http.Request) {
		if !migrationAuthOK(r) {
			writeJSONError(w, http.StatusForbidden, "forbidden")
			return
		}
		var req struct {
			VMID string `json:"vm_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil || strings.TrimSpace(req.VMID) == "" {
			writeJSONError(w, http.StatusBadRequest, "vm_id is required")
			return
		}
		if vmManager == nil {
			writeJSONError(w, http.StatusServiceUnavailable, "vm manager unavailable")
			return
		}
		if vm, err := vmManager.GetVM(req.VMID); err == nil && vm.State() == core_vm.StateRunning {
			writeJSONError(w, http.StatusConflict, "cannot abort migration for a running VM")
			return
		}
		drv, derr := vmManager.GetDriverForConfig(core_vm.VMConfig{Type: core_vm.VMTypeKVM})
		kd, ok := drv.(*core_vm.KVMDriverEnhanced)
		if derr != nil || !ok {
			writeJSONError(w, http.StatusServiceUnavailable, "kvm driver unavailable")
			return
		}
		statusCtx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
		defer cancel()
		if state, err := kd.GetStatus(statusCtx, req.VMID); err == nil && state == core_vm.StateRunning {
			writeJSONError(w, http.StatusConflict, "cannot abort migration for a running VM")
			return
		}
		destDir := filepath.Join(vmBase, req.VMID)
		if !incomingMigrationMarkerExists(destDir) {
			writeJSONError(w, http.StatusNotFound, "incoming migration not found")
			return
		}
		cleanupOrphanedIncoming(kd, req.VMID, destDir)
		writeJSON(w, http.StatusOK, map[string]interface{}{"aborted": true, "vm_id": req.VMID})
	}).Methods(http.MethodPost)
}

// freeMigrationPort asks the kernel for an unused TCP port for the incoming
// migration stream. ponytail: tiny TOCTOU between close and qemu's bind;
// acceptable, qemu surfaces a bind error if it loses the race.
func freeMigrationPort() (int, error) {
	ln, err := net.Listen("tcp", "0.0.0.0:0")
	if err != nil {
		return 0, err
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port, nil
}

// registerMigratedDest registers a VM that has just migrated ONTO this node into
// the manager (so control ops route) and the DB (so /api/vms lists it), from the
// config the source sent -- the driver already tracks the running qemu; this is
// the manager/DB half of ownership transfer. Idempotent via ON CONFLICT. Uses
// context.Background: it runs after the incoming request has returned.
// ponytail: does not re-increment this node's resource accounting for the
// migrated-in VM (AddVM is a pure map add) -- a minor dest-quota under-count,
// revisited when migration owns accounting end-to-end.
// registerMigratedDest persists a migrated VM into the destination node's
// manager and vms table. OrganizationID comes from the incoming-migration
// wire field OR from VMConfig.OrganizationID — without it the migrated row
// has NULL organization and is invisible to the destination's scoped
// listings (only admins see it), defeating cross-node tenant placement.
func registerMigratedDest(db *sql.DB, manager *core_vm.VMManager, vmID string, cfg core_vm.VMConfig, nodeID string, orgID string) {
	if manager == nil {
		return
	}
	cfg.ID = vmID
	// Fold the wire-level request field (explicit preference) into the config so
	// the SQL INSERT reads from one place, not two.
	if orgID == "" && cfg.OrganizationID != "" {
		orgID = cfg.OrganizationID
	}
	vm, err := core_vm.NewVM(cfg)
	if err != nil {
		logger.Warn("migrated-VM register skipped: rebuild failed", "vm", vmID, "error", err)
		return
	}
	vm.SetState(core_vm.StateRunning)
	if nodeID != "" {
		vm.SetNodeID(nodeID)
	}
	manager.AddVM(vm)

	// The migrating VM's owner exists in the SOURCE node's user directory, not
	// necessarily here; vms.owner_id is a local FK, so a foreign owner must
	// become NULL with the requested id preserved in vms.requested_owner_id
	// (same rule as createVMLocal). Without this the whole registration INSERT
	// failed and the migrated VM stayed invisible to this node's API (observed
	// live).
	owner := parseOwnerID(cfg.OwnerID)
	requestedOwner := owner
	if owner != nil && db != nil {
		var exists bool
		if err := db.QueryRow(`SELECT EXISTS (SELECT 1 FROM users WHERE id = $1)`, owner).Scan(&exists); err != nil || !exists {
			owner = nil
		}
	}
	if owner != nil {
		requestedOwner = nil // resolved locally: no divergence to record
	}
	// vms.node_id is the cluster node id string (NOVACRON_NODE_ID /
	// cluster_peers.node_id), not a nodes(id) UUID -- see novacron-ok7. It is
	// now written for real, so "which node is this VM on" is queryable via
	// plain SQL instead of every consumer having to parse metadata JSON.
	configPayload, _ := json.Marshal(map[string]interface{}{
		"cpu_shares":   cfg.CPUShares,
		"vcpus":        vcpusOrDefault(cfg.VCPUs),
		"memory_mb":    cfg.MemoryMB,
		"disk_size_gb": cfg.DiskSizeGB,
		"image":        cfg.Image,
	})
	if _, err := db.Exec(`
		INSERT INTO vms (id, name, state, cpu_cores, memory_mb, disk_gb, os_type, node_id, owner_id, requested_owner_id, metadata, organization_id, created_at, updated_at)
		VALUES ($1, $2, 'running', $3, $4, $5, $6, NULLIF($7, ''), NULLIF($8, '')::uuid, NULLIF($9, '')::uuid, $10, NULLIF($11, '')::uuid, NOW(), NOW())
		ON CONFLICT (id) DO UPDATE SET
			state = 'running', node_id = EXCLUDED.node_id, owner_id = EXCLUDED.owner_id,
			requested_owner_id = EXCLUDED.requested_owner_id, metadata = EXCLUDED.metadata,
			-- org passes source->dest via the migrate wire (novacron-wot): the
			-- destination stamps org from the IncomingMigrationRequest so a
			-- migrated VM lands inside its owner's org scope, never invisible.
			organization_id = COALESCE(EXCLUDED.organization_id, vms.organization_id), updated_at = NOW()
	`, vmID, cfg.Name, vcpusOrDefault(cfg.VCPUs), cfg.MemoryMB, cfg.DiskSizeGB, nullableStringValue(cfg.Image), nodeID, ownerString(owner), ownerString(requestedOwner), configPayload, orgID); err != nil {
		logger.Warn("migrated-VM DB register failed", "vm", vmID, "error", err)
		return
	}
	logger.Info("registered migrated-in VM on destination node", "vm", vmID, "node", nodeID)
}

// ownerString renders a parsed owner for the SQL NULLIF($8,”)::uuid idiom.
func ownerString(owner interface{}) string {
	if owner == nil {
		return ""
	}
	if s, ok := owner.(string); ok {
		return s
	}
	return ""
}

// registerConfiguredPeers registers migration peer nodes from the NOVACRON_PEERS
// env so a migrate request can resolve a bare target_node to its address, instead
// of the caller passing target_addr. Format: comma-separated id=host:port pairs,
// e.g. NOVACRON_PEERS="node2=10.0.0.2:9090,node3=10.0.0.3:9090". ponytail: a static
// env map -- a real cluster gets a gossip/registry with federation; this removes
// the manual escape hatch for a known-topology cluster with no new infra.
func registerConfiguredPeers(vmManager *core_vm.VMManager) {
	raw := strings.TrimSpace(os.Getenv("NOVACRON_PEERS"))
	if raw == "" || vmManager == nil {
		return
	}
	for _, pair := range strings.Split(raw, ",") {
		id, addr, ok := strings.Cut(strings.TrimSpace(pair), "=")
		id, addr = strings.TrimSpace(id), strings.TrimSpace(addr)
		if !ok || id == "" || addr == "" {
			logger.Warn("NOVACRON_PEERS entry ignored (want id=host:port)", "entry", pair)
			continue
		}
		if _, _, err := net.SplitHostPort(addr); err != nil {
			logger.Warn("NOVACRON_PEERS entry ignored (bad host:port)", "entry", pair, "error", err)
			continue
		}
		vmManager.RegisterMigrationPeer(id, addr)
		logger.Info("registered migration peer", "node", id, "migration_addr", addr)
	}
}

// parseOwnerID turns a config OwnerID string into a nullable uuid string for
// the vms.owner_id FK (users.id): NULL if absent/non-uuid, rather than failing
// registration.
func parseOwnerID(s string) interface{} {
	if _, err := uuid.Parse(strings.TrimSpace(s)); err != nil {
		return nil
	}
	return strings.TrimSpace(s)
}

// registerVMMigrateRoute wires POST /vms/{id}/migrate to the real manager's
// MigrateVM. Body: {target_node, migration_type?, uri?}. An explicit uri (QEMU
// migration URI, e.g. "tcp:host:port") flows straight through to the KVM driver;
// otherwise target_node is resolved to a URI inside the manager (see migrateVM).
// On success the DB row's node_id/state are updated to the observed reality.
func registerVMMigrateRoute(router *mux.Router, db *sql.DB, vmManager *core_vm.VMManager) {
	var migrateVM func(context.Context, string, string, map[string]string) error
	if vmManager != nil {
		migrateVM = vmManager.MigrateVM
	}
	registerVMMigrateRouteWithExecutor(router, db, migrateVM)
}

func registerVMMigrateRouteWithExecutor(router *mux.Router, db *sql.DB, migrateVM func(context.Context, string, string, map[string]string) error) {
	router.HandleFunc("/vms/{id}/migrate", func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]
		scopeOrg, isAdmin, visible := requireOrgScope(r.Context(), db, vmID)
		if !visible {
			writeJSONError(w, http.StatusNotFound, "vm not found")
			return
		}
		if migrateVM == nil {
			writeJSONError(w, http.StatusServiceUnavailable, "vm manager unavailable")
			return
		}

		var req struct {
			TargetNode    string `json:"target_node"`
			MigrationType string `json:"migration_type,omitempty"`
			URI           string `json:"uri,omitempty"`
			TargetAddr    string `json:"target_addr,omitempty"` // dest api host:port for block migration
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if strings.TrimSpace(req.TargetNode) == "" {
			writeJSONError(w, http.StatusBadRequest, "target_node is required")
			return
		}

		options := map[string]string{}
		if req.MigrationType != "" {
			options["migration_type"] = req.MigrationType
		}
		if req.URI != "" {
			options["uri"] = req.URI // explicit QEMU URI passes straight to the driver
		}
		if req.TargetAddr != "" {
			options["target_addr"] = req.TargetAddr // dest api addr for block migration
		}

		// Live migration can take well over 30s; give it a generous ceiling.
		ctx, cancel := context.WithTimeout(r.Context(), 10*time.Minute)
		defer cancel()

		if err := migrateVM(ctx, vmID, req.TargetNode, options); err != nil {
			writeJSONError(w, vmActionStatus(err), fmt.Sprintf("failed to migrate VM: %v", err))
			return
		}

		// The VM now runs on req.TargetNode; the manager already retired it here
		// (forgetVM). Remove the source DB row too so this node stops listing a
		// guest that has moved away -- the destination inserts its own row when the
		// incoming guest resumes (registerMigratedDest).
		var result sql.Result
		var err error
		if isAdmin {
			result, err = db.Exec(`DELETE FROM vms WHERE id = $1`, vmID)
		} else if scopeOrg != "" {
			result, err = db.Exec(`DELETE FROM vms WHERE id = $1 AND organization_id = $2`, vmID, scopeOrg)
		} else {
			result, err = db.Exec(`DELETE FROM vms WHERE id = $1 AND organization_id IS NULL`, vmID)
		}
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("migration succeeded to node %s but source VM row cleanup failed: %v", req.TargetNode, err))
			return
		}
		rowsAffected, err := result.RowsAffected()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("migration succeeded to node %s but source VM row cleanup result could not be confirmed: %v", req.TargetNode, err))
			return
		}
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("migration succeeded to node %s but source VM row cleanup failed: no source row was deleted", req.TargetNode))
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"id":      vmID,
			"state":   "migrated",
			"status":  "migrated",
			"node_id": req.TargetNode,
		})
	}).Methods(http.MethodPost)
}

// reconcileVMState corrects each persisted VM row to match whether its qemu
// process is actually alive (via the driver's pidfile) after a restart, and
// re-registers every still-running VM with the manager (see adoptManagerVM) so
// that a post-restart Stop/Delete reaches the re-adopted qemu instead of 404ing
// on an empty in-memory map.
func reconcileVMState(db *sql.DB, vmBase string, manager *core_vm.VMManager) {
	rows, err := db.Query(`SELECT id, state FROM vms`)
	if err != nil {
		logger.Warn("VM reconcile skipped: query failed", "error", err)
		return
	}
	type rec struct{ id, state string }
	var recs []rec
	for rows.Next() {
		var rc rec
		if err := rows.Scan(&rc.id, &rc.state); err == nil {
			recs = append(recs, rc)
		}
	}
	rows.Close()

	for _, rc := range recs {
		alive := pidFileAlive(filepath.Join(vmBase, rc.id, "qemu.pid"), rc.id)
		if alive {
			// Repopulate the manager so post-restart control ops (stop/delete)
			// route to the driver that re-adopted this qemu instead of 404ing.
			adoptManagerVM(manager, vmBase, rc.id)
		}
		var actual string
		switch {
		case alive && rc.state != "running":
			actual = "running"
		case !alive && rc.state == "running":
			actual = "stopped"
		default:
			continue
		}
		if _, err := db.Exec(`UPDATE vms SET state = $2, updated_at = NOW() WHERE id = $1`, rc.id, actual); err != nil {
			logger.Warn("VM reconcile update failed", "vm", rc.id, "error", err)
			continue
		}
		logger.Info("VM reconciled to actual qemu state", "vm", rc.id, "from", rc.state, "to", actual)
	}
}

// reconcileOrphanedMigrationDests finds VM directories under vmBase with a
// live qemu process but NO corresponding vms row at all -- a migration
// destination whose registerMigratedDest goroutine died with the prior
// api-server process before it ever ran (killed/restarted between the guest
// resuming and that goroutine reaching its INSERT). reconcileVMState above
// cannot see these: its query only iterates rows that already exist.
// Reuses registerMigratedDest itself (idempotent via ON CONFLICT) rather
// than duplicating its owner-resolution and INSERT logic (novacron-05h).
func reconcileOrphanedMigrationDests(db *sql.DB, vmBase string, manager *core_vm.VMManager) {
	if db == nil {
		return
	}
	known := map[string]bool{}
	rows, err := db.Query(`SELECT id FROM vms`)
	if err != nil {
		logger.Warn("orphaned migration dest reconcile skipped: query failed", "error", err)
		return
	}
	for rows.Next() {
		var id string
		if rows.Scan(&id) == nil {
			known[id] = true
		}
	}
	rows.Close()

	entries, err := os.ReadDir(vmBase)
	if err != nil {
		return // no vmBase directory yet (fresh node) is not an error
	}
	for _, e := range entries {
		if !e.IsDir() || known[e.Name()] {
			continue
		}
		id := e.Name()
		if !pidFileAlive(filepath.Join(vmBase, id, "qemu.pid"), id) {
			continue // no live process either -- not an orphan this reconcile owns
		}
		data, err := os.ReadFile(filepath.Join(vmBase, id, "config.json"))
		if err != nil {
			logger.Warn("orphaned migration dest has a live process but no readable config.json; cannot re-register it", "vm", id, "error", err)
			continue
		}
		var cfg core_vm.VMConfig
		if err := json.Unmarshal(data, &cfg); err != nil {
			logger.Warn("orphaned migration dest has a live process but an unparseable config.json; cannot re-register it", "vm", id, "error", err)
			continue
		}
		cfg.ID = id
		logger.Warn("adopting an orphaned migration destination with no DB row (its registerMigratedDest goroutine likely died with a prior api-server process)", "vm", id)
		registerMigratedDest(db, manager, id, cfg, selfNodeID(), cfg.OrganizationID)
	}
}

// adoptManagerVM rebuilds an in-memory VM object from its persisted config.json
// and registers it (state Running) with the manager, so that after an
// api-server restart a stop/delete call finds the VM and routes to the driver
// that re-adopted its live qemu. Without this the manager's in-memory map is
// empty on a fresh process and control ops 404 even though the qemu is alive.
// No-op if the manager already knows the VM or the config cannot be rebuilt.
func adoptManagerVM(manager *core_vm.VMManager, vmBase, id string) {
	if manager == nil {
		return
	}
	if _, err := manager.GetVM(id); err == nil {
		return
	}
	data, err := os.ReadFile(filepath.Join(vmBase, id, "config.json"))
	if err != nil {
		logger.Warn("VM re-adopt skipped: config unreadable", "vm", id, "error", err)
		return
	}
	var cfg core_vm.VMConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		logger.Warn("VM re-adopt skipped: config unparseable", "vm", id, "error", err)
		return
	}
	cfg.ID = id
	vm, err := core_vm.NewVM(cfg)
	if err != nil {
		logger.Warn("VM re-adopt skipped: rebuild failed", "vm", id, "error", err)
		return
	}
	vm.SetState(core_vm.StateRunning)
	manager.AddVM(vm)
	logger.Info("VM re-adopted into manager after restart", "vm", id)
}

// pidFileAlive reports whether the pid recorded in path is a live process that
// is actually this VM's qemu. The /proc cmdline check guards against a recycled
// PID (host reboot / PID reuse): a live-but-unrelated process must not be
// mistaken for the running VM and later signalled by a Stop/Delete.
func pidFileAlive(path, vmID string) bool {
	data, err := os.ReadFile(path)
	if err != nil {
		return false
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 0 {
		return false
	}
	if syscall.Kill(pid, 0) != nil { // signal 0 probes existence only
		return false
	}
	cmdline, err := os.ReadFile(fmt.Sprintf("/proc/%d/cmdline", pid))
	if err != nil {
		return false
	}
	return strings.Contains(string(cmdline), "qemu") && strings.Contains(string(cmdline), vmID)
}

func initializeCanonicalServices(cfg *config.Config, db *sql.DB, authManager *auth.SimpleAuthManager) (*canonicalServices, error) {
	auditLogger := audit.NewSimpleAuditLogger()
	twoFactorService := auth.NewTwoFactorService("NovaCron", []byte(authManager.GetJWTSecret()), db)
	securityHandlers := securityapi.NewSecurityHandlers(twoFactorService, auditLogger).WithRBACStore(securityapi.NewPostgresRBACStore(db))

	volumeStore, err := storage.NewStorageManager(storage.StorageManagerConfig{
		BasePath: filepath.Join(cfg.VM.StoragePath, "volumes"),
	})
	if err != nil {
		return nil, fmt.Errorf("failed to initialize storage-backed volume store: %w", err)
	}

	graphqlResolver := graphqlapi.NewResolverWithVolumeStore(nil, nil, volumeStore)
	websocketLogger := logrus.New()
	websocketLogger.SetLevel(logrus.InfoLevel)
	websocketHandler := websocketapi.NewWebSocketHandler(nil, nil, nil, nil, websocketLogger)

	// Initialize orchestration components
	orchLogger := logrus.New()
	orchLogger.SetLevel(logrus.InfoLevel)

	// Create event bus (noop for now, can be swapped for NATS)
	eventBus := events.NewNoopEventBus()

	// Create orchestration engine
	orchEngine := orchestration.NewDefaultOrchestrationEngine(orchLogger)
	orchEngine.SetEvacuationHandler(nil) // Can be wired later if needed

	// Create sub-components
	placementEngine := placement.NewDefaultPlacementEngine(orchLogger)
	autoScaler := autoscaling.NewDefaultAutoScaler(orchLogger, eventBus)
	if err := autoScaler.SetMetricsSource(func() (*autoscaling.MetricsData, error) {
		return metricsDataFromHost(hostMetrics(vmBasePath(cfg)))
	}); err != nil {
		orchLogger.Warnf("Failed to set auto-scaler metrics source: %v", err)
	}
	healingController := healing.NewDefaultHealingController(orchLogger, eventBus)
	policyEngine := policy.NewDefaultPolicyEngine(orchLogger, eventBus)

	// Wire components to orchestration engine
	// Note: The engine internally creates its own placement/eventbus but we can inject ours
	// For now, start the sub-components
	ctx := context.Background()
	if err := autoScaler.StartMonitoring(); err != nil {
		orchLogger.Warnf("Failed to start auto-scaler monitoring: %v", err)
	}
	if err := healingController.StartMonitoring(); err != nil {
		orchLogger.Warnf("Failed to start healing controller monitoring: %v", err)
	}
	if err := orchEngine.Start(ctx); err != nil {
		orchLogger.Warnf("Failed to start orchestration engine: %v", err)
	}

	// Create orchestration API handler
	orchAPI := orchestrationapi.NewOrchestrationAPI(
		orchLogger,
		orchEngine,
		autoScaler,
		healingController,
		policyEngine,
		placementEngine,
	)

	return &canonicalServices{
		twoFactorService:    twoFactorService,
		securityHandlers:    securityHandlers,
		websocketHandler:    websocketHandler,
		graphqlHandler:      graphqlapi.NewVolumeHTTPHandler(graphqlResolver),
		orchestrationAPI:    orchAPI,
		orchestrationEngine: orchEngine,
		shutdown: func() {
			websocketHandler.Shutdown()
			// Gracefully stop orchestration components
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			healingController.StopMonitoring()
			autoScaler.StopMonitoring()
			orchEngine.Stop(shutdownCtx)
		},
	}, nil
}

func registerCanonicalSecurityRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB, handlers *securityapi.SecurityHandlers) {
	twoFactorRouter := router.PathPrefix("/api/auth/2fa").Subrouter()
	twoFactorRouter.Use(requireAuth(authManager, db))
	twoFactorRouter.HandleFunc("/setup", handlers.Setup2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/qr", handlers.GenerateQRCode).Methods(http.MethodGet)
	twoFactorRouter.HandleFunc("/verify", handlers.Verify2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/enable", handlers.Enable2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/disable", handlers.Disable2FA).Methods(http.MethodPost)
	twoFactorRouter.HandleFunc("/status", handlers.Get2FAStatus).Methods(http.MethodGet)
	twoFactorRouter.HandleFunc("/backup-codes", handlers.GetBackupCodes).Methods(http.MethodGet)
	twoFactorRouter.HandleFunc("/backup-codes", handlers.RegenerateBackupCodes).Methods(http.MethodPost)

	registerSecurityRouteSet(router.PathPrefix("/api/security").Subrouter(), authManager, db, handlers)
	registerSecurityRouteSet(router.PathPrefix("/api/admin/security").Subrouter(), authManager, db, handlers)
}

func registerCanonicalAdminRoutes(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB) {
	adminRouter := router.PathPrefix("/api/admin").Subrouter()
	adminRouter.Use(requireAuth(authManager, db))
	adminRouter.Use(requireAnyRoleMiddleware("admin", "super-admin"))

	adminRouter.HandleFunc("/users", listCanonicalAdminUsers(db)).Methods(http.MethodGet)
	adminRouter.HandleFunc("/users", createCanonicalAdminUser(db)).Methods(http.MethodPost)
	adminRouter.HandleFunc("/users/{id}", updateCanonicalAdminUser(db)).Methods(http.MethodPut)
	adminRouter.HandleFunc("/users/{id}", deleteCanonicalAdminUser(db)).Methods(http.MethodDelete)
	adminRouter.HandleFunc("/users/{id}/roles", assignCanonicalAdminUserRoles(db)).Methods(http.MethodPost)
}

type canonicalAdminUser struct {
	ID        string    `json:"id"` // canonical users.id is a uuid
	Username  string    `json:"username"`
	Email     string    `json:"email"`
	Role      string    `json:"role"`   // canonical user_role enum: admin|operator|viewer
	Status    string    `json:"status"` // canonical user_status enum
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// active() reports account liveness from the canonical user_status enum; the
// legacy boolean "active" column does not exist in the canonical schema.
func (u canonicalAdminUser) active() bool {
	return u.Status == "active"
}

type canonicalAdminUserListResponse struct {
	Users      []canonicalAdminUser `json:"users"`
	Total      int                  `json:"total"`
	Page       int                  `json:"page"`
	PageSize   int                  `json:"page_size"`
	TotalPages int                  `json:"total_pages"`
}
type canonicalAdminCreateUserRequest struct {
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"password"`
	Role     string `json:"role"`
}

type canonicalAdminUpdateUserRequest struct {
	Username string `json:"username,omitempty"`
	Email    string `json:"email,omitempty"`
	Role     string `json:"role,omitempty"`
	Active   *bool  `json:"active,omitempty"`
}

func listCanonicalAdminUsers(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		page := parsePositiveInt(r.URL.Query().Get("page"), 1)
		pageSize := parsePositiveInt(r.URL.Query().Get("page_size"), 20)
		if pageSize > 100 {
			pageSize = 100
		}

		search := strings.TrimSpace(r.URL.Query().Get("search"))
		filterRole := ""
		if rawRole := strings.TrimSpace(r.URL.Query().Get("role")); rawRole != "" {
			role, ok := canonicalUserRoleForAdmin(rawRole)
			if !ok {
				writeJSONError(w, http.StatusBadRequest, "invalid role filter")
				return
			}
			filterRole = role
		}

		offset := (page - 1) * pageSize
		whereParts := make([]string, 0, 2)
		args := make([]interface{}, 0, 4)

		if search != "" {
			whereParts = append(whereParts, fmt.Sprintf("(username ILIKE $%d OR email ILIKE $%d)", len(args)+1, len(args)+2))
			args = append(args, "%"+search+"%", "%"+search+"%")
		}
		if filterRole != "" {
			whereParts = append(whereParts, fmt.Sprintf("role = $%d", len(args)+1))
			args = append(args, filterRole)
		}

		whereClause := ""
		if len(whereParts) > 0 {
			whereClause = " WHERE " + strings.Join(whereParts, " AND ")
		}

		var total int
		countQuery := "SELECT COUNT(*) FROM users" + whereClause
		if err := db.QueryRow(countQuery, args...).Scan(&total); err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to count users: %v", err))
			return
		}

		listQuery := fmt.Sprintf(`
			SELECT id, username, email, role, status, created_at, updated_at
			FROM users
			%s
			ORDER BY created_at DESC
			LIMIT $%d OFFSET $%d
		`, whereClause, len(args)+1, len(args)+2)
		listArgs := append(args, pageSize, offset)

		rows, err := db.Query(listQuery, listArgs...)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to list users: %v", err))
			return
		}
		defer rows.Close()

		users := make([]canonicalAdminUser, 0)
		for rows.Next() {
			var user canonicalAdminUser
			if err := rows.Scan(&user.ID, &user.Username, &user.Email, &user.Role, &user.Status, &user.CreatedAt, &user.UpdatedAt); err != nil {
				writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to scan user: %v", err))
				return
			}
			users = append(users, user)
		}
		if err := rows.Err(); err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to list users: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, canonicalAdminUserListResponse{
			Users:      users,
			Total:      total,
			Page:       page,
			PageSize:   pageSize,
			TotalPages: maxInt(1, (total+pageSize-1)/pageSize),
		})
	}
}

func createCanonicalAdminUser(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req canonicalAdminCreateUserRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		req.Username = strings.TrimSpace(req.Username)
		req.Email = strings.TrimSpace(req.Email)
		if req.Role == "" {
			req.Role = "viewer" // column default; empty means "no explicit role"
		}
		role, ok := canonicalUserRoleForAdmin(req.Role)
		if !ok {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid role: %s (valid: admin, operator, viewer; aliases: super-admin, user, readonly)", req.Role))
			return
		}
		req.Role = role

		if req.Username == "" || req.Email == "" || strings.TrimSpace(req.Password) == "" {
			writeJSONError(w, http.StatusBadRequest, "username, email, and password are required")
			return
		}
		if !strings.Contains(req.Email, "@") {
			writeJSONError(w, http.StatusBadRequest, "invalid email format")
			return
		}

		passwordHash, err := bcrypt.GenerateFromPassword([]byte(strings.TrimSpace(req.Password)), bcrypt.DefaultCost)
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to hash password: %v", err))
			return
		}

		var user canonicalAdminUser
		err = db.QueryRow(`
			INSERT INTO users (username, email, password_hash, role, status)
			VALUES ($1, $2, $3, $4, 'active')
			RETURNING id, username, email, role, status, created_at, updated_at
		`, req.Username, req.Email, string(passwordHash), req.Role).Scan(
			&user.ID,
			&user.Username,
			&user.Email,
			&user.Role,
			&user.Status,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			if canonicalAdminNotFoundError(err) {
				writeJSONError(w, http.StatusNotFound, "user not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to create user: %v", err))
			return
		}
		writeJSON(w, http.StatusCreated, user)
	}
}

func updateCanonicalAdminUser(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := canonicalAdminUserID(mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}

		var req canonicalAdminUpdateUserRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}

		updates := make([]string, 0, 4)
		args := make([]interface{}, 0, 5)

		if username := strings.TrimSpace(req.Username); username != "" {
			updates = append(updates, fmt.Sprintf("username = $%d", len(args)+1))
			args = append(args, username)
		}
		if email := strings.TrimSpace(req.Email); email != "" {
			if !strings.Contains(email, "@") {
				writeJSONError(w, http.StatusBadRequest, "invalid email format")
				return
			}
			updates = append(updates, fmt.Sprintf("email = $%d", len(args)+1))
			args = append(args, email)
		}
		if req.Role != "" {
			role, ok := canonicalUserRoleForAdmin(req.Role)
			if !ok {
				writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid role: %s (valid: admin, operator, viewer; aliases: super-admin, user, readonly)", req.Role))
				return
			}
			updates = append(updates, fmt.Sprintf("role = $%d", len(args)+1))
			args = append(args, role)
		}
		if req.Active != nil {
			// canonical user_status enum replaces the legacy boolean column
			status := "inactive"
			if *req.Active {
				status = "active"
			}
			updates = append(updates, fmt.Sprintf("status = $%d", len(args)+1))
			args = append(args, status)
		}
		if len(updates) == 0 {
			writeJSONError(w, http.StatusBadRequest, "no fields to update")
			return
		}

		updates = append(updates, "updated_at = NOW()")
		args = append(args, userID)

		query := fmt.Sprintf(`
			UPDATE users
			SET %s
			WHERE id = $%d
			RETURNING id, username, email, role, status, created_at, updated_at
		`, strings.Join(updates, ", "), len(args))

		var user canonicalAdminUser
		err = db.QueryRow(query, args...).Scan(
			&user.ID,
			&user.Username,
			&user.Email,
			&user.Role,
			&user.Status,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			if canonicalAdminNotFoundError(err) {
				writeJSONError(w, http.StatusNotFound, "user not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to update user: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, user)
	}
}

func deleteCanonicalAdminUser(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := canonicalAdminUserID(mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}

		result, err := db.Exec(`DELETE FROM users WHERE id = $1`, userID)
		if err != nil {
			if canonicalAdminNotFoundError(err) {
				writeJSONError(w, http.StatusNotFound, "user not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to delete user: %v", err))
			return
		}
		rowsAffected, err := result.RowsAffected()
		if err != nil {
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to confirm user deletion: %v", err))
			return
		}
		if rowsAffected == 0 {
			writeJSONError(w, http.StatusNotFound, "user not found")
			return
		}

		w.WriteHeader(http.StatusNoContent)
	}
}

func assignCanonicalAdminUserRoles(db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		userID, err := canonicalAdminUserID(mux.Vars(r)["id"])
		if err != nil {
			writeJSONError(w, http.StatusBadRequest, err.Error())
			return
		}

		var req struct {
			Roles []string `json:"roles"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, http.StatusBadRequest, "invalid request body")
			return
		}
		if len(req.Roles) == 0 {
			writeJSONError(w, http.StatusBadRequest, "at least one role is required")
			return
		}

		role, ok := canonicalUserRoleForAdmin(req.Roles[0])
		if !ok {
			writeJSONError(w, http.StatusBadRequest, fmt.Sprintf("invalid role: %s (valid: admin, operator, viewer; aliases: super-admin, user, readonly)", req.Roles[0]))
			return
		}
		var user canonicalAdminUser
		err = db.QueryRow(`
			UPDATE users
			SET role = $1, updated_at = NOW()
			WHERE id = $2
			RETURNING id, username, email, role, status, created_at, updated_at
		`, role, userID).Scan(
			&user.ID,
			&user.Username,
			&user.Email,
			&user.Role,
			&user.Status,
			&user.CreatedAt,
			&user.UpdatedAt,
		)
		if err != nil {
			if canonicalAdminNotFoundError(err) {
				writeJSONError(w, http.StatusNotFound, "user not found")
				return
			}
			writeJSONError(w, http.StatusInternalServerError, fmt.Sprintf("failed to assign user role: %v", err))
			return
		}

		writeJSON(w, http.StatusOK, map[string]interface{}{
			"user":    user,
			"message": "roles updated",
		})
	}
}

// canonicalAdminUserID validates the path id as a canonical users.id (uuid).
func canonicalAdminUserID(raw string) (string, error) {
	userID := strings.TrimSpace(raw)
	if _, err := uuid.Parse(userID); err != nil {
		return "", fmt.Errorf("invalid user ID")
	}
	return userID, nil
}

// canonicalAdminNotFoundError maps missing-row query results and database
// errors that explicitly report a missing user to the hidden 404 response.
func canonicalAdminNotFoundError(err error) bool {
	return errors.Is(err, sql.ErrNoRows) || strings.Contains(strings.ToLower(err.Error()), "not found")
}

// canonicalUserRoleForAdmin maps accepted role labels onto the canonical
// user_role enum ('admin','operator','viewer'); 'user'/'readonly' collapse to
// 'viewer' and 'super-admin' to 'admin', matching the RBAC catalog seeds.
// Anything unrecognized returns ok=false so callers reject the request (400)
// instead of silently creating a viewer.
func canonicalUserRoleForAdmin(raw string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "admin", "super-admin":
		return "admin", true
	case "operator":
		return "operator", true
	case "viewer", "user", "readonly":
		return "viewer", true
	default:
		return "", false
	}
}

func registerSecurityRouteSet(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB, handlers *securityapi.SecurityHandlers) {
	router.Use(requireAuth(authManager, db))
	router.Use(requireAnyRoleMiddleware("admin", "super-admin"))

	router.HandleFunc("/threats", handlers.GetThreats).Methods(http.MethodGet)
	router.HandleFunc("/vulnerabilities", handlers.GetVulnerabilities).Methods(http.MethodGet)
	router.HandleFunc("/compliance", handlers.GetComplianceStatus).Methods(http.MethodGet)
	router.HandleFunc("/compliance/check", handlers.TriggerComplianceCheck).Methods(http.MethodPost)
	router.HandleFunc("/compliance/export", handlers.ExportComplianceReport).Methods(http.MethodGet)
	router.HandleFunc("/incidents", handlers.GetIncidents).Methods(http.MethodGet)
	router.HandleFunc("/incidents", handlers.CreateSecurityIncident).Methods(http.MethodPost)
	router.HandleFunc("/events", handlers.GetSecurityEvents).Methods(http.MethodGet)
	router.HandleFunc("/events/{eventId}/acknowledge", handlers.AcknowledgeSecurityEvent).Methods(http.MethodPost)
	router.HandleFunc("/scan", handlers.StartVulnerabilityScan).Methods(http.MethodPost)
	router.HandleFunc("/scan/{scanId}", handlers.GetScanResults).Methods(http.MethodGet)
	router.HandleFunc("/cluster/{clusterId}/state", handlers.GetClusterSecurityState).Methods(http.MethodGet)
	router.HandleFunc("/audit/events", handlers.GetAuditEvents).Methods(http.MethodGet)
	router.HandleFunc("/audit/export", handlers.ExportAuditLog).Methods(http.MethodGet)
	router.HandleFunc("/audit/statistics", handlers.GetAuditStatistics).Methods(http.MethodGet)
	router.HandleFunc("/rbac/roles", handlers.GetRoles).Methods(http.MethodGet)
	router.HandleFunc("/rbac/roles", handlers.CreateRole).Methods(http.MethodPost)
	router.HandleFunc("/rbac/roles/{roleId}", handlers.UpdateRole).Methods(http.MethodPut)
	router.HandleFunc("/rbac/roles/{roleId}", handlers.DeleteRole).Methods(http.MethodDelete)
	router.HandleFunc("/rbac/permissions", handlers.GetPermissions).Methods(http.MethodGet)
	router.HandleFunc("/rbac/user/{userId}/roles", handlers.GetUserRoles).Methods(http.MethodGet)
	router.HandleFunc("/rbac/user/{userId}/roles", handlers.AssignUserRoles).Methods(http.MethodPost)
	router.HandleFunc("/rbac/user/{userId}/permissions", handlers.GetUserPermissions).Methods(http.MethodGet)
}

func registerCanonicalGraphQLRoute(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB, handler http.Handler) {
	router.Handle("/graphql", requireAuth(authManager, db)(handler)).Methods(http.MethodPost)
}

func registerSecurityWebSocketAliases(router *mux.Router, authManager *auth.SimpleAuthManager, db *sql.DB, handlers *securityapi.SecurityHandlers) {
	securityStream := requireAuth(authManager, db)(requireRoleHandler("admin", handlers.StreamSecurityEvents))
	router.Handle("/api/ws/security/events", securityStream).Methods(http.MethodGet)

	compatSecurityRouter := router.PathPrefix("/api/security").Subrouter()
	compatSecurityRouter.Use(requireAuth(authManager, db))
	compatSecurityRouter.Handle("/events/stream", requireRoleHandler("admin", handlers.StreamSecurityEvents)).Methods(http.MethodGet)
}

func healthCheckHandler(cfg *config.Config, db *sql.DB) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		checks := make(map[string]string)
		status := "healthy"

		ctx, cancel := context.WithTimeout(r.Context(), 2*time.Second)
		defer cancel()
		if err := db.PingContext(ctx); err != nil {
			checks["database"] = fmt.Sprintf("error: %v", err)
			status = "unhealthy"
		} else {
			checks["database"] = "ok"
		}

		if _, err := os.Stat(cfg.VM.StoragePath); err != nil {
			checks["storage"] = fmt.Sprintf("warning: %v", err)
		} else {
			checks["storage"] = "ok"
		}

		response := map[string]interface{}{
			"status":    status,
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"version":   "1.0.0",
			"service":   "novacron-api",
			"checks":    checks,
		}

		if status == "unhealthy" {
			writeJSON(w, http.StatusServiceUnavailable, response)
			return
		}

		writeJSON(w, http.StatusOK, response)
	}
}

func apiInfoHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"name":        "NovaCron API",
			"version":     "1.0.0",
			"description": "Distributed VM Management System",
			"endpoints": []string{
				"/api/auth/login",
				"/api/auth/register",
				"/api/auth/check-email",
				"/api/auth/forgot-password",
				"/api/auth/reset-password",
				"/api/auth/2fa/setup",
				"/api/auth/2fa/verify",
				"/api/auth/2fa/verify-login",
				"/api/auth/2fa/enable",
				"/api/auth/2fa/disable",
				"/api/auth/2fa/status",
				"/api/auth/2fa/backup-codes",
				"/api/v1/vms",
				"/api/v1/vms/{id}",
				"/api/v1/vms/{id}/start",
				"/api/v1/vms/{id}/stop",
				"/api/v1/vms/{id}/metrics",
				"/api/v1/monitoring/metrics",
				"/api/v1/monitoring/vms",
				"/api/v1/monitoring/alerts",
				"/api/security/threats",
				"/api/security/vulnerabilities",
				"/api/security/compliance",
				"/api/security/compliance/check",
				"/api/security/compliance/export",
				"/api/security/incidents",
				"/api/security/events",
				"/api/security/events/{eventId}/acknowledge",
				"/api/security/scan",
				"/api/security/audit/events",
				"/api/security/audit/export",
				"/api/security/audit/statistics",
				"/api/security/rbac/roles",
				"/api/security/rbac/permissions",
				"/api/admin/users",
				"/api/admin/users/{id}",
				"/api/admin/users/{id}/roles",
				"/api/admin/security/threats",
				"/api/admin/security/compliance",
				"/api/admin/security/compliance/check",
				"/api/admin/security/compliance/export",
				"/api/admin/security/incidents",
				"/api/admin/security/audit/export",
				"/api/admin/security/events/{eventId}/acknowledge",
				"/graphql",
				"/api/v1/networks",
				"/api/v1/networks/{id}",
				"/api/v1/vms/{vm_id}/interfaces",
				"/api/v1/vms/{vm_id}/interfaces/{id}",
				"/api/ws/metrics",
				"/api/ws/alerts",
				"/api/ws/logs",
				"/api/ws/security/events",
				"/api/ws/console/{vmId}",
				"/health",
			},
			"compatibility_endpoints": []string{
				"/auth/login",
				"/auth/register",
				"/api/vms",
				"/api/monitoring/metrics",
				"/api/monitoring/vms",
				"/api/monitoring/alerts",
				"/api/networks",
				"/api/networks/{id}",
				"/api/vms/{vm_id}/interfaces",
				"/api/vms/{vm_id}/interfaces/{id}",
				"/ws/metrics",
				"/ws/alerts",
				"/ws/logs",
				"/api/security/events/stream",
			},
			"unsupported_endpoints": []string{
				"unsupported GraphQL operations outside storage-backed volume queries and mutations",
			},
		})
	}
}

func extractBearerToken(authHeader string) (string, error) {
	if authHeader == "" {
		return "", fmt.Errorf("authorization header required")
	}
	if !strings.HasPrefix(authHeader, "Bearer ") {
		return "", fmt.Errorf("authorization header must start with 'Bearer '")
	}

	token := strings.TrimSpace(strings.TrimPrefix(authHeader, "Bearer "))
	if token == "" {
		return "", fmt.Errorf("token is required")
	}
	return token, nil
}

func validateJWT(tokenString, jwtSecret string) (jwt.MapClaims, error) {
	token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
		if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
			return nil, fmt.Errorf("unexpected signing method: %s", token.Method.Alg())
		}
		return []byte(jwtSecret), nil
	})
	if err != nil {
		return nil, err
	}
	if !token.Valid {
		return nil, fmt.Errorf("token is invalid")
	}

	claims, ok := token.Claims.(jwt.MapClaims)
	if !ok {
		return nil, fmt.Errorf("invalid claims")
	}
	return claims, nil
}

func resolveLoginUsername(db *sql.DB, identifier string) (string, error) {
	if !strings.Contains(identifier, "@") {
		return identifier, nil
	}

	var username string
	if err := db.QueryRow(`SELECT username FROM users WHERE email = $1`, identifier).Scan(&username); err != nil {
		return "", err
	}
	return username, nil
}

func defaultUsernameFromEmail(email string) string {
	localPart := strings.TrimSpace(strings.SplitN(email, "@", 2)[0])
	if localPart == "" {
		return fmt.Sprintf("user-%d", time.Now().Unix())
	}
	return localPart
}

func userHasEnabledTwoFactor(twoFactorService *auth.TwoFactorService, userID string) bool {
	if twoFactorService == nil || strings.TrimSpace(userID) == "" {
		return false
	}

	info, err := twoFactorService.GetUserTwoFactorInfo(userID)
	return err == nil && info != nil && info.Enabled
}

func issuePending2FAToken(secret string, user *auth.User) (string, error) {
	if user == nil {
		return "", fmt.Errorf("user is required")
	}

	claims := jwt.MapClaims{
		"purpose":   "pending_2fa",
		"user_id":   user.ID,
		"sub":       user.ID,
		"email":     user.Email,
		"username":  user.Username,
		"role":      primaryRole(user),
		"roles":     append([]string(nil), user.RoleIDs...),
		"tenant_id": user.TenantID,
		"exp":       time.Now().Add(10 * time.Minute).Unix(),
		"iat":       time.Now().Unix(),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secret))
}

func validatePending2FAToken(tokenString, secret string) (jwt.MapClaims, error) {
	claims, err := validateJWT(tokenString, secret)
	if err != nil {
		return nil, err
	}
	if stringClaim(claims, "purpose") != "pending_2fa" {
		return nil, fmt.Errorf("token is not a pending 2FA token")
	}
	return claims, nil
}

func issueSessionToken(secret string, user *auth.User) (string, error) {
	if user == nil {
		return "", fmt.Errorf("user is required")
	}

	role := primaryRole(user)
	roles := append([]string(nil), user.RoleIDs...)
	if len(roles) == 0 && role != "" {
		roles = []string{role}
	}

	claims := jwt.MapClaims{
		"user_id":   user.ID,
		"sub":       user.ID,
		"username":  user.Username,
		"email":     user.Email,
		"role":      role,
		"roles":     roles,
		"tenant_id": user.TenantID,
		"exp":       time.Now().Add(24 * time.Hour).Unix(),
		"iat":       time.Now().Unix(),
	}

	token := jwt.NewWithClaims(jwt.SigningMethodHS256, claims)
	return token.SignedString([]byte(secret))
}

func primaryRole(user *auth.User) string {
	if user == nil {
		return ""
	}
	if len(user.RoleIDs) > 0 && strings.TrimSpace(user.RoleIDs[0]) != "" {
		return strings.TrimSpace(user.RoleIDs[0])
	}
	for _, role := range user.Roles {
		if role != nil && strings.TrimSpace(role.Name) != "" {
			return strings.TrimSpace(role.Name)
		}
	}
	return "user"
}

func frontendUser(user *auth.User) map[string]interface{} {
	role := "user"
	roles := make([]string, 0, len(user.RoleIDs))
	if len(user.RoleIDs) > 0 {
		roles = append(roles, user.RoleIDs...)
		role = user.RoleIDs[0]
	}
	if len(roles) == 0 && len(user.Roles) > 0 {
		for _, r := range user.Roles {
			if r == nil || r.Name == "" {
				continue
			}
			roles = append(roles, r.Name)
		}
	}

	return map[string]interface{}{
		"id":        user.ID,
		"email":     user.Email,
		"firstName": "",
		"lastName":  "",
		// Tenancy is not persisted in the canonical schema; the frontend
		// contract still expects a tenant label, so default it.
		"tenantId":           defaultTenantLabel(user.TenantID),
		"tenant_id":          defaultTenantLabel(user.TenantID),
		"status":             "active",
		"role":               role,
		"roles":              roles,
		"two_factor_enabled": false,
	}
}

// defaultTenantLabel fills the tenant claim for API/UI compatibility; the
// canonical users table has no tenancy column so nothing is persisted.
func defaultTenantLabel(tenantID string) string {
	if strings.TrimSpace(tenantID) == "" {
		return "default"
	}
	return strings.TrimSpace(tenantID)
}
func hasAnyRole(ctx context.Context, requiredRoles ...string) bool {
	required := make(map[string]struct{}, len(requiredRoles))
	for _, role := range requiredRoles {
		normalized := strings.ToLower(strings.TrimSpace(role))
		if normalized != "" {
			required[normalized] = struct{}{}
		}
	}

	for _, role := range contextRoles(ctx) {
		if roleSatisfies(role, required) {
			return true
		}
	}
	return false
}

func contextRoles(ctx context.Context) []string {
	roleSet := make(map[string]struct{})
	if role, ok := ctx.Value("role").(string); ok && strings.TrimSpace(role) != "" {
		roleSet[strings.ToLower(strings.TrimSpace(role))] = struct{}{}
	}
	if roles, ok := ctx.Value("roles").([]string); ok {
		for _, role := range roles {
			if trimmed := strings.ToLower(strings.TrimSpace(role)); trimmed != "" {
				roleSet[trimmed] = struct{}{}
			}
		}
	}

	normalizedRoles := make([]string, 0, len(roleSet))
	for role := range roleSet {
		normalizedRoles = append(normalizedRoles, role)
	}
	return normalizedRoles
}

func roleSatisfies(userRole string, required map[string]struct{}) bool {
	if len(required) == 0 {
		return true
	}

	normalizedUserRole := strings.ToLower(strings.TrimSpace(userRole))
	if normalizedUserRole == "" {
		return false
	}
	if _, ok := required[normalizedUserRole]; ok {
		return true
	}

	for _, impliedRole := range impliedRoles(normalizedUserRole) {
		if _, ok := required[impliedRole]; ok {
			return true
		}
	}

	return false
}

func impliedRoles(role string) []string {
	switch role {
	case "super-admin":
		return []string{"admin", "operator", "viewer", "readonly", "user"}
	case "admin":
		return []string{"operator", "viewer", "readonly", "user"}
	case "operator":
		return []string{"viewer", "readonly"}
	case "viewer":
		return []string{"readonly"}
	default:
		return nil
	}
}

// authTokenSHA256Hex derives the stored token_hash for a raw auth token:
// sha256 hex of the raw token. The raw token exists only in the delivered
// email; the database never stores it.
func authTokenSHA256Hex(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}

// insertAuthToken invalidates any live token of the same purpose for the user
// (one live token per (user, purpose)), then stores the sha256 of a fresh
// 32-byte random token. It returns the raw token for delivery by email.
func insertAuthToken(db *sql.DB, userID, purpose string, validity time.Duration) (string, error) {
	raw := make([]byte, 32)
	if _, err := rand.Read(raw); err != nil {
		return "", fmt.Errorf("generate token: %w", err)
	}
	rawToken := hex.EncodeToString(raw)

	if _, err := db.Exec(`UPDATE auth_tokens SET used_at = NOW() WHERE user_id = $1 AND purpose = $2 AND used_at IS NULL`, userID, purpose); err != nil {
		return "", fmt.Errorf("invalidate live tokens: %w", err)
	}
	if _, err := db.Exec(`
		INSERT INTO auth_tokens (user_id, token_hash, purpose, expires_at)
		VALUES ($1, $2, $3, NOW() + make_interval(secs => $4::int))
	`, userID, authTokenSHA256Hex(rawToken), purpose, int(validity.Seconds())); err != nil {
		return "", fmt.Errorf("insert token: %w", err)
	}
	return rawToken, nil
}

// validateCanonicalPassword applies the same password policy the canonical
// registration flow enforces (config defaults: >=8 chars, mixed case, a
// number, and a special character).
func validateCanonicalPassword(password string) error {
	if len(password) < 8 {
		return fmt.Errorf("password must be at least 8 characters long")
	}
	hasUpper, hasLower, hasNumber, hasSpecial := false, false, false, false
	for _, ch := range password {
		switch {
		case ch >= 'A' && ch <= 'Z':
			hasUpper = true
		case ch >= 'a' && ch <= 'z':
			hasLower = true
		case ch >= '0' && ch <= '9':
			hasNumber = true
		default:
			hasSpecial = true
		}
	}
	if !hasUpper || !hasLower {
		return fmt.Errorf("password must contain both uppercase and lowercase characters")
	}
	if !hasNumber {
		return fmt.Errorf("password must contain at least one number")
	}
	if !hasSpecial {
		return fmt.Errorf("password must contain at least one special character")
	}
	return nil
}

func notImplementedJSON(message string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotImplemented, map[string]interface{}{
			"error":   "not_implemented",
			"message": message,
			"path":    r.URL.Path,
		})
	})
}

func writeJSONError(w http.ResponseWriter, status int, message string) {
	writeJSON(w, status, map[string]interface{}{"error": message})
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(payload)
}

func stringClaim(claims jwt.MapClaims, keys ...string) string {
	for _, key := range keys {
		value, ok := claims[key]
		if !ok {
			continue
		}
		if s, ok := value.(string); ok {
			return s
		}
	}
	return ""
}

func stringSliceClaim(claims jwt.MapClaims, key string) []string {
	value, ok := claims[key]
	if !ok {
		return nil
	}

	switch typed := value.(type) {
	case []string:
		return typed
	case []interface{}:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			if s, ok := item.(string); ok && s != "" {
				result = append(result, s)
			}
		}
		return result
	case string:
		if typed == "" {
			return nil
		}
		return []string{typed}
	default:
		return nil
	}
}

func nullableString(value sql.NullString) interface{} {
	if !value.Valid {
		return nil
	}
	return value.String
}

func nullableStringValue(value string) interface{} {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func emptyStringToNil(value string) interface{} {
	if strings.TrimSpace(value) == "" {
		return nil
	}
	return value
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func parsePositiveInt(raw string, fallback int) int {
	value, err := strconv.Atoi(strings.TrimSpace(raw))
	if err != nil || value <= 0 {
		return fallback
	}
	return value
}

func nullableIntValue(value int) interface{} {
	if value <= 0 {
		return nil
	}
	return value
}
