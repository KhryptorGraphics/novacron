package main

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/gorilla/mux"
	// Core-safe: do not import experimental packages

	api_orch "github.com/khryptorgraphics/novacron/backend/api/orchestration"
	api_vm "github.com/khryptorgraphics/novacron/backend/api/vm"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration"
	"github.com/khryptorgraphics/novacron/backend/pkg/config"
	applogger "github.com/khryptorgraphics/novacron/backend/pkg/logger"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
	"github.com/sirupsen/logrus"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

func main() {
	// Config & logger
	cfg, err := config.Load()
	if err != nil { panic(err) }
	appLogger := applogger.NewFromConfig(cfg.Logging.Level, cfg.Logging.Format, cfg.Logging.Output, cfg.Logging.Structured)

	// VM Manager (core-safe)
	vmCfg := core_vm.VMManagerConfig{ DefaultDriver: core_vm.VMTypeKVM, Drivers: map[core_vm.VMType]core_vm.VMDriverConfigManager{ core_vm.VMTypeKVM: { Enabled: true, Config: map[string]interface{}{} }, }, }
	vmManager, err := core_vm.NewVMManager(vmCfg)
	if err != nil { appLogger.Warn("VMManager init failed", "error", err); }

	// Orchestration engine (noop event bus in core mode)
	orchLogger := logrus.New()
	engine := orchestration.NewDefaultOrchestrationEngine(orchLogger)
	if vmManager != nil {
		adapters := &orchestration.OrchestrationAdapters{ VMManager: vmManager, PlacementEngine: engine.Placement() }
		evac := orchestration.NewDefaultEvacuationHandler(adapters.ListVMsByNodeAdapter, adapters.SelectTargetAdapter, adapters.MigrateAdapter, orchLogger)
		engine.SetEvacuationHandler(evac)
	}

	// Router and middleware
	r := mux.NewRouter()
	auth := middleware.NewAuthMiddleware(authManagerStub{secret: cfg.Auth.Secret})
	api := r.PathPrefix("/api/v1").Subrouter()
	api.Use(auth.RequireAuth)

	// RBAC wrappers
	require := func(role string, h http.HandlerFunc) http.Handler { return auth.RequireRole(role)(h) }

	// VM routes with RBAC wrappers
	requireFunc := func(role string, h http.HandlerFunc) http.Handler { return auth.RequireRole(role)(h) }
	api_vm.RegisterRoutes(api, vmManager, requireFunc)
	// api_vm.NewMigrationHandler(vmManager).RegisterRoutes(api) // experimental; core registers minimal routes below

	// Node evacuate route (operator+)
	// Core nodes endpoints
	api_vm.RegisterCoreNodeRoutes(api, engine)

	api.Handle("/nodes/{id}/evacuate", require("operator", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request){
		vars := mux.Vars(r)
		nodeID := vars["id"]
		ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second); defer cancel()
		if engine == nil { http.Error(w, "engine unavailable", http.StatusServiceUnavailable); return }
		// Use the already injected evacuation handler via engine
		if err := engine.EvacuateNode(ctx, nodeID); err != nil {
			http.Error(w, fmt.Sprintf("evacuation failed: %v", err), http.StatusInternalServerError); return
		}
		w.Header().Set("Content-Type","application/json")
		w.Write([]byte(`{"data": {"status":"started"}, "error": null}`))
	// Apply envelope for all API responses
	api.Use(middleware.ResponseEnvelopeMiddleware)

	})) ).Methods("POST")

	// Events REST (viewer+)
	api.Handle("/events", require("viewer", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request){
		w.Header().Set("Content-Type","application/json")
	// Response envelope for all /api responses
	api.Use(middleware.ResponseEnvelopeMiddleware)

		w.Write([]byte(`{"data": [], "error": null}`))
	})) ).Methods("GET")

	// WebSocket endpoint (viewer+)
	r.HandleFunc("/ws/events/v1", func(w http.ResponseWriter, r *http.Request){
		// In core mode, we allow unauthenticated access for simplicity if no secret
		if cfg.Auth.Secret != "" {
			// Require Authorization header
			if r.Header.Get("Authorization") == "" { http.Error(w, "unauthorized", http.StatusUnauthorized); return }
		}
		wsManager := api_orch.NewWebSocketManager(orchLogger, engine.EventBus())
	// Core-safe VM routes registration
	requireFunc := func(role string, h http.HandlerFunc) http.Handler { return auth.RequireRole(role)(h) }
	api_vm.RegisterRoutes(api, vmManager, requireFunc)
	// Minimal core migrations
	api_vm.RegisterCoreMigrationRoutes(api)

		wsManager.HandleWebSocket(w, r)
	}).Methods("GET")

	// Start server
	srv := &http.Server{ Addr: ":"+cfg.Server.APIPort, Handler: r, ReadTimeout: cfg.Server.ReadTimeout, WriteTimeout: cfg.Server.WriteTimeout }
	go func(){ if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed { panic(err) } }()
	appLogger.Info("Core server running", "port", cfg.Server.APIPort)

	// Graceful shutdown
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second); defer cancel()
	_ = srv.Shutdown(ctx)
}

type authManagerStub struct{ secret string }
func (a authManagerStub) GetJWTSecret() string { return a.secret }
func (a authManagerStub) GetUser(userID string) (interface{}, error) { return struct{}{}, nil }

