package vm

import (
	"net/http"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// require is a function that wraps a handler with RBAC by role
// type: func(role string, h http.HandlerFunc) http.Handler

// RegisterRoutes registers all VM API routes with RBAC wrappers
func RegisterRoutes(router *mux.Router, vmManager *vm.VMManager, require func(string, http.HandlerFunc) http.Handler) {
	vmHandler := NewHandler(vmManager)
	vmRouter := router.PathPrefix("/api/v1").Subrouter()

	// Viewer (read-only)
	vmRouter.Handle("/vms", require("viewer", vmHandler.ListVMs)).Methods("GET")
	vmRouter.Handle("/vms/{id}", require("viewer", vmHandler.GetVM)).Methods("GET")

	// Operator (CRUD + actions)
	vmRouter.Handle("/vms", require("operator", vmHandler.CreateVM)).Methods("POST")
	vmRouter.Handle("/vms/{id}", require("operator", vmHandler.UpdateVM)).Methods("PATCH")
	vmRouter.Handle("/vms/{id}", require("operator", vmHandler.DeleteVM)).Methods("DELETE")
	vmRouter.Handle("/vms/{id}/start", require("operator", vmHandler.StartVM)).Methods("POST")
	vmRouter.Handle("/vms/{id}/stop", require("operator", vmHandler.StopVM)).Methods("POST")
	vmRouter.Handle("/vms/{id}/restart", require("operator", vmHandler.RestartVM)).Methods("POST")
	vmRouter.Handle("/vms/{id}/pause", require("operator", vmHandler.PauseVM)).Methods("POST")
	vmRouter.Handle("/vms/{id}/resume", require("operator", vmHandler.ResumeVM)).Methods("POST")
}
