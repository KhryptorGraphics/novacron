package vm

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
)

// MigrationOperations provides HTTP handlers for migration-related API endpoints
type MigrationOperations struct {
	migrationManager MigrationManager
	logger           *logrus.Logger
}

// NewMigrationOperations creates a new MigrationOperations instance
func NewMigrationOperations(migrationManager MigrationManager, logger *logrus.Logger) *MigrationOperations {
	return &MigrationOperations{
		migrationManager: migrationManager,
		logger:           logger,
	}
}

// RegisterHandlers registers HTTP handlers for migration operations
func (o *MigrationOperations) RegisterHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/migrations", o.handleMigrations)
	mux.HandleFunc("/migrations/", o.handleMigrationByID)
}

// handleMigrations handles GET and POST requests to /migrations
func (o *MigrationOperations) handleMigrations(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		o.listMigrations(w, r)
	case http.MethodPost:
		o.createMigration(w, r)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleMigrationByID handles requests to /migrations/{id}
func (o *MigrationOperations) handleMigrationByID(w http.ResponseWriter, r *http.Request) {
	// Extract migration ID from URL
	id := extractIDFromPath(r.URL.Path, "/migrations/")
	if id == "" {
		http.Error(w, "Invalid migration ID", http.StatusBadRequest)
		return
	}

	switch r.Method {
	case http.MethodGet:
		o.getMigration(w, r, id)
	case http.MethodDelete:
		o.cancelMigration(w, r, id)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

// listMigrations handles GET /migrations
func (o *MigrationOperations) listMigrations(w http.ResponseWriter, r *http.Request) {
	// Parse query parameters
	vmID := r.URL.Query().Get("vm_id")

	var migrations []*MigrationStatus
	var err error

	// List migrations, optionally filtered by VM ID
	if vmID != "" {
		migrations, err = o.migrationManager.ListMigrationsForVM(vmID)
	} else {
		migrations, err = o.migrationManager.ListMigrations()
	}

	if err != nil {
		o.logger.WithError(err).Error("Failed to list migrations")
		http.Error(w, fmt.Sprintf("Failed to list migrations: %s", err), http.StatusInternalServerError)
		return
	}

	// Return the list of migrations
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"migrations": migrations,
		"count":      len(migrations),
	})
}

// getMigration handles GET /migrations/{id}
func (o *MigrationOperations) getMigration(w http.ResponseWriter, r *http.Request, id string) {
	// Get migration status
	migration, err := o.migrationManager.GetMigrationStatus(id)
	if err != nil {
		if err == ErrMigrationNotFound {
			http.Error(w, "Migration not found", http.StatusNotFound)
		} else {
			o.logger.WithError(err).Error("Failed to get migration status")
			http.Error(w, fmt.Sprintf("Failed to get migration status: %s", err), http.StatusInternalServerError)
		}
		return
	}

	// Return the migration status
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(migration)
}

// createMigration handles POST /migrations
func (o *MigrationOperations) createMigration(w http.ResponseWriter, r *http.Request) {
	// Parse request body
	var req struct {
		VMID        string          `json:"vmId"`
		TargetNode  string          `json:"targetNode"`
		Type        string          `json:"type,omitempty"`
		BandwidthLimit int64        `json:"bandwidthLimit,omitempty"`
		Compression int             `json:"compression,omitempty"`
		Iterations  int             `json:"iterations,omitempty"`
		Priority    int             `json:"priority,omitempty"`
		Force       bool            `json:"force,omitempty"`
		SkipVerify  bool            `json:"skipVerify,omitempty"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.VMID == "" {
		http.Error(w, "Missing required field: vmId", http.StatusBadRequest)
		return
	}
	if req.TargetNode == "" {
		http.Error(w, "Missing required field: targetNode", http.StatusBadRequest)
		return
	}

	// Create migration options
	options := DefaultMigrationOptions()
	if req.Type != "" {
		options.Type = req.Type
	}
	if req.BandwidthLimit > 0 {
		options.BandwidthLimit = req.BandwidthLimit
	}
	if req.Compression > 0 {
		options.CompressionLevel = req.Compression
	}
	if req.Iterations > 0 {
		options.MemoryIterations = req.Iterations
	}
	if req.Priority > 0 {
		options.Priority = req.Priority
	}
	options.Force = req.Force
	options.SkipVerification = req.SkipVerify

	// Create the migration
	record, err := o.migrationManager.Migrate(req.VMID, req.TargetNode, options)
	if err != nil {
		o.logger.WithError(err).Error("Failed to create migration")
		http.Error(w, fmt.Sprintf("Failed to create migration: %s", err), http.StatusInternalServerError)
		return
	}

	// Return the migration record
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(record)
}

// cancelMigration handles DELETE /migrations/{id}
func (o *MigrationOperations) cancelMigration(w http.ResponseWriter, r *http.Request, id string) {
	// Cancel the migration
	if err := o.migrationManager.CancelMigration(id); err != nil {
		if err == ErrMigrationNotFound {
			http.Error(w, "Migration not found", http.StatusNotFound)
		} else {
			o.logger.WithError(err).Error("Failed to cancel migration")
			http.Error(w, fmt.Sprintf("Failed to cancel migration: %s", err), http.StatusInternalServerError)
		}
		return
	}

	// Return success
	w.WriteHeader(http.StatusNoContent)
}

// Helper function to extract ID from URL path
func extractIDFromPath(path, prefix string) string {
	if !strings.HasPrefix(path, prefix) {
		return ""
	}
	return strings.TrimPrefix(path, prefix)
}

// StreamMigrationEvents streams migration events over a websocket connection
func (o *MigrationOperations) StreamMigrationEvents(w http.ResponseWriter, r *http.Request) {
	// Parse migration ID
	id := r.URL.Query().Get("id")
	if id == "" {
		http.Error(w, "Missing migration ID", http.StatusBadRequest)
		return
	}

	// Upgrade to websocket connection
	upgrader := newWebSocketUpgrader()
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		o.logger.WithError(err).Error("Failed to upgrade to WebSocket connection")
		return
	}
	defer conn.Close()

	// Subscribe to migration events
	eventsChan, unsubscribe, err := o.migrationManager.SubscribeToMigrationEvents(id)
	if err != nil {
		o.logger.WithError(err).Error("Failed to subscribe to migration events")
		conn.WriteMessage(websocketTextMessage, []byte(fmt.Sprintf("Error: %s", err)))
		return
	}
	defer unsubscribe()

	// Create a ping ticker to keep connection alive
	pingTicker := time.NewTicker(30 * time.Second)
	defer pingTicker.Stop()

	// Create a channel to receive close signal
	done := make(chan struct{})

	// Handle incoming messages (to detect client disconnect)
	go func() {
		defer close(done)
		for {
			_, _, err := conn.ReadMessage()
			if err != nil {
				return
			}
		}
	}()

	// Stream events
	for {
		select {
		case event, ok := <-eventsChan:
			// Check if channel is closed
			if !ok {
				return
			}

			// Send event as JSON
			eventJSON, err := json.Marshal(event)
			if err != nil {
				o.logger.WithError(err).Error("Failed to marshal migration event")
				continue
			}

			if err := conn.WriteMessage(websocketTextMessage, eventJSON); err != nil {
				o.logger.WithError(err).Error("Failed to write WebSocket message")
				return
			}
		case <-pingTicker.C:
			// Send ping message
			if err := conn.WriteMessage(websocketPingMessage, []byte{}); err != nil {
				o.logger.WithError(err).Error("Failed to write WebSocket ping")
				return
			}
		case <-done:
			// Client disconnected
			return
		}
	}
}

// WebSocket message types
const (
	websocketTextMessage = 1
	websocketPingMessage = 9
)

// Placeholder for WebSocket upgrader
func newWebSocketUpgrader() *mockWebSocketUpgrader {
	return &mockWebSocketUpgrader{}
}

// Mock WebSocket types for compilation
type mockWebSocketUpgrader struct{}
func (u *mockWebSocketUpgrader) Upgrade(w http.ResponseWriter, r *http.Request, responseHeader http.Header) (*mockWebSocketConn, error) {
	return &mockWebSocketConn{}, nil
}
type mockWebSocketConn struct{}
func (c *mockWebSocketConn) WriteMessage(messageType int, data []byte) error { return nil }
func (c *mockWebSocketConn) ReadMessage() (messageType int, p []byte, err error) { return 0, nil, nil }
func (c *mockWebSocketConn) Close() error { return nil }
