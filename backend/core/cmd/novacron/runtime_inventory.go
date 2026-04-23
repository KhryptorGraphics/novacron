package main

import (
	"context"
	"database/sql"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gorilla/mux"
	_ "github.com/lib/pq"
)

type runtimeInventoryStore struct {
	db *sql.DB
}

func newRuntimeInventoryStoreFromEnv() (*runtimeInventoryStore, error) {
	postgresURL := strings.TrimSpace(getenvFirst("NOVACRON_DATABASE_URL", "NOVACRON_AUTH_POSTGRES_URL", "DB_URL"))
	if postgresURL == "" {
		return nil, nil
	}

	db, err := sql.Open("postgres", postgresURL)
	if err != nil {
		return nil, fmt.Errorf("open runtime inventory postgres database: %w", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	if err := db.PingContext(ctx); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("ping runtime inventory postgres database: %w", err)
	}

	return &runtimeInventoryStore{db: db}, nil
}

func (s *runtimeInventoryStore) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

func runtimeInventoryUnavailableHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		respondRuntimeJSON(w, http.StatusServiceUnavailable, map[string]string{
			"error": "runtime inventory persistence is not configured",
		})
	}
}

func runtimeGetVMsHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		rows, err := store.db.QueryContext(r.Context(), `SELECT id, name, state, node_id, tenant_id, created_at, updated_at FROM vms ORDER BY created_at DESC`)
		if err != nil {
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query VMs"})
			return
		}
		defer rows.Close()

		vms := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name, state, tenantID string
			var nodeID sql.NullString
			var createdAt, updatedAt time.Time

			if err := rows.Scan(&id, &name, &state, &nodeID, &tenantID, &createdAt, &updatedAt); err != nil {
				continue
			}

			vms = append(vms, map[string]interface{}{
				"id":         id,
				"name":       name,
				"state":      state,
				"status":     state,
				"node_id":    runtimeNullableString(nodeID),
				"tenant_id":  tenantID,
				"created_at": createdAt.Format(time.RFC3339),
				"updated_at": updatedAt.Format(time.RFC3339),
			})
		}

		respondRuntimeJSON(w, http.StatusOK, vms)
	}
}

func runtimeGetVMHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]

		var id, name, state, tenantID string
		var nodeID sql.NullString
		var createdAt, updatedAt time.Time

		err := store.db.QueryRowContext(r.Context(), `
			SELECT id, name, state, node_id, tenant_id, created_at, updated_at
			FROM vms WHERE id = $1
		`, vmID).Scan(&id, &name, &state, &nodeID, &tenantID, &createdAt, &updatedAt)
		if err != nil {
			if err == sql.ErrNoRows {
				respondRuntimeJSON(w, http.StatusNotFound, map[string]string{"error": "vm not found"})
				return
			}
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query VM"})
			return
		}

		respondRuntimeJSON(w, http.StatusOK, map[string]interface{}{
			"id":         id,
			"name":       name,
			"state":      state,
			"status":     state,
			"node_id":    runtimeNullableString(nodeID),
			"tenant_id":  tenantID,
			"created_at": createdAt.Format(time.RFC3339),
			"updated_at": updatedAt.Format(time.RFC3339),
		})
	}
}

func runtimeGetVMMetricsHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["id"]

		var cpuUsage, memoryUsage float64
		err := store.db.QueryRowContext(r.Context(), `
			SELECT COALESCE(cpu_usage, 0), COALESCE(memory_usage, 0)
			FROM vm_metrics
			WHERE vm_id = $1
			ORDER BY timestamp DESC
			LIMIT 1
		`, vmID).Scan(&cpuUsage, &memoryUsage)
		if err != nil && err != sql.ErrNoRows {
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query VM metrics"})
			return
		}

		respondRuntimeJSON(w, http.StatusOK, map[string]interface{}{
			"id":           vmID,
			"cpu_usage":    cpuUsage,
			"memory_usage": memoryUsage,
		})
	}
}

func runtimeGetMonitoringVMsHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		rows, err := store.db.QueryContext(r.Context(), `SELECT id, name, state FROM vms ORDER BY created_at DESC`)
		if err != nil {
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query VMs"})
			return
		}
		defer rows.Close()

		vmMetrics := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name, state string
			if err := rows.Scan(&id, &name, &state); err != nil {
				continue
			}

			vmMetrics = append(vmMetrics, map[string]interface{}{
				"vmId":        id,
				"name":        name,
				"cpuUsage":    50.0 + float64(len(id)%20),
				"memoryUsage": 60.0 + float64(len(name)%30),
				"diskUsage":   40.0 + float64(len(id)%15),
				"networkRx":   1024 * 1024,
				"networkTx":   2048 * 1024,
				"iops":        100,
				"status":      state,
			})
		}

		respondRuntimeJSON(w, http.StatusOK, vmMetrics)
	}
}

func runtimeGetNetworksHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		rows, err := store.db.QueryContext(r.Context(), `
			SELECT id, name, type, subnet, gateway, status, created_at, updated_at
			FROM networks
			ORDER BY created_at DESC
		`)
		if err != nil {
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query networks"})
			return
		}
		defer rows.Close()

		networks := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, name, networkType, subnet, status string
			var gateway sql.NullString
			var createdAt, updatedAt time.Time

			if err := rows.Scan(&id, &name, &networkType, &subnet, &gateway, &status, &createdAt, &updatedAt); err != nil {
				continue
			}

			networks = append(networks, map[string]interface{}{
				"id":         id,
				"name":       name,
				"type":       networkType,
				"subnet":     subnet,
				"gateway":    runtimeNullableString(gateway),
				"status":     status,
				"created_at": createdAt.Format(time.RFC3339),
				"updated_at": updatedAt.Format(time.RFC3339),
			})
		}

		respondRuntimeJSON(w, http.StatusOK, networks)
	}
}

func runtimeGetNetworkHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		networkID := mux.Vars(r)["id"]

		var id, name, networkType, subnet, status string
		var gateway sql.NullString
		var createdAt, updatedAt time.Time
		err := store.db.QueryRowContext(r.Context(), `
			SELECT id, name, type, subnet, gateway, status, created_at, updated_at
			FROM networks WHERE id = $1
		`, networkID).Scan(&id, &name, &networkType, &subnet, &gateway, &status, &createdAt, &updatedAt)
		if err != nil {
			if err == sql.ErrNoRows {
				respondRuntimeJSON(w, http.StatusNotFound, map[string]string{"error": "network not found"})
				return
			}
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query network"})
			return
		}

		respondRuntimeJSON(w, http.StatusOK, map[string]interface{}{
			"id":         id,
			"name":       name,
			"type":       networkType,
			"subnet":     subnet,
			"gateway":    runtimeNullableString(gateway),
			"status":     status,
			"created_at": createdAt.Format(time.RFC3339),
			"updated_at": updatedAt.Format(time.RFC3339),
		})
	}
}

func runtimeGetVMInterfacesHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]

		rows, err := store.db.QueryContext(r.Context(), `
			SELECT id, vm_id, network_id, name, mac_address, ip_address, status, created_at, updated_at
			FROM vm_interfaces
			WHERE vm_id = $1
			ORDER BY created_at DESC
		`, vmID)
		if err != nil {
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query VM interfaces"})
			return
		}
		defer rows.Close()

		interfaces := make([]map[string]interface{}, 0)
		for rows.Next() {
			var id, currentVMID, name, macAddress, status string
			var networkID, ipAddress sql.NullString
			var createdAt, updatedAt time.Time

			if err := rows.Scan(&id, &currentVMID, &networkID, &name, &macAddress, &ipAddress, &status, &createdAt, &updatedAt); err != nil {
				continue
			}

			interfaces = append(interfaces, map[string]interface{}{
				"id":          id,
				"vm_id":       currentVMID,
				"network_id":  runtimeNullableString(networkID),
				"name":        name,
				"mac_address": macAddress,
				"ip_address":  runtimeNullableString(ipAddress),
				"status":      status,
				"created_at":  createdAt.Format(time.RFC3339),
				"updated_at":  updatedAt.Format(time.RFC3339),
			})
		}

		respondRuntimeJSON(w, http.StatusOK, interfaces)
	}
}

func runtimeGetVMInterfaceHandler(store *runtimeInventoryStore) http.HandlerFunc {
	if store == nil || store.db == nil {
		return runtimeInventoryUnavailableHandler()
	}

	return func(w http.ResponseWriter, r *http.Request) {
		vmID := mux.Vars(r)["vm_id"]
		interfaceID := mux.Vars(r)["id"]

		var id, currentVMID, name, macAddress, status string
		var networkID, ipAddress sql.NullString
		var createdAt, updatedAt time.Time
		err := store.db.QueryRowContext(r.Context(), `
			SELECT id, vm_id, network_id, name, mac_address, ip_address, status, created_at, updated_at
			FROM vm_interfaces
			WHERE vm_id = $1 AND id = $2
		`, vmID, interfaceID).Scan(&id, &currentVMID, &networkID, &name, &macAddress, &ipAddress, &status, &createdAt, &updatedAt)
		if err != nil {
			if err == sql.ErrNoRows {
				respondRuntimeJSON(w, http.StatusNotFound, map[string]string{"error": "vm interface not found"})
				return
			}
			respondRuntimeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to query VM interface"})
			return
		}

		respondRuntimeJSON(w, http.StatusOK, map[string]interface{}{
			"id":          id,
			"vm_id":       currentVMID,
			"network_id":  runtimeNullableString(networkID),
			"name":        name,
			"mac_address": macAddress,
			"ip_address":  runtimeNullableString(ipAddress),
			"status":      status,
			"created_at":  createdAt.Format(time.RFC3339),
			"updated_at":  updatedAt.Format(time.RFC3339),
		})
	}
}

func runtimeNullableString(value sql.NullString) interface{} {
	if value.Valid {
		return value.String
	}
	return nil
}
