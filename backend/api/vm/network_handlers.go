package vm

import (
	"context"
	"encoding/json"
	"net/http"
	"strconv"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

// NetworkHandler handles VM network API requests
type NetworkHandler struct {
	networkManager *vm.VMNetworkManager
}

// NewNetworkHandler creates a new VM network API handler
func NewNetworkHandler(networkManager *vm.VMNetworkManager) *NetworkHandler {
	return &NetworkHandler{
		networkManager: networkManager,
	}
}

// RegisterRoutes registers VM network API routes
func (h *NetworkHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/networks", h.ListNetworks).Methods("GET")
	router.HandleFunc("/networks", h.CreateNetwork).Methods("POST")
	router.HandleFunc("/networks/{id}", h.GetNetwork).Methods("GET")
	router.HandleFunc("/networks/{id}", h.DeleteNetwork).Methods("DELETE")
	router.HandleFunc("/vms/{vm_id}/interfaces", h.ListNetworkInterfaces).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/interfaces", h.AttachNetworkInterface).Methods("POST")
	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", h.GetNetworkInterface).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", h.UpdateNetworkInterface).Methods("PUT")
	router.HandleFunc("/vms/{vm_id}/interfaces/{id}", h.DetachNetworkInterface).Methods("DELETE")
}

// ListNetworks handles GET /networks
func (h *NetworkHandler) ListNetworks(w http.ResponseWriter, r *http.Request) {
	// Get networks
	networks := h.networkManager.ListNetworks()
	
	// Convert to response format
	response := make([]map[string]interface{}, 0, len(networks))
	for _, network := range networks {
		response = append(response, map[string]interface{}{
			"id":         network.ID,
			"name":       network.Name,
			"type":       network.Type,
			"subnet":     network.Subnet,
			"gateway":    network.Gateway,
			"dhcp":       network.DHCP,
			"dhcp_range": network.DHCPRange,
			"bridge":     network.Bridge,
			"mtu":        network.MTU,
			"vlan":       network.VLAN,
			"created_at": network.CreatedAt,
			"updated_at": network.UpdatedAt,
			"tags":       network.Tags,
			"metadata":   network.Metadata,
		})
	}
	
	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateNetwork handles POST /networks
func (h *NetworkHandler) CreateNetwork(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Name      string            `json:"name"`
		Type      string            `json:"type"`
		Subnet    string            `json:"subnet"`
		Gateway   string            `json:"gateway"`
		DHCP      bool              `json:"dhcp"`
		DHCPRange string            `json:"dhcp_range"`
		Bridge    string            `json:"bridge"`
		MTU       int               `json:"mtu"`
		VLAN      int               `json:"vlan"`
		Tags      []string          `json:"tags"`
		Metadata  map[string]string `json:"metadata"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Validate network type
	var networkType vm.NetworkType
	switch request.Type {
	case "bridge":
		networkType = vm.NetworkTypeBridge
	case "nat":
		networkType = vm.NetworkTypeNAT
	case "host":
		networkType = vm.NetworkTypeHost
	case "isolated":
		networkType = vm.NetworkTypeIsolated
	default:
		http.Error(w, "Invalid network type. Must be 'bridge', 'nat', 'host', or 'isolated'.", http.StatusBadRequest)
		return
	}
	
	// Create network
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	network, err := h.networkManager.CreateNetwork(ctx, request.Name, networkType, request.Subnet, request.Gateway, request.DHCP, request.DHCPRange, request.Bridge, request.MTU, request.VLAN, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         network.ID,
		"name":       network.Name,
		"type":       network.Type,
		"subnet":     network.Subnet,
		"gateway":    network.Gateway,
		"dhcp":       network.DHCP,
		"dhcp_range": network.DHCPRange,
		"bridge":     network.Bridge,
		"mtu":        network.MTU,
		"vlan":       network.VLAN,
		"created_at": network.CreatedAt,
		"updated_at": network.UpdatedAt,
		"tags":       network.Tags,
		"metadata":   network.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetNetwork handles GET /networks/{id}
func (h *NetworkHandler) GetNetwork(w http.ResponseWriter, r *http.Request) {
	// Get network ID from URL
	vars := mux.Vars(r)
	networkID := vars["id"]
	
	// Get network
	network, err := h.networkManager.GetNetwork(networkID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":         network.ID,
		"name":       network.Name,
		"type":       network.Type,
		"subnet":     network.Subnet,
		"gateway":    network.Gateway,
		"dhcp":       network.DHCP,
		"dhcp_range": network.DHCPRange,
		"bridge":     network.Bridge,
		"mtu":        network.MTU,
		"vlan":       network.VLAN,
		"created_at": network.CreatedAt,
		"updated_at": network.UpdatedAt,
		"tags":       network.Tags,
		"metadata":   network.Metadata,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteNetwork handles DELETE /networks/{id}
func (h *NetworkHandler) DeleteNetwork(w http.ResponseWriter, r *http.Request) {
	// Get network ID from URL
	vars := mux.Vars(r)
	networkID := vars["id"]
	
	// Delete network
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.networkManager.DeleteNetwork(ctx, networkID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// ListNetworkInterfaces handles GET /vms/{vm_id}/interfaces
func (h *NetworkHandler) ListNetworkInterfaces(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Get interfaces
	interfaces, err := h.networkManager.ListNetworkInterfaces(vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Convert to response format
	response := make([]map[string]interface{}, 0, len(interfaces))
	for _, iface := range interfaces {
		response = append(response, map[string]interface{}{
			"id":          iface.ID,
			"vm_id":       iface.VMID,
			"network_id":  iface.NetworkID,
			"mac_address": iface.MACAddress,
			"ip_address":  iface.IPAddress,
			"model":       iface.Model,
			"mtu":         iface.MTU,
			"index":       iface.Index,
			"created_at":  iface.CreatedAt,
			"updated_at":  iface.UpdatedAt,
		})
	}
	
	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// AttachNetworkInterface handles POST /vms/{vm_id}/interfaces
func (h *NetworkHandler) AttachNetworkInterface(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	
	// Parse request
	var request struct {
		NetworkID   string `json:"network_id"`
		MACAddress  string `json:"mac_address"`
		IPAddress   string `json:"ip_address"`
		Model       string `json:"model"`
		MTU         int    `json:"mtu"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Attach interface
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	iface, err := h.networkManager.AttachNetworkInterface(ctx, vmID, request.NetworkID, request.MACAddress, request.IPAddress, request.Model, request.MTU)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":          iface.ID,
		"vm_id":       iface.VMID,
		"network_id":  iface.NetworkID,
		"mac_address": iface.MACAddress,
		"ip_address":  iface.IPAddress,
		"model":       iface.Model,
		"mtu":         iface.MTU,
		"index":       iface.Index,
		"created_at":  iface.CreatedAt,
		"updated_at":  iface.UpdatedAt,
	}
	
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetNetworkInterface handles GET /vms/{vm_id}/interfaces/{id}
func (h *NetworkHandler) GetNetworkInterface(w http.ResponseWriter, r *http.Request) {
	// Get VM ID and interface ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	interfaceID := vars["id"]
	
	// Get interface
	iface, err := h.networkManager.GetNetworkInterface(vmID, interfaceID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":          iface.ID,
		"vm_id":       iface.VMID,
		"network_id":  iface.NetworkID,
		"mac_address": iface.MACAddress,
		"ip_address":  iface.IPAddress,
		"model":       iface.Model,
		"mtu":         iface.MTU,
		"index":       iface.Index,
		"created_at":  iface.CreatedAt,
		"updated_at":  iface.UpdatedAt,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// UpdateNetworkInterface handles PUT /vms/{vm_id}/interfaces/{id}
func (h *NetworkHandler) UpdateNetworkInterface(w http.ResponseWriter, r *http.Request) {
	// Get VM ID and interface ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	interfaceID := vars["id"]
	
	// Parse request
	var request struct {
		IPAddress string `json:"ip_address"`
		MTU       int    `json:"mtu"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Update interface
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	iface, err := h.networkManager.UpdateNetworkInterface(ctx, vmID, interfaceID, request.IPAddress, request.MTU)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	response := map[string]interface{}{
		"id":          iface.ID,
		"vm_id":       iface.VMID,
		"network_id":  iface.NetworkID,
		"mac_address": iface.MACAddress,
		"ip_address":  iface.IPAddress,
		"model":       iface.Model,
		"mtu":         iface.MTU,
		"index":       iface.Index,
		"created_at":  iface.CreatedAt,
		"updated_at":  iface.UpdatedAt,
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DetachNetworkInterface handles DELETE /vms/{vm_id}/interfaces/{id}
func (h *NetworkHandler) DetachNetworkInterface(w http.ResponseWriter, r *http.Request) {
	// Get VM ID and interface ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]
	interfaceID := vars["id"]
	
	// Detach interface
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	
	if err := h.networkManager.DetachNetworkInterface(ctx, vmID, interfaceID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	
	// Write response
	w.WriteHeader(http.StatusNoContent)
}
