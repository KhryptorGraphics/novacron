//go:build experimental

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

// SecurityHandler handles VM security API requests
type SecurityHandler struct {
	securityManager *vm.VMSecurityManager
}

// NewSecurityHandler creates a new VM security API handler
func NewSecurityHandler(securityManager *vm.VMSecurityManager) *SecurityHandler {
	return &SecurityHandler{
		securityManager: securityManager,
	}
}

// RegisterRoutes registers VM security API routes
func (h *SecurityHandler) RegisterRoutes(router *mux.Router) {
	router.HandleFunc("/security/profiles", h.ListSecurityProfiles).Methods("GET")
	router.HandleFunc("/security/profiles", h.CreateSecurityProfile).Methods("POST")
	router.HandleFunc("/security/profiles/{id}", h.GetSecurityProfile).Methods("GET")
	router.HandleFunc("/security/profiles/{id}", h.UpdateSecurityProfile).Methods("PUT")
	router.HandleFunc("/security/profiles/{id}", h.DeleteSecurityProfile).Methods("DELETE")
	router.HandleFunc("/security/certificates", h.ListCertificates).Methods("GET")
	router.HandleFunc("/security/certificates", h.CreateCertificate).Methods("POST")
	router.HandleFunc("/security/certificates/{id}", h.GetCertificate).Methods("GET")
	router.HandleFunc("/security/certificates/{id}", h.DeleteCertificate).Methods("DELETE")
	router.HandleFunc("/vms/{vm_id}/security", h.GetVMSecurityInfo).Methods("GET")
	router.HandleFunc("/vms/{vm_id}/security/profile", h.ApplySecurityProfile).Methods("POST")
}

// ListSecurityProfiles handles GET /security/profiles
func (h *SecurityHandler) ListSecurityProfiles(w http.ResponseWriter, r *http.Request) {
	// Get profiles
	profiles := h.securityManager.ListSecurityProfiles()

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(profiles))
	for _, profile := range profiles {
		response = append(response, map[string]interface{}{
			"id":                 profile.ID,
			"name":               profile.Name,
			"description":        profile.Description,
			"secure_boot":        profile.SecureBoot,
			"tpm_enabled":        profile.TPMEnabled,
			"encryption_enabled": profile.EncryptionEnabled,
			"encryption_type":    profile.EncryptionType,
			"created_at":         profile.CreatedAt,
			"updated_at":         profile.UpdatedAt,
			"tags":               profile.Tags,
			"metadata":           profile.Metadata,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateSecurityProfile handles POST /security/profiles
func (h *SecurityHandler) CreateSecurityProfile(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Name              string            `json:"name"`
		Description       string            `json:"description"`
		SecureBoot        bool              `json:"secure_boot"`
		TPMEnabled        bool              `json:"tpm_enabled"`
		EncryptionEnabled bool              `json:"encryption_enabled"`
		EncryptionType    string            `json:"encryption_type"`
		Tags              []string          `json:"tags"`
		Metadata          map[string]string `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Create profile
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	profile, err := h.securityManager.CreateSecurityProfile(ctx, request.Name, request.Description, request.SecureBoot, request.TPMEnabled, request.EncryptionEnabled, request.EncryptionType, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":                 profile.ID,
		"name":               profile.Name,
		"description":        profile.Description,
		"secure_boot":        profile.SecureBoot,
		"tpm_enabled":        profile.TPMEnabled,
		"encryption_enabled": profile.EncryptionEnabled,
		"encryption_type":    profile.EncryptionType,
		"created_at":         profile.CreatedAt,
		"updated_at":         profile.UpdatedAt,
		"tags":               profile.Tags,
		"metadata":           profile.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetSecurityProfile handles GET /security/profiles/{id}
func (h *SecurityHandler) GetSecurityProfile(w http.ResponseWriter, r *http.Request) {
	// Get profile ID from URL
	vars := mux.Vars(r)
	profileID := vars["id"]

	// Get profile
	profile, err := h.securityManager.GetSecurityProfile(profileID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":                 profile.ID,
		"name":               profile.Name,
		"description":        profile.Description,
		"secure_boot":        profile.SecureBoot,
		"tpm_enabled":        profile.TPMEnabled,
		"encryption_enabled": profile.EncryptionEnabled,
		"encryption_type":    profile.EncryptionType,
		"created_at":         profile.CreatedAt,
		"updated_at":         profile.UpdatedAt,
		"tags":               profile.Tags,
		"metadata":           profile.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// UpdateSecurityProfile handles PUT /security/profiles/{id}
func (h *SecurityHandler) UpdateSecurityProfile(w http.ResponseWriter, r *http.Request) {
	// Get profile ID from URL
	vars := mux.Vars(r)
	profileID := vars["id"]

	// Parse request
	var request struct {
		Name              string            `json:"name"`
		Description       string            `json:"description"`
		SecureBoot        bool              `json:"secure_boot"`
		TPMEnabled        bool              `json:"tpm_enabled"`
		EncryptionEnabled bool              `json:"encryption_enabled"`
		EncryptionType    string            `json:"encryption_type"`
		Tags              []string          `json:"tags"`
		Metadata          map[string]string `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Update profile
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	profile, err := h.securityManager.UpdateSecurityProfile(ctx, profileID, request.Name, request.Description, request.SecureBoot, request.TPMEnabled, request.EncryptionEnabled, request.EncryptionType, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":                 profile.ID,
		"name":               profile.Name,
		"description":        profile.Description,
		"secure_boot":        profile.SecureBoot,
		"tpm_enabled":        profile.TPMEnabled,
		"encryption_enabled": profile.EncryptionEnabled,
		"encryption_type":    profile.EncryptionType,
		"created_at":         profile.CreatedAt,
		"updated_at":         profile.UpdatedAt,
		"tags":               profile.Tags,
		"metadata":           profile.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteSecurityProfile handles DELETE /security/profiles/{id}
func (h *SecurityHandler) DeleteSecurityProfile(w http.ResponseWriter, r *http.Request) {
	// Get profile ID from URL
	vars := mux.Vars(r)
	profileID := vars["id"]

	// Delete profile
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	if err := h.securityManager.DeleteSecurityProfile(ctx, profileID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// ListCertificates handles GET /security/certificates
func (h *SecurityHandler) ListCertificates(w http.ResponseWriter, r *http.Request) {
	// Get certificates
	certificates := h.securityManager.ListCertificates()

	// Convert to response format
	response := make([]map[string]interface{}, 0, len(certificates))
	for _, cert := range certificates {
		response = append(response, map[string]interface{}{
			"id":          cert.ID,
			"name":        cert.Name,
			"description": cert.Description,
			"type":        cert.Type,
			"fingerprint": cert.Fingerprint,
			"not_before":  cert.NotBefore,
			"not_after":   cert.NotAfter,
			"created_at":  cert.CreatedAt,
			"updated_at":  cert.UpdatedAt,
			"tags":        cert.Tags,
			"metadata":    cert.Metadata,
		})
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// CreateCertificate handles POST /security/certificates
func (h *SecurityHandler) CreateCertificate(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Name         string            `json:"name"`
		Description  string            `json:"description"`
		Type         string            `json:"type"`
		ValidityDays int               `json:"validity_days"`
		Tags         []string          `json:"tags"`
		Metadata     map[string]string `json:"metadata"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Create certificate
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	cert, err := h.securityManager.CreateCertificate(ctx, request.Name, request.Description, request.Type, request.ValidityDays, request.Tags, request.Metadata)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          cert.ID,
		"name":        cert.Name,
		"description": cert.Description,
		"type":        cert.Type,
		"fingerprint": cert.Fingerprint,
		"not_before":  cert.NotBefore,
		"not_after":   cert.NotAfter,
		"created_at":  cert.CreatedAt,
		"updated_at":  cert.UpdatedAt,
		"tags":        cert.Tags,
		"metadata":    cert.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// GetCertificate handles GET /security/certificates/{id}
func (h *SecurityHandler) GetCertificate(w http.ResponseWriter, r *http.Request) {
	// Get certificate ID from URL
	vars := mux.Vars(r)
	certID := vars["id"]

	// Get certificate
	cert, err := h.securityManager.GetCertificate(certID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusNotFound)
		return
	}

	// Write response
	response := map[string]interface{}{
		"id":          cert.ID,
		"name":        cert.Name,
		"description": cert.Description,
		"type":        cert.Type,
		"fingerprint": cert.Fingerprint,
		"not_before":  cert.NotBefore,
		"not_after":   cert.NotAfter,
		"created_at":  cert.CreatedAt,
		"updated_at":  cert.UpdatedAt,
		"tags":        cert.Tags,
		"metadata":    cert.Metadata,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// DeleteCertificate handles DELETE /security/certificates/{id}
func (h *SecurityHandler) DeleteCertificate(w http.ResponseWriter, r *http.Request) {
	// Get certificate ID from URL
	vars := mux.Vars(r)
	certID := vars["id"]

	// Delete certificate
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	if err := h.securityManager.DeleteCertificate(ctx, certID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.WriteHeader(http.StatusNoContent)
}

// GetVMSecurityInfo handles GET /vms/{vm_id}/security
func (h *SecurityHandler) GetVMSecurityInfo(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Get VM security info
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	info, err := h.securityManager.GetVMSecurityInfo(ctx, vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}

// ApplySecurityProfile handles POST /vms/{vm_id}/security/profile
func (h *SecurityHandler) ApplySecurityProfile(w http.ResponseWriter, r *http.Request) {
	// Get VM ID from URL
	vars := mux.Vars(r)
	vmID := vars["vm_id"]

	// Parse request
	var request struct {
		ProfileID string `json:"profile_id"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Apply security profile
	ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()

	if err := h.securityManager.ApplySecurityProfile(ctx, vmID, request.ProfileID); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Get VM security info
	info, err := h.securityManager.GetVMSecurityInfo(ctx, vmID)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(info)
}
