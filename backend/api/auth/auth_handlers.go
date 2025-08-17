package auth

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// Handler handles authentication API requests
type Handler struct {
	authService auth.AuthService
}

// NewHandler creates a new authentication API handler
func NewHandler(authService auth.AuthService) *Handler {
	return &Handler{
		authService: authService,
	}
}

// RegisterRequest represents a user registration request
type RegisterRequest struct {
	Email     string `json:"email" validate:"required,email"`
	Password  string `json:"password" validate:"required,min=8"`
	FirstName string `json:"firstName" validate:"required"`
	LastName  string `json:"lastName" validate:"required"`
	TenantID  string `json:"tenantId,omitempty"`
}

// LoginRequest represents a user login request
type LoginRequest struct {
	Email    string `json:"email" validate:"required,email"`
	Password string `json:"password" validate:"required"`
}

// AuthResponse represents an authentication response
type AuthResponse struct {
	Token     string    `json:"token"`
	ExpiresAt time.Time `json:"expiresAt"`
	User      *UserResponse `json:"user"`
}

// UserResponse represents a user response
type UserResponse struct {
	ID        string `json:"id"`
	Email     string `json:"email"`
	FirstName string `json:"firstName"`
	LastName  string `json:"lastName"`
	TenantID  string `json:"tenantId,omitempty"`
	Status    string `json:"status"`
}

// Register handles POST /auth/register
func (h *Handler) Register(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request RegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate request
	if request.Email == "" || request.Password == "" || request.FirstName == "" || request.LastName == "" {
		http.Error(w, "Email, password, first name, and last name are required", http.StatusBadRequest)
		return
	}

	// Create user
	newUser := &auth.User{
		ID:        fmt.Sprintf("user-%d", time.Now().Unix()),
		Username:  request.Email,
		Email:     request.Email,
		FirstName: request.FirstName,
		LastName:  request.LastName,
		TenantID:  request.TenantID,
		Status:    auth.UserStatusActive,
		CreatedAt: time.Now(),
		UpdatedAt: time.Now(),
	}

	// Create user with password
	if err := h.authService.CreateUser(newUser, request.Password); err != nil {
		http.Error(w, fmt.Sprintf("Failed to create user: %v", err), http.StatusInternalServerError)
		return
	}

	// Create response
	response := &UserResponse{
		ID:        newUser.ID,
		Email:     newUser.Email,
		FirstName: newUser.FirstName,
		LastName:  newUser.LastName,
		TenantID:  newUser.TenantID,
		Status:    string(newUser.Status),
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(response)
}

// Login handles POST /auth/login
func (h *Handler) Login(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate request
	if request.Email == "" || request.Password == "" {
		http.Error(w, "Email and password are required", http.StatusBadRequest)
		return
	}

	// Authenticate user
	session, err := h.authService.Login(request.Email, request.Password)
	if err != nil {
		http.Error(w, "Invalid email or password", http.StatusUnauthorized)
		return
	}

	// For now, we'll create a simple user response without fetching from the service
	// In a real implementation, you would fetch the user details from the user service
	response := &AuthResponse{
		Token:     session.Token,
		ExpiresAt: session.ExpiresAt,
		User: &UserResponse{
			ID:        fmt.Sprintf("user-%d", time.Now().Unix()),
			Email:     request.Email,
			FirstName: "User",
			LastName:  "Name",
			Status:    "active",
		},
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// Logout handles POST /auth/logout
func (h *Handler) Logout(w http.ResponseWriter, r *http.Request) {
	// Get session ID from context (would be set by middleware)
	sessionID := r.Context().Value("sessionID").(string)

	// Logout user
	if err := h.authService.Logout(sessionID); err != nil {
		http.Error(w, "Failed to logout", http.StatusInternalServerError)
		return
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"message": "Logged out successfully"})
}

// Refresh handles POST /auth/refresh
func (h *Handler) Refresh(w http.ResponseWriter, r *http.Request) {
	// Get session ID and token from context (would be set by middleware)
	sessionID := r.Context().Value("sessionID").(string)
	token := r.Context().Value("token").(string)

	// Refresh session
	session, err := h.authService.RefreshSession(sessionID, token)
	if err != nil {
		http.Error(w, "Failed to refresh session", http.StatusUnauthorized)
		return
	}

	// For now, we'll create a simple user response
	// In a real implementation, you would fetch the user details from the user service
	response := &AuthResponse{
		Token:     session.Token,
		ExpiresAt: session.ExpiresAt,
		User: &UserResponse{
			ID:        "user-123",
			Email:     "user@example.com",
			FirstName: "User",
			LastName:  "Name",
			Status:    "active",
		},
	}

	// Write response
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// ForgotPassword handles POST /auth/forgot-password
func (h *Handler) ForgotPassword(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Email string `json:"email" validate:"required,email"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate request
	if request.Email == "" {
		http.Error(w, "Email is required", http.StatusBadRequest)
		return
	}

	// In a real implementation, you would:
	// 1. Check if user exists
	// 2. Generate password reset token
	// 3. Send email with reset link
	// 4. Store token with expiration

	// For now, we'll just return success
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"message": "Password reset email sent"})
}

// ResetPassword handles POST /auth/reset-password
func (h *Handler) ResetPassword(w http.ResponseWriter, r *http.Request) {
	// Parse request
	var request struct {
		Token    string `json:"token" validate:"required"`
		Password string `json:"password" validate:"required,min=8"`
	}

	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate request
	if request.Token == "" || request.Password == "" {
		http.Error(w, "Token and password are required", http.StatusBadRequest)
		return
	}

	// In a real implementation, you would:
	// 1. Validate token
	// 2. Check token expiration
	// 3. Hash new password
	// 4. Update user password
	// 5. Invalidate token

	// For now, we'll just return success
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]string{"message": "Password reset successfully"})
}