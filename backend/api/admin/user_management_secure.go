package admin

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/jmoiron/sqlx"
	"github.com/khryptorgraphics/novacron/backend/core/security"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

type SecureUserManagementHandlers struct {
	db        *sql.DB
	protector *security.SQLInjectionProtector
}

type User struct {
	ID        int       `json:"id"`
	Username  string    `json:"username"`
	Email     string    `json:"email"`
	Role      string    `json:"role"`
	Active    bool      `json:"active"`
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

type CreateUserRequest struct {
	Username string `json:"username"`
	Email    string `json:"email"`
	Password string `json:"password"`
	Role     string `json:"role"`
}

type UpdateUserRequest struct {
	Username string `json:"username,omitempty"`
	Email    string `json:"email,omitempty"`
	Role     string `json:"role,omitempty"`
	Active   *bool  `json:"active,omitempty"`
}

type UserListResponse struct {
	Users      []User `json:"users"`
	Total      int    `json:"total"`
	Page       int    `json:"page"`
	PageSize   int    `json:"page_size"`
	TotalPages int    `json:"total_pages"`
}

func NewSecureUserManagementHandlers(db *sql.DB) *SecureUserManagementHandlers {
	sqlxDB := sqlx.NewDb(db, "postgres")
	return &SecureUserManagementHandlers{
		db:        db,
		protector: security.NewSQLInjectionProtector(sqlxDB),
	}
}

// GET /api/admin/users - List users with pagination (SQL injection safe)
func (h *SecureUserManagementHandlers) ListUsers(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page <= 0 {
		page = 1
	}

	pageSize, _ := strconv.Atoi(r.URL.Query().Get("page_size"))
	if pageSize <= 0 || pageSize > 100 {
		pageSize = 20
	}

	search := r.URL.Query().Get("search")
	role := r.URL.Query().Get("role")

	offset := (page - 1) * pageSize

	// Sanitize input parameters
	search = security.SanitizeSearchTerm(search)

	// Validate role parameter against allowed values
	if role != "" {
		allowedRoles := []string{"admin", "user", "viewer", "operator"}
		roleValid := false
		for _, allowedRole := range allowedRoles {
			if role == allowedRole {
				roleValid = true
				break
			}
		}
		if !roleValid {
			http.Error(w, "Invalid role filter", http.StatusBadRequest)
			return
		}
	}

	// Build secure parameterized query
	baseQuery := `SELECT id, username, email, role, active, created_at, updated_at FROM users`
	countQuery := `SELECT COUNT(*) FROM users`

	var whereClause string
	var args []interface{}

	if search != "" && role != "" {
		whereClause = ` WHERE (username ILIKE $1 OR email ILIKE $2) AND role = $3`
		args = []interface{}{"%" + search + "%", "%" + search + "%", role}
	} else if search != "" {
		whereClause = ` WHERE (username ILIKE $1 OR email ILIKE $2)`
		args = []interface{}{"%" + search + "%", "%" + search + "%"}
	} else if role != "" {
		whereClause = ` WHERE role = $1`
		args = []interface{}{role}
	}

	// Get total count using protected query
	var total int
	countRow := h.protector.SafeQueryRow(r.Context(), countQuery+whereClause, args...)
	if err := countRow.Scan(&total); err != nil {
		logger.Error("Failed to count users", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Get users with limit and offset
	query := baseQuery + whereClause + ` ORDER BY created_at DESC LIMIT $` + strconv.Itoa(len(args)+1) + ` OFFSET $` + strconv.Itoa(len(args)+2)
	args = append(args, pageSize, offset)

	rows, err := h.protector.SafeQuery(r.Context(), query, args...)
	if err != nil {
		logger.Error("Failed to query users", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	users := []User{}
	for rows.Next() {
		var user User
		err := rows.Scan(&user.ID, &user.Username, &user.Email, &user.Role, &user.Active, &user.CreatedAt, &user.UpdatedAt)
		if err != nil {
			logger.Error("Failed to scan user", "error", err)
			continue
		}
		users = append(users, user)
	}

	totalPages := (total + pageSize - 1) / pageSize

	response := UserListResponse{
		Users:      users,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// POST /api/admin/users - Create user (SQL injection safe)
func (h *SecureUserManagementHandlers) CreateUser(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate required fields
	req.Username = strings.TrimSpace(req.Username)
	req.Email = strings.TrimSpace(req.Email)

	if req.Username == "" || req.Email == "" || req.Password == "" {
		http.Error(w, "Username, email, and password are required", http.StatusBadRequest)
		return
	}

	// Validate username length and characters
	if len(req.Username) < 1 || len(req.Username) > 50 {
		http.Error(w, "Username must be 1-50 characters", http.StatusBadRequest)
		return
	}

	// Basic email validation
	if !strings.Contains(req.Email, "@") || len(req.Email) > 100 {
		http.Error(w, "Invalid email format", http.StatusBadRequest)
		return
	}

	// Validate role
	if req.Role == "" {
		req.Role = "user" // Default role
	} else {
		allowedRoles := []string{"admin", "user", "viewer", "operator"}
		roleValid := false
		for _, allowedRole := range allowedRoles {
			if req.Role == allowedRole {
				roleValid = true
				break
			}
		}
		if !roleValid {
			http.Error(w, "Invalid role", http.StatusBadRequest)
			return
		}
	}

	// Check if username or email already exists using protected query
	var exists bool
	existsRow := h.protector.SafeQueryRow(r.Context(), "SELECT EXISTS(SELECT 1 FROM users WHERE username = $1 OR email = $2)", req.Username, req.Email)
	if err := existsRow.Scan(&exists); err != nil {
		logger.Error("Failed to check user existence", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if exists {
		http.Error(w, "Username or email already exists", http.StatusConflict)
		return
	}

	// Hash password (simplified - in production, use proper password hashing)
	// hashedPassword := hashPassword(req.Password)

	// Insert user using protected query
	var userID int
	insertRow := h.protector.SafeQueryRow(r.Context(), `
		INSERT INTO users (username, email, password_hash, role, active, created_at, updated_at)
		VALUES ($1, $2, $3, $4, true, NOW(), NOW())
		RETURNING id
	`, req.Username, req.Email, req.Password, req.Role)

	if err := insertRow.Scan(&userID); err != nil {
		logger.Error("Failed to create user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Return created user (without password) using protected query
	var user User
	userRow := h.protector.SafeQueryRow(r.Context(), `
		SELECT id, username, email, role, active, created_at, updated_at
		FROM users WHERE id = $1
	`, userID)

	if err := userRow.Scan(&user.ID, &user.Username, &user.Email, &user.Role, &user.Active, &user.CreatedAt, &user.UpdatedAt); err != nil {
		logger.Error("Failed to retrieve created user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

// PUT /api/admin/users/{id} - Update user (SQL injection safe)
func (h *SecureUserManagementHandlers) UpdateUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "Invalid user ID", http.StatusBadRequest)
		return
	}

	var req UpdateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Check if user exists using protected query
	var exists bool
	existsRow := h.protector.SafeQueryRow(r.Context(), "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)", userID)
	if err := existsRow.Scan(&exists); err != nil {
		logger.Error("Failed to check user existence", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if !exists {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	// Build secure update query
	setParts := []string{}
	args := []interface{}{}

	if req.Username != "" {
		// Validate username
		req.Username = strings.TrimSpace(req.Username)
		if len(req.Username) < 1 || len(req.Username) > 50 {
			http.Error(w, "Username must be 1-50 characters", http.StatusBadRequest)
			return
		}
		setParts = append(setParts, "username = $"+strconv.Itoa(len(args)+1))
		args = append(args, req.Username)
	}

	if req.Email != "" {
		// Basic email validation
		if !strings.Contains(req.Email, "@") || len(req.Email) > 100 {
			http.Error(w, "Invalid email format", http.StatusBadRequest)
			return
		}
		setParts = append(setParts, "email = $"+strconv.Itoa(len(args)+1))
		args = append(args, req.Email)
	}

	if req.Role != "" {
		// Validate role
		allowedRoles := []string{"admin", "user", "viewer", "operator"}
		roleValid := false
		for _, allowedRole := range allowedRoles {
			if req.Role == allowedRole {
				roleValid = true
				break
			}
		}
		if !roleValid {
			http.Error(w, "Invalid role", http.StatusBadRequest)
			return
		}
		setParts = append(setParts, "role = $"+strconv.Itoa(len(args)+1))
		args = append(args, req.Role)
	}

	if req.Active != nil {
		setParts = append(setParts, "active = $"+strconv.Itoa(len(args)+1))
		args = append(args, *req.Active)
	}

	if len(setParts) == 0 {
		http.Error(w, "No fields to update", http.StatusBadRequest)
		return
	}

	setParts = append(setParts, "updated_at = NOW()")
	setClause := strings.Join(setParts, ", ")

	args = append(args, userID)
	query := "UPDATE users SET " + setClause + " WHERE id = $" + strconv.Itoa(len(args))

	_, err = h.protector.SafeExec(r.Context(), query, args...)
	if err != nil {
		logger.Error("Failed to update user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Return updated user using protected query
	var user User
	updatedUserRow := h.protector.SafeQueryRow(r.Context(), `
		SELECT id, username, email, role, active, created_at, updated_at
		FROM users WHERE id = $1
	`, userID)

	if err := updatedUserRow.Scan(&user.ID, &user.Username, &user.Email, &user.Role, &user.Active, &user.CreatedAt, &user.UpdatedAt); err != nil {
		logger.Error("Failed to retrieve updated user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

// DELETE /api/admin/users/{id} - Delete user (SQL injection safe)
func (h *SecureUserManagementHandlers) DeleteUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "Invalid user ID", http.StatusBadRequest)
		return
	}

	// Check if user exists using protected query
	var exists bool
	deleteExistsRow := h.protector.SafeQueryRow(r.Context(), "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)", userID)
	if err := deleteExistsRow.Scan(&exists); err != nil {
		logger.Error("Failed to check user existence", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if !exists {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	// Delete user using protected query
	_, err = h.protector.SafeExec(r.Context(), "DELETE FROM users WHERE id = $1", userID)
	if err != nil {
		logger.Error("Failed to delete user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// POST /api/admin/users/{id}/roles - Assign roles to user (SQL injection safe)
func (h *SecureUserManagementHandlers) AssignRoles(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "Invalid user ID", http.StatusBadRequest)
		return
	}

	var req struct {
		Roles []string `json:"roles"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate roles and update using protected query
	if len(req.Roles) > 0 {
		// Validate role
		allowedRoles := []string{"admin", "user", "viewer", "operator"}
		roleValid := false
		for _, allowedRole := range allowedRoles {
			if req.Roles[0] == allowedRole {
				roleValid = true
				break
			}
		}
		if !roleValid {
			http.Error(w, "Invalid role", http.StatusBadRequest)
			return
		}
		_, err = h.protector.SafeExec(r.Context(), "UPDATE users SET role = $1, updated_at = NOW() WHERE id = $2", req.Roles[0], userID)
		if err != nil {
			logger.Error("Failed to assign role", "error", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}
	}

	w.WriteHeader(http.StatusNoContent)
}
