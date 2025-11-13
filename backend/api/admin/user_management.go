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

type UserManagementHandlers struct {
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

func NewUserManagementHandlers(db *sql.DB) *UserManagementHandlers {
	sqlxDB := sqlx.NewDb(db, "postgres")
	return &UserManagementHandlers{
		db:        db,
		protector: security.NewSQLInjectionProtector(sqlxDB),
	}
}

// GET /api/admin/users - List users with pagination
func (h *UserManagementHandlers) ListUsers(w http.ResponseWriter, r *http.Request) {
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

	// Build query with filters
	baseQuery := `SELECT id, username, email, role, active, created_at, updated_at FROM users`
	countQuery := `SELECT COUNT(*) FROM users`

	conditions := []string{}
	args := []interface{}{}
	argIndex := 1

	if search != "" {
		conditions = append(conditions, fmt.Sprintf("(username ILIKE $%d OR email ILIKE $%d)", argIndex, argIndex))
		args = append(args, "%"+search+"%")
		argIndex++
	}

	if role != "" {
		conditions = append(conditions, fmt.Sprintf("role = $%d", argIndex))
		args = append(args, role)
		argIndex++
	}

	whereClause := ""
	if len(conditions) > 0 {
		whereClause = " WHERE " + conditions[0]
		for _, condition := range conditions[1:] {
			whereClause += " AND " + condition
		}
	}

	// Get total count
	var total int
	err := h.db.QueryRow(countQuery+whereClause, args...).Scan(&total)
	if err != nil {
		logger.Error("Failed to count users", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Get users
	query := baseQuery + whereClause + fmt.Sprintf(" ORDER BY created_at DESC LIMIT $%d OFFSET $%d", argIndex, argIndex+1)
	args = append(args, pageSize, offset)

	rows, err := h.db.Query(query, args...)
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

// POST /api/admin/users - Create user
func (h *UserManagementHandlers) CreateUser(w http.ResponseWriter, r *http.Request) {
	var req CreateUserRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Username == "" || req.Email == "" || req.Password == "" {
		http.Error(w, "Username, email, and password are required", http.StatusBadRequest)
		return
	}

	if req.Role == "" {
		req.Role = "user" // Default role
	}

	// Check if username or email already exists
	var exists bool
	err := h.db.QueryRow("SELECT EXISTS(SELECT 1 FROM users WHERE username = $1 OR email = $2)", req.Username, req.Email).Scan(&exists)
	if err != nil {
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

	// Insert user
	var userID int
	err = h.db.QueryRow(`
		INSERT INTO users (username, email, password_hash, role, active, created_at, updated_at)
		VALUES ($1, $2, $3, $4, true, NOW(), NOW())
		RETURNING id
	`, req.Username, req.Email, req.Password, req.Role).Scan(&userID)

	if err != nil {
		logger.Error("Failed to create user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Return created user (without password)
	var user User
	err = h.db.QueryRow(`
		SELECT id, username, email, role, active, created_at, updated_at
		FROM users WHERE id = $1
	`, userID).Scan(&user.ID, &user.Username, &user.Email, &user.Role, &user.Active, &user.CreatedAt, &user.UpdatedAt)

	if err != nil {
		logger.Error("Failed to retrieve created user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

// PUT /api/admin/users/{id} - Update user
func (h *UserManagementHandlers) UpdateUser(w http.ResponseWriter, r *http.Request) {
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

	// Check if user exists
	var exists bool
	err = h.db.QueryRow("SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)", userID).Scan(&exists)
	if err != nil {
		logger.Error("Failed to check user existence", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if !exists {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	// Build update query dynamically
	setParts := []string{}
	args := []interface{}{}
	argIndex := 1

	if req.Username != "" {
		setParts = append(setParts, fmt.Sprintf("username = $%d", argIndex))
		args = append(args, req.Username)
		argIndex++
	}

	if req.Email != "" {
		setParts = append(setParts, fmt.Sprintf("email = $%d", argIndex))
		args = append(args, req.Email)
		argIndex++
	}

	if req.Role != "" {
		setParts = append(setParts, fmt.Sprintf("role = $%d", argIndex))
		args = append(args, req.Role)
		argIndex++
	}

	if req.Active != nil {
		setParts = append(setParts, fmt.Sprintf("active = $%d", argIndex))
		args = append(args, *req.Active)
		argIndex++
	}

	if len(setParts) == 0 {
		http.Error(w, "No fields to update", http.StatusBadRequest)
		return
	}

	setParts = append(setParts, fmt.Sprintf("updated_at = NOW()"))
	setClause := setParts[0]
	for _, part := range setParts[1:] {
		setClause += ", " + part
	}

	args = append(args, userID)
	query := fmt.Sprintf("UPDATE users SET %s WHERE id = $%d", setClause, argIndex)

	_, err = h.db.Exec(query, args...)
	if err != nil {
		logger.Error("Failed to update user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	// Return updated user
	var user User
	err = h.db.QueryRow(`
		SELECT id, username, email, role, active, created_at, updated_at
		FROM users WHERE id = $1
	`, userID).Scan(&user.ID, &user.Username, &user.Email, &user.Role, &user.Active, &user.CreatedAt, &user.UpdatedAt)

	if err != nil {
		logger.Error("Failed to retrieve updated user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(user)
}

// DELETE /api/admin/users/{id} - Delete user
func (h *UserManagementHandlers) DeleteUser(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	userID, err := strconv.Atoi(vars["id"])
	if err != nil {
		http.Error(w, "Invalid user ID", http.StatusBadRequest)
		return
	}

	// Check if user exists
	var exists bool
	err = h.db.QueryRow("SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)", userID).Scan(&exists)
	if err != nil {
		logger.Error("Failed to check user existence", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	if !exists {
		http.Error(w, "User not found", http.StatusNotFound)
		return
	}

	// Delete user
	_, err = h.db.Exec("DELETE FROM users WHERE id = $1", userID)
	if err != nil {
		logger.Error("Failed to delete user", "error", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusNoContent)
}

// POST /api/admin/users/{id}/roles - Assign roles to user
func (h *UserManagementHandlers) AssignRoles(w http.ResponseWriter, r *http.Request) {
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

	// For simplicity, we'll just update the main role field
	// In a more complex system, you'd have a separate user_roles table
	if len(req.Roles) > 0 {
		_, err = h.db.Exec("UPDATE users SET role = $1, updated_at = NOW() WHERE id = $2", req.Roles[0], userID)
		if err != nil {
			logger.Error("Failed to assign role", "error", err)
			http.Error(w, "Internal server error", http.StatusInternalServerError)
			return
		}
	}

	w.WriteHeader(http.StatusNoContent)
}
