package admin

import (
	"database/sql"
	"encoding/json"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

type TemplateHandlers struct {
	db *sql.DB
}

type VMTemplate struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	OS          string                 `json:"os"`
	OSVersion   string                 `json:"os_version"`
	CPUCores    int                    `json:"cpu_cores"`
	MemoryMB    int                    `json:"memory_mb"`
	DiskGB      int                    `json:"disk_gb"`
	ImagePath   string                 `json:"image_path"`
	IsPublic    bool                   `json:"is_public"`
	UsageCount  int                    `json:"usage_count"`
	Tags        []string               `json:"tags,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	CreatedBy   string                 `json:"created_by"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

type CreateTemplateRequest struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	OS          string                 `json:"os"`
	OSVersion   string                 `json:"os_version"`
	CPUCores    int                    `json:"cpu_cores"`
	MemoryMB    int                    `json:"memory_mb"`
	DiskGB      int                    `json:"disk_gb"`
	ImagePath   string                 `json:"image_path"`
	IsPublic    bool                   `json:"is_public"`
	Tags        []string               `json:"tags,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

type UpdateTemplateRequest struct {
	Name        *string                 `json:"name,omitempty"`
	Description *string                 `json:"description,omitempty"`
	OS          *string                 `json:"os,omitempty"`
	OSVersion   *string                 `json:"os_version,omitempty"`
	CPUCores    *int                    `json:"cpu_cores,omitempty"`
	MemoryMB    *int                    `json:"memory_mb,omitempty"`
	DiskGB      *int                    `json:"disk_gb,omitempty"`
	ImagePath   *string                 `json:"image_path,omitempty"`
	IsPublic    *bool                   `json:"is_public,omitempty"`
	Tags        []string                `json:"tags,omitempty"`
	Metadata    map[string]interface{}  `json:"metadata,omitempty"`
}

type TemplateListResponse struct {
	Templates  []VMTemplate `json:"templates"`
	Total      int          `json:"total"`
	Page       int          `json:"page"`
	PageSize   int          `json:"page_size"`
	TotalPages int          `json:"total_pages"`
}

func NewTemplateHandlers(db *sql.DB) *TemplateHandlers {
	return &TemplateHandlers{db: db}
}

// GET /api/admin/templates - List VM templates
func (h *TemplateHandlers) ListTemplates(w http.ResponseWriter, r *http.Request) {
	page, _ := strconv.Atoi(r.URL.Query().Get("page"))
	if page <= 0 {
		page = 1
	}

	pageSize, _ := strconv.Atoi(r.URL.Query().Get("page_size"))
	if pageSize <= 0 || pageSize > 100 {
		pageSize = 20
	}

	search := r.URL.Query().Get("search")
	os := r.URL.Query().Get("os")
	isPublic := r.URL.Query().Get("public")

	offset := (page - 1) * pageSize

	// Build query
	query := `
		SELECT id, name, description, os, os_version, cpu_cores, memory_mb, disk_gb,
		       image_path, is_public, usage_count, tags, metadata, created_by, created_at, updated_at
		FROM vm_templates
		WHERE 1=1
	`
	args := []interface{}{}
	argIndex := 1

	if search != "" {
		query += ` AND (name ILIKE $` + strconv.Itoa(argIndex) + ` OR description ILIKE $` + strconv.Itoa(argIndex) + `)`
		args = append(args, "%"+search+"%")
		argIndex++
	}

	if os != "" {
		query += ` AND os = $` + strconv.Itoa(argIndex)
		args = append(args, os)
		argIndex++
	}

	if isPublic != "" {
		public := isPublic == "true"
		query += ` AND is_public = $` + strconv.Itoa(argIndex)
		args = append(args, public)
		argIndex++
	}

	// Count total
	countQuery := `SELECT COUNT(*) FROM (` + query + `) AS count_query`
	var total int
	err := h.db.QueryRow(countQuery, args...).Scan(&total)
	if err != nil {
		logger.Error("Failed to count templates", "error", err)
		http.Error(w, "Failed to count templates", http.StatusInternalServerError)
		return
	}

	// Add ordering and pagination
	query += ` ORDER BY created_at DESC LIMIT $` + strconv.Itoa(argIndex) + ` OFFSET $` + strconv.Itoa(argIndex+1)
	args = append(args, pageSize, offset)

	rows, err := h.db.Query(query, args...)
	if err != nil {
		logger.Error("Failed to query templates", "error", err)
		http.Error(w, "Failed to query templates", http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	templates := []VMTemplate{}
	for rows.Next() {
		var t VMTemplate
		var tagsJSON, metadataJSON []byte

		err := rows.Scan(
			&t.ID, &t.Name, &t.Description, &t.OS, &t.OSVersion,
			&t.CPUCores, &t.MemoryMB, &t.DiskGB, &t.ImagePath,
			&t.IsPublic, &t.UsageCount, &tagsJSON, &metadataJSON,
			&t.CreatedBy, &t.CreatedAt, &t.UpdatedAt,
		)
		if err != nil {
			logger.Error("Failed to scan template", "error", err)
			continue
		}

		// Parse JSON fields
		if tagsJSON != nil {
			json.Unmarshal(tagsJSON, &t.Tags)
		}
		if metadataJSON != nil {
			json.Unmarshal(metadataJSON, &t.Metadata)
		}

		templates = append(templates, t)
	}

	totalPages := (total + pageSize - 1) / pageSize

	response := TemplateListResponse{
		Templates:  templates,
		Total:      total,
		Page:       page,
		PageSize:   pageSize,
		TotalPages: totalPages,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

// POST /api/admin/templates - Create a new VM template
func (h *TemplateHandlers) CreateTemplate(w http.ResponseWriter, r *http.Request) {
	var req CreateTemplateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate required fields
	if req.Name == "" || req.OS == "" {
		http.Error(w, "Name and OS are required", http.StatusBadRequest)
		return
	}

	// Get user from context (assuming middleware sets this)
	username := r.Header.Get("X-User-Email")
	if username == "" {
		username = "system"
	}

	// Convert tags and metadata to JSON
	tagsJSON, _ := json.Marshal(req.Tags)
	metadataJSON, _ := json.Marshal(req.Metadata)

	// Insert template
	query := `
		INSERT INTO vm_templates (name, description, os, os_version, cpu_cores, memory_mb, disk_gb,
		                          image_path, is_public, tags, metadata, created_by, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, NOW(), NOW())
		RETURNING id, created_at, updated_at
	`

	var template VMTemplate
	template.Name = req.Name
	template.Description = req.Description
	template.OS = req.OS
	template.OSVersion = req.OSVersion
	template.CPUCores = req.CPUCores
	template.MemoryMB = req.MemoryMB
	template.DiskGB = req.DiskGB
	template.ImagePath = req.ImagePath
	template.IsPublic = req.IsPublic
	template.Tags = req.Tags
	template.Metadata = req.Metadata
	template.CreatedBy = username

	err := h.db.QueryRow(query,
		req.Name, req.Description, req.OS, req.OSVersion,
		req.CPUCores, req.MemoryMB, req.DiskGB, req.ImagePath,
		req.IsPublic, tagsJSON, metadataJSON, username,
	).Scan(&template.ID, &template.CreatedAt, &template.UpdatedAt)

	if err != nil {
		logger.Error("Failed to create template", "error", err)
		http.Error(w, "Failed to create template", http.StatusInternalServerError)
		return
	}

	logger.Info("Template created", "id", template.ID, "name", template.Name, "created_by", username)

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(template)
}

// GET /api/admin/templates/{id} - Get template by ID
func (h *TemplateHandlers) GetTemplate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	query := `
		SELECT id, name, description, os, os_version, cpu_cores, memory_mb, disk_gb,
		       image_path, is_public, usage_count, tags, metadata, created_by, created_at, updated_at
		FROM vm_templates
		WHERE id = $1
	`

	var t VMTemplate
	var tagsJSON, metadataJSON []byte

	err := h.db.QueryRow(query, id).Scan(
		&t.ID, &t.Name, &t.Description, &t.OS, &t.OSVersion,
		&t.CPUCores, &t.MemoryMB, &t.DiskGB, &t.ImagePath,
		&t.IsPublic, &t.UsageCount, &tagsJSON, &metadataJSON,
		&t.CreatedBy, &t.CreatedAt, &t.UpdatedAt,
	)

	if err == sql.ErrNoRows {
		http.Error(w, "Template not found", http.StatusNotFound)
		return
	} else if err != nil {
		logger.Error("Failed to get template", "error", err, "id", id)
		http.Error(w, "Failed to get template", http.StatusInternalServerError)
		return
	}

	// Parse JSON fields
	if tagsJSON != nil {
		json.Unmarshal(tagsJSON, &t.Tags)
	}
	if metadataJSON != nil {
		json.Unmarshal(metadataJSON, &t.Metadata)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(t)
}

// PUT /api/admin/templates/{id} - Update template
func (h *TemplateHandlers) UpdateTemplate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	var req UpdateTemplateRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Build dynamic update query
	updates := []string{}
	args := []interface{}{}
	argIndex := 1

	if req.Name != nil {
		updates = append(updates, `name = $`+strconv.Itoa(argIndex))
		args = append(args, *req.Name)
		argIndex++
	}
	if req.Description != nil {
		updates = append(updates, `description = $`+strconv.Itoa(argIndex))
		args = append(args, *req.Description)
		argIndex++
	}
	if req.OS != nil {
		updates = append(updates, `os = $`+strconv.Itoa(argIndex))
		args = append(args, *req.OS)
		argIndex++
	}
	if req.OSVersion != nil {
		updates = append(updates, `os_version = $`+strconv.Itoa(argIndex))
		args = append(args, *req.OSVersion)
		argIndex++
	}
	if req.CPUCores != nil {
		updates = append(updates, `cpu_cores = $`+strconv.Itoa(argIndex))
		args = append(args, *req.CPUCores)
		argIndex++
	}
	if req.MemoryMB != nil {
		updates = append(updates, `memory_mb = $`+strconv.Itoa(argIndex))
		args = append(args, *req.MemoryMB)
		argIndex++
	}
	if req.DiskGB != nil {
		updates = append(updates, `disk_gb = $`+strconv.Itoa(argIndex))
		args = append(args, *req.DiskGB)
		argIndex++
	}
	if req.ImagePath != nil {
		updates = append(updates, `image_path = $`+strconv.Itoa(argIndex))
		args = append(args, *req.ImagePath)
		argIndex++
	}
	if req.IsPublic != nil {
		updates = append(updates, `is_public = $`+strconv.Itoa(argIndex))
		args = append(args, *req.IsPublic)
		argIndex++
	}
	if req.Tags != nil {
		tagsJSON, _ := json.Marshal(req.Tags)
		updates = append(updates, `tags = $`+strconv.Itoa(argIndex))
		args = append(args, tagsJSON)
		argIndex++
	}
	if req.Metadata != nil {
		metadataJSON, _ := json.Marshal(req.Metadata)
		updates = append(updates, `metadata = $`+strconv.Itoa(argIndex))
		args = append(args, metadataJSON)
		argIndex++
	}

	if len(updates) == 0 {
		http.Error(w, "No fields to update", http.StatusBadRequest)
		return
	}

	// Add updated_at
	updates = append(updates, `updated_at = NOW()`)

	// Add ID to args
	args = append(args, id)

	query := `UPDATE vm_templates SET ` + strings.Join(updates, ", ") + ` WHERE id = $` + strconv.Itoa(argIndex)

	result, err := h.db.Exec(query, args...)
	if err != nil {
		logger.Error("Failed to update template", "error", err, "id", id)
		http.Error(w, "Failed to update template", http.StatusInternalServerError)
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		http.Error(w, "Template not found", http.StatusNotFound)
		return
	}

	logger.Info("Template updated", "id", id)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Template updated successfully",
		"id":      id,
	})
}

// DELETE /api/admin/templates/{id} - Delete template
func (h *TemplateHandlers) DeleteTemplate(w http.ResponseWriter, r *http.Request) {
	vars := mux.Vars(r)
	id := vars["id"]

	result, err := h.db.Exec("DELETE FROM vm_templates WHERE id = $1", id)
	if err != nil {
		logger.Error("Failed to delete template", "error", err, "id", id)
		http.Error(w, "Failed to delete template", http.StatusInternalServerError)
		return
	}

	rowsAffected, _ := result.RowsAffected()
	if rowsAffected == 0 {
		http.Error(w, "Template not found", http.StatusNotFound)
		return
	}

	logger.Info("Template deleted", "id", id)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"message": "Template deleted successfully",
		"id":      id,
	})
}
