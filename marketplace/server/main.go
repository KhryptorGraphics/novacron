package main

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"
	_ "github.com/lib/pq"
)

// Template represents a VM or infrastructure template
type Template struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Category    string                 `json:"category"`
	Version     string                 `json:"version"`
	Author      string                 `json:"author"`
	Rating      float64                `json:"rating"`
	Downloads   int64                  `json:"downloads"`
	Verified    bool                   `json:"verified"`
	Tags        []string               `json:"tags"`
	Config      map[string]interface{} `json:"config"`
	Icon        string                 `json:"icon"`
	CreatedAt   time.Time              `json:"created_at"`
	UpdatedAt   time.Time              `json:"updated_at"`
}

// MarketplaceServer manages the template marketplace
type MarketplaceServer struct {
	db     *sql.DB
	router *mux.Router
}

// NewMarketplaceServer creates a new marketplace server
func NewMarketplaceServer(dbURL string) (*MarketplaceServer, error) {
	db, err := sql.Open("postgres", dbURL)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to database: %w", err)
	}

	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	server := &MarketplaceServer{
		db:     db,
		router: mux.NewRouter(),
	}

	server.setupRoutes()

	return server, nil
}

func (s *MarketplaceServer) setupRoutes() {
	api := s.router.PathPrefix("/api/v1").Subrouter()

	// Template routes
	api.HandleFunc("/templates", s.listTemplates).Methods("GET")
	api.HandleFunc("/templates/{id}", s.getTemplate).Methods("GET")
	api.HandleFunc("/templates", s.createTemplate).Methods("POST")
	api.HandleFunc("/templates/{id}", s.updateTemplate).Methods("PUT")
	api.HandleFunc("/templates/{id}", s.deleteTemplate).Methods("DELETE")
	api.HandleFunc("/templates/{id}/download", s.downloadTemplate).Methods("POST")
	api.HandleFunc("/templates/{id}/rate", s.rateTemplate).Methods("POST")

	// Search and filter
	api.HandleFunc("/templates/search", s.searchTemplates).Methods("GET")
	api.HandleFunc("/templates/categories", s.getCategories).Methods("GET")
	api.HandleFunc("/templates/featured", s.getFeaturedTemplates).Methods("GET")

	// Validation
	api.HandleFunc("/templates/validate", s.validateTemplate).Methods("POST")
}

func (s *MarketplaceServer) listTemplates(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	// Query parameters
	category := r.URL.Query().Get("category")
	sortBy := r.URL.Query().Get("sort")
	if sortBy == "" {
		sortBy = "downloads"
	}

	query := `
		SELECT id, name, description, category, version, author, rating, downloads, verified, tags, icon, created_at, updated_at
		FROM templates
		WHERE ($1 = '' OR category = $1)
		ORDER BY ` + sortBy + ` DESC
		LIMIT 100
	`

	rows, err := s.db.QueryContext(ctx, query, category)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer rows.Close()

	templates := []Template{}
	for rows.Next() {
		var t Template
		var tags string
		err := rows.Scan(&t.ID, &t.Name, &t.Description, &t.Category, &t.Version,
			&t.Author, &t.Rating, &t.Downloads, &t.Verified, &tags, &t.Icon,
			&t.CreatedAt, &t.UpdatedAt)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		json.Unmarshal([]byte(tags), &t.Tags)
		templates = append(templates, t)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(templates)
}

func (s *MarketplaceServer) getTemplate(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	vars := mux.Vars(r)
	id := vars["id"]

	query := `
		SELECT id, name, description, category, version, author, rating, downloads, verified, tags, config, icon, created_at, updated_at
		FROM templates
		WHERE id = $1
	`

	var t Template
	var tags, config string
	err := s.db.QueryRowContext(ctx, query, id).Scan(
		&t.ID, &t.Name, &t.Description, &t.Category, &t.Version,
		&t.Author, &t.Rating, &t.Downloads, &t.Verified, &tags, &config, &t.Icon,
		&t.CreatedAt, &t.UpdatedAt)

	if err == sql.ErrNoRows {
		http.Error(w, "Template not found", http.StatusNotFound)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	json.Unmarshal([]byte(tags), &t.Tags)
	json.Unmarshal([]byte(config), &t.Config)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(t)
}

func (s *MarketplaceServer) createTemplate(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	var t Template
	if err := json.NewDecoder(r.Body).Decode(&t); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// Validate template
	if err := s.validateTemplateConfig(t.Config); err != nil {
		http.Error(w, fmt.Sprintf("Invalid template: %v", err), http.StatusBadRequest)
		return
	}

	// Generate ID
	t.ID = fmt.Sprintf("tpl-%d", time.Now().Unix())
	t.CreatedAt = time.Now()
	t.UpdatedAt = time.Now()
	t.Rating = 0
	t.Downloads = 0
	t.Verified = false

	tags, _ := json.Marshal(t.Tags)
	config, _ := json.Marshal(t.Config)

	query := `
		INSERT INTO templates (id, name, description, category, version, author, rating, downloads, verified, tags, config, icon, created_at, updated_at)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
	`

	_, err := s.db.ExecContext(ctx, query, t.ID, t.Name, t.Description, t.Category, t.Version,
		t.Author, t.Rating, t.Downloads, t.Verified, string(tags), string(config), t.Icon,
		t.CreatedAt, t.UpdatedAt)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteStatus(http.StatusCreated)
	json.NewEncoder(w).Encode(t)
}

func (s *MarketplaceServer) updateTemplate(w http.ResponseWriter, r *http.Request) {
	// Update template implementation
}

func (s *MarketplaceServer) deleteTemplate(w http.ResponseWriter, r *http.Request) {
	// Delete template implementation
}

func (s *MarketplaceServer) downloadTemplate(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()
	vars := mux.Vars(r)
	id := vars["id"]

	// Increment download counter
	query := `UPDATE templates SET downloads = downloads + 1 WHERE id = $1`
	_, err := s.db.ExecContext(ctx, query, id)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
}

func (s *MarketplaceServer) rateTemplate(w http.ResponseWriter, r *http.Request) {
	// Rating implementation
}

func (s *MarketplaceServer) searchTemplates(w http.ResponseWriter, r *http.Request) {
	// Search implementation with full-text search
}

func (s *MarketplaceServer) getCategories(w http.ResponseWriter, r *http.Request) {
	categories := []string{
		"web-applications",
		"databases",
		"ml-workloads",
		"development-environments",
		"game-servers",
		"infrastructure",
		"security",
		"monitoring",
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(categories)
}

func (s *MarketplaceServer) getFeaturedTemplates(w http.ResponseWriter, r *http.Request) {
	// Return featured/verified templates
}

func (s *MarketplaceServer) validateTemplate(w http.ResponseWriter, r *http.Request) {
	var config map[string]interface{}
	if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	if err := s.validateTemplateConfig(config); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"valid": false,
			"error": err.Error(),
		})
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"valid": true,
	})
}

func (s *MarketplaceServer) validateTemplateConfig(config map[string]interface{}) error {
	// Required fields
	requiredFields := []string{"name", "memory", "cpus", "disk", "image"}
	for _, field := range requiredFields {
		if _, exists := config[field]; !exists {
			return fmt.Errorf("missing required field: %s", field)
		}
	}

	// Validate resource limits
	if memory, ok := config["memory"].(float64); ok {
		if memory < 512*1024*1024 || memory > 512*1024*1024*1024 {
			return fmt.Errorf("memory must be between 512MB and 512GB")
		}
	}

	if cpus, ok := config["cpus"].(float64); ok {
		if cpus < 1 || cpus > 128 {
			return fmt.Errorf("cpus must be between 1 and 128")
		}
	}

	// Security validation
	if _, exists := config["unsafe_options"]; exists {
		return fmt.Errorf("unsafe options are not allowed")
	}

	return nil
}

func (s *MarketplaceServer) Start(addr string) error {
	log.Printf("Starting marketplace server on %s", addr)
	return http.ListenAndServe(addr, s.router)
}

func main() {
	dbURL := "postgres://marketplace:password@localhost/marketplace?sslmode=disable"
	server, err := NewMarketplaceServer(dbURL)
	if err != nil {
		log.Fatalf("Failed to create server: %v", err)
	}

	if err := server.Start(":8080"); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
