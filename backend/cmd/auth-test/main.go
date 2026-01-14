package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/golang-jwt/jwt/v5"
	"github.com/gorilla/handlers"
	"github.com/gorilla/mux"
	_ "github.com/lib/pq"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
	"github.com/khryptorgraphics/novacron/backend/pkg/middleware"
)

func main() {
	// Initialize in-memory database for testing
	db, err := sql.Open("postgres", "postgres://user:pass@localhost:5432/novacron?sslmode=disable")
	if err != nil {
		// Fall back to in-memory testing without real database
		log.Printf("Database connection failed, using mock authentication: %v", err)
		startMockAuthServer()
		return
	}
	defer db.Close()

	// Test database connection
	if err := db.Ping(); err != nil {
		log.Printf("Database ping failed, using mock authentication: %v", err)
		startMockAuthServer()
		return
	}

	// Initialize authentication manager
	authManager := auth.NewSimpleAuthManager("test-jwt-secret-key", db)

	// Create router
	router := mux.NewRouter()

	// Add CORS middleware
	corsHandler := handlers.CORS(
		handlers.AllowedOrigins([]string{"*"}),
		handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
		handlers.AllowedHeaders([]string{"Content-Type", "Authorization"}),
	)

	// Add authentication middleware
	authMiddleware := middleware.NewAuthMiddleware(authManager)

	// Public routes (no auth required)
	registerAuthRoutes(router, authManager)

	// Protected routes (auth required)
	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.Use(authMiddleware.RequireAuth)

	// Test protected endpoint
	apiRouter.HandleFunc("/test", func(w http.ResponseWriter, r *http.Request) {
		userID, _ := r.Context().Value("user_id").(string)
		tenantID, _ := r.Context().Value("tenant_id").(string)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message":   "Protected endpoint accessed successfully",
			"user_id":   userID,
			"tenant_id": tenantID,
			"timestamp": time.Now().UTC(),
		})
	}).Methods("GET")

	// Health check
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status": "healthy",
			"service": "auth-test",
		})
	}).Methods("GET")

	log.Println("Starting auth test server on :8090...")
	log.Fatal(http.ListenAndServe(":8090", corsHandler(router)))
}

func registerAuthRoutes(router *mux.Router, authManager *auth.SimpleAuthManager) {
	// Login endpoint
	router.HandleFunc("/auth/login", func(w http.ResponseWriter, r *http.Request) {
		var loginReq struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}

		if err := json.NewDecoder(r.Body).Decode(&loginReq); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		user, token, err := authManager.Authenticate(loginReq.Username, loginReq.Password)
		if err != nil {
			http.Error(w, "Invalid credentials", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		response := map[string]interface{}{
			"token": token,
			"user": map[string]interface{}{
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"tenant_id": user.TenantID,
			},
		}

		// Add role information if available
		if len(user.Roles) > 0 {
			response["user"].(map[string]interface{})["role"] = user.Roles[0].Name
		}

		json.NewEncoder(w).Encode(response)
	}).Methods("POST")

	// Register endpoint
	router.HandleFunc("/auth/register", func(w http.ResponseWriter, r *http.Request) {
		var registerReq struct {
			Username string `json:"username"`
			Email    string `json:"email"`
			Password string `json:"password"`
			TenantID string `json:"tenant_id,omitempty"`
		}

		if err := json.NewDecoder(r.Body).Decode(&registerReq); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if registerReq.TenantID == "" {
			registerReq.TenantID = "default"
		}

		user, err := authManager.CreateUser(registerReq.Username, registerReq.Email, registerReq.Password, "user", registerReq.TenantID)
		if err != nil {
			http.Error(w, fmt.Sprintf("Failed to create user: %v", err), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		response := map[string]interface{}{
			"user": map[string]interface{}{
				"id":        user.ID,
				"username":  user.Username,
				"email":     user.Email,
				"tenant_id": user.TenantID,
			},
		}

		// Add role information if available
		if len(user.Roles) > 0 {
			response["user"].(map[string]interface{})["role"] = user.Roles[0].Name
		}

		json.NewEncoder(w).Encode(response)
	}).Methods("POST")

	// Token validation endpoint
	router.HandleFunc("/auth/validate", func(w http.ResponseWriter, r *http.Request) {
		authHeader := r.Header.Get("Authorization")
		if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
			http.Error(w, "Invalid or missing token", http.StatusUnauthorized)
			return
		}

		tokenString := strings.TrimPrefix(authHeader, "Bearer ")

		// Parse and validate the JWT token
		token, err := jwt.Parse(tokenString, func(token *jwt.Token) (interface{}, error) {
			if _, ok := token.Method.(*jwt.SigningMethodHMAC); !ok {
				return nil, jwt.ErrSignatureInvalid
			}
			return []byte(authManager.GetJWTSecret()), nil
		})

		if err != nil || !token.Valid {
			http.Error(w, "Invalid token", http.StatusUnauthorized)
			return
		}

		claims, ok := token.Claims.(jwt.MapClaims)
		if !ok {
			http.Error(w, "Invalid token claims", http.StatusUnauthorized)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"valid": true,
			"user": map[string]interface{}{
				"id":        claims["user_id"],
				"username":  claims["username"],
				"email":     claims["email"],
				"role":      claims["role"],
				"tenant_id": claims["tenant_id"],
			},
		})
	}).Methods("GET")
}

func startMockAuthServer() {
	router := mux.NewRouter()

	// Add CORS middleware
	corsHandler := handlers.CORS(
		handlers.AllowedOrigins([]string{"*"}),
		handlers.AllowedMethods([]string{"GET", "POST", "PUT", "DELETE", "OPTIONS"}),
		handlers.AllowedHeaders([]string{"Content-Type", "Authorization"}),
	)

	// Mock authentication endpoints
	router.HandleFunc("/auth/login", func(w http.ResponseWriter, r *http.Request) {
		var loginReq struct {
			Username string `json:"username"`
			Password string `json:"password"`
		}

		if err := json.NewDecoder(r.Body).Decode(&loginReq); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		// Mock authentication - accept any username/password combo
		if loginReq.Username == "" || loginReq.Password == "" {
			http.Error(w, "Username and password required", http.StatusBadRequest)
			return
		}

		// Generate a simple mock JWT token
		mockToken := "mock-jwt-token-" + loginReq.Username + "-" + fmt.Sprintf("%d", time.Now().Unix())

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"token": mockToken,
			"user": map[string]interface{}{
				"id":        "mock-user-id",
				"username":  loginReq.Username,
				"email":     loginReq.Username + "@example.com",
				"role":      "user",
				"tenant_id": "default",
			},
		})
	}).Methods("POST")

	router.HandleFunc("/auth/register", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"message": "Mock user registered successfully",
			"user": map[string]interface{}{
				"id":        "mock-user-id",
				"username":  "mockuser",
				"email":     "mockuser@example.com",
				"role":      "user",
				"tenant_id": "default",
			},
		})
	}).Methods("POST")

	// Health check
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{
			"status": "healthy",
			"service": "auth-test-mock",
		})
	}).Methods("GET")

	log.Println("Starting mock auth test server on :8090...")
	log.Fatal(http.ListenAndServe(":8090", corsHandler(router)))
}