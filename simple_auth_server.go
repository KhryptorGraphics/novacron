package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/mux"
)

type User struct {
	ID        string `json:"id"`
	Email     string `json:"email"`
	FirstName string `json:"firstName"`
	LastName  string `json:"lastName"`
	Status    string `json:"status"`
}

type AuthResponse struct {
	Token     string    `json:"token"`
	ExpiresAt time.Time `json:"expiresAt"`
	User      User      `json:"user"`
}

type LoginRequest struct {
	Email    string `json:"email"`
	Password string `json:"password"`
}

type RegisterRequest struct {
	FirstName string `json:"firstName"`
	LastName  string `json:"lastName"`
	Email     string `json:"email"`
	Password  string `json:"password"`
}

func loginHandler(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Simple validation
	if req.Email == "" || req.Password == "" {
		http.Error(w, "Email and password are required", http.StatusBadRequest)
		return
	}

	// For demo purposes, accept any login
	response := AuthResponse{
		Token:     "demo-token-12345",
		ExpiresAt: time.Now().Add(24 * time.Hour),
		User: User{
			ID:        "user-123",
			Email:     req.Email,
			FirstName: "Demo",
			LastName:  "User",
			Status:    "active",
		},
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}

func registerHandler(w http.ResponseWriter, r *http.Request) {
	var req RegisterRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	// Simple validation
	if req.Email == "" || req.Password == "" || req.FirstName == "" || req.LastName == "" {
		http.Error(w, "All fields are required", http.StatusBadRequest)
		return
	}

	// For demo purposes, accept any registration
	user := User{
		ID:        "user-" + fmt.Sprintf("%d", time.Now().Unix()),
		Email:     req.Email,
		FirstName: req.FirstName,
		LastName:  req.LastName,
		Status:    "active",
	}

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(user)
}

func main() {
	router := mux.NewRouter()

	// Auth routes
	router.HandleFunc("/api/auth/login", loginHandler).Methods("POST")
	router.HandleFunc("/api/auth/register", registerHandler).Methods("POST")

	// Health check
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status": "ok"}`))
	}).Methods("GET")

	fmt.Println("Starting simple auth server on :8090")
	log.Fatal(http.ListenAndServe(":8090", router))
}