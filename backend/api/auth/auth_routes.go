package auth

import (
	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/auth"
)

// RegisterRoutes registers authentication API routes
func RegisterRoutes(router *mux.Router, authService auth.AuthService) {
	// Create API handlers
	authHandler := NewHandler(authService)
	
	// Register auth routes
	authRouter := router.PathPrefix("/api/auth").Subrouter()
	
	// Public routes
	authRouter.HandleFunc("/register", authHandler.Register).Methods("POST")
	authRouter.HandleFunc("/login", authHandler.Login).Methods("POST")
	authRouter.HandleFunc("/forgot-password", authHandler.ForgotPassword).Methods("POST")
	authRouter.HandleFunc("/reset-password", authHandler.ResetPassword).Methods("POST")
	
	// Protected routes (would require auth middleware)
	authRouter.HandleFunc("/logout", authHandler.Logout).Methods("POST")
	authRouter.HandleFunc("/refresh", authHandler.Refresh).Methods("POST")
}