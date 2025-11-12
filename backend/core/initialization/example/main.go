// Package main provides an example of using the NovaCron initialization system
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"novacron/backend/core/initialization"
	"novacron/backend/core/initialization/config"
	"novacron/backend/core/initialization/orchestrator"
)

// ExampleComponent demonstrates a simple component implementation
type ExampleComponent struct {
	name   string
	config *config.Config
	logger Logger
}

// Logger interface for logging
type Logger interface {
	Info(msg string, keysAndValues ...interface{})
	Error(msg string, err error, keysAndValues ...interface{})
	Debug(msg string, keysAndValues ...interface{})
	Warn(msg string, keysAndValues ...interface{})
}

// NewExampleComponent creates a new example component
func NewExampleComponent(name string, cfg *config.Config, logger Logger) *ExampleComponent {
	return &ExampleComponent{
		name:   name,
		config: cfg,
		logger: logger,
	}
}

// Name returns the component name
func (e *ExampleComponent) Name() string {
	return e.name
}

// Dependencies returns component dependencies
func (e *ExampleComponent) Dependencies() []string {
	return []string{} // No dependencies
}

// Initialize initializes the component
func (e *ExampleComponent) Initialize(ctx context.Context) error {
	e.logger.Info("Initializing component", "name", e.name)

	// Simulate initialization work
	time.Sleep(100 * time.Millisecond)

	e.logger.Info("Component initialized", "name", e.name)
	return nil
}

// Shutdown shuts down the component
func (e *ExampleComponent) Shutdown(ctx context.Context) error {
	e.logger.Info("Shutting down component", "name", e.name)

	// Simulate cleanup work
	time.Sleep(50 * time.Millisecond)

	e.logger.Info("Component shutdown complete", "name", e.name)
	return nil
}

// HealthCheck checks component health
func (e *ExampleComponent) HealthCheck(ctx context.Context) error {
	// Implement health check logic
	return nil
}

func main() {
	// Example 1: Generate default configuration
	if len(os.Args) > 1 && os.Args[1] == "generate-config" {
		configPath := "config.yaml"
		if len(os.Args) > 2 {
			configPath = os.Args[2]
		}

		if err := initialization.GenerateDefaultConfig(configPath); err != nil {
			log.Fatalf("Failed to generate config: %v", err)
		}

		fmt.Printf("Generated default configuration: %s\n", configPath)
		return
	}

	// Example 2: Initialize system with custom config
	configPath := os.Getenv("NOVACRON_CONFIG")
	if configPath == "" {
		configPath = "config.yaml"
	}

	// Check if config exists, generate if not
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		fmt.Printf("Config not found, generating default: %s\n", configPath)
		if err := initialization.GenerateDefaultConfig(configPath); err != nil {
			log.Fatalf("Failed to generate config: %v", err)
		}
	}

	// Create initializer
	init, err := initialization.NewInitializer(configPath)
	if err != nil {
		log.Fatalf("Failed to create initializer: %v", err)
	}

	// Register custom components
	container := init.GetContainer()
	logger := init.GetLogger()
	cfg := init.GetConfig()

	// Example: Register and initialize custom components
	// This demonstrates how to extend the initialization system

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Initialize system
	fmt.Println("Initializing NovaCron system...")
	if err := init.Initialize(ctx); err != nil {
		log.Fatalf("Initialization failed: %v", err)
	}

	fmt.Printf("NovaCron initialized successfully (Node ID: %s)\n", cfg.System.NodeID)

	// Print status
	status := init.GetStatus()
	fmt.Println("\nSystem Status:")
	if components, ok := status["components"].(map[string]orchestrator.ComponentStatus); ok {
		for name, st := range components {
			fmt.Printf("  - %s: %s\n", name, st)
		}
	}

	// Setup signal handling for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	fmt.Println("\nNovaCron is running. Press Ctrl+C to stop.")

	// Wait for signal
	sig := <-sigChan
	fmt.Printf("\nReceived signal %v, shutting down...\n", sig)

	// Create shutdown context with timeout
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer shutdownCancel()

	// Graceful shutdown
	if err := init.Shutdown(shutdownCtx); err != nil {
		log.Printf("Shutdown error: %v", err)
		os.Exit(1)
	}

	fmt.Println("NovaCron stopped successfully")
}

// Example usage of DI container
func exampleDIUsage(container *initialization.Initializer) {
	// Get services from container
	cfg := container.GetConfig()
	logger := container.GetLogger()

	logger.Info("Example DI usage",
		"node_id", cfg.System.NodeID,
		"log_level", cfg.System.LogLevel,
	)

	// Create component with dependencies
	exampleComp := NewExampleComponent("example", cfg, logger)

	ctx := context.Background()
	if err := exampleComp.Initialize(ctx); err != nil {
		logger.Error("Component init failed", err)
	}
}
