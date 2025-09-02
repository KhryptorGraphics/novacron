// +build ignore

package main

import (
	"context"
	"fmt"
	"log"
	"os"
	
	"github.com/novacron/backend/core/security"
)

func main() {
	ctx := context.Background()
	
	// Get Vault configuration from environment
	vaultAddr := getEnvOrDefault("VAULT_ADDR", "http://localhost:8200")
	rootToken := getEnvOrDefault("VAULT_ROOT_TOKEN", "root")
	
	fmt.Println("🔐 NovaCron Vault Setup")
	fmt.Println("========================")
	fmt.Printf("Vault Address: %s\n", vaultAddr)
	fmt.Println()
	
	// Initialize Vault
	initializer, err := security.NewVaultInitializer(vaultAddr, rootToken)
	if err != nil {
		log.Fatalf("❌ Failed to connect to Vault: %v", err)
	}
	
	// Setup NovaCron secrets
	fmt.Println("📝 Creating NovaCron secrets...")
	if err := initializer.SetupNovaCronSecrets(ctx); err != nil {
		log.Fatalf("❌ Failed to setup secrets: %v", err)
	}
	
	// Create application token
	fmt.Println("🔑 Creating application token...")
	appToken, err := initializer.CreateAppToken(ctx)
	if err != nil {
		log.Fatalf("❌ Failed to create app token: %v", err)
	}
	
	fmt.Println()
	fmt.Println("✅ Vault setup complete!")
	fmt.Println()
	fmt.Println("Add this to your environment or .env file:")
	fmt.Println("============================================")
	fmt.Printf("export VAULT_ADDR=%s\n", vaultAddr)
	fmt.Printf("export VAULT_TOKEN=%s\n", appToken)
	fmt.Println()
	fmt.Println("For development, you can also use:")
	fmt.Println("export NOVACRON_ENV=development")
	fmt.Println()
	
	// Save token to file for easy access
	tokenFile := ".vault-token"
	if err := os.WriteFile(tokenFile, []byte(appToken), 0600); err != nil {
		log.Printf("⚠️  Warning: Could not save token to file: %v", err)
	} else {
		fmt.Printf("Token saved to: %s\n", tokenFile)
	}
}

func getEnvOrDefault(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}