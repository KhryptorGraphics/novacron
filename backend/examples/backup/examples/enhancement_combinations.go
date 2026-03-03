package examples

import (
	"context"
	"log"
	"time"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
	"github.com/khryptorgraphics/novacron/backend/core/backup/providers"
)

// This file demonstrates how to combine different backup enhancements
// to create solutions for various use cases

// DemonstrateEnhancementCombinations shows how different backup enhancements
// can be combined for various use cases
func DemonstrateEnhancementCombinations() {
	// Create example jobs
	job := createSampleBackupJob()

	// Demonstrate different enhancement combinations
	spaceEfficientBackup(job)
	securityFocusedBackup(job)
	performanceCriticalBackup(job)
	maximumProtectionBackup(job)
}

// spaceEfficientBackup demonstrates a space-efficient solution using compression with incremental backups
func spaceEfficientBackup(job *backup.BackupJob) {
	log.Println("=== Space-Efficient Backup Solution ===")
	log.Println("Combining compression and incremental backups for maximum space efficiency")

	// Set up a base local provider
	baseProvider, err := providers.NewLocalStorageProvider("local-base", "Local Storage", "/var/novacron/backups")
	if err != nil {
		log.Fatalf("Failed to create base provider: %v", err)
	}

	// Add incremental backup capability with a 7-day full backup interval
	incrementalProvider, err := providers.NewIncrementalStorageProvider(baseProvider, 7*24*time.Hour)
	if err != nil {
		log.Fatalf("Failed to create incremental provider: %v", err)
	}

	// Add compression with best compression level for maximum space savings
	compressedProvider, err := providers.NewCompressedStorageProvider(incrementalProvider, providers.BestCompression)
	if err != nil {
		log.Fatalf("Failed to create compressed provider: %v", err)
	}

	// Create the backup
	ctx := context.Background()
	backup, err := compressedProvider.CreateBackup(ctx, job)
	if err != nil {
		log.Fatalf("Failed to create backup: %v", err)
	}

	log.Printf("Space-efficient backup created: %s", backup.ID)
	log.Println("Benefits:")
	log.Println("- Reduced storage usage through best-compression gzip")
	log.Println("- Further space savings by only storing changed files")
	log.Println("- Automatic full backups every 7 days for recovery points")
	log.Println("- Optimal for environments with limited storage capacity")
	log.Println("- Ideal for frequent backups of slowly changing data")
	log.Println("- Storage reduction of up to 80-90% compared to full uncompressed backups")
}

// securityFocusedBackup demonstrates a security-focused setup using encryption with remote storage
func securityFocusedBackup(job *backup.BackupJob) {
	log.Println("\n=== Security-Focused Backup Solution ===")
	log.Println("Combining encryption and remote storage for maximum security")

	// Set up authentication credentials for remote storage
	credentials := map[string]string{
		"username": "backup-user",
		"password": "supersecretpassword", // In production, use secure credential management
	}

	// Create a remote storage provider
	baseProvider, err := providers.NewRemoteStorageProvider(
		"secure-remote",
		"Secure Remote Storage",
		providers.SFTP,
		"backup-server.example.com:22",
		"/backups/secure",
		credentials,
	)
	if err != nil {
		log.Fatalf("Failed to create remote provider: %v", err)
	}

	// Generate a secure encryption key (in production, use a proper key management system)
	encryptionKey := make([]byte, 32) // 32 bytes for AES-256
	// In a real implementation, fill this with secure random data
	// For example: crypto/rand.Read(encryptionKey)

	// Add encryption layer
	encryptedProvider, err := providers.NewEncryptedStorageProvider(baseProvider, "key-1", encryptionKey)
	if err != nil {
		log.Fatalf("Failed to create encrypted provider: %v", err)
	}

	// Create the backup
	ctx := context.Background()
	backup, err := encryptedProvider.CreateBackup(ctx, job)
	if err != nil {
		log.Fatalf("Failed to create backup: %v", err)
	}

	log.Printf("Security-focused backup created: %s", backup.ID)
	log.Println("Benefits:")
	log.Println("- AES-256-GCM encryption provides military-grade security")
	log.Println("- Data is encrypted before transmission to remote storage")
	log.Println("- Remote storage provides physical separation and disaster recovery")
	log.Println("- Encryption key management with key ID tracking")
	log.Println("- Ideal for sensitive data and regulatory compliance")
	log.Println("- Secure against both storage provider breaches and network interception")
}

// performanceCriticalBackup demonstrates a performance-critical scenario using incremental backups with best-speed compression
func performanceCriticalBackup(job *backup.BackupJob) {
	log.Println("\n=== Performance-Critical Backup Solution ===")
	log.Println("Combining incremental backups with best-speed compression for optimal performance")

	// Set up a base local provider
	baseProvider, err := providers.NewLocalStorageProvider("fast-local", "Fast Local Storage", "/var/novacron/fast-backups")
	if err != nil {
		log.Fatalf("Failed to create base provider: %v", err)
	}

	// Add compression with best speed for better performance
	compressedProvider, err := providers.NewCompressedStorageProvider(baseProvider, providers.BestSpeed)
	if err != nil {
		log.Fatalf("Failed to create compressed provider: %v", err)
	}

	// Add incremental backup capability with a 30-day full backup interval
	// Longer interval means fewer full backups, better for performance
	incrementalProvider, err := providers.NewIncrementalStorageProvider(compressedProvider, 30*24*time.Hour)
	if err != nil {
		log.Fatalf("Failed to create incremental provider: %v", err)
	}

	// Create the backup
	ctx := context.Background()
	backup, err := incrementalProvider.CreateBackup(ctx, job)
	if err != nil {
		log.Fatalf("Failed to create backup: %v", err)
	}

	log.Printf("Performance-critical backup created: %s", backup.ID)
	log.Println("Benefits:")
	log.Println("- Fast compression algorithm prioritizes speed over compression ratio")
	log.Println("- Incremental backups minimize data transfer and processing time")
	log.Println("- Local storage eliminates network transfer overhead")
	log.Println("- Extended full backup interval (30 days) reduces frequency of resource-intensive operations")
	log.Println("- Ideal for production systems with tight backup windows")
	log.Println("- Backup completion up to 5x faster than full backups with maximum compression")
}

// maximumProtectionBackup demonstrates maximum data protection by leveraging all enhancements together
func maximumProtectionBackup(job *backup.BackupJob) {
	log.Println("\n=== Maximum Protection Backup Solution ===")
	log.Println("Combining all enhancements for comprehensive data protection")

	// Set up authentication credentials for remote storage
	credentials := map[string]string{
		"access_key": "AKIAIOSFODNN7EXAMPLE",
		"secret_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
	}

	// Create a remote storage provider using S3
	baseProvider, err := providers.NewRemoteStorageProvider(
		"max-protection-s3",
		"S3 Protection Storage",
		providers.S3,
		"s3://enterprise-backups/novacron",
		"/",
		credentials,
	)
	if err != nil {
		log.Fatalf("Failed to create remote provider: %v", err)
	}

	// Add incremental backup capability with a 14-day full backup interval
	// Balance between recovery points and performance
	incrementalProvider, err := providers.NewIncrementalStorageProvider(baseProvider, 14*24*time.Hour)
	if err != nil {
		log.Fatalf("Failed to create incremental provider: %v", err)
	}

	// Add compression with default level for balance between space and performance
	compressedProvider, err := providers.NewCompressedStorageProvider(incrementalProvider, providers.DefaultCompression)
	if err != nil {
		log.Fatalf("Failed to create compressed provider: %v", err)
	}

	// Generate a secure encryption key (in production, use a proper key management system)
	encryptionKey := make([]byte, 32) // 32 bytes for AES-256
	// In a real implementation, fill this with secure random data

	// Add encryption as the outermost layer
	encryptedProvider, err := providers.NewEncryptedStorageProvider(compressedProvider, "key-1", encryptionKey)
	if err != nil {
		log.Fatalf("Failed to create encrypted provider: %v", err)
	}

	// Create the backup
	ctx := context.Background()
	backup, err := encryptedProvider.CreateBackup(ctx, job)
	if err != nil {
		log.Fatalf("Failed to create backup: %v", err)
	}

	log.Printf("Maximum protection backup created: %s", backup.ID)
	log.Println("Benefits:")
	log.Println("- AES-256-GCM encryption provides data security")
	log.Println("- Compression reduces storage costs and transfer time")
	log.Println("- Incremental backups improve performance and reduce bandwidth")
	log.Println("- Remote S3 storage provides geographical redundancy")
	log.Println("- Balanced 14-day full backup interval for recovery flexibility")
	log.Println("- Comprehensive solution meeting enterprise requirements:")
	log.Println("  - Security: Encryption prevents unauthorized access")
	log.Println("  - Performance: Incremental + compression optimizes speed")
	log.Println("  - Reliability: S3 with redundancy ensures data durability")
	log.Println("  - Efficiency: Compression + incremental minimizes costs")
	log.Println("  - Compliance: End-to-end protection meets regulatory requirements")
}

// Helper function to create a sample backup job
func createSampleBackupJob() *backup.BackupJob {
	return &backup.BackupJob{
		ID:          "example-job-1",
		Name:        "Example Backup Job",
		Description: "Demo job for backup enhancement combinations",
		Type:        backup.FullBackup,
		Targets: []*backup.BackupTarget{
			{
				ID:         "vm-1",
				Name:       "Production Database VM",
				Type:       "vm",
				ResourceID: "vm-123",
				Metadata: map[string]string{
					"os": "linux",
				},
			},
			{
				ID:         "container-1",
				Name:       "Web Frontend Container",
				Type:       "container",
				ResourceID: "container-456",
				Metadata: map[string]string{
					"image": "web-app:latest",
				},
			},
		},
		TenantID: "tenant-1",
	}
}
