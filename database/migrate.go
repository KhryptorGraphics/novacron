package main

import (
	"database/sql"
	"embed"
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	_ "github.com/lib/pq"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

func main() {
	var (
		dbURL     = flag.String("db", "", "Database URL (can also use DB_URL env var)")
		direction = flag.String("direction", "up", "Migration direction: up, down, force, version")
		steps     = flag.Int("steps", 0, "Number of migrations to run (0 = all)")
		version   = flag.Int("version", 0, "Force migrate to version")
		create    = flag.String("create", "", "Create a new migration with the given name")
	)
	flag.Parse()

	// Get database URL from flag or environment
	databaseURL := *dbURL
	if databaseURL == "" {
		databaseURL = os.Getenv("DB_URL")
	}
	if databaseURL == "" {
		databaseURL = os.Getenv("DATABASE_URL")
	}

	// Handle create migration command
	if *create != "" {
		if err := createMigration(*create); err != nil {
			log.Fatal("Failed to create migration:", err)
		}
		return
	}

	if databaseURL == "" {
		log.Fatal("Database URL is required. Use -db flag or set DB_URL/DATABASE_URL environment variable")
	}

	// Open database connection
	db, err := sql.Open("postgres", databaseURL)
	if err != nil {
		log.Fatal("Failed to open database:", err)
	}
	defer db.Close()

	// Create driver instance
	driver, err := postgres.WithInstance(db, &postgres.Config{})
	if err != nil {
		log.Fatal("Failed to create driver:", err)
	}

	// Create source instance from embedded filesystem
	source, err := iofs.New(migrationsFS, "migrations")
	if err != nil {
		log.Fatal("Failed to create source:", err)
	}

	// Create migrate instance
	m, err := migrate.NewWithInstance("iofs", source, "postgres", driver)
	if err != nil {
		log.Fatal("Failed to create migrate instance:", err)
	}

	// Execute migration based on direction
	switch *direction {
	case "up":
		if *steps > 0 {
			err = m.Steps(*steps)
		} else {
			err = m.Up()
		}
	case "down":
		if *steps > 0 {
			err = m.Steps(-*steps)
		} else {
			err = m.Down()
		}
	case "force":
		if *version == 0 {
			log.Fatal("Version is required for force migration")
		}
		err = m.Force(*version)
	case "version":
		v, dirty, verr := m.Version()
		if verr != nil {
			log.Fatal("Failed to get version:", verr)
		}
		fmt.Printf("Current version: %d, Dirty: %v\n", v, dirty)
		return
	case "drop":
		err = m.Drop()
	default:
		log.Fatal("Invalid direction. Use: up, down, force, version, or drop")
	}

	if err != nil && err != migrate.ErrNoChange {
		log.Fatal("Migration failed:", err)
	}

	if err == migrate.ErrNoChange {
		fmt.Println("No migrations to run")
	} else {
		fmt.Printf("Migration %s completed successfully\n", *direction)
	}
}

func createMigration(name string) error {
	timestamp := time.Now().Format("20060102150405")
	upFile := fmt.Sprintf("migrations/%s_%s.up.sql", timestamp, name)
	downFile := fmt.Sprintf("migrations/%s_%s.down.sql", timestamp, name)

	// Create up migration file
	upContent := fmt.Sprintf("-- Migration: %s\n-- Created: %s\n-- Direction: UP\n\n", name, time.Now().Format(time.RFC3339))
	if err := os.WriteFile(upFile, []byte(upContent), 0644); err != nil {
		return fmt.Errorf("failed to create up migration: %w", err)
	}

	// Create down migration file
	downContent := fmt.Sprintf("-- Migration: %s\n-- Created: %s\n-- Direction: DOWN\n\n", name, time.Now().Format(time.RFC3339))
	if err := os.WriteFile(downFile, []byte(downContent), 0644); err != nil {
		return fmt.Errorf("failed to create down migration: %w", err)
	}

	fmt.Printf("Created migration files:\n  %s\n  %s\n", upFile, downFile)
	return nil
}