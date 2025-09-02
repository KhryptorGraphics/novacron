package migrations

import (
	"database/sql"
	"embed"
	"fmt"
	"log"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source/iofs"
)

//go:embed *.sql
var fs embed.FS

// Manager handles database migrations
type Manager struct {
	db *sql.DB
	m  *migrate.Migrate
}

// NewManager creates a new migration manager
func NewManager(db *sql.DB, dbName string) (*Manager, error) {
	driver, err := postgres.WithInstance(db, &postgres.Config{})
	if err != nil {
		return nil, fmt.Errorf("failed to create postgres driver: %w", err)
	}

	d, err := iofs.New(fs, ".")
	if err != nil {
		return nil, fmt.Errorf("failed to create iofs driver: %w", err)
	}

	m, err := migrate.NewWithInstance("iofs", d, dbName, driver)
	if err != nil {
		return nil, fmt.Errorf("failed to create migrate instance: %w", err)
	}

	return &Manager{
		db: db,
		m:  m,
	}, nil
}

// Up runs all pending migrations
func (m *Manager) Up() error {
	if err := m.m.Up(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("failed to run migrations: %w", err)
	}
	
	version, dirty, err := m.m.Version()
	if err != nil && err != migrate.ErrNilVersion {
		return fmt.Errorf("failed to get migration version: %w", err)
	}
	
	log.Printf("Database migration completed. Version: %d, Dirty: %v", version, dirty)
	return nil
}

// Down rolls back the last migration
func (m *Manager) Down() error {
	if err := m.m.Steps(-1); err != nil {
		return fmt.Errorf("failed to rollback migration: %w", err)
	}
	return nil
}

// Version returns the current migration version
func (m *Manager) Version() (uint, bool, error) {
	return m.m.Version()
}

// Force sets a specific migration version
func (m *Manager) Force(version int) error {
	if err := m.m.Force(version); err != nil {
		return fmt.Errorf("failed to force migration version: %w", err)
	}
	return nil
}

// Close closes the migration manager
func (m *Manager) Close() error {
	sourceErr, dbErr := m.m.Close()
	if sourceErr != nil {
		return fmt.Errorf("failed to close source: %w", sourceErr)
	}
	if dbErr != nil {
		return fmt.Errorf("failed to close database: %w", dbErr)
	}
	return nil
}