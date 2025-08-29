package vm

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sync"
)

// MigrationStorage defines the interface for migration storage operations
type MigrationStorage interface {
	// SaveMigrationRecord persists a migration record
	SaveMigrationRecord(record *MigrationRecord) error

	// LoadMigrationRecord retrieves a migration record by ID
	LoadMigrationRecord(migrationID string) (*MigrationRecord, error)

	// ListMigrationRecords retrieves all migration records
	ListMigrationRecords() ([]*MigrationRecord, error)

	// ListMigrationRecordsForVM retrieves all migration records for a specific VM
	ListMigrationRecordsForVM(vmID string) ([]*MigrationRecord, error)

	// DeleteMigrationRecord removes a migration record
	DeleteMigrationRecord(migrationID string) error
}

// FileMigrationStorage implements MigrationStorage using the filesystem
type FileMigrationStorage struct {
	storageDir string
	mu         sync.RWMutex
}

// NewFileMigrationStorage creates a new FileMigrationStorage
func NewFileMigrationStorage(storageDir string) (*FileMigrationStorage, error) {
	// Ensure the storage directory exists
	if err := os.MkdirAll(storageDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	return &FileMigrationStorage{
		storageDir: storageDir,
	}, nil
}

// SaveMigrationRecord persists a migration record to a file
func (s *FileMigrationStorage) SaveMigrationRecord(record *MigrationRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Marshal the record to JSON
	data, err := json.MarshalIndent(record, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal migration record: %w", err)
	}

	// Write to file
	filename := filepath.Join(s.storageDir, fmt.Sprintf("migration_%s.json", record.ID))
	if err := ioutil.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write migration record to file: %w", err)
	}

	return nil
}

// LoadMigrationRecord retrieves a migration record from a file
func (s *FileMigrationStorage) LoadMigrationRecord(migrationID string) (*MigrationRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Read from file
	filename := filepath.Join(s.storageDir, fmt.Sprintf("migration_%s.json", migrationID))
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, ErrMigrationNotFound
		}
		return nil, fmt.Errorf("failed to read migration record from file: %w", err)
	}

	// Unmarshal the record
	record := &MigrationRecord{}
	if err := json.Unmarshal(data, record); err != nil {
		return nil, fmt.Errorf("failed to unmarshal migration record: %w", err)
	}

	return record, nil
}

// ListMigrationRecords retrieves all migration records
func (s *FileMigrationStorage) ListMigrationRecords() ([]*MigrationRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Find all migration record files
	pattern := filepath.Join(s.storageDir, "migration_*.json")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return nil, fmt.Errorf("failed to list migration records: %w", err)
	}

	// Load each record
	records := make([]*MigrationRecord, 0, len(matches))
	for _, filename := range matches {
		data, err := ioutil.ReadFile(filename)
		if err != nil {
			continue // Skip files that can't be read
		}

		record := &MigrationRecord{}
		if err := json.Unmarshal(data, record); err != nil {
			continue // Skip files that can't be unmarshaled
		}

		records = append(records, record)
	}

	return records, nil
}

// ListMigrationRecordsForVM retrieves all migration records for a specific VM
func (s *FileMigrationStorage) ListMigrationRecordsForVM(vmID string) ([]*MigrationRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Get all records
	allRecords, err := s.ListMigrationRecords()
	if err != nil {
		return nil, err
	}

	// Filter records for the specified VM
	vmRecords := make([]*MigrationRecord, 0)
	for _, record := range allRecords {
		if record.VMID == vmID {
			vmRecords = append(vmRecords, record)
		}
	}

	return vmRecords, nil
}

// DeleteMigrationRecord removes a migration record
func (s *FileMigrationStorage) DeleteMigrationRecord(migrationID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Delete the file
	filename := filepath.Join(s.storageDir, fmt.Sprintf("migration_%s.json", migrationID))
	if err := os.Remove(filename); err != nil {
		if os.IsNotExist(err) {
			return ErrMigrationNotFound
		}
		return fmt.Errorf("failed to delete migration record: %w", err)
	}

	return nil
}

// InMemoryMigrationStorage implements MigrationStorage in memory (for testing)
type InMemoryMigrationStorage struct {
	records map[string]*MigrationRecord
	mu      sync.RWMutex
}

// NewInMemoryMigrationStorage creates a new InMemoryMigrationStorage
func NewInMemoryMigrationStorage() *InMemoryMigrationStorage {
	return &InMemoryMigrationStorage{
		records: make(map[string]*MigrationRecord),
	}
}

// SaveMigrationRecord stores a migration record in memory
func (s *InMemoryMigrationStorage) SaveMigrationRecord(record *MigrationRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Make a copy of the record
	recordCopy := *record
	s.records[record.ID] = &recordCopy

	return nil
}

// LoadMigrationRecord retrieves a migration record from memory
func (s *InMemoryMigrationStorage) LoadMigrationRecord(migrationID string) (*MigrationRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	record, ok := s.records[migrationID]
	if !ok {
		return nil, ErrMigrationNotFound
	}

	// Make a copy of the record
	recordCopy := *record
	return &recordCopy, nil
}

// ListMigrationRecords retrieves all migration records from memory
func (s *InMemoryMigrationStorage) ListMigrationRecords() ([]*MigrationRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	records := make([]*MigrationRecord, 0, len(s.records))
	for _, record := range s.records {
		recordCopy := *record
		records = append(records, &recordCopy)
	}

	return records, nil
}

// ListMigrationRecordsForVM retrieves all migration records for a specific VM from memory
func (s *InMemoryMigrationStorage) ListMigrationRecordsForVM(vmID string) ([]*MigrationRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	records := make([]*MigrationRecord, 0)
	for _, record := range s.records {
		if record.VMID == vmID {
			recordCopy := *record
			records = append(records, &recordCopy)
		}
	}

	return records, nil
}

// DeleteMigrationRecord removes a migration record from memory
func (s *InMemoryMigrationStorage) DeleteMigrationRecord(migrationID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.records[migrationID]; !ok {
		return ErrMigrationNotFound
	}

	delete(s.records, migrationID)
	return nil
}
