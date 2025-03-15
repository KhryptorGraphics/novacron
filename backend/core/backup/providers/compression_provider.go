package providers

import (
	"bytes"
	"compress/gzip"
	"context"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
)

// CompressionLevel defines the compression level
type CompressionLevel int

const (
	// DefaultCompression is the default compression level
	DefaultCompression CompressionLevel = 0
	// NoCompression indicates no compression
	NoCompression CompressionLevel = 1
	// BestSpeed indicates the fastest compression
	BestSpeed CompressionLevel = 2
	// BestCompression indicates the highest compression
	BestCompression CompressionLevel = 3
)

// CompressedStorageProvider wraps an existing provider and adds compression
type CompressedStorageProvider struct {
	// baseProvider is the underlying provider to wrap
	baseProvider backup.BackupProvider

	// compressionLevel is the level of compression to use
	compressionLevel CompressionLevel
}

// NewCompressedStorageProvider creates a new compressed storage provider
func NewCompressedStorageProvider(baseProvider backup.BackupProvider, level CompressionLevel) (*CompressedStorageProvider, error) {
	return &CompressedStorageProvider{
		baseProvider:     baseProvider,
		compressionLevel: level,
	}, nil
}

// ID returns the provider ID
func (p *CompressedStorageProvider) ID() string {
	return fmt.Sprintf("compressed-%s", p.baseProvider.ID())
}

// Name returns the provider name
func (p *CompressedStorageProvider) Name() string {
	return fmt.Sprintf("Compressed %s", p.baseProvider.Name())
}

// Type returns the type of storage this provider supports
func (p *CompressedStorageProvider) Type() backup.StorageType {
	return p.baseProvider.Type()
}

// CreateBackup creates a compressed backup
func (p *CompressedStorageProvider) CreateBackup(ctx context.Context, job *backup.BackupJob) (*backup.Backup, error) {
	// Create the backup with the base provider
	b, err := p.baseProvider.CreateBackup(ctx, job)
	if err != nil {
		return nil, err
	}

	// Add compression metadata
	if b.Metadata == nil {
		b.Metadata = make(map[string]string)
	}
	b.Metadata["compressed"] = "true"
	b.Metadata["compression_algorithm"] = "gzip"
	b.Metadata["compression_level"] = p.getCompressionLevelString()

	return b, nil
}

// getCompressionLevelString returns a string representation of the compression level
func (p *CompressedStorageProvider) getCompressionLevelString() string {
	switch p.compressionLevel {
	case DefaultCompression:
		return "default"
	case NoCompression:
		return "none"
	case BestSpeed:
		return "best_speed"
	case BestCompression:
		return "best_compression"
	default:
		return "unknown"
	}
}

// DeleteBackup deletes a backup
func (p *CompressedStorageProvider) DeleteBackup(ctx context.Context, backupID string) error {
	return p.baseProvider.DeleteBackup(ctx, backupID)
}

// RestoreBackup restores a backup
func (p *CompressedStorageProvider) RestoreBackup(ctx context.Context, job *backup.RestoreJob) error {
	return p.baseProvider.RestoreBackup(ctx, job)
}

// ListBackups lists backups
func (p *CompressedStorageProvider) ListBackups(ctx context.Context, filter map[string]interface{}) ([]*backup.Backup, error) {
	return p.baseProvider.ListBackups(ctx, filter)
}

// GetBackup gets a backup by ID
func (p *CompressedStorageProvider) GetBackup(ctx context.Context, backupID string) (*backup.Backup, error) {
	return p.baseProvider.GetBackup(ctx, backupID)
}

// ValidateBackup validates a backup
func (p *CompressedStorageProvider) ValidateBackup(ctx context.Context, backupID string) error {
	return p.baseProvider.ValidateBackup(ctx, backupID)
}

// CompressedBackupSession represents an active compressed backup operation
type CompressedBackupSession struct {
	// baseSession is the underlying session
	baseSession *BackupSession

	// provider is the compressed provider
	provider *CompressedStorageProvider
}

// NewCompressedBackupSession creates a new compressed backup session
func (p *CompressedStorageProvider) NewCompressedBackupSession(b *backup.Backup, targetID string) (*CompressedBackupSession, error) {
	// Get the base provider
	baseProvider, ok := p.baseProvider.(*LocalStorageProvider)
	if !ok {
		return nil, fmt.Errorf("base provider must be LocalStorageProvider")
	}

	// Create a base session
	baseSession, err := baseProvider.NewBackupSession(b, targetID)
	if err != nil {
		return nil, err
	}

	return &CompressedBackupSession{
		baseSession: baseSession,
		provider:    p,
	}, nil
}

// WriteFile writes a compressed file to the backup
func (s *CompressedBackupSession) WriteFile(filename string, data []byte) error {
	// Compress the data
	compressedData, err := s.compressData(data)
	if err != nil {
		return fmt.Errorf("failed to compress data: %w", err)
	}

	// Write the compressed data
	return s.baseSession.WriteFile(filename, compressedData)
}

// CopyFile copies and compresses a file from a source path to the backup
func (s *CompressedBackupSession) CopyFile(sourcePath, destFilename string) error {
	// Read the source file
	data, err := ioutil.ReadFile(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to read source file: %w", err)
	}

	// Write the compressed file
	return s.WriteFile(destFilename, data)
}

// Complete marks the backup session as complete
func (s *CompressedBackupSession) Complete() error {
	return s.baseSession.Complete()
}

// compressData compresses data using gzip
func (s *CompressedBackupSession) compressData(data []byte) ([]byte, error) {
	// Convert compression level to gzip level
	var level int
	switch s.provider.compressionLevel {
	case DefaultCompression:
		level = gzip.DefaultCompression
	case NoCompression:
		level = gzip.NoCompression
	case BestSpeed:
		level = gzip.BestSpeed
	case BestCompression:
		level = gzip.BestCompression
	default:
		level = gzip.DefaultCompression
	}

	// Create a buffer to write the compressed data to
	var buf bytes.Buffer

	// Create a gzip writer
	gzipWriter, err := gzip.NewWriterLevel(&buf, level)
	if err != nil {
		return nil, err
	}

	// Write the data to the gzip writer
	if _, err := gzipWriter.Write(data); err != nil {
		gzipWriter.Close()
		return nil, err
	}

	// Close the gzip writer to flush the data
	if err := gzipWriter.Close(); err != nil {
		return nil, err
	}

	// Return the compressed data
	return buf.Bytes(), nil
}

// decompressData decompresses data using gzip
func (s *CompressedBackupSession) decompressData(compressedData []byte) ([]byte, error) {
	// Create a reader for the compressed data
	gzipReader, err := gzip.NewReader(bytes.NewReader(compressedData))
	if err != nil {
		return nil, err
	}
	defer gzipReader.Close()

	// Read the decompressed data
	decompressedData, err := io.ReadAll(gzipReader)
	if err != nil {
		return nil, err
	}

	return decompressedData, nil
}
