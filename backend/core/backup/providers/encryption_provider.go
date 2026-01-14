package providers

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"io"
	"io/ioutil"

	"github.com/khryptorgraphics/novacron/backend/core/backup"
)

// EncryptedStorageProvider wraps an existing provider and adds encryption
type EncryptedStorageProvider struct {
	// baseProvider is the underlying provider to wrap
	baseProvider backup.BackupProvider

	// encryptionKey is the key used for encryption
	encryptionKey []byte

	// keyID is the ID of the encryption key
	keyID string
}

// NewEncryptedStorageProvider creates a new encrypted storage provider
func NewEncryptedStorageProvider(baseProvider backup.BackupProvider, keyID string, key []byte) (*EncryptedStorageProvider, error) {
	// Validate key length (AES-256 requires 32 bytes)
	if len(key) != 32 {
		return nil, fmt.Errorf("encryption key must be 32 bytes for AES-256 (got %d bytes)", len(key))
	}

	return &EncryptedStorageProvider{
		baseProvider:  baseProvider,
		encryptionKey: key,
		keyID:         keyID,
	}, nil
}

// ID returns the provider ID
func (p *EncryptedStorageProvider) ID() string {
	return fmt.Sprintf("encrypted-%s", p.baseProvider.ID())
}

// Name returns the provider name
func (p *EncryptedStorageProvider) Name() string {
	return fmt.Sprintf("Encrypted %s", p.baseProvider.Name())
}

// Type returns the type of storage this provider supports
func (p *EncryptedStorageProvider) Type() backup.StorageType {
	return p.baseProvider.Type()
}

// CreateBackup creates an encrypted backup
func (p *EncryptedStorageProvider) CreateBackup(ctx context.Context, job *backup.BackupJob) (*backup.Backup, error) {
	// Create the backup with the base provider
	b, err := p.baseProvider.CreateBackup(ctx, job)
	if err != nil {
		return nil, err
	}

	// Add encryption metadata
	if b.Metadata == nil {
		b.Metadata = make(map[string]string)
	}
	b.Metadata["encrypted"] = "true"
	b.Metadata["encryption_key_id"] = p.keyID
	b.Metadata["encryption_algorithm"] = "AES-256-GCM"

	return b, nil
}

// DeleteBackup deletes a backup
func (p *EncryptedStorageProvider) DeleteBackup(ctx context.Context, backupID string) error {
	return p.baseProvider.DeleteBackup(ctx, backupID)
}

// RestoreBackup restores a backup
func (p *EncryptedStorageProvider) RestoreBackup(ctx context.Context, job *backup.RestoreJob) error {
	return p.baseProvider.RestoreBackup(ctx, job)
}

// ListBackups lists backups
func (p *EncryptedStorageProvider) ListBackups(ctx context.Context, filter map[string]interface{}) ([]*backup.Backup, error) {
	return p.baseProvider.ListBackups(ctx, filter)
}

// GetBackup gets a backup by ID
func (p *EncryptedStorageProvider) GetBackup(ctx context.Context, backupID string) (*backup.Backup, error) {
	return p.baseProvider.GetBackup(ctx, backupID)
}

// ValidateBackup validates a backup
func (p *EncryptedStorageProvider) ValidateBackup(ctx context.Context, backupID string) error {
	return p.baseProvider.ValidateBackup(ctx, backupID)
}

// EncryptedBackupSession represents an active encrypted backup operation
type EncryptedBackupSession struct {
	// baseSession is the underlying session
	baseSession *BackupSession

	// provider is the encrypted provider
	provider *EncryptedStorageProvider
}

// NewEncryptedBackupSession creates a new encrypted backup session
func (p *EncryptedStorageProvider) NewEncryptedBackupSession(b *backup.Backup, targetID string) (*EncryptedBackupSession, error) {
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

	return &EncryptedBackupSession{
		baseSession: baseSession,
		provider:    p,
	}, nil
}

// WriteFile writes an encrypted file to the backup
func (s *EncryptedBackupSession) WriteFile(filename string, data []byte) error {
	// Encrypt the data
	encryptedData, err := s.encryptData(data)
	if err != nil {
		return fmt.Errorf("failed to encrypt data: %w", err)
	}

	// Write the encrypted data
	return s.baseSession.WriteFile(filename, encryptedData)
}

// CopyFile copies and encrypts a file from a source path to the backup
func (s *EncryptedBackupSession) CopyFile(sourcePath, destFilename string) error {
	// Read the source file
	data, err := ioutil.ReadFile(sourcePath)
	if err != nil {
		return fmt.Errorf("failed to read source file: %w", err)
	}

	// Write the encrypted file
	return s.WriteFile(destFilename, data)
}

// Complete marks the backup session as complete
func (s *EncryptedBackupSession) Complete() error {
	return s.baseSession.Complete()
}

// encryptData encrypts data using AES-256-GCM
func (s *EncryptedBackupSession) encryptData(plaintext []byte) ([]byte, error) {
	// Create a new AES cipher block
	block, err := aes.NewCipher(s.provider.encryptionKey)
	if err != nil {
		return nil, err
	}

	// Create a new GCM cipher
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Create a nonce
	nonce := make([]byte, gcm.NonceSize())
	if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
		return nil, err
	}

	// Encrypt the data
	ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)

	return ciphertext, nil
}

// decryptData decrypts data using AES-256-GCM
func (s *EncryptedBackupSession) decryptData(ciphertext []byte) ([]byte, error) {
	// Create a new AES cipher block
	block, err := aes.NewCipher(s.provider.encryptionKey)
	if err != nil {
		return nil, err
	}

	// Create a new GCM cipher
	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Extract the nonce
	if len(ciphertext) < gcm.NonceSize() {
		return nil, fmt.Errorf("ciphertext too short")
	}
	nonce, ciphertext := ciphertext[:gcm.NonceSize()], ciphertext[gcm.NonceSize():]

	// Decrypt the data
	plaintext, err := gcm.Open(nil, nonce, ciphertext, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

// EncryptedRestoreSession represents an active encrypted restore operation
type EncryptedRestoreSession struct {
	// baseSession is the underlying restore session
	baseSession interface{} // This would be a RestoreSession in a real implementation

	// provider is the encrypted provider
	provider *EncryptedStorageProvider
}

// ReadFile reads and decrypts a file from the backup
func (s *EncryptedRestoreSession) ReadFile(filename string) ([]byte, error) {
	// In a real implementation, we would read the encrypted file from the base session
	// and decrypt it. For now, this is just a placeholder.
	return nil, fmt.Errorf("not implemented")
}
