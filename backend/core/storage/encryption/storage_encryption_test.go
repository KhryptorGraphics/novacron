package encryption

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"testing"
)

func TestEncryption(t *testing.T) {
	// Generate a random master key
	masterKey, err := GenerateRandomKey(32)
	if err != nil {
		t.Fatalf("Failed to generate master key: %v", err)
	}

	// Create test data of different types and sizes
	testCases := []struct {
		name        string
		data        []byte
		volumeID    string
		algorithm   EncryptionAlgorithm
		mode        EncryptionMode
		description string
	}{
		{
			name:        "Small Text AES-256 GCM",
			data:        []byte("This is a small piece of text that will be encrypted."),
			volumeID:    "vol-1",
			algorithm:   EncryptionAES256,
			mode:        EncryptionModeGCM,
			description: "Small text data with AES-256 GCM",
		},
		{
			name:        "Medium Text AES-192 CBC",
			data:        bytes.Repeat([]byte("Medium sized text for encryption testing. "), 10),
			volumeID:    "vol-2",
			algorithm:   EncryptionAES192,
			mode:        EncryptionModeCBC,
			description: "Medium text data with AES-192 CBC",
		},
		{
			name:        "Large Text AES-128 CTR",
			data:        bytes.Repeat([]byte("This is a larger piece of text that will be used to test encryption with CTR mode. "), 100),
			volumeID:    "vol-3",
			algorithm:   EncryptionAES128,
			mode:        EncryptionModeCTR,
			description: "Large text data with AES-128 CTR",
		},
		{
			name:        "Binary Data AES-256 GCM",
			data:        generateRandomData(1024),
			volumeID:    "vol-4",
			algorithm:   EncryptionAES256,
			mode:        EncryptionModeGCM,
			description: "Random binary data with AES-256 GCM",
		},
		{
			name:        "Very Small Data AES-256 GCM",
			data:        []byte("abc"),
			volumeID:    "vol-5",
			algorithm:   EncryptionAES256,
			mode:        EncryptionModeGCM,
			description: "Very small data with AES-256 GCM",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create config for this test case
			config := EncryptionConfig{
				Algorithm:           tc.algorithm,
				Mode:                tc.mode,
				MasterKey:           masterKey,
				SaltPrefix:          "test",
				Authenticate:        true,
				MinSizeBytes:        1,
				IncludeVerification: true,
				StoreMetadata:       true,
			}

			// Create encryptor
			encryptor, err := NewEncryptor(config)
			if err != nil {
				t.Fatalf("Failed to create encryptor: %v", err)
			}

			// Encrypt the data
			encData, err := encryptor.Encrypt(tc.data, tc.volumeID)
			if err != nil {
				t.Fatalf("Encryption failed: %v", err)
			}

			// Verify encryption metadata
			if encData.Algorithm != tc.algorithm {
				t.Errorf("Expected algorithm %s, got %s", tc.algorithm, encData.Algorithm)
			}

			if encData.Mode != tc.mode {
				t.Errorf("Expected mode %s, got %s", tc.mode, encData.Mode)
			}

			if encData.OriginalSize != len(tc.data) {
				t.Errorf("Expected original size %d, got %d", len(tc.data), encData.OriginalSize)
			}

			// Verify data was actually encrypted (should be different from original)
			if bytes.Equal(encData.Data, tc.data) {
				t.Errorf("Encrypted data is identical to original data")
			}

			// Verify IV was generated
			if len(encData.IV) == 0 {
				t.Errorf("No initialization vector was generated")
			}

			// Decrypt the data
			decrypted, err := encryptor.Decrypt(encData, tc.volumeID)
			if err != nil {
				t.Fatalf("Decryption failed: %v", err)
			}

			// Verify decrypted data matches original
			if !bytes.Equal(decrypted, tc.data) {
				t.Errorf("Decrypted data doesn't match original")
				t.Logf("Original (first 50 bytes): %s", tc.data[:min(50, len(tc.data))])
				t.Logf("Decrypted (first 50 bytes): %s", decrypted[:min(50, len(decrypted))])
			}
		})
	}
}

func TestKeyVerification(t *testing.T) {
	// Generate a legitimate master key
	masterKey, err := GenerateRandomKey(32)
	if err != nil {
		t.Fatalf("Failed to generate master key: %v", err)
	}

	// Generate an incorrect master key
	wrongKey, err := GenerateRandomKey(32)
	if err != nil {
		t.Fatalf("Failed to generate wrong key: %v", err)
	}

	// Create test data
	data := []byte("This data will be encrypted with verification enabled")
	volumeID := "vol-verify"

	// Create config with verification enabled
	config := EncryptionConfig{
		Algorithm:           EncryptionAES256,
		Mode:                EncryptionModeGCM,
		MasterKey:           masterKey,
		SaltPrefix:          "test",
		Authenticate:        true,
		MinSizeBytes:        1,
		IncludeVerification: true,
		StoreMetadata:       true,
	}

	// Create encryptor with the correct key
	encryptor, err := NewEncryptor(config)
	if err != nil {
		t.Fatalf("Failed to create encryptor: %v", err)
	}

	// Encrypt the data
	encData, err := encryptor.Encrypt(data, volumeID)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Try to decrypt with the wrong key
	wrongConfig := config
	wrongConfig.MasterKey = wrongKey
	wrongEncryptor, err := NewEncryptor(wrongConfig)
	if err != nil {
		t.Fatalf("Failed to create wrong encryptor: %v", err)
	}

	// Decryption should fail because of verification
	_, err = wrongEncryptor.Decrypt(encData, volumeID)
	if err == nil {
		t.Errorf("Decryption with wrong key succeeded, but should have failed")
	} else {
		t.Logf("Correctly failed to decrypt with wrong key: %v", err)
	}

	// Now try with the correct key again
	decrypted, err := encryptor.Decrypt(encData, volumeID)
	if err != nil {
		t.Fatalf("Decryption with correct key failed: %v", err)
	}

	// Verify decrypted data matches original
	if !bytes.Equal(decrypted, data) {
		t.Errorf("Decrypted data doesn't match original")
	}
}

func TestDifferentVolumesHaveDifferentKeys(t *testing.T) {
	// Generate a master key
	masterKey, err := GenerateRandomKey(32)
	if err != nil {
		t.Fatalf("Failed to generate master key: %v", err)
	}

	// Create config
	config := EncryptionConfig{
		Algorithm:           EncryptionAES256,
		Mode:                EncryptionModeGCM,
		MasterKey:           masterKey,
		SaltPrefix:          "test",
		Authenticate:        true,
		MinSizeBytes:        1,
		IncludeVerification: true,
		StoreMetadata:       true,
	}

	// Create encryptor
	encryptor, err := NewEncryptor(config)
	if err != nil {
		t.Fatalf("Failed to create encryptor: %v", err)
	}

	// Generate keys for different volumes
	vol1Key, err := encryptor.GenerateKey("vol-1")
	if err != nil {
		t.Fatalf("Failed to generate key for vol-1: %v", err)
	}

	vol2Key, err := encryptor.GenerateKey("vol-2")
	if err != nil {
		t.Fatalf("Failed to generate key for vol-2: %v", err)
	}

	// Keys should be different
	if bytes.Equal(vol1Key, vol2Key) {
		t.Errorf("Keys for different volumes are identical, but should be different")
		t.Logf("vol-1 key: %s", hex.EncodeToString(vol1Key))
		t.Logf("vol-2 key: %s", hex.EncodeToString(vol2Key))
	} else {
		t.Logf("Correctly generated different keys for different volumes")
	}

	// Now try encrypting the same data with different volume IDs
	data := []byte("This is the same data")

	enc1, err := encryptor.Encrypt(data, "vol-1")
	if err != nil {
		t.Fatalf("Failed to encrypt for vol-1: %v", err)
	}

	enc2, err := encryptor.Encrypt(data, "vol-2")
	if err != nil {
		t.Fatalf("Failed to encrypt for vol-2: %v", err)
	}

	// Ciphertexts should be different
	if bytes.Equal(enc1.Data, enc2.Data) {
		t.Errorf("Ciphertexts for different volumes are identical, but should be different")
	}

	// Try decrypting with wrong volume ID
	_, err = encryptor.Decrypt(enc1, "vol-2")
	if err == nil {
		t.Errorf("Decryption with wrong volume ID succeeded, but should have failed")
	}

	_, err = encryptor.Decrypt(enc2, "vol-1")
	if err == nil {
		t.Errorf("Decryption with wrong volume ID succeeded, but should have failed")
	}
}

func TestEncryptionModes(t *testing.T) {
	// Generate a master key
	masterKey, err := GenerateRandomKey(32)
	if err != nil {
		t.Fatalf("Failed to generate master key: %v", err)
	}

	// Test data
	data := []byte("This is test data for different encryption modes")
	volumeID := "vol-modes"

	// Test each mode
	modes := []EncryptionMode{
		EncryptionModeGCM,
		EncryptionModeCBC,
		EncryptionModeCTR,
	}

	for _, mode := range modes {
		t.Run(string(mode), func(t *testing.T) {
			// Create config
			config := EncryptionConfig{
				Algorithm:           EncryptionAES256,
				Mode:                mode,
				MasterKey:           masterKey,
				SaltPrefix:          "test",
				Authenticate:        true,
				MinSizeBytes:        1,
				IncludeVerification: true,
				StoreMetadata:       true,
			}

			// Create encryptor
			encryptor, err := NewEncryptor(config)
			if err != nil {
				t.Fatalf("Failed to create encryptor: %v", err)
			}

			// Encrypt and decrypt
			encData, err := encryptor.Encrypt(data, volumeID)
			if err != nil {
				t.Fatalf("Encryption failed: %v", err)
			}

			if encData.Mode != mode {
				t.Errorf("Expected mode %s, got %s", mode, encData.Mode)
			}

			decrypted, err := encryptor.Decrypt(encData, volumeID)
			if err != nil {
				t.Fatalf("Decryption failed: %v", err)
			}

			// Verify decrypted data matches original
			if !bytes.Equal(decrypted, data) {
				t.Errorf("Decrypted data doesn't match original")
			}
		})
	}
}

func TestNoEncryption(t *testing.T) {
	// Create config with no encryption
	config := EncryptionConfig{
		Algorithm:           EncryptionNone,
		Mode:                EncryptionModeGCM, // Should be ignored
		MasterKey:           "",                // No key needed
		SaltPrefix:          "test",
		Authenticate:        true,
		MinSizeBytes:        1,
		IncludeVerification: true,
		StoreMetadata:       true,
	}

	// Create encryptor
	encryptor, err := NewEncryptor(config)
	if err != nil {
		t.Fatalf("Failed to create encryptor: %v", err)
	}

	// Test data
	data := []byte("This data should not be encrypted")
	volumeID := "vol-none"

	// "Encrypt" the data
	encData, err := encryptor.Encrypt(data, volumeID)
	if err != nil {
		t.Fatalf("Encryption failed: %v", err)
	}

	// Verify "encrypted" data is the same as original
	if !bytes.Equal(encData.Data, data) {
		t.Errorf("Data was modified despite EncryptionNone")
	}

	if encData.Algorithm != EncryptionNone {
		t.Errorf("Expected algorithm %s, got %s", EncryptionNone, encData.Algorithm)
	}

	// "Decrypt" the data
	decrypted, err := encryptor.Decrypt(encData, volumeID)
	if err != nil {
		t.Fatalf("Decryption failed: %v", err)
	}

	// Verify decrypted data matches original
	if !bytes.Equal(decrypted, data) {
		t.Errorf("Decrypted data doesn't match original")
	}
}

func TestMinimumSize(t *testing.T) {
	// Generate a master key
	masterKey, err := GenerateRandomKey(32)
	if err != nil {
		t.Fatalf("Failed to generate master key: %v", err)
	}

	// Create config with minimum size of 100 bytes
	config := EncryptionConfig{
		Algorithm:           EncryptionAES256,
		Mode:                EncryptionModeGCM,
		MasterKey:           masterKey,
		SaltPrefix:          "test",
		Authenticate:        true,
		MinSizeBytes:        100, // Only encrypt data >= 100 bytes
		IncludeVerification: true,
		StoreMetadata:       true,
	}

	// Create encryptor
	encryptor, err := NewEncryptor(config)
	if err != nil {
		t.Fatalf("Failed to create encryptor: %v", err)
	}

	// Test small data (less than minimum size)
	smallData := []byte("This data is less than 100 bytes")
	smallVolumeID := "vol-small"

	// Encrypt small data
	smallEncData, err := encryptor.Encrypt(smallData, smallVolumeID)
	if err != nil {
		t.Fatalf("Encryption of small data failed: %v", err)
	}

	// Small data should not be encrypted
	if !bytes.Equal(smallEncData.Data, smallData) {
		t.Errorf("Small data was encrypted despite being below minimum size")
	}

	if smallEncData.Algorithm != EncryptionNone {
		t.Errorf("Expected algorithm %s for small data, got %s", EncryptionNone, smallEncData.Algorithm)
	}

	// Test large data (more than minimum size)
	largeData := bytes.Repeat([]byte("This data is more than 100 bytes. "), 5)
	largeVolumeID := "vol-large"

	// Encrypt large data
	largeEncData, err := encryptor.Encrypt(largeData, largeVolumeID)
	if err != nil {
		t.Fatalf("Encryption of large data failed: %v", err)
	}

	// Large data should be encrypted
	if bytes.Equal(largeEncData.Data, largeData) {
		t.Errorf("Large data was not encrypted despite being above minimum size")
	}

	if largeEncData.Algorithm != EncryptionAES256 {
		t.Errorf("Expected algorithm %s for large data, got %s", EncryptionAES256, largeEncData.Algorithm)
	}

	// Decrypt both data sets
	smallDecrypted, err := encryptor.Decrypt(smallEncData, smallVolumeID)
	if err != nil {
		t.Fatalf("Decryption of small data failed: %v", err)
	}

	largeDecrypted, err := encryptor.Decrypt(largeEncData, largeVolumeID)
	if err != nil {
		t.Fatalf("Decryption of large data failed: %v", err)
	}

	// Verify decrypted data matches original
	if !bytes.Equal(smallDecrypted, smallData) {
		t.Errorf("Decrypted small data doesn't match original")
	}

	if !bytes.Equal(largeDecrypted, largeData) {
		t.Errorf("Decrypted large data doesn't match original")
	}
}

// Helper function to generate random binary data
func generateRandomData(size int) []byte {
	data := make([]byte, size)
	rand.Read(data)
	return data
}

// Min function
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
