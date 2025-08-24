package vm

import (
	// "bytes" // Currently unused
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	// "io/ioutil" // Currently unused
	"net"
	"os"
	"runtime"
	"time"

	"github.com/sirupsen/logrus"
)

// MigrationProgressTracker tracks and reports progress for VM migrations
type MigrationProgressTracker struct {
	logger           *logrus.Entry
	startTime        time.Time
	totalBytes       int64
	transferredBytes int64
	lastUpdateTime   time.Time
	lastBytes        int64
	currentRate      int64 // bytes per second
}

// NewMigrationProgressTracker creates a new progress tracker
func NewMigrationProgressTracker(logger *logrus.Entry, totalBytes int64) *MigrationProgressTracker {
	now := time.Now()
	return &MigrationProgressTracker{
		logger:         logger,
		startTime:      now,
		lastUpdateTime: now,
		totalBytes:     totalBytes,
	}
}

// Update updates the progress tracker with current progress
func (t *MigrationProgressTracker) Update(transferredBytes int64) {
	now := time.Now()
	duration := now.Sub(t.lastUpdateTime)
	
	// Only update rate if at least 1 second has passed
	if duration.Seconds() >= 1.0 {
		bytesInPeriod := transferredBytes - t.lastBytes
		t.currentRate = int64(float64(bytesInPeriod) / duration.Seconds())
		t.lastUpdateTime = now
		t.lastBytes = transferredBytes
	}
	
	t.transferredBytes = transferredBytes
}

// GetProgress returns the current progress as a float between 0 and 1
func (t *MigrationProgressTracker) GetProgress() float64 {
	if t.totalBytes <= 0 {
		return 0.0
	}
	return float64(t.transferredBytes) / float64(t.totalBytes)
}

// GetRate returns the current transfer rate in bytes per second
func (t *MigrationProgressTracker) GetRate() int64 {
	return t.currentRate
}

// GetETA returns the estimated time remaining in seconds
func (t *MigrationProgressTracker) GetETA() int64 {
	if t.currentRate <= 0 {
		return -1 // Cannot estimate
	}
	
	bytesRemaining := t.totalBytes - t.transferredBytes
	if bytesRemaining <= 0 {
		return 0
	}
	
	return bytesRemaining / t.currentRate
}

// GetElapsedTime returns the elapsed time in seconds
func (t *MigrationProgressTracker) GetElapsedTime() float64 {
	return time.Since(t.startTime).Seconds()
}

// LogProgress logs the current progress
func (t *MigrationProgressTracker) LogProgress() {
	progress := t.GetProgress() * 100
	rate := float64(t.GetRate()) / (1024 * 1024) // Convert to MB/s
	eta := t.GetETA()
	
	if eta >= 0 {
		t.logger.WithFields(logrus.Fields{
			"progress":  fmt.Sprintf("%.2f%%", progress),
			"rate":      fmt.Sprintf("%.2f MB/s", rate),
			"eta":       fmt.Sprintf("%ds", eta),
			"bytes":     t.transferredBytes,
			"totalBytes": t.totalBytes,
		}).Info("Migration progress")
	} else {
		t.logger.WithFields(logrus.Fields{
			"progress":  fmt.Sprintf("%.2f%%", progress),
			"rate":      fmt.Sprintf("%.2f MB/s", rate),
			"bytes":     t.transferredBytes,
			"totalBytes": t.totalBytes,
		}).Info("Migration progress")
	}
}

// CompressFile compresses a file using gzip at the specified compression level
func CompressFile(sourcePath, destPath string, level int) (int64, error) {
	// Open source file
	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return 0, fmt.Errorf("failed to open source file: %w", err)
	}
	defer sourceFile.Close()
	
	// Create destination file
	destFile, err := os.Create(destPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()
	
	// Create gzip writer
	gzipWriter, err := gzip.NewWriterLevel(destFile, level)
	if err != nil {
		return 0, fmt.Errorf("failed to create gzip writer: %w", err)
	}
	defer gzipWriter.Close()
	
	// Copy data from source to gzip writer
	written, err := io.Copy(gzipWriter, sourceFile)
	if err != nil {
		return 0, fmt.Errorf("failed to compress file: %w", err)
	}
	
	// Ensure all data is written
	if err := gzipWriter.Flush(); err != nil {
		return written, fmt.Errorf("failed to flush gzip writer: %w", err)
	}
	
	return written, nil
}

// DecompressFile decompresses a gzip compressed file
func DecompressFile(sourcePath, destPath string) (int64, error) {
	// Open source file
	sourceFile, err := os.Open(sourcePath)
	if err != nil {
		return 0, fmt.Errorf("failed to open source file: %w", err)
	}
	defer sourceFile.Close()
	
	// Create gzip reader
	gzipReader, err := gzip.NewReader(sourceFile)
	if err != nil {
		return 0, fmt.Errorf("failed to create gzip reader: %w", err)
	}
	defer gzipReader.Close()
	
	// Create destination file
	destFile, err := os.Create(destPath)
	if err != nil {
		return 0, fmt.Errorf("failed to create destination file: %w", err)
	}
	defer destFile.Close()
	
	// Copy data from gzip reader to destination
	written, err := io.Copy(destFile, gzipReader)
	if err != nil {
		return 0, fmt.Errorf("failed to decompress file: %w", err)
	}
	
	return written, nil
}

// CalculateFileChecksum calculates SHA-256 checksum of a file
func CalculateFileChecksum(filePath string) (string, error) {
	// Open file
	file, err := os.Open(filePath)
	if err != nil {
		return "", fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	// Create hash
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", fmt.Errorf("failed to calculate checksum: %w", err)
	}
	
	// Return hex encoded hash
	return hex.EncodeToString(hash.Sum(nil)), nil
}

// VerifyFileIntegrity verifies a file's integrity by comparing its checksum
func VerifyFileIntegrity(filePath, expectedChecksum string) (bool, error) {
	actualChecksum, err := CalculateFileChecksum(filePath)
	if err != nil {
		return false, err
	}
	
	return actualChecksum == expectedChecksum, nil
}

// RateLimitedReader wraps an io.Reader with rate limiting functionality
type RateLimitedReader struct {
	reader       io.Reader
	bytesPerSec  int64
	bytesRead    int64
	lastReadTime time.Time
}

// NewRateLimitedReader creates a new rate limited reader
func NewRateLimitedReader(reader io.Reader, bytesPerSec int64) *RateLimitedReader {
	return &RateLimitedReader{
		reader:       reader,
		bytesPerSec:  bytesPerSec,
		lastReadTime: time.Now(),
	}
}

// Read implements io.Reader with rate limiting
func (r *RateLimitedReader) Read(p []byte) (int, error) {
	if r.bytesPerSec <= 0 {
		// No rate limiting
		return r.reader.Read(p)
	}
	
	now := time.Now()
	elapsed := now.Sub(r.lastReadTime).Seconds()
	
	// Reset counter if more than a second has passed
	if elapsed >= 1.0 {
		r.bytesRead = 0
		r.lastReadTime = now
	}
	
	// Check if we've exceeded the rate limit
	if r.bytesRead >= r.bytesPerSec {
		// Sleep until the next second starts
		time.Sleep(time.Until(r.lastReadTime.Add(time.Second)))
		r.bytesRead = 0
		r.lastReadTime = time.Now()
	}
	
	// Limit read size to not exceed the rate limit
	maxBytes := r.bytesPerSec - r.bytesRead
	if int64(len(p)) > maxBytes {
		p = p[:maxBytes]
	}
	
	// Read data
	n, err := r.reader.Read(p)
	r.bytesRead += int64(n)
	
	return n, err
}

// CheckSystemResources checks if the system has enough resources for migration
func CheckSystemResources(requiredMemoryMB int) (bool, map[string]interface{}) {
	resources := make(map[string]interface{})
	
	// Get available memory
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	
	// Convert to MB
	availableMemoryMB := int64(m.Sys) / (1024 * 1024)
	resources["available_memory_mb"] = availableMemoryMB
	resources["required_memory_mb"] = requiredMemoryMB
	
	// Check if we have enough memory
	hasEnoughMemory := availableMemoryMB >= int64(requiredMemoryMB)
	resources["has_enough_memory"] = hasEnoughMemory
	
	// Get CPU count
	cpuCount := runtime.NumCPU()
	resources["cpu_count"] = cpuCount
	
	return hasEnoughMemory, resources
}

// CheckNetworkConnectivity checks if there's network connectivity between two nodes
func CheckNetworkConnectivity(address string, port int, timeoutSec int) (bool, error) {
	conn, err := net.DialTimeout("tcp", fmt.Sprintf("%s:%d", address, port), time.Duration(timeoutSec)*time.Second)
	if err != nil {
		return false, err
	}
	defer conn.Close()
	
	return true, nil
}

// CalculateBandwidthRequirements calculates the required bandwidth for a migration
func CalculateBandwidthRequirements(totalBytes int64, maxDowntimeSeconds int) int64 {
	if maxDowntimeSeconds <= 0 {
		return 0 // Unlimited
	}
	
	// Required bandwidth in bytes per second
	return totalBytes / int64(maxDowntimeSeconds)
}

// EstimateMigrationTime estimates the time required for migration
func EstimateMigrationTime(totalBytes int64, bandwidthBytesPerSec int64) time.Duration {
	if bandwidthBytesPerSec <= 0 {
		return time.Duration(0) // Cannot estimate
	}
	
	seconds := float64(totalBytes) / float64(bandwidthBytesPerSec)
	return time.Duration(seconds * float64(time.Second))
}

// EstimateMemoryDirtyRate simulates memory dirty rate calculation for pre-copy migrations
func EstimateMemoryDirtyRate(vm *VM) (int64, error) {
	// This is a placeholder implementation that would be replaced with actual 
	// measurements of memory write patterns in a real system.
	// 
	// In a real implementation, this would:
	// 1. Take memory page fault counts over time
	// 2. Analyze VM's memory write patterns
	// 3. Calculate the rate at which memory is modified
	
	// Placeholder: assume a random dirty rate between 10-50 MB/sec
	dirtyRateMBPerSec := 10 + (time.Now().UnixNano() % 40)
	return dirtyRateMBPerSec * 1024 * 1024, nil // Convert to bytes per second
}

// CalculateOptimalIterations calculates the optimal number of pre-copy iterations
func CalculateOptimalIterations(memorySizeMB int, dirtyRateMBPerSec int64, bandwidthMBPerSec int64, maxDowntimeMS int) int {
	if bandwidthMBPerSec <= dirtyRateMBPerSec {
		// Bandwidth cannot keep up with dirty rate
		return 1 // Minimal iterations
	}
	
	// Calculate transfer time for each iteration
	transferTimeMS := (int64(memorySizeMB) * 1024 * 1024 * 1000) / (bandwidthMBPerSec * 1024 * 1024)
	
	// Calculate dirty pages accumulated during transfer
	dirtyMB := (dirtyRateMBPerSec * transferTimeMS) / 1000
	
	// Calculate iterations needed to get dirty pages below max downtime threshold
	iterations := 1
	for dirtyMB*1024*1024*1000/(bandwidthMBPerSec*1024*1024) > int64(maxDowntimeMS) && iterations < 30 {
		iterations++
		dirtyMB = (dirtyRateMBPerSec * dirtyMB * 1024 * 1024 * 1000) / (bandwidthMBPerSec * 1024 * 1024 * 1000)
	}
	
	return iterations
}
