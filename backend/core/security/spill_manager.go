package security

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// SpillManager manages spilling events to disk when queues are full
type SpillManager struct {
	spillDir    string
	maxFiles    int
	mu          sync.RWMutex
	fileCounter int64
}

// NewSpillManager creates a new spill manager
func NewSpillManager(spillDir string, maxFiles int) *SpillManager {
	if err := os.MkdirAll(spillDir, 0755); err != nil {
		// If we can't create the directory, disable spilling
		return nil
	}

	return &SpillManager{
		spillDir: spillDir,
		maxFiles: maxFiles,
	}
}

// SpillEvent writes an event to disk
func (sm *SpillManager) SpillEvent(event *PrioritySecurityEvent) error {
	if sm == nil {
		return fmt.Errorf("spill manager not available")
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	// Create unique filename
	filename := fmt.Sprintf("spill_%d_%s_%d.json",
		time.Now().Unix(),
		event.Priority.String(),
		sm.fileCounter)
	sm.fileCounter++

	filepath := filepath.Join(sm.spillDir, filename)

	// Marshal event to JSON
	data, err := json.Marshal(event)
	if err != nil {
		return fmt.Errorf("failed to marshal event: %w", err)
	}

	// Write to file
	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write spill file: %w", err)
	}

	// Clean up old files if we exceed the limit
	sm.cleanupOldFiles()

	return nil
}

// RecoverSpilledEvents recovers spilled events and tries to re-enqueue them
func (sm *SpillManager) RecoverSpilledEvents(pq *PriorityQueue) error {
	if sm == nil {
		return nil
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	files, err := os.ReadDir(sm.spillDir)
	if err != nil {
		return fmt.Errorf("failed to read spill directory: %w", err)
	}

	// Sort files by modification time (oldest first)
	sort.Slice(files, func(i, j int) bool {
		infoI, _ := files[i].Info()
		infoJ, _ := files[j].Info()
		return infoI.ModTime().Before(infoJ.ModTime())
	})

	recovered := 0
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			if sm.recoverFile(filepath.Join(sm.spillDir, file.Name()), pq) {
				recovered++
			}

			// Don't overwhelm the system - recover a few at a time
			if recovered >= 10 {
				break
			}
		}
	}

	if recovered > 0 {
		fmt.Printf("Recovered %d spilled events\n", recovered)
	}

	return nil
}

// recoverFile recovers a single spilled event file
func (sm *SpillManager) recoverFile(filePath string, pq *PriorityQueue) bool {
	data, err := os.ReadFile(filePath)
	if err != nil {
		fmt.Printf("Failed to read spill file %s: %v\n", filePath, err)
		return false
	}

	var event PrioritySecurityEvent
	if err := json.Unmarshal(data, &event); err != nil {
		fmt.Printf("Failed to unmarshal spill file %s: %v\n", filePath, err)
		// Remove corrupted file
		os.Remove(filePath)
		return false
	}

	// Try to re-enqueue
	if err := pq.Enqueue(&event); err != nil {
		// Queue still full, leave file for later
		return false
	}

	// Successfully re-enqueued, remove the file
	if err := os.Remove(filePath); err != nil {
		fmt.Printf("Failed to remove spill file %s: %v\n", filePath, err)
	}

	return true
}

// cleanupOldFiles removes old spill files if we exceed the limit
func (sm *SpillManager) cleanupOldFiles() {
	files, err := os.ReadDir(sm.spillDir)
	if err != nil {
		return
	}

	if len(files) <= sm.maxFiles {
		return
	}

	// Get file info with modification times
	type fileInfo struct {
		name    string
		modTime time.Time
	}

	var fileInfos []fileInfo
	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			info, err := file.Info()
			if err != nil {
				continue
			}
			fileInfos = append(fileInfos, fileInfo{
				name:    file.Name(),
				modTime: info.ModTime(),
			})
		}
	}

	// Sort by modification time (oldest first)
	sort.Slice(fileInfos, func(i, j int) bool {
		return fileInfos[i].modTime.Before(fileInfos[j].modTime)
	})

	// Remove oldest files
	filesToRemove := len(fileInfos) - sm.maxFiles
	for i := 0; i < filesToRemove; i++ {
		filePath := filepath.Join(sm.spillDir, fileInfos[i].name)
		os.Remove(filePath)
	}
}

// Close cleans up the spill manager
func (sm *SpillManager) Close() error {
	// Optionally clean up all spill files on shutdown
	// For now, we'll keep them for recovery on restart
	return nil
}

// GetSpillStats returns statistics about spilled events
func (sm *SpillManager) GetSpillStats() map[string]interface{} {
	if sm == nil {
		return map[string]interface{}{
			"enabled":     false,
			"spill_count": 0,
		}
	}

	sm.mu.RLock()
	defer sm.mu.RUnlock()

	files, err := os.ReadDir(sm.spillDir)
	if err != nil {
		return map[string]interface{}{
			"enabled": true,
			"error":   err.Error(),
		}
	}

	spillCount := 0
	totalSize := int64(0)

	for _, file := range files {
		if !file.IsDir() && filepath.Ext(file.Name()) == ".json" {
			spillCount++
			if info, err := file.Info(); err == nil {
				totalSize += info.Size()
			}
		}
	}

	return map[string]interface{}{
		"enabled":         true,
		"spill_directory": sm.spillDir,
		"spill_count":     spillCount,
		"total_size":      totalSize,
		"max_files":       sm.maxFiles,
	}
}