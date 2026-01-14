package plugins

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// MarketplaceConfig contains configuration for the plugin marketplace
type MarketplaceConfig struct {
	// URLs of marketplace repositories
	RepositoryURLs []string

	// Local directory for marketplace metadata
	CacheDirectory string

	// Cache expiration in hours
	CacheExpirationHours int

	// Require signature verification for plugins
	RequireSignatureVerification bool

	// Signature verification public key
	SignaturePublicKey string

	// Enable plugin auto-update
	EnableAutoUpdate bool

	// Auto-update interval in hours
	AutoUpdateIntervalHours int
}

// DefaultMarketplaceConfig returns a default marketplace configuration
func DefaultMarketplaceConfig() MarketplaceConfig {
	return MarketplaceConfig{
		RepositoryURLs:               []string{"https://marketplace.novacron.io/plugins"},
		CacheDirectory:               "./marketplace-cache",
		CacheExpirationHours:         24,
		RequireSignatureVerification: true,
		EnableAutoUpdate:             false,
		AutoUpdateIntervalHours:      168, // 7 days
	}
}

// MarketplacePluginInfo extends PluginInfo with marketplace-specific information
type MarketplacePluginInfo struct {
	// Base plugin information
	PluginInfo

	// Download URL
	DownloadURL string

	// Size in bytes
	SizeBytes int64

	// SHA256 checksum
	Checksum string

	// Digital signature
	Signature string

	// Release date
	ReleaseDate time.Time

	// Download count
	DownloadCount int

	// Rating (0.0 to 5.0)
	Rating float64

	// Number of ratings
	RatingCount int

	// Screenshots URLs
	Screenshots []string

	// Project homepage
	Homepage string

	// Documentation URL
	DocumentationURL string

	// Support contact
	SupportContact string

	// Repository URL
	RepositoryURL string

	// Reviews
	Reviews []PluginReview

	// Pricing information
	Pricing *PluginPricing
}

// PluginReview represents a user review for a plugin
type PluginReview struct {
	// User who left the review
	Username string

	// Rating (0.0 to 5.0)
	Rating float64

	// Review text
	Text string

	// Date of the review
	Date time.Time
}

// PluginPricing represents pricing information for a plugin
type PluginPricing struct {
	// Whether the plugin is free
	IsFree bool

	// One-time price in USD
	PriceUSD float64

	// Subscription pricing
	SubscriptionPriceUSD float64

	// Subscription period (monthly, yearly)
	SubscriptionPeriod string

	// Whether there's a trial period
	HasTrial bool

	// Trial period in days
	TrialDays int
}

// DownloadProgress tracks plugin download progress
type DownloadProgress struct {
	// Total bytes to download
	TotalBytes int64

	// Bytes downloaded so far
	BytesDownloaded int64

	// Progress percentage (0-100)
	PercentComplete float64

	// Download speed in bytes per second
	SpeedBytesPerSecond int64

	// Estimated time remaining in seconds
	EstimatedTimeSeconds int

	// Current status (downloading, verifying, installing)
	Status string
}

// PluginMarketplace provides a centralized repository for discovering and installing plugins
type PluginMarketplace struct {
	// Marketplace configuration
	config MarketplaceConfig

	// Plugin manager to register plugins with
	pluginManager *PluginManager

	// HTTP client for marketplace requests
	client *http.Client

	// Cache of marketplace plugins
	pluginsCache map[string]MarketplacePluginInfo

	// Last cache update time
	lastCacheUpdate time.Time

	// Active downloads
	activeDownloads map[string]*DownloadProgress

	// Lock for concurrent access
	lock sync.RWMutex
}

// NewPluginMarketplace creates a new plugin marketplace
func NewPluginMarketplace(config MarketplaceConfig, pluginManager *PluginManager) (*PluginMarketplace, error) {
	// Create the cache directory if it doesn't exist
	if err := os.MkdirAll(config.CacheDirectory, 0755); err != nil {
		return nil, fmt.Errorf("failed to create marketplace cache directory: %v", err)
	}

	return &PluginMarketplace{
		config:          config,
		pluginManager:   pluginManager,
		client:          &http.Client{Timeout: 30 * time.Second},
		pluginsCache:    make(map[string]MarketplacePluginInfo),
		activeDownloads: make(map[string]*DownloadProgress),
	}, nil
}

// RefreshCache updates the cache of available plugins from marketplace repositories
func (pm *PluginMarketplace) RefreshCache() error {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	// Check each repository URL
	for _, repoURL := range pm.config.RepositoryURLs {
		// Fetch the plugin index
		indexURL := fmt.Sprintf("%s/index.json", repoURL)
		resp, err := pm.client.Get(indexURL)
		if err != nil {
			return fmt.Errorf("failed to fetch plugin index from %s: %v", indexURL, err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("failed to fetch plugin index from %s: status %d", indexURL, resp.StatusCode)
		}

		// Parse the index
		var plugins []MarketplacePluginInfo
		if err := json.NewDecoder(resp.Body).Decode(&plugins); err != nil {
			return fmt.Errorf("failed to parse plugin index from %s: %v", indexURL, err)
		}

		// Update cache
		for _, plugin := range plugins {
			pm.pluginsCache[plugin.ID] = plugin
		}
	}

	// Update cache timestamp
	pm.lastCacheUpdate = time.Now()

	// Write cache to disk
	return pm.writeCacheToDisk()
}

// writeCacheToDisk persists the plugin cache to disk
func (pm *PluginMarketplace) writeCacheToDisk() error {
	cacheFile := filepath.Join(pm.config.CacheDirectory, "plugins.json")

	// Convert map to slice for JSON serialization
	plugins := make([]MarketplacePluginInfo, 0, len(pm.pluginsCache))
	for _, plugin := range pm.pluginsCache {
		plugins = append(plugins, plugin)
	}

	// Marshal to JSON with pretty printing
	data, err := json.MarshalIndent(plugins, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal plugins cache: %v", err)
	}

	// Write to file
	if err := os.WriteFile(cacheFile, data, 0644); err != nil {
		return fmt.Errorf("failed to write plugins cache to disk: %v", err)
	}

	return nil
}

// loadCacheFromDisk loads the plugin cache from disk
func (pm *PluginMarketplace) loadCacheFromDisk() error {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	cacheFile := filepath.Join(pm.config.CacheDirectory, "plugins.json")

	// Check if cache file exists
	if _, err := os.Stat(cacheFile); os.IsNotExist(err) {
		return nil // No cache file, not an error
	}

	// Read the cache file
	data, err := os.ReadFile(cacheFile)
	if err != nil {
		return fmt.Errorf("failed to read plugins cache from disk: %v", err)
	}

	// Unmarshal from JSON
	var plugins []MarketplacePluginInfo
	if err := json.Unmarshal(data, &plugins); err != nil {
		return fmt.Errorf("failed to unmarshal plugins cache: %v", err)
	}

	// Update cache map
	for _, plugin := range plugins {
		pm.pluginsCache[plugin.ID] = plugin
	}

	return nil
}

// ListAvailablePlugins returns a list of available plugins from the marketplace
func (pm *PluginMarketplace) ListAvailablePlugins() ([]MarketplacePluginInfo, error) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	// Check if cache needs to be refreshed
	cacheExpiration := time.Hour * time.Duration(pm.config.CacheExpirationHours)
	if time.Since(pm.lastCacheUpdate) > cacheExpiration {
		pm.lock.RUnlock()
		if err := pm.RefreshCache(); err != nil {
			// If refresh fails, log the error but continue with what we have
			fmt.Printf("Failed to refresh plugin cache: %v\n", err)
		}
		pm.lock.RLock()
	}

	// Convert map to slice for return
	plugins := make([]MarketplacePluginInfo, 0, len(pm.pluginsCache))
	for _, plugin := range pm.pluginsCache {
		plugins = append(plugins, plugin)
	}

	return plugins, nil
}

// verifyChecksum verifies the checksum of a downloaded plugin
func (m *PluginMarketplace) verifyChecksum(filePath, expectedChecksum string) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file for checksum verification: %w", err)
	}
	defer file.Close()

	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return fmt.Errorf("failed to calculate checksum: %w", err)
	}

	actualChecksum := fmt.Sprintf("%x", hash.Sum(nil))
	if actualChecksum != expectedChecksum {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expectedChecksum, actualChecksum)
	}

	return nil
}

// verifySignature verifies the digital signature of a downloaded plugin
func (m *PluginMarketplace) verifySignature(filePath, signature string) error {
	// Stub implementation - in production this would verify digital signatures
	// using public key cryptography (e.g., RSA, ECDSA)
	if signature == "" {
		return fmt.Errorf("empty signature provided")
	}
	
	// TODO: Implement actual signature verification with public keys
	log.Printf("Signature verification for %s: %s (stub implementation)", filePath, signature)
	return nil
}

// SearchPlugins searches for plugins based on criteria
func (pm *PluginMarketplace) SearchPlugins(query string, category string, tags []string) ([]MarketplacePluginInfo, error) {
	// Get all available plugins
	allPlugins, err := pm.ListAvailablePlugins()
	if err != nil {
		return nil, err
	}

	// Filter based on criteria
	var results []MarketplacePluginInfo
	for _, plugin := range allPlugins {
		// Check if plugin matches the query
		if query != "" {
			if !containsIgnoreCase(plugin.Name, query) && !containsIgnoreCase(plugin.Description, query) {
				continue
			}
		}

		// Check if plugin matches the category
		if category != "" {
			categoryMatched := false
			for _, tag := range plugin.Tags {
				if tag == category {
					categoryMatched = true
					break
				}
			}
			if !categoryMatched {
				continue
			}
		}

		// Check if plugin has all the required tags
		if len(tags) > 0 {
			tagsMatched := true
			for _, requiredTag := range tags {
				tagFound := false
				for _, pluginTag := range plugin.Tags {
					if pluginTag == requiredTag {
						tagFound = true
						break
					}
				}
				if !tagFound {
					tagsMatched = false
					break
				}
			}
			if !tagsMatched {
				continue
			}
		}

		// Plugin matches all criteria
		results = append(results, plugin)
	}

	return results, nil
}

// containsIgnoreCase checks if a string contains a substring, ignoring case
func containsIgnoreCase(s, substr string) bool {
	s, substr = strings.ToLower(s), strings.ToLower(substr)
	return strings.Contains(s, substr)
}

// GetPluginDetails returns detailed information about a specific plugin
func (pm *PluginMarketplace) GetPluginDetails(pluginID string) (*MarketplacePluginInfo, error) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	plugin, exists := pm.pluginsCache[pluginID]
	if !exists {
		return nil, fmt.Errorf("plugin %s not found in marketplace", pluginID)
	}

	return &plugin, nil
}

// DownloadPlugin downloads a plugin from the marketplace
func (pm *PluginMarketplace) DownloadPlugin(pluginID string, progressCallback func(*DownloadProgress), autoEnable bool) error {
	// Get plugin details
	pluginInfo, err := pm.GetPluginDetails(pluginID)
	if err != nil {
		return err
	}

	// Create progress tracker
	progress := &DownloadProgress{
		TotalBytes:      pluginInfo.SizeBytes,
		BytesDownloaded: 0,
		PercentComplete: 0,
		Status:          "starting",
	}

	// Store in active downloads
	pm.lock.Lock()
	pm.activeDownloads[pluginID] = progress
	pm.lock.Unlock()

	// Ensure we remove from active downloads when done
	defer func() {
		pm.lock.Lock()
		delete(pm.activeDownloads, pluginID)
		pm.lock.Unlock()
	}()

	// Create the plugins directory if it doesn't exist
	pluginsDir := filepath.Join(pm.pluginManager.paths[0])
	if err := os.MkdirAll(pluginsDir, 0755); err != nil {
		return fmt.Errorf("failed to create plugins directory: %v", err)
	}

	// Determine destination path
	destPath := filepath.Join(pluginsDir, fmt.Sprintf("%s.so", pluginID))

	// Start download
	progress.Status = "downloading"
	if progressCallback != nil {
		progressCallback(progress)
	}

	// Create HTTP request
	req, err := http.NewRequest("GET", pluginInfo.DownloadURL, nil)
	if err != nil {
		return fmt.Errorf("failed to create download request: %v", err)
	}

	// Execute request
	resp, err := pm.client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download plugin: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download plugin: status %d", resp.StatusCode)
	}

	// Create destination file
	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create destination file: %v", err)
	}
	defer destFile.Close()

	// Download with progress tracking
	startTime := time.Now()
	progress.Status = "downloading"

	buf := make([]byte, 32*1024) // 32KB buffer
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			// Write to file
			if _, err := destFile.Write(buf[:n]); err != nil {
				return fmt.Errorf("failed to write to destination file: %v", err)
			}

			// Update progress
			progress.BytesDownloaded += int64(n)
			progress.PercentComplete = float64(progress.BytesDownloaded) / float64(progress.TotalBytes) * 100

			// Calculate speed
			elapsed := time.Since(startTime).Seconds()
			if elapsed > 0 {
				progress.SpeedBytesPerSecond = int64(float64(progress.BytesDownloaded) / elapsed)

				// Estimate time remaining
				if progress.SpeedBytesPerSecond > 0 {
					bytesRemaining := progress.TotalBytes - progress.BytesDownloaded
					progress.EstimatedTimeSeconds = int(float64(bytesRemaining) / float64(progress.SpeedBytesPerSecond))
				}
			}

			// Report progress
			if progressCallback != nil {
				progressCallback(progress)
			}
		}

		if err == io.EOF {
			break
		}

		if err != nil {
			return fmt.Errorf("error during download: %v", err)
		}
	}

	// Verify checksum
	progress.Status = "verifying"
	if progressCallback != nil {
		progressCallback(progress)
	}

	// Implement checksum verification
	if plugin.Checksum != "" {
		if err := m.verifyChecksum(destPath, plugin.Checksum); err != nil {
			os.Remove(destPath)
			return fmt.Errorf("checksum verification failed: %w", err)
		}
	}

	// Implement signature verification if required
	if plugin.Signature != "" {
		if err := m.verifySignature(destPath, plugin.Signature); err != nil {
			os.Remove(destPath)
			return fmt.Errorf("signature verification failed: %w", err)
		}
	}

	// Load the plugin
	progress.Status = "loading"
	if progressCallback != nil {
		progressCallback(progress)
	}

	// Load the plugin using the plugin manager
	plugin, err := pm.pluginManager.LoadPlugin(destPath)
	if err != nil {
		// Cleanup on error
		os.Remove(destPath)
		return fmt.Errorf("failed to load plugin: %v", err)
	}

	// Enable the plugin if auto-enable is specified
	if autoEnable {
		progress.Status = "enabling"
		if progressCallback != nil {
			progressCallback(progress)
		}

		if err := pm.pluginManager.EnablePlugin(plugin.Info.ID); err != nil {
			return fmt.Errorf("failed to enable plugin: %v", err)
		}
	}

	progress.Status = "completed"
	if progressCallback != nil {
		progressCallback(progress)
	}

	return nil
}

// GetDownloadProgress returns the progress of an active download
func (pm *PluginMarketplace) GetDownloadProgress(pluginID string) (*DownloadProgress, error) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	progress, exists := pm.activeDownloads[pluginID]
	if !exists {
		return nil, fmt.Errorf("no active download for plugin %s", pluginID)
	}

	return progress, nil
}

// SubmitPluginReview submits a review for a plugin
func (pm *PluginMarketplace) SubmitPluginReview(pluginID string, username string, rating float64, text string) error {
	// This would typically involve a network request to the marketplace API
	// For now, just a placeholder implementation
	return fmt.Errorf("plugin review submission not implemented")
}

// CheckForUpdates checks for updates to installed plugins
func (pm *PluginMarketplace) CheckForUpdates() (map[string]string, error) {
	// Get installed plugins
	installedPlugins := pm.pluginManager.ListPlugins()

	// Get available plugins from marketplace
	availablePlugins, err := pm.ListAvailablePlugins()
	if err != nil {
		return nil, fmt.Errorf("failed to list available plugins: %v", err)
	}

	// Build map of available plugins by ID
	availablePluginsMap := make(map[string]MarketplacePluginInfo)
	for _, plugin := range availablePlugins {
		availablePluginsMap[plugin.ID] = plugin
	}

	// Check for updates
	updates := make(map[string]string)
	for _, installedPlugin := range installedPlugins {
		availablePlugin, exists := availablePluginsMap[installedPlugin.Info.ID]
		if exists && availablePlugin.Version != installedPlugin.Info.Version {
			updates[installedPlugin.Info.ID] = availablePlugin.Version
		}
	}

	return updates, nil
}

// UpdatePlugin updates a plugin to the latest version
func (pm *PluginMarketplace) UpdatePlugin(pluginID string, progressCallback func(*DownloadProgress), autoEnable bool) error {
	// First, disable the existing plugin
	if err := pm.pluginManager.DisablePlugin(pluginID); err != nil {
		return fmt.Errorf("failed to disable plugin before update: %v", err)
	}

	// Then download the new version
	return pm.DownloadPlugin(pluginID, progressCallback, autoEnable)
}
