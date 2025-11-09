package loadbalancing

import (
	"math"
	"testing"
)

func TestGeoRouterCreation(t *testing.T) {
	config := DefaultConfig()
	router, err := NewGeoRouter(config)
	if err != nil {
		t.Fatalf("Failed to create geo router: %v", err)
	}

	if router == nil {
		t.Fatal("Expected non-nil router")
	}
}

func TestGeoRouterGetClientLocation(t *testing.T) {
	config := DefaultConfig()
	router, _ := NewGeoRouter(config)

	tests := []struct {
		name     string
		clientIP string
		wantErr  bool
	}{
		{
			name:     "Valid IP",
			clientIP: "1.2.3.4",
			wantErr:  false,
		},
		{
			name:     "Localhost",
			clientIP: "127.0.0.1",
			wantErr:  false,
		},
		{
			name:     "Invalid IP",
			clientIP: "invalid",
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loc, err := router.GetClientLocation(tt.clientIP)
			if tt.wantErr && err == nil {
				t.Error("Expected error but got none")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
			if !tt.wantErr && loc == nil {
				t.Error("Expected location but got nil")
			}
		})
	}
}

func TestGeoRouterCalculateDistance(t *testing.T) {
	config := DefaultConfig()
	router, _ := NewGeoRouter(config)

	// Test distance between known cities
	// New York to Los Angeles (approximately 3936 km)
	nyLat, nyLon := 40.7128, -74.0060
	laLat, laLon := 34.0522, -118.2437

	distance := router.calculateDistance(nyLat, nyLon, laLat, laLon)

	// Allow 10% margin of error
	expectedDistance := 3936.0
	margin := expectedDistance * 0.1

	if math.Abs(distance-expectedDistance) > margin {
		t.Errorf("Distance calculation incorrect: expected ~%.0f km, got %.0f km",
			expectedDistance, distance)
	}

	// Test same location
	sameDistance := router.calculateDistance(nyLat, nyLon, nyLat, nyLon)
	if sameDistance > 0.1 {
		t.Errorf("Same location distance should be ~0, got %.2f", sameDistance)
	}
}

func TestGeoRouterFindNearestServer(t *testing.T) {
	config := DefaultConfig()
	router, _ := NewGeoRouter(config)

	// Client location in New York
	clientLoc := &GeoLocation{
		Latitude:  40.7128,
		Longitude: -74.0060,
		Region:    "us-east-1",
	}

	// Servers in different locations
	servers := []*Server{
		{
			ID:        "1",
			Region:    "us-east-1",
			Latitude:  39.0438,  // Virginia
			Longitude: -77.4874,
		},
		{
			ID:        "2",
			Region:    "us-west-1",
			Latitude:  37.7749,  // California
			Longitude: -122.4194,
		},
		{
			ID:        "3",
			Region:    "eu-west-1",
			Latitude:  53.3498,  // Dublin
			Longitude: -6.2603,
		},
	}

	nearest := router.FindNearestServer(clientLoc, servers)
	if nearest == nil {
		t.Fatal("Expected nearest server but got nil")
	}

	// Should select US East (Virginia) as it's closest to New York
	if nearest.ID != "1" {
		t.Errorf("Expected server 1 (Virginia), got server %s", nearest.ID)
	}
}

func TestGeoRouterFindServersByProximity(t *testing.T) {
	config := DefaultConfig()
	router, _ := NewGeoRouter(config)

	clientLoc := &GeoLocation{
		Latitude:  40.7128,
		Longitude: -74.0060,
	}

	servers := []*Server{
		{ID: "1", Latitude: 39.0438, Longitude: -77.4874},   // Virginia
		{ID: "2", Latitude: 37.7749, Longitude: -122.4194},  // California
		{ID: "3", Latitude: 53.3498, Longitude: -6.2603},    // Dublin
		{ID: "4", Latitude: 35.6762, Longitude: 139.6503},   // Tokyo
	}

	// Get top 2 servers
	proximate := router.FindServersByProximity(clientLoc, servers, 2)
	if len(proximate) != 2 {
		t.Fatalf("Expected 2 servers, got %d", len(proximate))
	}

	// First should be Virginia (closest)
	if proximate[0].ID != "1" {
		t.Errorf("Expected first server to be 1 (Virginia), got %s", proximate[0].ID)
	}

	// Second should be California
	if proximate[1].ID != "2" {
		t.Errorf("Expected second server to be 2 (California), got %s", proximate[1].ID)
	}
}

func TestGeoRouterClearCache(t *testing.T) {
	config := DefaultConfig()
	router, _ := NewGeoRouter(config)

	// Add entry to cache
	router.GetClientLocation("1.2.3.4")

	// Verify cache has entry
	router.cache.mu.RLock()
	cacheSize := len(router.cache.entries)
	router.cache.mu.RUnlock()

	if cacheSize == 0 {
		t.Error("Expected cache to have entries")
	}

	// Clear cache
	router.ClearCache()

	// Verify cache is empty
	router.cache.mu.RLock()
	cacheSize = len(router.cache.entries)
	router.cache.mu.RUnlock()

	if cacheSize != 0 {
		t.Errorf("Expected cache to be empty, got %d entries", cacheSize)
	}
}
