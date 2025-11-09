package loadbalancing

import (
	"math"
	"net"
	"sync"
)

// GeoRouter provides geographic routing capabilities
type GeoRouter struct {
	config *LoadBalancerConfig
	geoip  *GeoIPDatabase
	cache  *geoCache
	mu     sync.RWMutex
}

// geoCache caches GeoIP lookups
type geoCache struct {
	entries map[string]*GeoLocation
	mu      sync.RWMutex
}

// GeoIPDatabase represents a GeoIP database interface
type GeoIPDatabase struct {
	// In production, this would integrate with MaxMind GeoIP2 or similar
	// For now, we'll use a simplified version
	regions map[string]*GeoLocation
	mu      sync.RWMutex
}

// NewGeoRouter creates a new geographic router
func NewGeoRouter(config *LoadBalancerConfig) (*GeoRouter, error) {
	geoip, err := loadGeoIPDatabase(config.GeoIPDatabasePath)
	if err != nil {
		// If GeoIP database is not available, create an empty one
		geoip = &GeoIPDatabase{
			regions: make(map[string]*GeoLocation),
		}
	}

	return &GeoRouter{
		config: config,
		geoip:  geoip,
		cache: &geoCache{
			entries: make(map[string]*GeoLocation),
		},
	}, nil
}

// loadGeoIPDatabase loads the GeoIP database
func loadGeoIPDatabase(path string) (*GeoIPDatabase, error) {
	// In production, load actual GeoIP database
	// For now, create a mock database with common regions
	db := &GeoIPDatabase{
		regions: map[string]*GeoLocation{
			"us-east-1":    {Latitude: 39.0438, Longitude: -77.4874, Region: "us-east-1", Country: "US", City: "Virginia"},
			"us-west-1":    {Latitude: 37.7749, Longitude: -122.4194, Region: "us-west-1", Country: "US", City: "California"},
			"eu-west-1":    {Latitude: 53.3498, Longitude: -6.2603, Region: "eu-west-1", Country: "IE", City: "Dublin"},
			"eu-central-1": {Latitude: 50.1109, Longitude: 8.6821, Region: "eu-central-1", Country: "DE", City: "Frankfurt"},
			"ap-southeast-1": {Latitude: 1.3521, Longitude: 103.8198, Region: "ap-southeast-1", Country: "SG", City: "Singapore"},
			"ap-northeast-1": {Latitude: 35.6762, Longitude: 139.6503, Region: "ap-northeast-1", Country: "JP", City: "Tokyo"},
		},
	}
	return db, nil
}

// GetClientLocation determines the geographic location of a client IP
func (gr *GeoRouter) GetClientLocation(clientIP string) (*GeoLocation, error) {
	// Check cache first
	gr.cache.mu.RLock()
	if loc, exists := gr.cache.entries[clientIP]; exists {
		gr.cache.mu.RUnlock()
		return loc, nil
	}
	gr.cache.mu.RUnlock()

	// Parse IP
	ip := net.ParseIP(clientIP)
	if ip == nil {
		return nil, ErrInvalidGeoLocation
	}

	// Lookup in GeoIP database
	location := gr.lookupIP(ip)
	if location == nil {
		// Default to a fallback location
		location = &GeoLocation{
			Latitude:  0.0,
			Longitude: 0.0,
			Region:    "unknown",
			Country:   "XX",
			City:      "Unknown",
		}
	}

	// Cache the result
	gr.cache.mu.Lock()
	gr.cache.entries[clientIP] = location
	gr.cache.mu.Unlock()

	return location, nil
}

// lookupIP looks up IP in GeoIP database
func (gr *GeoRouter) lookupIP(ip net.IP) *GeoLocation {
	// In production, use actual GeoIP2 database lookup
	// For now, use heuristics based on IP ranges

	// This is a simplified mock implementation
	// Real implementation would use MaxMind GeoIP2 or similar

	// Default to US East for private/local IPs
	if ip.IsPrivate() || ip.IsLoopback() {
		gr.geoip.mu.RLock()
		defer gr.geoip.mu.RUnlock()
		if loc, ok := gr.geoip.regions["us-east-1"]; ok {
			return loc
		}
	}

	// For demonstration, map some IP ranges to regions
	// In production, use proper GeoIP database
	return &GeoLocation{
		Latitude:  39.0438,
		Longitude: -77.4874,
		Region:    "us-east-1",
		Country:   "US",
		City:      "Default",
	}
}

// FindNearestServer finds the geographically nearest server to a location
func (gr *GeoRouter) FindNearestServer(clientLoc *GeoLocation, servers []*Server) *Server {
	if len(servers) == 0 {
		return nil
	}

	var nearestServer *Server
	minDistance := math.MaxFloat64

	for _, server := range servers {
		distance := gr.calculateDistance(clientLoc.Latitude, clientLoc.Longitude,
			server.Latitude, server.Longitude)

		if distance < minDistance {
			minDistance = distance
			nearestServer = server
		}
	}

	return nearestServer
}

// FindServersByProximity returns servers sorted by proximity to client location
func (gr *GeoRouter) FindServersByProximity(clientLoc *GeoLocation, servers []*Server, maxResults int) []*Server {
	if len(servers) == 0 {
		return nil
	}

	// Calculate distances
	type serverDistance struct {
		server   *Server
		distance float64
	}

	distances := make([]serverDistance, 0, len(servers))
	for _, server := range servers {
		distance := gr.calculateDistance(clientLoc.Latitude, clientLoc.Longitude,
			server.Latitude, server.Longitude)
		distances = append(distances, serverDistance{server: server, distance: distance})
	}

	// Sort by distance using simple insertion sort (efficient for small lists)
	for i := 1; i < len(distances); i++ {
		key := distances[i]
		j := i - 1
		for j >= 0 && distances[j].distance > key.distance {
			distances[j+1] = distances[j]
			j--
		}
		distances[j+1] = key
	}

	// Return top N servers
	limit := maxResults
	if limit > len(distances) {
		limit = len(distances)
	}

	result := make([]*Server, limit)
	for i := 0; i < limit; i++ {
		result[i] = distances[i].server
	}

	return result
}

// calculateDistance calculates the great-circle distance between two points
// using the Haversine formula. Returns distance in kilometers.
func (gr *GeoRouter) calculateDistance(lat1, lon1, lat2, lon2 float64) float64 {
	const earthRadius = 6371.0 // Earth's radius in kilometers

	// Convert to radians
	lat1Rad := lat1 * math.Pi / 180.0
	lon1Rad := lon1 * math.Pi / 180.0
	lat2Rad := lat2 * math.Pi / 180.0
	lon2Rad := lon2 * math.Pi / 180.0

	// Haversine formula
	dLat := lat2Rad - lat1Rad
	dLon := lon2Rad - lon1Rad

	a := math.Sin(dLat/2)*math.Sin(dLat/2) +
		math.Cos(lat1Rad)*math.Cos(lat2Rad)*
			math.Sin(dLon/2)*math.Sin(dLon/2)

	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))

	return earthRadius * c
}

// GetRegionLocation returns the geographic location of a region
func (gr *GeoRouter) GetRegionLocation(region string) (*GeoLocation, error) {
	gr.geoip.mu.RLock()
	defer gr.geoip.mu.RUnlock()

	if loc, ok := gr.geoip.regions[region]; ok {
		return loc, nil
	}

	return nil, ErrNoRegionMatch
}

// ClearCache clears the GeoIP cache
func (gr *GeoRouter) ClearCache() {
	gr.cache.mu.Lock()
	defer gr.cache.mu.Unlock()
	gr.cache.entries = make(map[string]*GeoLocation)
}
