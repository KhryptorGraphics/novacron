package cluster

import (
	"errors"
	"time"
)

// Common errors
var (
	ErrClusterNotFound = errors.New("cluster not found")
)

// ClusterInfo represents information about a cluster
type ClusterInfo struct {
	ID       string    `json:"id"`
	Name     string    `json:"name"`
	Endpoint string    `json:"endpoint"`
	Status   string    `json:"status"`
	Capacity Capacity  `json:"capacity"`
	Created  time.Time `json:"created,omitempty"`
	Updated  time.Time `json:"updated,omitempty"`
}

// Capacity represents resource capacity
type Capacity struct {
	CPU     int64 `json:"cpu"`
	Memory  int64 `json:"memory"`
	Storage int64 `json:"storage"`
}

// Resource represents a cluster resource
type Resource struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	ClusterID string    `json:"cluster_id"`
	Status    string    `json:"status"`
	Created   time.Time `json:"created,omitempty"`
	Updated   time.Time `json:"updated,omitempty"`
}

// FederationStatus represents the status of the federation
type FederationStatus struct {
	TotalClusters  int       `json:"total_clusters"`
	ActiveClusters int       `json:"active_clusters"`
	HealthStatus   string    `json:"health_status"`
	LastSync       time.Time `json:"last_sync"`
}
