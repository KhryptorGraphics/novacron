package graphql

import (
	"time"
)

// GraphQL type definitions

// StorageVolume represents a storage volume
type StorageVolume struct {
	ID            string         `json:"id"`
	Name          string         `json:"name"`
	Size          int            `json:"size"`
	Tier          string         `json:"tier"`
	VMID          string         `json:"vmId,omitempty"`
	CreatedAt     time.Time      `json:"createdAt"`
	UpdatedAt     time.Time      `json:"updatedAt"`
	AccessPattern *AccessPattern `json:"accessPattern,omitempty"`
}

// AccessPattern represents volume access patterns
type AccessPattern struct {
	Temperature   string    `json:"temperature"`
	AccessRate    float64   `json:"accessRate"`
	LastAccessed  time.Time `json:"lastAccessed"`
	PredictedTier string    `json:"predictedTier"`
}

// Input types

// CreateVolumeInput represents input for creating a volume
type CreateVolumeInput struct {
	Name string  `json:"name"`
	Size int     `json:"size"`
	Tier string  `json:"tier"`
	VMID *string `json:"vmId,omitempty"`
}

// PaginationInput represents pagination parameters
type PaginationInput struct {
	Page     int `json:"page"`
	PageSize int `json:"pageSize"`
}
