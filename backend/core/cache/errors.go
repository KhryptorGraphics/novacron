package cache

import "errors"

var (
	// ErrCacheMiss indicates the requested key was not found in cache
	ErrCacheMiss = errors.New("cache miss")

	// ErrCacheNotAvailable indicates the cache service is not available
	ErrCacheNotAvailable = errors.New("cache not available")

	// ErrInvalidKey indicates an invalid cache key was provided
	ErrInvalidKey = errors.New("invalid cache key")

	// ErrKeyTooLong indicates the cache key exceeds maximum length
	ErrKeyTooLong = errors.New("cache key too long")

	// ErrValueTooLarge indicates the cache value exceeds maximum size
	ErrValueTooLarge = errors.New("cache value too large")

	// ErrTTLTooLong indicates the TTL exceeds maximum allowed
	ErrTTLTooLong = errors.New("TTL too long")

	// ErrCacheFull indicates the cache is full and cannot accept new entries
	ErrCacheFull = errors.New("cache full")

	// ErrSerializationFailed indicates data serialization failed
	ErrSerializationFailed = errors.New("serialization failed")

	// ErrDeserializationFailed indicates data deserialization failed
	ErrDeserializationFailed = errors.New("deserialization failed")

	// ErrConnectionFailed indicates connection to cache backend failed
	ErrConnectionFailed = errors.New("cache connection failed")

	// ErrTimeout indicates cache operation timed out
	ErrTimeout = errors.New("cache operation timeout")

	// ErrClusterDown indicates the cache cluster is down
	ErrClusterDown = errors.New("cache cluster down")

	// ErrReadOnly indicates the cache is in read-only mode
	ErrReadOnly = errors.New("cache in read-only mode")
)