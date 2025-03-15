package storage

import (
	"bytes"
	"context"
	"fmt"
	"path"
	"strconv"
	"sync"
)

// DistributedStorage provides an interface for distributed storage systems
type DistributedStorage interface {
	// Get retrieves a value for a key
	Get(ctx context.Context, key string) ([]byte, error)

	// Put stores a value for a key
	Put(ctx context.Context, key string, value []byte) error

	// Delete removes a key
	Delete(ctx context.Context, key string) error

	// List lists keys matching a pattern
	List(ctx context.Context, pattern string) ([]string, error)

	// BatchGet gets multiple values for keys
	BatchGet(ctx context.Context, keys []string) (map[string][]byte, error)

	// BatchPut stores multiple values for keys
	BatchPut(ctx context.Context, keyValues map[string][]byte) error

	// BatchDelete removes multiple keys
	BatchDelete(ctx context.Context, keys []string) error

	// Watch watches for changes to a key
	Watch(ctx context.Context, key string) (<-chan []byte, error)

	// Atomic operations
	CompareAndSwap(ctx context.Context, key string, oldValue, newValue []byte) (bool, error)
	IncrementCounter(ctx context.Context, key string, increment int64) (int64, error)
}

// InMemoryStorage implements DistributedStorage with in-memory storage
// This is primarily for testing and development
type InMemoryStorage struct {
	data      map[string][]byte
	watchers  map[string][]chan []byte
	dataMutex sync.RWMutex
}

// NewInMemoryStorage creates a new in-memory storage
func NewInMemoryStorage() *InMemoryStorage {
	return &InMemoryStorage{
		data:     make(map[string][]byte),
		watchers: make(map[string][]chan []byte),
	}
}

// Get retrieves a value for a key
func (s *InMemoryStorage) Get(ctx context.Context, key string) ([]byte, error) {
	s.dataMutex.RLock()
	defer s.dataMutex.RUnlock()

	value, exists := s.data[key]
	if !exists {
		return nil, fmt.Errorf("key not found: %s", key)
	}

	// Return a copy to avoid data races
	result := make([]byte, len(value))
	copy(result, value)
	return result, nil
}

// Put stores a value for a key
func (s *InMemoryStorage) Put(ctx context.Context, key string, value []byte) error {
	s.dataMutex.Lock()
	defer s.dataMutex.Unlock()

	// Store a copy to avoid data races
	valueCopy := make([]byte, len(value))
	copy(valueCopy, value)
	s.data[key] = valueCopy

	// Notify watchers
	s.notifyWatchers(key, valueCopy)

	return nil
}

// Delete removes a key
func (s *InMemoryStorage) Delete(ctx context.Context, key string) error {
	s.dataMutex.Lock()
	defer s.dataMutex.Unlock()

	delete(s.data, key)

	// Notify watchers with nil value
	s.notifyWatchers(key, nil)

	return nil
}

// List lists keys matching a pattern
func (s *InMemoryStorage) List(ctx context.Context, pattern string) ([]string, error) {
	s.dataMutex.RLock()
	defer s.dataMutex.RUnlock()

	var result []string
	for key := range s.data {
		matched, err := path.Match(pattern, key)
		if err != nil {
			return nil, fmt.Errorf("invalid pattern: %s", pattern)
		}
		if matched {
			result = append(result, key)
		}
	}

	return result, nil
}

// BatchGet gets multiple values for keys
func (s *InMemoryStorage) BatchGet(ctx context.Context, keys []string) (map[string][]byte, error) {
	s.dataMutex.RLock()
	defer s.dataMutex.RUnlock()

	result := make(map[string][]byte)
	for _, key := range keys {
		value, exists := s.data[key]
		if exists {
			// Return a copy to avoid data races
			valueCopy := make([]byte, len(value))
			copy(valueCopy, value)
			result[key] = valueCopy
		}
	}

	return result, nil
}

// BatchPut stores multiple values for keys
func (s *InMemoryStorage) BatchPut(ctx context.Context, keyValues map[string][]byte) error {
	s.dataMutex.Lock()
	defer s.dataMutex.Unlock()

	for key, value := range keyValues {
		// Store a copy to avoid data races
		valueCopy := make([]byte, len(value))
		copy(valueCopy, value)
		s.data[key] = valueCopy

		// Notify watchers
		s.notifyWatchers(key, valueCopy)
	}

	return nil
}

// BatchDelete removes multiple keys
func (s *InMemoryStorage) BatchDelete(ctx context.Context, keys []string) error {
	s.dataMutex.Lock()
	defer s.dataMutex.Unlock()

	for _, key := range keys {
		delete(s.data, key)

		// Notify watchers with nil value
		s.notifyWatchers(key, nil)
	}

	return nil
}

// Watch watches for changes to a key
func (s *InMemoryStorage) Watch(ctx context.Context, key string) (<-chan []byte, error) {
	s.dataMutex.Lock()
	defer s.dataMutex.Unlock()

	// Create a channel for this watcher
	ch := make(chan []byte, 10) // Buffer a few updates

	// Add to watchers
	s.watchers[key] = append(s.watchers[key], ch)

	// Start a goroutine to clean up when context is done
	go func() {
		<-ctx.Done()
		s.dataMutex.Lock()
		defer s.dataMutex.Unlock()

		// Remove the channel from watchers
		watchers := s.watchers[key]
		for i, watcher := range watchers {
			if watcher == ch {
				s.watchers[key] = append(watchers[:i], watchers[i+1:]...)
				break
			}
		}

		// Close the channel
		close(ch)
	}()

	return ch, nil
}

// CompareAndSwap atomically replaces oldValue with newValue for key
func (s *InMemoryStorage) CompareAndSwap(ctx context.Context, key string, oldValue, newValue []byte) (bool, error) {
	s.dataMutex.Lock()
	defer s.dataMutex.Unlock()

	value, exists := s.data[key]
	if !exists {
		if oldValue == nil {
			// Key doesn't exist and oldValue is nil, so we can set it
			valueCopy := make([]byte, len(newValue))
			copy(valueCopy, newValue)
			s.data[key] = valueCopy

			// Notify watchers
			s.notifyWatchers(key, valueCopy)

			return true, nil
		}
		return false, nil
	}

	// Compare values
	if !bytes.Equal(value, oldValue) {
		return false, nil
	}

	// Values match, update
	valueCopy := make([]byte, len(newValue))
	copy(valueCopy, newValue)
	s.data[key] = valueCopy

	// Notify watchers
	s.notifyWatchers(key, valueCopy)

	return true, nil
}

// IncrementCounter atomically increments a counter
func (s *InMemoryStorage) IncrementCounter(ctx context.Context, key string, increment int64) (int64, error) {
	s.dataMutex.Lock()
	defer s.dataMutex.Unlock()

	var counter int64
	value, exists := s.data[key]
	if exists {
		// Parse the existing counter
		var err error
		counter, err = strconv.ParseInt(string(value), 10, 64)
		if err != nil {
			return 0, fmt.Errorf("invalid counter value: %s", string(value))
		}
	}

	// Increment the counter
	counter += increment

	// Update the value
	newValue := []byte(strconv.FormatInt(counter, 10))
	s.data[key] = newValue

	// Notify watchers
	s.notifyWatchers(key, newValue)

	return counter, nil
}

// Helper functions

// notifyWatchers notifies all watchers for a key
func (s *InMemoryStorage) notifyWatchers(key string, value []byte) {
	watchers := s.watchers[key]
	for _, watcher := range watchers {
		// Make a copy for each watcher to avoid data races
		valueCopy := make([]byte, len(value))
		copy(valueCopy, value)

		// Try to send non-blocking
		select {
		case watcher <- valueCopy:
		default:
			// Channel is full, drop the update
		}
	}
}
