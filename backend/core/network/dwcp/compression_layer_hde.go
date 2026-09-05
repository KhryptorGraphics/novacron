// compression_layer_hde.go — HDE-backed implementation of the DWCP
// CompressionLayer interface (novacron-349).
//
// The DWCP Manager's compression field was never assigned: DefaultConfig()
// ships Compression.Enabled == true, but nothing constructed a
// CompressionLayer, so startPhase0Components silently skipped the
// "Compression Layer (HDE)" step and the manager's compression metrics stayed
// zero forever. This adapter wires the package's existing HDE engine behind
// the CompressionLayer contract so the enabled-by-default configuration
// actually compresses.
//
// Lifecycle ownership lives here rather than in HDE itself: NewHDE constructs
// the engine already running (its context is set in the constructor), and
// HDE.Stop() releases the zstd codec maps that HDE.Start() never rebuilds —
// so a single shared instance cannot survive a Stop→Start cycle (compression
// would fail with "encoder not found"). The adapter therefore builds a fresh
// HDE on every Start, which also makes the manager's component-recovery path
// (Stop followed by Start) correct.
package dwcp

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// Compile-time interface conformance check.
var _ CompressionLayer = (*hdeCompressionLayer)(nil)

// hdeCompressionLayer adapts the root-package HDE engine to the
// CompressionLayer interface consumed by the DWCP Manager.
type hdeCompressionLayer struct {
	// cfg is the HDE configuration every Start rebuilds the engine from.
	cfg HDEConfig

	// level is the configured DWCP compression level, reported in metrics.
	level CompressionLevel

	// mu guards hde across lifecycle calls (Start/Stop, which race with
	// Encode/Decode/GetMetrics through the manager's health-monitoring
	// recovery goroutines).
	mu  sync.RWMutex
	hde *HDE
}

// newHDECompressionLayer builds a compression layer from the manager's
// CompressionConfig. Only the delta/dictionary knobs map onto HDEConfig;
// every other HDEConfig field is left zero so NewHDE applies its documented
// defaults.
func newHDECompressionLayer(cfg CompressionConfig) (*hdeCompressionLayer, error) {
	hdeCfg := HDEConfig{
		EnableDelta:      cfg.EnableDeltaEncoding,
		EnableDictionary: cfg.EnableDictionary,
		MaxDeltaHistory:  cfg.MaxDeltaChain,
	}

	// No probe construction here: NewHDE immediately spawns its cleanup
	// goroutine (hde.go cleanupLoop) and Stop must not race it — the layer
	// only ever constructs engines inside Start, and a broken HDE
	// configuration surfaces as a Phase 0 startup error there (which still
	// fails the manager deterministically).
	return &hdeCompressionLayer{
		cfg:   hdeCfg,
		level: cfg.Level,
	}, nil
}

// Start starts the compression layer by building a fresh HDE engine.
// Implements Lifecycle. The ctx parameter is part of the Lifecycle contract;
// HDE wires its own internal context in NewHDE, and shutdown is driven by the
// manager's explicit Stop path (the same way the transport layer is handled),
// so ctx cancellation is honored through Manager.Stop rather than here.
func (h *hdeCompressionLayer) Start(ctx context.Context) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.hde != nil {
		return nil // already running; Stop nils the engine
	}

	engine, err := NewHDE(h.cfg)
	if err != nil {
		return fmt.Errorf("failed to construct HDE compression engine: %w", err)
	}
	h.hde = engine
	return nil
}

// Stop shuts down the compression layer and releases the HDE engine.
// Implements Lifecycle. Idempotent.
func (h *hdeCompressionLayer) Stop() error {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.hde == nil {
		return nil
	}
	err := h.hde.Stop()
	h.hde = nil
	return err
}

// IsRunning reports whether the layer has a running HDE engine.
// Implements Lifecycle.
func (h *hdeCompressionLayer) IsRunning() bool {
	h.mu.RLock()
	defer h.mu.RUnlock()
	return h.hde != nil && h.hde.IsRunning()
}

// HealthCheck returns nil while the layer is running.
// Implements HealthChecker.
func (h *hdeCompressionLayer) HealthCheck() error {
	if !h.IsRunning() {
		return errors.New("hde compression layer not running")
	}
	return nil
}

// IsHealthy reports whether the layer is healthy.
// Implements HealthChecker.
func (h *hdeCompressionLayer) IsHealthy() bool {
	return h.HealthCheck() == nil
}

// Encode compresses data with HDE at the given tier.
// Implements CompressionLayer.
func (h *hdeCompressionLayer) Encode(key string, data []byte, tier int) (*EncodedData, error) {
	h.mu.RLock()
	hde := h.hde
	h.mu.RUnlock()

	if hde == nil || !hde.IsRunning() {
		return nil, errors.New("hde compression layer not running")
	}

	out, err := hde.CompressMemory(key, data, CompressionLevel(tier))
	if err != nil {
		return nil, fmt.Errorf("hde compression failed: %w", err)
	}
	return &EncodedData{
		Data:           out,
		OriginalSize:   len(data),
		CompressedSize: len(out),
		Tier:           tier,
		Timestamp:      time.Now(),
	}, nil
}

// Decode decompresses data produced by Encode.
// Implements CompressionLayer.
func (h *hdeCompressionLayer) Decode(key string, data *EncodedData) ([]byte, error) {
	h.mu.RLock()
	hde := h.hde
	h.mu.RUnlock()

	if hde == nil || !hde.IsRunning() {
		return nil, errors.New("hde compression layer not running")
	}
	if data == nil {
		return nil, errors.New("no encoded data to decode")
	}
	return hde.Decompress(data.Data)
}

// GetMetrics maps the HDE engine's metrics onto CompressionMetrics.
// Implements CompressionLayer. Returns nil while the layer is stopped.
func (h *hdeCompressionLayer) GetMetrics() *CompressionMetrics {
	h.mu.RLock()
	hde := h.hde
	h.mu.RUnlock()

	if hde == nil {
		return nil
	}

	raw := hde.GetMetrics()
	metrics := &CompressionMetrics{
		Level:     h.level,
		Timestamp: time.Now(),
	}
	// Type-assert each key to the concrete atomic Load() types HDE stores;
	// a missing or unexpected key leaves the zero value.
	if v, ok := raw["bytes_original"].(int64); ok {
		metrics.BytesIn = uint64(v)
	}
	if v, ok := raw["bytes_compressed"].(int64); ok {
		metrics.BytesOut = uint64(v)
	}
	if v, ok := raw["compression_ratio"].(float64); ok {
		metrics.CompressionRatio = v
	}
	if v, ok := raw["delta_hit_rate"].(float64); ok {
		metrics.DeltaHitRate = v
	}
	return metrics
}
