package resilience

import (
	"context"
	"errors"
	"math/rand"
	"sync"
	"time"

	"go.uber.org/zap"
)

var (
	// ErrEmergencyMode is returned when system is in emergency mode
	ErrEmergencyMode = errors.New("system in emergency mode")
)

// DegradationLevel represents system degradation levels
type DegradationLevel int

const (
	// LevelNormal indicates normal operation
	LevelNormal DegradationLevel = iota
	// LevelDegraded indicates degraded performance
	LevelDegraded
	// LevelSeverelyDegraded indicates severely degraded performance
	LevelSeverelyDegraded
	// LevelEmergency indicates emergency mode
	LevelEmergency
)

// String returns string representation
func (dl DegradationLevel) String() string {
	switch dl {
	case LevelNormal:
		return "normal"
	case LevelDegraded:
		return "degraded"
	case LevelSeverelyDegraded:
		return "severely_degraded"
	case LevelEmergency:
		return "emergency"
	default:
		return "unknown"
	}
}

// DegradationManager manages graceful degradation
type DegradationManager struct {
	name          string
	currentLevel  DegradationLevel
	componentLevels map[string]DegradationLevel
	logger        *zap.Logger
	mu            sync.RWMutex

	// Callbacks
	onLevelChange func(old, new DegradationLevel)

	// Metrics
	levelChanges     int64
	timeInDegraded   time.Duration
	lastLevelChange  time.Time
	levelStartTime   time.Time
}

// NewDegradationManager creates a new degradation manager
func NewDegradationManager(name string, logger *zap.Logger) *DegradationManager {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &DegradationManager{
		name:            name,
		currentLevel:    LevelNormal,
		componentLevels: make(map[string]DegradationLevel),
		logger:          logger,
		levelStartTime:  time.Now(),
	}
}

// SetComponentLevel sets degradation level for a component
func (dm *DegradationManager) SetComponentLevel(component string, level DegradationLevel) {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	oldLevel := dm.componentLevels[component]
	dm.componentLevels[component] = level

	dm.logger.Info("Component degradation level changed",
		zap.String("manager", dm.name),
		zap.String("component", component),
		zap.String("oldLevel", oldLevel.String()),
		zap.String("newLevel", level.String()))

	// Recalculate overall level
	dm.recalculateLevel()
}

// GetComponentLevel returns degradation level for a component
func (dm *DegradationManager) GetComponentLevel(component string) DegradationLevel {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	level, exists := dm.componentLevels[component]
	if !exists {
		return LevelNormal
	}
	return level
}

// GetLevel returns current overall degradation level
func (dm *DegradationManager) GetLevel() DegradationLevel {
	dm.mu.RLock()
	defer dm.mu.RUnlock()
	return dm.currentLevel
}

// SetLevelChangeCallback sets callback for level changes
func (dm *DegradationManager) SetLevelChangeCallback(fn func(old, new DegradationLevel)) {
	dm.onLevelChange = fn
}

// recalculateLevel recalculates overall degradation level
func (dm *DegradationManager) recalculateLevel() {
	// Find maximum degradation level across all components
	maxLevel := LevelNormal
	for _, level := range dm.componentLevels {
		if level > maxLevel {
			maxLevel = level
		}
	}

	oldLevel := dm.currentLevel
	if maxLevel != oldLevel {
		dm.currentLevel = maxLevel
		dm.levelChanges++
		dm.lastLevelChange = time.Now()

		// Track time in degraded state
		if oldLevel != LevelNormal {
			dm.timeInDegraded += time.Since(dm.levelStartTime)
		}
		dm.levelStartTime = time.Now()

		dm.logger.Warn("System degradation level changed",
			zap.String("manager", dm.name),
			zap.String("oldLevel", oldLevel.String()),
			zap.String("newLevel", maxLevel.String()))

		// Call callback
		if dm.onLevelChange != nil {
			go dm.onLevelChange(oldLevel, maxLevel)
		}
	}
}

// Execute runs functions based on degradation level
func (dm *DegradationManager) Execute(normal, degraded, emergency func() error) error {
	level := dm.GetLevel()

	switch level {
	case LevelNormal, LevelDegraded:
		if normal != nil {
			return normal()
		}
		return nil

	case LevelSeverelyDegraded:
		if degraded != nil {
			return degraded()
		}
		if normal != nil {
			return normal()
		}
		return nil

	case LevelEmergency:
		if emergency != nil {
			return emergency()
		}
		return ErrEmergencyMode
	}

	return nil
}

// ExecuteWithFallback executes with automatic fallback based on level
func (dm *DegradationManager) ExecuteWithFallback(ctx context.Context, operations map[DegradationLevel]func(context.Context) error) error {
	level := dm.GetLevel()

	// Try current level
	if op, exists := operations[level]; exists && op != nil {
		return op(ctx)
	}

	// Fallback to lower levels
	for l := level; l <= LevelEmergency; l++ {
		if op, exists := operations[l]; exists && op != nil {
			dm.logger.Debug("Falling back to degraded operation",
				zap.String("manager", dm.name),
				zap.String("targetLevel", level.String()),
				zap.String("actualLevel", l.String()))
			return op(ctx)
		}
	}

	return ErrEmergencyMode
}

// ShouldDegrade determines if operation should degrade based on metrics
func (dm *DegradationManager) ShouldDegrade(component string, errorRate, latency float64, errorThreshold, latencyThreshold float64) DegradationLevel {
	if errorRate > errorThreshold*2 || latency > latencyThreshold*2 {
		return LevelEmergency
	} else if errorRate > errorThreshold*1.5 || latency > latencyThreshold*1.5 {
		return LevelSeverelyDegraded
	} else if errorRate > errorThreshold || latency > latencyThreshold {
		return LevelDegraded
	}
	return LevelNormal
}

// AutoDegrade automatically degrades component based on metrics
func (dm *DegradationManager) AutoDegrade(component string, errorRate, latency float64, errorThreshold, latencyThreshold float64) {
	level := dm.ShouldDegrade(component, errorRate, latency, errorThreshold, latencyThreshold)
	dm.SetComponentLevel(component, level)
}

// Recover attempts to recover component to normal level
func (dm *DegradationManager) Recover(component string) {
	dm.SetComponentLevel(component, LevelNormal)
}

// RecoverAll recovers all components to normal level
func (dm *DegradationManager) RecoverAll() {
	dm.mu.Lock()
	defer dm.mu.Unlock()

	for component := range dm.componentLevels {
		dm.componentLevels[component] = LevelNormal
	}

	dm.recalculateLevel()
	dm.logger.Info("All components recovered to normal",
		zap.String("manager", dm.name))
}

// GetMetrics returns degradation manager metrics
func (dm *DegradationManager) GetMetrics() DegradationMetrics {
	dm.mu.RLock()
	defer dm.mu.RUnlock()

	componentLevels := make(map[string]string)
	for component, level := range dm.componentLevels {
		componentLevels[component] = level.String()
	}

	timeInCurrentLevel := time.Since(dm.levelStartTime)
	if dm.currentLevel != LevelNormal {
		timeInCurrentLevel = dm.timeInDegraded + timeInCurrentLevel
	}

	return DegradationMetrics{
		Name:               dm.name,
		CurrentLevel:       dm.currentLevel.String(),
		ComponentLevels:    componentLevels,
		LevelChanges:       dm.levelChanges,
		TimeInDegraded:     dm.timeInDegraded,
		TimeInCurrentLevel: timeInCurrentLevel,
		LastLevelChange:    dm.lastLevelChange,
	}
}

// FeatureFlagManager manages feature flags for graceful degradation
type FeatureFlagManager struct {
	name   string
	flags  map[string]bool
	logger *zap.Logger
	mu     sync.RWMutex
}

// NewFeatureFlagManager creates a new feature flag manager
func NewFeatureFlagManager(name string, logger *zap.Logger) *FeatureFlagManager {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &FeatureFlagManager{
		name:   name,
		flags:  make(map[string]bool),
		logger: logger,
	}
}

// SetFlag sets a feature flag
func (ffm *FeatureFlagManager) SetFlag(feature string, enabled bool) {
	ffm.mu.Lock()
	defer ffm.mu.Unlock()

	ffm.flags[feature] = enabled
	ffm.logger.Info("Feature flag updated",
		zap.String("manager", ffm.name),
		zap.String("feature", feature),
		zap.Bool("enabled", enabled))
}

// IsEnabled checks if a feature is enabled
func (ffm *FeatureFlagManager) IsEnabled(feature string) bool {
	ffm.mu.RLock()
	defer ffm.mu.RUnlock()

	enabled, exists := ffm.flags[feature]
	if !exists {
		return true // Features enabled by default
	}
	return enabled
}

// DisableFeature disables a feature
func (ffm *FeatureFlagManager) DisableFeature(feature string) {
	ffm.SetFlag(feature, false)
}

// EnableFeature enables a feature
func (ffm *FeatureFlagManager) EnableFeature(feature string) {
	ffm.SetFlag(feature, true)
}

// GetAllFlags returns all feature flags
func (ffm *FeatureFlagManager) GetAllFlags() map[string]bool {
	ffm.mu.RLock()
	defer ffm.mu.RUnlock()

	flags := make(map[string]bool, len(ffm.flags))
	for k, v := range ffm.flags {
		flags[k] = v
	}
	return flags
}

// LoadSheddingManager manages load shedding
type LoadSheddingManager struct {
	name             string
	enabled          bool
	shedProbability  float64
	maxLoad          float64
	currentLoad      float64
	logger           *zap.Logger
	mu               sync.RWMutex

	// Metrics
	totalRequests   int64
	shedRequests    int64
}

// NewLoadSheddingManager creates a new load shedding manager
func NewLoadSheddingManager(name string, maxLoad float64, logger *zap.Logger) *LoadSheddingManager {
	if logger == nil {
		logger = zap.NewNop()
	}

	return &LoadSheddingManager{
		name:    name,
		enabled: false,
		maxLoad: maxLoad,
		logger:  logger,
	}
}

// Enable enables load shedding
func (lsm *LoadSheddingManager) Enable() {
	lsm.mu.Lock()
	defer lsm.mu.Unlock()

	lsm.enabled = true
	lsm.logger.Warn("Load shedding enabled",
		zap.String("manager", lsm.name))
}

// Disable disables load shedding
func (lsm *LoadSheddingManager) Disable() {
	lsm.mu.Lock()
	defer lsm.mu.Unlock()

	lsm.enabled = false
	lsm.logger.Info("Load shedding disabled",
		zap.String("manager", lsm.name))
}

// UpdateLoad updates current load
func (lsm *LoadSheddingManager) UpdateLoad(load float64) {
	lsm.mu.Lock()
	defer lsm.mu.Unlock()

	lsm.currentLoad = load

	// Calculate shed probability based on load
	if load > lsm.maxLoad {
		lsm.shedProbability = (load - lsm.maxLoad) / lsm.maxLoad
		if lsm.shedProbability > 0.9 {
			lsm.shedProbability = 0.9 // Cap at 90%
		}
	} else {
		lsm.shedProbability = 0
	}
}

// ShouldShed determines if request should be shed
func (lsm *LoadSheddingManager) ShouldShed() bool {
	lsm.mu.RLock()
	defer lsm.mu.RUnlock()

	lsm.totalRequests++

	if !lsm.enabled || lsm.shedProbability == 0 {
		return false
	}

	// Probabilistic load shedding
	if randFloat64() < lsm.shedProbability {
		lsm.shedRequests++
		lsm.logger.Debug("Load shedding request",
			zap.String("manager", lsm.name),
			zap.Float64("load", lsm.currentLoad),
			zap.Float64("probability", lsm.shedProbability))
		return true
	}

	return false
}

// GetMetrics returns load shedding metrics
func (lsm *LoadSheddingManager) GetMetrics() LoadSheddingMetrics {
	lsm.mu.RLock()
	defer lsm.mu.RUnlock()

	shedRate := float64(0)
	if lsm.totalRequests > 0 {
		shedRate = float64(lsm.shedRequests) / float64(lsm.totalRequests)
	}

	return LoadSheddingMetrics{
		Name:            lsm.name,
		Enabled:         lsm.enabled,
		CurrentLoad:     lsm.currentLoad,
		MaxLoad:         lsm.maxLoad,
		ShedProbability: lsm.shedProbability,
		TotalRequests:   lsm.totalRequests,
		ShedRequests:    lsm.shedRequests,
		ShedRate:        shedRate,
	}
}

// Metrics types

// DegradationMetrics contains degradation manager metrics
type DegradationMetrics struct {
	Name               string
	CurrentLevel       string
	ComponentLevels    map[string]string
	LevelChanges       int64
	TimeInDegraded     time.Duration
	TimeInCurrentLevel time.Duration
	LastLevelChange    time.Time
}

// LoadSheddingMetrics contains load shedding metrics
type LoadSheddingMetrics struct {
	Name            string
	Enabled         bool
	CurrentLoad     float64
	MaxLoad         float64
	ShedProbability float64
	TotalRequests   int64
	ShedRequests    int64
	ShedRate        float64
}

// Helper for random number generation
var dmRand = struct {
	r  *rand.Rand
	mu sync.Mutex
}{
	r: rand.New(rand.NewSource(time.Now().UnixNano())),
}

// Float64 returns a random float64
func randFloat64() float64 {
	dmRand.mu.Lock()
	defer dmRand.mu.Unlock()
	return dmRand.r.Float64()
}