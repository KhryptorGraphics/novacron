package performance

import "time"

// PerformanceConfig defines auto-tuning configuration
type PerformanceConfig struct {
	Profiling             ProfilingConfig
	AutoRightSizing       bool
	AutoCPUPinning        bool
	AutoNumaOptimization  bool
	AutoIOTuning          bool
	AutoNetworkTuning     bool
	TuningConvergence     time.Duration // 30 minutes
	OverheadTarget        float64       // 0.02 (2%)
	MaxConcurrentTuning   int           // 5
	TuningInterval        time.Duration // 5 minutes
	ValidationPeriod      time.Duration // 10 minutes
	RollbackThreshold     float64       // 0.1 (10% degradation)
}

// ProfilingConfig defines profiling settings
type ProfilingConfig struct {
	Enabled       bool
	SamplingRate  int      // 100 Hz
	Targets       []string // ["cpu", "memory", "mutex", "block", "goroutine"]
	RetentionDays int      // 7
	OutputDir     string
	CPUProfile    bool
	MemProfile    bool
	MutexProfile  bool
	BlockProfile  bool
	GoRoutine     bool
	Heap          bool
	Allocs        bool
}

// RightSizingConfig defines VM right-sizing parameters
type RightSizingConfig struct {
	Enabled             bool
	CPUTargetMin        float64 // 0.60 (60%)
	CPUTargetMax        float64 // 0.80 (80%)
	MemoryTargetMin     float64 // 0.70 (70%)
	MemoryTargetMax     float64 // 0.85 (85%)
	ObservationPeriod   time.Duration
	ConfidenceThreshold float64 // 0.90
	CostSavingsMin      float64 // 0.10 (10%)
}

// NumaConfig defines NUMA optimization settings
type NumaConfig struct {
	Enabled                 bool
	AutoTopologyDetection   bool
	MemoryPlacementStrategy string // "local", "interleave", "preferred"
	CacheLocalityOptimize   bool
	CrossNumaTrafficTarget  float64 // 0.10 (10%)
}

// CPUPinningConfig defines CPU affinity settings
type CPUPinningConfig struct {
	Enabled          bool
	Strategy         string  // "dedicated", "shared", "mixed"
	OvercommitRatio  float64 // 1.0 (no overcommit), 2.0 (2:1)
	HyperthreadingOpt bool
	CacheAffinity    bool
	IsolateNoisy     bool
}

// IOTuningConfig defines I/O optimization settings
type IOTuningConfig struct {
	Enabled            bool
	AutoSchedulerSelect bool
	Schedulers         []string // ["noop", "deadline", "cfq", "bfq"]
	QueueDepthAuto     bool
	ReadAheadAuto      bool
	PrioritizationAuto bool
}

// NetworkTuningConfig defines network optimization settings
type NetworkTuningConfig struct {
	Enabled              bool
	TCPWindowAutoTune    bool
	CongestionControl    string // "bbr", "cubic", "reno"
	BufferAutoSize       bool
	RDMAOptimize         bool
	RingBufferAutoSize   bool
	OffloadOptimize      bool
}

// CostOptimizerConfig defines cost optimization settings
type CostOptimizerConfig struct {
	Enabled                bool
	MultiObjectiveOptimize bool
	ParetoFrontierAnalysis bool
	SpotInstanceRecommend  bool
	ReservedInstancePlan   bool
	SavingsPlansAnalyze    bool
	CostPredictionEnabled  bool
	SLAConstraints         map[string]float64
}

// BenchmarkConfig defines benchmark suite settings
type BenchmarkConfig struct {
	Enabled           bool
	SyntheticTests    []string // ["cpu", "memory", "io", "network"]
	ApplicationTests  []string
	CompareBaseline   bool
	RegressionDetect  bool
	AutoRun           bool
	RunInterval       time.Duration
}

// DefaultConfig returns default performance configuration
func DefaultConfig() *PerformanceConfig {
	return &PerformanceConfig{
		Profiling: ProfilingConfig{
			Enabled:       true,
			SamplingRate:  100,
			Targets:       []string{"cpu", "memory", "mutex", "block"},
			RetentionDays: 7,
			OutputDir:     "/var/lib/novacron/profiles",
			CPUProfile:    true,
			MemProfile:    true,
			MutexProfile:  true,
			BlockProfile:  true,
		},
		AutoRightSizing:      true,
		AutoCPUPinning:       true,
		AutoNumaOptimization: true,
		AutoIOTuning:         true,
		AutoNetworkTuning:    true,
		TuningConvergence:    30 * time.Minute,
		OverheadTarget:       0.02,
		MaxConcurrentTuning:  5,
		TuningInterval:       5 * time.Minute,
		ValidationPeriod:     10 * time.Minute,
		RollbackThreshold:    0.1,
	}
}

// RightSizingDefaults returns default right-sizing config
func RightSizingDefaults() *RightSizingConfig {
	return &RightSizingConfig{
		Enabled:             true,
		CPUTargetMin:        0.60,
		CPUTargetMax:        0.80,
		MemoryTargetMin:     0.70,
		MemoryTargetMax:     0.85,
		ObservationPeriod:   24 * time.Hour,
		ConfidenceThreshold: 0.90,
		CostSavingsMin:      0.10,
	}
}

// NumaDefaults returns default NUMA config
func NumaDefaults() *NumaConfig {
	return &NumaConfig{
		Enabled:                 true,
		AutoTopologyDetection:   true,
		MemoryPlacementStrategy: "local",
		CacheLocalityOptimize:   true,
		CrossNumaTrafficTarget:  0.10,
	}
}

// CPUPinningDefaults returns default CPU pinning config
func CPUPinningDefaults() *CPUPinningConfig {
	return &CPUPinningConfig{
		Enabled:          true,
		Strategy:         "mixed",
		OvercommitRatio:  1.5,
		HyperthreadingOpt: true,
		CacheAffinity:    true,
		IsolateNoisy:     true,
	}
}
