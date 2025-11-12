package transport

import (
	"math"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
)

// CongestionController implements BBR and CUBIC congestion control algorithms
type CongestionController struct {
	algorithm string // "bbr" or "cubic"

	// BBR state
	bbrState        *BBRState
	bbrPacingGain   float64
	bbrCwndGain     float64

	// CUBIC state
	cubicState      *CUBICState

	// Common state
	cwnd            atomic.Int32   // congestion window (packets)
	ssthresh        atomic.Int32   // slow start threshold
	rtt             atomic.Int64   // RTT in microseconds
	minRTT          atomic.Int64   // minimum RTT observed
	bandwidth       atomic.Int64   // bytes per second
	maxBandwidth    atomic.Int64   // maximum bandwidth observed
	packetsInFlight atomic.Int32   // current packets in flight
	packetsSent     atomic.Uint64  // total packets sent
	packetsAcked    atomic.Uint64  // total packets acknowledged
	packetsLost     atomic.Uint64  // total packets lost

	// Pacing
	pacingRate      int64          // bytes per second
	lastSendTime    atomic.Value   // time.Time

	// Synchronization
	mu              sync.RWMutex
	logger          *zap.Logger
}

// BBRState represents BBR congestion control state
type BBRState struct {
	// BBR phases
	mode            BBRMode
	cycleIndex      int
	cycleStart      time.Time

	// Bandwidth estimation
	btlBW           int64          // bottleneck bandwidth
	btlBWFilter     []int64        // bandwidth filter
	roundCount      uint64
	roundStart      bool

	// RTT estimation
	probeRTT        bool
	probeRTTStart   time.Time
	probeRTTRounds  int
	idleRestart     bool

	// Delivery rate
	delivered       uint64
	deliveredTime   time.Time
}

// CUBICState represents CUBIC congestion control state
type CUBICState struct {
	// CUBIC parameters
	beta            float64        // multiplicative decrease factor
	c               float64        // CUBIC constant
	wMax            float64        // max window before reduction
	k               float64        // time to reach wMax
	originPoint     float64        // origin point of CUBIC function
	epoch           time.Time      // epoch start time
	tcpFriendly     bool           // TCP friendly mode
	fastConvergence bool           // fast convergence mode
	lastCwnd        float64        // last congestion window
	lastTime        time.Time      // last update time
}

// BBRMode represents BBR algorithm phases
type BBRMode int

const (
	BBRModeStartup BBRMode = iota
	BBRModeDrain
	BBRModeProbeBW
	BBRModeProbeRTT
)

// NewCongestionController creates a new congestion controller
func NewCongestionController(algorithm string, pacingRate int64, logger *zap.Logger) *CongestionController {
	if logger == nil {
		logger, _ = zap.NewProduction()
	}

	cc := &CongestionController{
		algorithm:  algorithm,
		pacingRate: pacingRate,
		logger:     logger,
	}

	// Initialize congestion window
	cc.cwnd.Store(10) // Initial window: 10 packets
	cc.ssthresh.Store(65535) // Large initial ssthresh

	cc.lastSendTime.Store(time.Now())

	switch algorithm {
	case "bbr":
		cc.initBBR()
	case "cubic":
		cc.initCUBIC()
	default:
		logger.Warn("Unknown congestion algorithm, defaulting to BBR",
			zap.String("algorithm", algorithm))
		cc.algorithm = "bbr"
		cc.initBBR()
	}

	return cc
}

// initBBR initializes BBR state
func (cc *CongestionController) initBBR() {
	cc.bbrState = &BBRState{
		mode:          BBRModeStartup,
		cycleIndex:    0,
		cycleStart:    time.Now(),
		btlBWFilter:   make([]int64, 0, 10),
		roundStart:    true,
		delivered:     0,
		deliveredTime: time.Now(),
	}

	// BBR pacing and cwnd gains for different phases
	cc.bbrPacingGain = 2.77 // High gain during startup
	cc.bbrCwndGain = 2.0

	cc.logger.Info("BBR congestion control initialized")
}

// initCUBIC initializes CUBIC state
func (cc *CongestionController) initCUBIC() {
	cc.cubicState = &CUBICState{
		beta:            0.7,  // 30% reduction
		c:               0.4,  // CUBIC constant
		wMax:            0,
		k:               0,
		originPoint:     0,
		epoch:           time.Now(),
		tcpFriendly:     true,
		fastConvergence: true,
		lastCwnd:        10,
		lastTime:        time.Now(),
	}

	cc.logger.Info("CUBIC congestion control initialized")
}

// GetSendDelay calculates delay before sending based on pacing
func (cc *CongestionController) GetSendDelay(packetSize int) time.Duration {
	if cc.pacingRate <= 0 {
		return 0
	}

	lastSend := cc.lastSendTime.Load().(time.Time)
	now := time.Now()

	// Calculate ideal inter-packet gap
	pacingDelay := time.Duration(float64(packetSize) / float64(cc.pacingRate) * float64(time.Second))

	// Calculate actual elapsed time since last send
	elapsed := now.Sub(lastSend)

	// If we're behind schedule, send immediately
	if elapsed >= pacingDelay {
		return 0
	}

	// Otherwise, wait for the remaining time
	return pacingDelay - elapsed
}

// OnPacketSent notifies controller of packet send
func (cc *CongestionController) OnPacketSent(packetSize int) {
	cc.packetsInFlight.Add(1)
	cc.packetsSent.Add(1)
	cc.lastSendTime.Store(time.Now())

	// Update BBR delivered counter
	if cc.algorithm == "bbr" && cc.bbrState != nil {
		cc.mu.Lock()
		cc.bbrState.delivered++
		cc.mu.Unlock()
	}
}

// OnPacketAcked notifies controller of packet acknowledgment
func (cc *CongestionController) OnPacketAcked(packetSize int, rtt time.Duration) {
	cc.packetsInFlight.Add(-1)
	cc.packetsAcked.Add(1)

	// Update RTT measurements
	rttMicros := rtt.Microseconds()
	cc.rtt.Store(rttMicros)

	// Update min RTT
	currentMinRTT := cc.minRTT.Load()
	if currentMinRTT == 0 || rttMicros < currentMinRTT {
		cc.minRTT.Store(rttMicros)
	}

	// Calculate bandwidth
	elapsed := time.Since(cc.lastSendTime.Load().(time.Time))
	if elapsed > 0 {
		bw := int64(float64(packetSize) / elapsed.Seconds())
		cc.bandwidth.Store(bw)

		// Update max bandwidth
		currentMaxBW := cc.maxBandwidth.Load()
		if bw > currentMaxBW {
			cc.maxBandwidth.Store(bw)
		}
	}

	// Update congestion window based on algorithm
	switch cc.algorithm {
	case "bbr":
		cc.updateBBR(packetSize, rtt)
	case "cubic":
		cc.updateCUBIC(packetSize, rtt)
	}
}

// OnPacketLost notifies controller of packet loss
func (cc *CongestionController) OnPacketLost(packetSize int) {
	cc.packetsInFlight.Add(-1)
	cc.packetsLost.Add(1)

	// Handle loss based on algorithm
	switch cc.algorithm {
	case "bbr":
		cc.handleBBRLoss()
	case "cubic":
		cc.handleCUBICLoss()
	}
}

// updateBBR updates BBR state
func (cc *CongestionController) updateBBR(packetSize int, rtt time.Duration) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if cc.bbrState == nil {
		return
	}

	state := cc.bbrState

	// Update bandwidth estimate
	bw := cc.bandwidth.Load()
	state.btlBWFilter = append(state.btlBWFilter, bw)
	if len(state.btlBWFilter) > 10 {
		state.btlBWFilter = state.btlBWFilter[1:]
	}

	// Calculate bottleneck bandwidth (max of recent measurements)
	maxBW := int64(0)
	for _, bw := range state.btlBWFilter {
		if bw > maxBW {
			maxBW = bw
		}
	}
	state.btlBW = maxBW

	// Update BBR mode
	cc.updateBBRMode()

	// Update pacing rate based on mode and bandwidth
	bdp := float64(state.btlBW) * float64(cc.minRTT.Load()) / 1e6 // BDP = BW * RTT
	targetCwnd := int32(bdp * cc.bbrCwndGain)

	if targetCwnd < 4 {
		targetCwnd = 4 // Minimum window
	}

	cc.cwnd.Store(targetCwnd)

	// Update pacing rate
	cc.pacingRate = int64(float64(state.btlBW) * cc.bbrPacingGain)
}

// updateBBRMode updates BBR mode (Startup -> Drain -> ProbeBW -> ProbeRTT)
func (cc *CongestionController) updateBBRMode() {
	state := cc.bbrState
	if state == nil {
		return
	}

	switch state.mode {
	case BBRModeStartup:
		// Exit startup if bandwidth plateaus
		if cc.isStartupComplete() {
			state.mode = BBRModeDrain
			cc.bbrPacingGain = 0.75 // Drain excess queue
			cc.bbrCwndGain = 1.0
			cc.logger.Debug("BBR: Startup -> Drain")
		}

	case BBRModeDrain:
		// Exit drain when inflight <= BDP
		inflight := cc.packetsInFlight.Load()
		bdp := cc.calculateBDP()
		if float64(inflight) <= bdp {
			state.mode = BBRModeProbeBW
			state.cycleStart = time.Now()
			cc.bbrPacingGain = 1.0
			cc.bbrCwndGain = 2.0
			cc.logger.Debug("BBR: Drain -> ProbeBW")
		}

	case BBRModeProbeBW:
		// Cycle through pacing gains to probe bandwidth
		if time.Since(state.cycleStart) > cc.getRTTProbeDuration() {
			state.cycleIndex = (state.cycleIndex + 1) % 8
			state.cycleStart = time.Now()

			// Cycle gains: [1.25, 0.75, 1, 1, 1, 1, 1, 1]
			gains := []float64{1.25, 0.75, 1, 1, 1, 1, 1, 1}
			cc.bbrPacingGain = gains[state.cycleIndex]
		}

		// Periodically enter ProbeRTT
		if time.Since(state.cycleStart) > 10*time.Second {
			state.mode = BBRModeProbeRTT
			state.probeRTTStart = time.Now()
			cc.logger.Debug("BBR: ProbeBW -> ProbeRTT")
		}

	case BBRModeProbeRTT:
		// Reduce cwnd to probe min RTT
		cc.cwnd.Store(4) // Minimum window

		// Exit ProbeRTT after 200ms
		if time.Since(state.probeRTTStart) > 200*time.Millisecond {
			state.mode = BBRModeProbeBW
			state.cycleStart = time.Now()
			cc.bbrPacingGain = 1.0
			cc.logger.Debug("BBR: ProbeRTT -> ProbeBW")
		}
	}
}

// isStartupComplete checks if BBR startup phase is complete
func (cc *CongestionController) isStartupComplete() bool {
	// Exit startup if bandwidth hasn't grown by 25% in 3 rounds
	maxBW := cc.maxBandwidth.Load()
	currentBW := cc.bandwidth.Load()

	return float64(currentBW) < float64(maxBW)*1.25
}

// calculateBDP calculates bandwidth-delay product
func (cc *CongestionController) calculateBDP() float64 {
	bw := float64(cc.bandwidth.Load())
	rtt := float64(cc.minRTT.Load()) / 1e6 // Convert to seconds
	return bw * rtt / 1500 // Packets (assuming 1500 byte MTU)
}

// getRTTProbeDuration returns duration for RTT probe
func (cc *CongestionController) getRTTProbeDuration() time.Duration {
	rtt := time.Duration(cc.rtt.Load()) * time.Microsecond
	if rtt < 10*time.Millisecond {
		return 10 * time.Millisecond
	}
	return rtt
}

// handleBBRLoss handles packet loss in BBR
func (cc *CongestionController) handleBBRLoss() {
	// BBR doesn't reduce cwnd on loss unless persistent congestion
	packetsLost := cc.packetsLost.Load()
	packetsSent := cc.packetsSent.Load()

	lossRate := float64(packetsLost) / float64(packetsSent)
	if lossRate > 0.1 { // >10% loss
		cc.logger.Warn("High packet loss in BBR mode",
			zap.Float64("loss_rate", lossRate))
	}
}

// updateCUBIC updates CUBIC state
func (cc *CongestionController) updateCUBIC(packetSize int, rtt time.Duration) {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if cc.cubicState == nil {
		return
	}

	state := cc.cubicState
	now := time.Now()
	t := now.Sub(state.epoch).Seconds()

	// CUBIC function: W(t) = C * (t - K)^3 + Wmax
	cwnd := state.c * math.Pow(t-state.k, 3) + state.wMax

	// TCP friendly check
	if state.tcpFriendly {
		tcpCwnd := state.wMax*state.beta + (3*(1-state.beta)/(1+state.beta)) * t
		if tcpCwnd > cwnd {
			cwnd = tcpCwnd
		}
	}

	// Update congestion window
	if cwnd < 4 {
		cwnd = 4
	}
	cc.cwnd.Store(int32(cwnd))

	state.lastCwnd = cwnd
	state.lastTime = now
}

// handleCUBICLoss handles packet loss in CUBIC
func (cc *CongestionController) handleCUBICLoss() {
	cc.mu.Lock()
	defer cc.mu.Unlock()

	if cc.cubicState == nil {
		return
	}

	state := cc.cubicState
	currentCwnd := float64(cc.cwnd.Load())

	// Fast convergence
	if state.fastConvergence && currentCwnd < state.wMax {
		state.wMax = currentCwnd * (2 - state.beta) / 2
	} else {
		state.wMax = currentCwnd
	}

	// Multiplicative decrease
	newCwnd := currentCwnd * state.beta
	if newCwnd < 4 {
		newCwnd = 4
	}
	cc.cwnd.Store(int32(newCwnd))

	// Calculate K (time to reach Wmax)
	state.k = math.Cbrt((state.wMax - newCwnd) / state.c)
	state.epoch = time.Now()

	cc.logger.Debug("CUBIC multiplicative decrease",
		zap.Float64("old_cwnd", currentCwnd),
		zap.Float64("new_cwnd", newCwnd),
		zap.Float64("wmax", state.wMax))
}

// GetCongestionWindow returns current congestion window
func (cc *CongestionController) GetCongestionWindow() int32 {
	return cc.cwnd.Load()
}

// GetPacketLossRate returns current packet loss rate
func (cc *CongestionController) GetPacketLossRate() float64 {
	sent := cc.packetsSent.Load()
	if sent == 0 {
		return 0
	}
	lost := cc.packetsLost.Load()
	return float64(lost) / float64(sent)
}

// GetRTT returns current RTT
func (cc *CongestionController) GetRTT() time.Duration {
	return time.Duration(cc.rtt.Load()) * time.Microsecond
}

// GetBandwidth returns current bandwidth estimate
func (cc *CongestionController) GetBandwidth() int64 {
	return cc.bandwidth.Load()
}

// GetMetrics returns congestion controller metrics
func (cc *CongestionController) GetMetrics() map[string]interface{} {
	cc.mu.RLock()
	defer cc.mu.RUnlock()

	metrics := map[string]interface{}{
		"algorithm":         cc.algorithm,
		"cwnd":              cc.cwnd.Load(),
		"ssthresh":          cc.ssthresh.Load(),
		"rtt_us":            cc.rtt.Load(),
		"min_rtt_us":        cc.minRTT.Load(),
		"bandwidth_bps":     cc.bandwidth.Load(),
		"max_bandwidth_bps": cc.maxBandwidth.Load(),
		"packets_in_flight": cc.packetsInFlight.Load(),
		"packets_sent":      cc.packetsSent.Load(),
		"packets_acked":     cc.packetsAcked.Load(),
		"packets_lost":      cc.packetsLost.Load(),
		"loss_rate":         cc.GetPacketLossRate(),
		"pacing_rate_bps":   cc.pacingRate,
	}

	if cc.algorithm == "bbr" && cc.bbrState != nil {
		metrics["bbr_mode"] = cc.bbrState.mode
		metrics["bbr_btl_bw"] = cc.bbrState.btlBW
		metrics["bbr_pacing_gain"] = cc.bbrPacingGain
		metrics["bbr_cwnd_gain"] = cc.bbrCwndGain
	}

	if cc.algorithm == "cubic" && cc.cubicState != nil {
		metrics["cubic_wmax"] = cc.cubicState.wMax
		metrics["cubic_k"] = cc.cubicState.k
		metrics["cubic_beta"] = cc.cubicState.beta
	}

	return metrics
}
