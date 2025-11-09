package qos

import (
	"context"
	"fmt"
	"math"
	"sync"
	"time"
)

// TrafficClass represents different types of network traffic
type TrafficClass int

const (
	TrafficRealTime TrafficClass = iota
	TrafficInteractive
	TrafficBulk
	TrafficBestEffort
)

// QoSPolicy defines quality of service parameters
type QoSPolicy struct {
	Class            TrafficClass
	Priority         int     // 0-7, higher is better
	MinBandwidth     float64 // Minimum guaranteed bandwidth (Mbps)
	MaxBandwidth     float64 // Maximum allowed bandwidth (Mbps)
	MaxLatency       float64 // Maximum tolerable latency (ms)
	MaxJitter        float64 // Maximum jitter (ms)
	PacketLossRate   float64 // Maximum acceptable loss rate
	DSCP             int     // Differentiated Services Code Point
}

// FlowFeatures represents features extracted from a network flow
type FlowFeatures struct {
	SrcPort          int
	DstPort          int
	Protocol         string
	PacketSize       int
	InterArrivalTime float64 // Microseconds
	BurstSize        int
	FlowDuration     time.Duration
	PacketCount      int
	ByteCount        int64
	TCPFlags         uint8
	PayloadEntropy   float64 // For encrypted traffic detection
}

// AdaptiveQoS implements ML-based traffic classification and QoS
type AdaptiveQoS struct {
	mu sync.RWMutex

	// ML classifier
	classifier      *TrafficClassifier
	featureExtractor *FeatureExtractor

	// QoS policies
	policies       map[TrafficClass]*QoSPolicy
	dynamicPolicies map[string]*QoSPolicy // Per-application policies

	// Application fingerprints
	fingerprints map[string]ApplicationFingerprint

	// Performance metrics
	classificationCount int64
	classificationAccuracy float64
	policyUpdates      int64
	avgClassificationTime time.Duration

	// Real-time adaptation
	adaptationEnabled bool
	adaptationInterval time.Duration
	lastAdaptation    time.Time
}

// TrafficClassifier uses ML to classify traffic
type TrafficClassifier struct {
	// Random Forest classifier
	trees         []*DecisionTree
	numTrees      int
	maxDepth      int
	minSamples    int
	featureImportance []float64

	// Training data
	trainingData []LabeledFlow
	accuracy     float64
}

// DecisionTree for Random Forest
type DecisionTree struct {
	Root     *TreeNode
	Features []int // Selected features for this tree
}

// TreeNode represents a node in decision tree
type TreeNode struct {
	IsLeaf       bool
	Class        TrafficClass
	FeatureIndex int
	Threshold    float64
	Left         *TreeNode
	Right        *TreeNode
	Samples      int
	Gini         float64
}

// LabeledFlow for training
type LabeledFlow struct {
	Features FlowFeatures
	Class    TrafficClass
}

// ApplicationFingerprint identifies specific applications
type ApplicationFingerprint struct {
	Name            string
	Ports           []int
	PayloadPatterns [][]byte
	TLSSignatures   []string
	BehaviorPattern string // ML-learned pattern
	DefaultClass    TrafficClass
}

// FeatureExtractor extracts features from raw packets
type FeatureExtractor struct {
	windowSize int
	history    []PacketInfo
}

// PacketInfo stores packet metadata
type PacketInfo struct {
	Timestamp time.Time
	Size      int
	SrcPort   int
	DstPort   int
	Protocol  string
	Flags     uint8
}

// NewAdaptiveQoS creates a new adaptive QoS system
func NewAdaptiveQoS() *AdaptiveQoS {
	qos := &AdaptiveQoS{
		policies:           make(map[TrafficClass]*QoSPolicy),
		dynamicPolicies:    make(map[string]*QoSPolicy),
		fingerprints:       make(map[string]ApplicationFingerprint),
		adaptationEnabled:  true,
		adaptationInterval: 30 * time.Second,
	}

	// Initialize default policies
	qos.initializeDefaultPolicies()

	// Initialize classifier
	qos.classifier = &TrafficClassifier{
		numTrees:   100,
		maxDepth:   10,
		minSamples: 5,
	}

	// Initialize feature extractor
	qos.featureExtractor = &FeatureExtractor{
		windowSize: 100,
		history:    make([]PacketInfo, 0, 100),
	}

	// Load application fingerprints
	qos.loadApplicationFingerprints()

	return qos
}

// Initialize initializes the adaptive QoS system
func (q *AdaptiveQoS) Initialize(ctx context.Context) error {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Train classifier with initial dataset
	if err := q.trainClassifier(); err != nil {
		return fmt.Errorf("failed to train classifier: %w", err)
	}

	// Start adaptation loop
	if q.adaptationEnabled {
		go q.adaptationLoop(ctx)
	}

	return nil
}

// ClassifyFlow classifies a network flow using ML
func (q *AdaptiveQoS) ClassifyFlow(ctx context.Context, features FlowFeatures) (TrafficClass, float64, error) {
	start := time.Now()
	defer func() {
		q.updateClassificationMetrics(time.Since(start))
	}()

	q.mu.RLock()
	defer q.mu.RUnlock()

	// Check for application fingerprint match first
	if class, confidence := q.matchFingerprint(features); confidence > 0.9 {
		q.classificationCount++
		return class, confidence, nil
	}

	// Use ML classifier
	class, confidence := q.classifier.classify(features)

	// Update metrics
	q.classificationCount++

	if confidence < 0.5 {
		// Low confidence, use heuristics
		class = q.heuristicClassification(features)
		confidence = 0.6
	}

	return class, confidence, nil
}

// ApplyQoS applies QoS policy to a flow
func (q *AdaptiveQoS) ApplyQoS(flowID string, class TrafficClass, features FlowFeatures) (*QoSPolicy, error) {
	q.mu.RLock()
	policy, exists := q.policies[class]
	q.mu.RUnlock()

	if !exists {
		return nil, fmt.Errorf("no policy for class %v", class)
	}

	// Check for dynamic policy override
	q.mu.RLock()
	if dynPolicy, exists := q.dynamicPolicies[flowID]; exists {
		policy = dynPolicy
	}
	q.mu.RUnlock()

	// Apply traffic shaping
	if err := q.applyTrafficShaping(flowID, policy); err != nil {
		return nil, err
	}

	// Mark packets with DSCP
	if err := q.markPackets(flowID, policy.DSCP); err != nil {
		return nil, err
	}

	return policy, nil
}

// UpdatePolicy dynamically updates QoS policy
func (q *AdaptiveQoS) UpdatePolicy(class TrafficClass, policy *QoSPolicy) {
	q.mu.Lock()
	defer q.mu.Unlock()

	q.policies[class] = policy
	q.policyUpdates++
}

// trainClassifier trains the Random Forest classifier
func (q *AdaptiveQoS) trainClassifier() error {
	// Generate or load training data
	q.classifier.trainingData = q.generateTrainingData()

	if len(q.classifier.trainingData) < 100 {
		return fmt.Errorf("insufficient training data")
	}

	// Build Random Forest
	q.classifier.trees = make([]*DecisionTree, q.classifier.numTrees)

	for i := 0; i < q.classifier.numTrees; i++ {
		// Bootstrap sampling
		sample := q.bootstrapSample(q.classifier.trainingData)

		// Random feature selection
		features := q.selectRandomFeatures(12, 8) // Select 8 out of 12 features

		// Build tree
		tree := &DecisionTree{
			Features: features,
		}
		tree.Root = q.buildTree(sample, features, 0)

		q.classifier.trees[i] = tree
	}

	// Calculate feature importance
	q.classifier.calculateFeatureImportance()

	// Validate accuracy
	q.validateClassifier()

	return nil
}

// classify performs classification using Random Forest
func (c *TrafficClassifier) classify(features FlowFeatures) (TrafficClass, float64) {
	votes := make(map[TrafficClass]int)

	// Get votes from all trees
	for _, tree := range c.trees {
		class := tree.classify(features)
		votes[class]++
	}

	// Find majority class
	var maxVotes int
	var bestClass TrafficClass

	for class, count := range votes {
		if count > maxVotes {
			maxVotes = count
			bestClass = class
		}
	}

	confidence := float64(maxVotes) / float64(len(c.trees))
	return bestClass, confidence
}

// classify traverses decision tree
func (t *DecisionTree) classify(features FlowFeatures) TrafficClass {
	node := t.Root

	for !node.IsLeaf {
		featureValue := t.getFeatureValue(features, node.FeatureIndex)

		if featureValue <= node.Threshold {
			node = node.Left
		} else {
			node = node.Right
		}

		if node == nil {
			return TrafficBestEffort // Default
		}
	}

	return node.Class
}

// buildTree recursively builds decision tree
func (q *AdaptiveQoS) buildTree(data []LabeledFlow, features []int, depth int) *TreeNode {
	// Check stopping criteria
	if depth >= q.classifier.maxDepth || len(data) < q.classifier.minSamples {
		return &TreeNode{
			IsLeaf: true,
			Class:  q.majorityClass(data),
			Samples: len(data),
		}
	}

	// Check if all samples have same class
	if q.isPure(data) {
		return &TreeNode{
			IsLeaf: true,
			Class:  data[0].Class,
			Samples: len(data),
		}
	}

	// Find best split
	bestFeature, bestThreshold, bestGini := q.findBestSplit(data, features)

	if bestGini >= q.calculateGini(data) {
		// No improvement
		return &TreeNode{
			IsLeaf: true,
			Class:  q.majorityClass(data),
			Samples: len(data),
		}
	}

	// Split data
	leftData, rightData := q.splitData(data, bestFeature, bestThreshold)

	// Recursively build subtrees
	node := &TreeNode{
		IsLeaf:       false,
		FeatureIndex: bestFeature,
		Threshold:    bestThreshold,
		Gini:         bestGini,
		Samples:      len(data),
	}

	node.Left = q.buildTree(leftData, features, depth+1)
	node.Right = q.buildTree(rightData, features, depth+1)

	return node
}

// matchFingerprint matches flow against application fingerprints
func (q *AdaptiveQoS) matchFingerprint(features FlowFeatures) (TrafficClass, float64) {
	for _, fp := range q.fingerprints {
		// Check port match
		portMatch := false
		for _, port := range fp.Ports {
			if features.DstPort == port || features.SrcPort == port {
				portMatch = true
				break
			}
		}

		if portMatch {
			// Additional checks could be added here
			return fp.DefaultClass, 0.95
		}
	}

	return TrafficBestEffort, 0.0
}

// heuristicClassification uses heuristics for classification
func (q *AdaptiveQoS) heuristicClassification(features FlowFeatures) TrafficClass {
	// Real-time: small packets, low inter-arrival time
	if features.PacketSize < 200 && features.InterArrivalTime < 20000 {
		return TrafficRealTime
	}

	// Interactive: SSH, RDP, Telnet ports
	interactivePorts := []int{22, 23, 3389, 5900}
	for _, port := range interactivePorts {
		if features.DstPort == port || features.SrcPort == port {
			return TrafficInteractive
		}
	}

	// Bulk: large packets, high byte count
	if features.PacketSize > 1400 && features.ByteCount > 1000000 {
		return TrafficBulk
	}

	return TrafficBestEffort
}

// initializeDefaultPolicies sets up default QoS policies
func (q *AdaptiveQoS) initializeDefaultPolicies() {
	// Real-time traffic (VoIP, video conferencing)
	q.policies[TrafficRealTime] = &QoSPolicy{
		Class:          TrafficRealTime,
		Priority:       7,
		MinBandwidth:   5,    // 5 Mbps minimum
		MaxBandwidth:   100,  // 100 Mbps maximum
		MaxLatency:     10,   // 10ms max latency
		MaxJitter:      2,    // 2ms max jitter
		PacketLossRate: 0.01, // 1% max loss
		DSCP:           46,   // EF (Expedited Forwarding)
	}

	// Interactive traffic (SSH, remote desktop)
	q.policies[TrafficInteractive] = &QoSPolicy{
		Class:          TrafficInteractive,
		Priority:       5,
		MinBandwidth:   2,
		MaxBandwidth:   50,
		MaxLatency:     50,
		MaxJitter:      10,
		PacketLossRate: 0.1,
		DSCP:           34, // AF41
	}

	// Bulk traffic (file transfer, backup)
	q.policies[TrafficBulk] = &QoSPolicy{
		Class:          TrafficBulk,
		Priority:       3,
		MinBandwidth:   10,
		MaxBandwidth:   1000,
		MaxLatency:     1000,
		MaxJitter:      100,
		PacketLossRate: 1,
		DSCP:           10, // AF11
	}

	// Best effort
	q.policies[TrafficBestEffort] = &QoSPolicy{
		Class:          TrafficBestEffort,
		Priority:       0,
		MinBandwidth:   0,
		MaxBandwidth:   100,
		MaxLatency:     5000,
		MaxJitter:      500,
		PacketLossRate: 5,
		DSCP:           0, // BE
	}
}

// loadApplicationFingerprints loads known application signatures
func (q *AdaptiveQoS) loadApplicationFingerprints() {
	// VoIP applications
	q.fingerprints["sip"] = ApplicationFingerprint{
		Name:         "SIP",
		Ports:        []int{5060, 5061},
		DefaultClass: TrafficRealTime,
	}

	q.fingerprints["rtp"] = ApplicationFingerprint{
		Name:         "RTP",
		Ports:        []int{16384, 16385, 16386, 16387}, // Common RTP ports
		DefaultClass: TrafficRealTime,
	}

	// Video conferencing
	q.fingerprints["zoom"] = ApplicationFingerprint{
		Name:         "Zoom",
		Ports:        []int{8801, 8802},
		DefaultClass: TrafficRealTime,
	}

	// Interactive applications
	q.fingerprints["ssh"] = ApplicationFingerprint{
		Name:         "SSH",
		Ports:        []int{22},
		DefaultClass: TrafficInteractive,
	}

	q.fingerprints["rdp"] = ApplicationFingerprint{
		Name:         "RDP",
		Ports:        []int{3389},
		DefaultClass: TrafficInteractive,
	}

	// Bulk transfer
	q.fingerprints["ftp"] = ApplicationFingerprint{
		Name:         "FTP",
		Ports:        []int{20, 21},
		DefaultClass: TrafficBulk,
	}
}

// adaptationLoop continuously adapts QoS policies
func (q *AdaptiveQoS) adaptationLoop(ctx context.Context) {
	ticker := time.NewTicker(q.adaptationInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			q.adaptPolicies()
		}
	}
}

// adaptPolicies adjusts QoS policies based on network conditions
func (q *AdaptiveQoS) adaptPolicies() {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Analyze recent classification accuracy
	if q.classificationAccuracy < 0.95 {
		// Retrain classifier if accuracy drops
		go q.trainClassifier()
	}

	// Update policy parameters based on network conditions
	// This would integrate with network monitoring
	q.lastAdaptation = time.Now()
}

// Helper methods
func (q *AdaptiveQoS) generateTrainingData() []LabeledFlow {
	// Generate synthetic training data
	// In production, this would use real labeled flows
	data := make([]LabeledFlow, 1000)

	for i := range data {
		class := TrafficClass(i % 4)
		data[i] = LabeledFlow{
			Features: q.generateSyntheticFeatures(class),
			Class:    class,
		}
	}

	return data
}

func (q *AdaptiveQoS) generateSyntheticFeatures(class TrafficClass) FlowFeatures {
	switch class {
	case TrafficRealTime:
		return FlowFeatures{
			PacketSize:       100 + randInt(100),
			InterArrivalTime: 10000 + randFloat()*10000,
			BurstSize:        1,
			PacketCount:      1000,
			ByteCount:        100000,
		}
	case TrafficInteractive:
		return FlowFeatures{
			PacketSize:       200 + randInt(300),
			InterArrivalTime: 50000 + randFloat()*50000,
			BurstSize:        5,
			PacketCount:      500,
			ByteCount:        150000,
		}
	case TrafficBulk:
		return FlowFeatures{
			PacketSize:       1400 + randInt(100),
			InterArrivalTime: 100000 + randFloat()*100000,
			BurstSize:        100,
			PacketCount:      10000,
			ByteCount:        14000000,
		}
	default:
		return FlowFeatures{
			PacketSize:       500 + randInt(500),
			InterArrivalTime: 200000 + randFloat()*200000,
			BurstSize:        10,
			PacketCount:      100,
			ByteCount:        50000,
		}
	}
}

func (q *AdaptiveQoS) bootstrapSample(data []LabeledFlow) []LabeledFlow {
	sample := make([]LabeledFlow, len(data))
	for i := range sample {
		sample[i] = data[randInt(len(data))]
	}
	return sample
}

func (q *AdaptiveQoS) selectRandomFeatures(total, count int) []int {
	features := make([]int, count)
	for i := range features {
		features[i] = randInt(total)
	}
	return features
}

func (q *AdaptiveQoS) calculateGini(data []LabeledFlow) float64 {
	if len(data) == 0 {
		return 0
	}

	counts := make(map[TrafficClass]int)
	for _, flow := range data {
		counts[flow.Class]++
	}

	gini := 1.0
	total := float64(len(data))

	for _, count := range counts {
		p := float64(count) / total
		gini -= p * p
	}

	return gini
}

func (q *AdaptiveQoS) isPure(data []LabeledFlow) bool {
	if len(data) == 0 {
		return true
	}

	class := data[0].Class
	for _, flow := range data[1:] {
		if flow.Class != class {
			return false
		}
	}
	return true
}

func (q *AdaptiveQoS) majorityClass(data []LabeledFlow) TrafficClass {
	counts := make(map[TrafficClass]int)
	for _, flow := range data {
		counts[flow.Class]++
	}

	var maxCount int
	var majorityClass TrafficClass

	for class, count := range counts {
		if count > maxCount {
			maxCount = count
			majorityClass = class
		}
	}

	return majorityClass
}

func (q *AdaptiveQoS) findBestSplit(data []LabeledFlow, features []int) (int, float64, float64) {
	bestFeature := features[0]
	bestThreshold := 0.0
	bestGini := 1.0

	// Simplified - would be more sophisticated in production
	for _, feature := range features {
		for _, flow := range data {
			threshold := q.getFeatureValueByIndex(flow.Features, feature)

			left, right := q.splitData(data, feature, threshold)
			if len(left) == 0 || len(right) == 0 {
				continue
			}

			gini := q.weightedGini(left, right)
			if gini < bestGini {
				bestGini = gini
				bestFeature = feature
				bestThreshold = threshold
			}
		}
	}

	return bestFeature, bestThreshold, bestGini
}

func (q *AdaptiveQoS) splitData(data []LabeledFlow, feature int, threshold float64) ([]LabeledFlow, []LabeledFlow) {
	var left, right []LabeledFlow

	for _, flow := range data {
		value := q.getFeatureValueByIndex(flow.Features, feature)
		if value <= threshold {
			left = append(left, flow)
		} else {
			right = append(right, flow)
		}
	}

	return left, right
}

func (q *AdaptiveQoS) weightedGini(left, right []LabeledFlow) float64 {
	total := len(left) + len(right)
	if total == 0 {
		return 0
	}

	leftGini := q.calculateGini(left)
	rightGini := q.calculateGini(right)

	return (float64(len(left))*leftGini + float64(len(right))*rightGini) / float64(total)
}

func (q *AdaptiveQoS) getFeatureValueByIndex(features FlowFeatures, index int) float64 {
	// Map index to feature value
	switch index {
	case 0:
		return float64(features.PacketSize)
	case 1:
		return features.InterArrivalTime
	case 2:
		return float64(features.BurstSize)
	case 3:
		return float64(features.PacketCount)
	case 4:
		return float64(features.ByteCount)
	case 5:
		return features.PayloadEntropy
	default:
		return 0
	}
}

func (t *DecisionTree) getFeatureValue(features FlowFeatures, index int) float64 {
	// Same as above - would be shared in production
	switch index {
	case 0:
		return float64(features.PacketSize)
	case 1:
		return features.InterArrivalTime
	case 2:
		return float64(features.BurstSize)
	case 3:
		return float64(features.PacketCount)
	case 4:
		return float64(features.ByteCount)
	case 5:
		return features.PayloadEntropy
	default:
		return 0
	}
}

func (c *TrafficClassifier) calculateFeatureImportance() {
	c.featureImportance = make([]float64, 12) // Number of features
	// Calculate based on tree splits - simplified
	for i := range c.featureImportance {
		c.featureImportance[i] = 1.0 / 12.0
	}
}

func (q *AdaptiveQoS) validateClassifier() {
	// Cross-validation on training data
	// Simplified - would use k-fold in production
	correct := 0
	for _, flow := range q.classifier.trainingData[:100] {
		class, _ := q.classifier.classify(flow.Features)
		if class == flow.Class {
			correct++
		}
	}
	q.classifier.accuracy = float64(correct) / 100.0
	q.classificationAccuracy = q.classifier.accuracy
}

func (q *AdaptiveQoS) applyTrafficShaping(flowID string, policy *QoSPolicy) error {
	// Implement token bucket or leaky bucket algorithm
	// This would interface with actual network stack
	return nil
}

func (q *AdaptiveQoS) markPackets(flowID string, dscp int) error {
	// Mark packets with DSCP value
	// This would interface with netfilter/iptables
	return nil
}

func (q *AdaptiveQoS) updateClassificationMetrics(duration time.Duration) {
	q.mu.Lock()
	defer q.mu.Unlock()

	// Exponential moving average
	alpha := 0.1
	q.avgClassificationTime = time.Duration(float64(q.avgClassificationTime)*(1-alpha) + float64(duration)*alpha)
}

// Helper functions
func randInt(max int) int {
	return int(randFloat() * float64(max))
}

func randFloat() float64 {
	return math.Float64frombits(0x3FF<<52 | uint64(time.Now().UnixNano()&0xFFFFFFFFFFFFF)) - 1
}

// GetMetrics returns QoS metrics
func (q *AdaptiveQoS) GetMetrics() map[string]interface{} {
	q.mu.RLock()
	defer q.mu.RUnlock()

	return map[string]interface{}{
		"classification_count":    q.classificationCount,
		"classification_accuracy": q.classificationAccuracy * 100, // Percentage
		"policy_updates":          q.policyUpdates,
		"avg_classification_time": q.avgClassificationTime.Microseconds(),
		"num_policies":            len(q.policies),
		"num_fingerprints":        len(q.fingerprints),
		"classifier_accuracy":     q.classifier.accuracy * 100,
	}
}