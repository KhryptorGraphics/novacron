package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/mux"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
	"github.com/khryptorgraphics/novacron/backend/core/vm"
)

type runtimeOrchestrationRuntime struct {
	config           runtimeConfig
	vmManager        *vm.VMManager
	migrationManager *vm.VMMigrationManager
	schedulerService *scheduler.Scheduler
	startedAt        time.Time
	mu               sync.Mutex
	policies         map[string]map[string]interface{}
	decisions        []runtimeOrchestrationDecision
	scalingEvents    []runtimeOrchestrationScalingEvent
	models           map[string]runtimeOrchestrationModel
	nextPolicyID     int
	nextJobID        int
}

type runtimeOrchestrationDecision struct {
	ID             string                 `json:"id"`
	Type           string                 `json:"type"`
	Target         string                 `json:"target"`
	CreatedAt      time.Time              `json:"createdAt"`
	Metadata       map[string]interface{} `json:"metadata,omitempty"`
	DecisionType   string                 `json:"decisionType"`
	Recommendation string                 `json:"recommendation"`
	Score          float64                `json:"score"`
	Confidence     float64                `json:"confidence"`
	Explanation    string                 `json:"explanation"`
	Timestamp      time.Time              `json:"timestamp"`
	Status         string                 `json:"status"`
}

type runtimeOrchestrationScalingEvent struct {
	ID                string    `json:"id"`
	Type              string    `json:"type"`
	Target            string    `json:"target"`
	CreatedAt         time.Time `json:"createdAt"`
	Timestamp         time.Time `json:"timestamp"`
	Action            string    `json:"action"`
	VMID              string    `json:"vmId"`
	BeforeCount       int       `json:"beforeCount"`
	AfterCount        int       `json:"afterCount"`
	Reason            string    `json:"reason"`
	CPUUtilization    float64   `json:"cpuUtilization"`
	MemoryUtilization float64   `json:"memoryUtilization"`
	RequestRate       float64   `json:"requestRate"`
	ResponseTime      float64   `json:"responseTime"`
	Status            string    `json:"status"`
}

type runtimeOrchestrationModel struct {
	Type         string    `json:"type"`
	ModelType    string    `json:"modelType"`
	Version      string    `json:"version"`
	Status       string    `json:"status"`
	Accuracy     float64   `json:"accuracy"`
	Throughput   float64   `json:"throughput"`
	Latency      float64   `json:"latency"`
	LastTraining time.Time `json:"lastTraining"`
	UpdatedAt    time.Time `json:"updatedAt"`
	LastJobID    string    `json:"lastJobId,omitempty"`
	ArtifactID   string    `json:"artifactId,omitempty"`
}

func registerRuntimeOrchestrationRoutes(router *mux.Router, config runtimeConfig, vmManager *vm.VMManager, migrationManager *vm.VMMigrationManager, schedulerService *scheduler.Scheduler) {
	runtime := newRuntimeOrchestrationRuntime(config, vmManager, migrationManager, schedulerService)
	router.HandleFunc("/internal/runtime/v1/orchestration/status", runtime.getStatus).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/orchestration/decisions", runtime.listDecisions).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/orchestration/policies", runtime.listPolicies).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/orchestration/policies", runtime.createPolicy).Methods(http.MethodPost)
	router.HandleFunc("/internal/runtime/v1/orchestration/policies/{id}", runtime.updatePolicy).Methods(http.MethodPut, http.MethodPatch)
	router.HandleFunc("/internal/runtime/v1/orchestration/policies/{id}", runtime.deletePolicy).Methods(http.MethodDelete)
	router.HandleFunc("/internal/runtime/v1/orchestration/ml-models", runtime.listModels).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/orchestration/ml-models/{modelType}/retrain", runtime.retrainModel).Methods(http.MethodPost)
	router.HandleFunc("/internal/runtime/v1/orchestration/ml-models/{modelType}/download", runtime.downloadModel).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/orchestration/metrics/realtime", runtime.getRealtimeMetrics).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/orchestration/scaling/metrics", runtime.listScalingMetrics).Methods(http.MethodGet)
	router.HandleFunc("/internal/runtime/v1/orchestration/scaling/events", runtime.listScalingEvents).Methods(http.MethodGet)
}

func newRuntimeOrchestrationRuntime(config runtimeConfig, vmManager *vm.VMManager, migrationManager *vm.VMMigrationManager, schedulerService *scheduler.Scheduler) *runtimeOrchestrationRuntime {
	now := time.Now().UTC()
	return &runtimeOrchestrationRuntime{
		config:           config,
		vmManager:        vmManager,
		migrationManager: migrationManager,
		schedulerService: schedulerService,
		startedAt:        now,
		policies:         map[string]map[string]interface{}{},
		decisions: []runtimeOrchestrationDecision{{
			ID:             "decision-bootstrap",
			Type:           "runtime-observation",
			Target:         config.Hypervisor.ID,
			CreatedAt:      now,
			Metadata:       map[string]interface{}{"nodeId": config.Hypervisor.ID},
			DecisionType:   "optimization",
			Recommendation: "Observe runtime state before taking action",
			Score:          1.0,
			Confidence:     1.0,
			Explanation:    "Runtime orchestration producer initialized",
			Timestamp:      now,
			Status:         "executed",
		}},
		scalingEvents: []runtimeOrchestrationScalingEvent{{
			ID:          "scaling-bootstrap",
			Type:        "observation",
			Target:      config.Hypervisor.ID,
			CreatedAt:   now,
			Timestamp:   now,
			Action:      "no_change",
			BeforeCount: 0,
			AfterCount:  0,
			Reason:      "runtime scaling feed initialized",
			Status:      "completed",
		}},
		models: map[string]runtimeOrchestrationModel{
			"resource-prediction": {Type: "resource-prediction", ModelType: "resource-prediction", Version: "runtime-v1", Status: "deployed", Accuracy: 0.0, LastTraining: now, UpdatedAt: now},
			"placement":           {Type: "placement", ModelType: "placement", Version: "runtime-v1", Status: "deployed", Accuracy: 0.0, LastTraining: now, UpdatedAt: now},
			"anomaly-detection":   {Type: "anomaly-detection", ModelType: "anomaly-detection", Version: "runtime-v1", Status: "deployed", Accuracy: 0.0, LastTraining: now, UpdatedAt: now},
		},
		nextPolicyID: 1,
		nextJobID:    1,
	}
}

func (r *runtimeOrchestrationRuntime) getStatus(w http.ResponseWriter, req *http.Request) {
	metrics := r.metricsSnapshot()
	r.mu.Lock()
	activePolicies := 0
	for _, policy := range r.policies {
		if enabled, _ := policy["enabled"].(bool); enabled {
			activePolicies++
		}
	}
	eventsProcessed := len(r.decisions) + len(r.scalingEvents)
	r.mu.Unlock()

	respondRuntimeJSON(w, http.StatusOK, map[string]interface{}{
		"state":           "running",
		"nodeId":          r.config.Hypervisor.ID,
		"startTime":       r.startedAt,
		"activePolicies":  activePolicies,
		"eventsProcessed": eventsProcessed,
		"metrics":         metrics,
	})
}

func (r *runtimeOrchestrationRuntime) listDecisions(w http.ResponseWriter, req *http.Request) {
	limit := runtimeOrchestrationLimit(req, 50)
	r.mu.Lock()
	defer r.mu.Unlock()

	decisions := append([]runtimeOrchestrationDecision(nil), r.decisions...)
	sort.SliceStable(decisions, func(i, j int) bool {
		return decisions[i].CreatedAt.After(decisions[j].CreatedAt)
	})
	if len(decisions) > limit {
		decisions = decisions[:limit]
	}
	respondRuntimeJSON(w, http.StatusOK, decisions)
}

func (r *runtimeOrchestrationRuntime) listPolicies(w http.ResponseWriter, req *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()

	policies := make([]map[string]interface{}, 0, len(r.policies))
	for _, policy := range r.policies {
		policies = append(policies, runtimeOrchestrationCloneMap(policy))
	}
	sort.SliceStable(policies, func(i, j int) bool {
		return fmt.Sprint(policies[i]["createdAt"]) < fmt.Sprint(policies[j]["createdAt"])
	})
	respondRuntimeJSON(w, http.StatusOK, map[string]interface{}{
		"policies": policies,
		"count":    len(policies),
	})
}

func (r *runtimeOrchestrationRuntime) createPolicy(w http.ResponseWriter, req *http.Request) {
	var request map[string]interface{}
	if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid policy payload"})
		return
	}

	now := time.Now().UTC()
	r.mu.Lock()
	id := runtimeOrchestrationString(request, "id")
	if id == "" {
		id = fmt.Sprintf("policy-%d", r.nextPolicyID)
		r.nextPolicyID++
	}
	policy := runtimeOrchestrationCloneMap(request)
	policy["id"] = id
	if _, ok := policy["name"].(string); !ok || runtimeOrchestrationString(policy, "name") == "" {
		policy["name"] = id
	}
	if _, ok := policy["type"].(string); !ok || runtimeOrchestrationString(policy, "type") == "" {
		policy["type"] = "custom"
	}
	if _, ok := policy["enabled"].(bool); !ok {
		policy["enabled"] = false
	}
	if _, ok := policy["createdAt"].(string); !ok {
		policy["createdAt"] = now
	}
	policy["updatedAt"] = now
	r.policies[id] = policy
	r.decisions = append(r.decisions, runtimeOrchestrationDecision{
		ID:             fmt.Sprintf("decision-policy-%s-created", id),
		Type:           "policy-created",
		Target:         id,
		CreatedAt:      now,
		DecisionType:   "optimization",
		Recommendation: "Apply updated orchestration policy inventory",
		Score:          1.0,
		Confidence:     1.0,
		Explanation:    "Policy registered through runtime producer endpoint",
		Timestamp:      now,
		Status:         "executed",
	})
	r.mu.Unlock()

	respondRuntimeJSON(w, http.StatusCreated, policy)
}

func (r *runtimeOrchestrationRuntime) updatePolicy(w http.ResponseWriter, req *http.Request) {
	policyID := mux.Vars(req)["id"]
	var request map[string]interface{}
	if err := json.NewDecoder(req.Body).Decode(&request); err != nil {
		respondRuntimeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid policy payload"})
		return
	}

	r.mu.Lock()
	policy, ok := r.policies[policyID]
	if !ok {
		r.mu.Unlock()
		respondRuntimeJSON(w, http.StatusNotFound, map[string]string{"error": "policy not found"})
		return
	}
	updated := runtimeOrchestrationCloneMap(policy)
	for key, value := range request {
		updated[key] = value
	}
	updated["id"] = policyID
	if _, ok := updated["createdAt"]; !ok {
		updated["createdAt"] = policy["createdAt"]
	}
	now := time.Now().UTC()
	updated["updatedAt"] = now
	r.policies[policyID] = updated
	r.decisions = append(r.decisions, runtimeOrchestrationDecision{
		ID:             fmt.Sprintf("decision-policy-%s-updated", policyID),
		Type:           "policy-updated",
		Target:         policyID,
		CreatedAt:      now,
		DecisionType:   "optimization",
		Recommendation: "Reconcile orchestration policy inventory",
		Score:          1.0,
		Confidence:     1.0,
		Explanation:    "Policy updated through runtime producer endpoint",
		Timestamp:      now,
		Status:         "executed",
	})
	r.mu.Unlock()

	respondRuntimeJSON(w, http.StatusOK, updated)
}

func (r *runtimeOrchestrationRuntime) deletePolicy(w http.ResponseWriter, req *http.Request) {
	policyID := mux.Vars(req)["id"]
	now := time.Now().UTC()

	r.mu.Lock()
	if _, ok := r.policies[policyID]; !ok {
		r.mu.Unlock()
		respondRuntimeJSON(w, http.StatusNotFound, map[string]string{"error": "policy not found"})
		return
	}
	delete(r.policies, policyID)
	r.decisions = append(r.decisions, runtimeOrchestrationDecision{
		ID:             fmt.Sprintf("decision-policy-%s-deleted", policyID),
		Type:           "policy-deleted",
		Target:         policyID,
		CreatedAt:      now,
		DecisionType:   "optimization",
		Recommendation: "Remove orchestration policy from active inventory",
		Score:          1.0,
		Confidence:     1.0,
		Explanation:    "Policy deleted through runtime producer endpoint",
		Timestamp:      now,
		Status:         "executed",
	})
	r.mu.Unlock()

	respondRuntimeJSON(w, http.StatusOK, map[string]interface{}{"id": policyID, "deleted": true})
}

func (r *runtimeOrchestrationRuntime) listModels(w http.ResponseWriter, req *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()

	models := make([]runtimeOrchestrationModel, 0, len(r.models))
	for _, model := range r.models {
		models = append(models, model)
	}
	sort.SliceStable(models, func(i, j int) bool {
		return models[i].Type < models[j].Type
	})
	respondRuntimeJSON(w, http.StatusOK, models)
}

func (r *runtimeOrchestrationRuntime) retrainModel(w http.ResponseWriter, req *http.Request) {
	modelType := mux.Vars(req)["modelType"]
	now := time.Now().UTC()

	r.mu.Lock()
	model, ok := r.models[modelType]
	if !ok {
		model = runtimeOrchestrationModel{Type: modelType, ModelType: modelType, Version: "runtime-v1", Accuracy: 0.0}
	}
	jobID := fmt.Sprintf("model-job-%d", r.nextJobID)
	r.nextJobID++
	model.Status = "training"
	model.UpdatedAt = now
	model.LastTraining = now
	model.LastJobID = jobID
	r.models[modelType] = model
	r.decisions = append(r.decisions, runtimeOrchestrationDecision{
		ID:             fmt.Sprintf("decision-%s", jobID),
		Type:           "model-retrain",
		Target:         modelType,
		CreatedAt:      now,
		DecisionType:   "optimization",
		Recommendation: "Retrain orchestration model",
		Score:          1.0,
		Confidence:     1.0,
		Explanation:    "Model retraining requested",
		Timestamp:      now,
		Status:         "pending",
	})
	r.mu.Unlock()

	respondRuntimeJSON(w, http.StatusAccepted, map[string]interface{}{
		"status":    "queued",
		"modelType": modelType,
		"jobId":     jobID,
		"queuedAt":  now,
	})
}

func (r *runtimeOrchestrationRuntime) downloadModel(w http.ResponseWriter, req *http.Request) {
	modelType := mux.Vars(req)["modelType"]
	respondRuntimeJSON(w, http.StatusNotImplemented, map[string]interface{}{
		"status":    "unavailable",
		"modelType": modelType,
		"error":     "runtime model artifact export is not configured",
	})
}

func (r *runtimeOrchestrationRuntime) getRealtimeMetrics(w http.ResponseWriter, req *http.Request) {
	respondRuntimeJSON(w, http.StatusOK, r.metricsSnapshot())
}

func (r *runtimeOrchestrationRuntime) listScalingMetrics(w http.ResponseWriter, req *http.Request) {
	now := time.Now().UTC()
	metrics := r.metricsSnapshot()
	respondRuntimeJSON(w, http.StatusOK, []map[string]interface{}{{
		"timestamp":         now,
		"range":             req.URL.Query().Get("range"),
		"totalVMs":          metrics["activeVMs"],
		"cpuUtilization":    metrics["cpuUsage"],
		"memoryUtilization": metrics["memoryUsage"],
		"requestRate":       float64(0),
		"responseTime":      metrics["responseTime"],
		"throughput":        float64(0),
		"errorRate":         float64(0),
		"scalingEvents":     len(r.scalingEvents),
	}})
}

func (r *runtimeOrchestrationRuntime) listScalingEvents(w http.ResponseWriter, req *http.Request) {
	limit := runtimeOrchestrationLimit(req, 50)
	r.mu.Lock()
	defer r.mu.Unlock()

	events := append([]runtimeOrchestrationScalingEvent(nil), r.scalingEvents...)
	sort.SliceStable(events, func(i, j int) bool {
		return events[i].CreatedAt.After(events[j].CreatedAt)
	})
	if len(events) > limit {
		events = events[:limit]
	}
	respondRuntimeJSON(w, http.StatusOK, events)
}

func (r *runtimeOrchestrationRuntime) metricsSnapshot() map[string]interface{} {
	nodes := []*vm.NodeResourceInfo(nil)
	if r.vmManager != nil {
		nodes = r.vmManager.ListSchedulerNodes()
	}

	var totalCPU, usedCPU int
	var totalMemoryMB, usedMemoryMB int
	var totalDiskGB, usedDiskGB int
	activeVMs := 0
	for _, node := range nodes {
		if node == nil {
			continue
		}
		totalCPU += node.TotalCPU
		usedCPU += node.UsedCPU
		totalMemoryMB += node.TotalMemoryMB
		usedMemoryMB += node.UsedMemoryMB
		totalDiskGB += node.TotalDiskGB
		usedDiskGB += node.UsedDiskGB
		activeVMs += node.VMCount
	}

	r.mu.Lock()
	decisionCount := len(r.decisions)
	r.mu.Unlock()

	cpuUsage := percent(usedCPU, totalCPU)
	memoryUsage := percent(usedMemoryMB, totalMemoryMB)
	diskIO := percent(usedDiskGB, totalDiskGB)
	return map[string]interface{}{
		"timestamp":            time.Now().UTC(),
		"cpuUsage":             cpuUsage,
		"memoryUsage":          memoryUsage,
		"networkIO":            float64(0),
		"diskIO":               diskIO,
		"decisionsPerMinute":   decisionCount,
		"responseTime":         float64(0),
		"activeVMs":            activeVMs,
		"nodeCount":            len(nodes),
		"cpu_usage":            cpuUsage,
		"memory_usage":         memoryUsage,
		"network_io":           float64(0),
		"disk_io":              diskIO,
		"decisions_per_minute": decisionCount,
		"response_time":        float64(0),
	}
}

func runtimeOrchestrationLimit(req *http.Request, fallback int) int {
	value := req.URL.Query().Get("limit")
	if value == "" {
		return fallback
	}
	limit, err := strconv.Atoi(value)
	if err != nil || limit <= 0 {
		return fallback
	}
	return limit
}

func runtimeOrchestrationString(payload map[string]interface{}, key string) string {
	value, _ := payload[key].(string)
	return strings.TrimSpace(value)
}

func runtimeOrchestrationCloneMap(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{}, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}
