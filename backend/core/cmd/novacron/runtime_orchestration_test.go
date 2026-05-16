package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRuntimeOrchestrationStatusAndMetricsEndpoints(t *testing.T) {
	t.Parallel()

	router := newRuntimeOrchestrationTestRouter(t)

	status := getRuntimeOrchestrationJSON[map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/status", http.StatusOK)
	if got, want := status["state"], "running"; got != want {
		t.Fatalf("status state = %q, want %q", got, want)
	}
	if _, ok := status["metrics"].(map[string]interface{}); !ok {
		t.Fatalf("status metrics missing or wrong type: %#v", status["metrics"])
	}

	realtime := getRuntimeOrchestrationJSON[map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/metrics/realtime", http.StatusOK)
	for _, key := range []string{"cpuUsage", "memoryUsage", "networkIO", "diskIO", "decisionsPerMinute", "responseTime"} {
		if _, ok := realtime[key]; !ok {
			t.Fatalf("realtime metrics missing %q in %#v", key, realtime)
		}
	}
}

func TestRuntimeOrchestrationPolicyLifecycle(t *testing.T) {
	t.Parallel()

	router := newRuntimeOrchestrationTestRouter(t)

	created := postRuntimeOrchestrationJSON[map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/policies", map[string]interface{}{
		"name":      "scale-on-pressure",
		"type":      "autoscaling",
		"enabled":   true,
		"threshold": 80,
	}, http.StatusCreated)
	policyID, ok := created["id"].(string)
	if !ok || policyID == "" {
		t.Fatalf("created policy id missing: %#v", created)
	}
	if got, want := created["name"], "scale-on-pressure"; got != want {
		t.Fatalf("created policy name = %q, want %q", got, want)
	}

	listed := getRuntimeOrchestrationJSON[map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/policies", http.StatusOK)
	if got, want := listed["count"], float64(1); got != want {
		t.Fatalf("policy count = %v, want %v", got, want)
	}

	updated := methodRuntimeOrchestrationJSON[map[string]interface{}](t, router, http.MethodPatch, "/internal/runtime/v1/orchestration/policies/"+policyID, map[string]interface{}{
		"enabled": false,
	}, http.StatusOK)
	if got, want := updated["enabled"], false; got != want {
		t.Fatalf("updated policy enabled = %v, want %v", got, want)
	}

	deleted := methodRuntimeOrchestrationJSON[map[string]interface{}](t, router, http.MethodDelete, "/internal/runtime/v1/orchestration/policies/"+policyID, nil, http.StatusOK)
	if got, want := deleted["deleted"], true; got != want {
		t.Fatalf("deleted flag = %v, want %v", got, want)
	}
}

func TestRuntimeOrchestrationDecisionsAndScalingFeeds(t *testing.T) {
	t.Parallel()

	router := newRuntimeOrchestrationTestRouter(t)

	decisions := getRuntimeOrchestrationJSON[[]map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/decisions?limit=5", http.StatusOK)
	if len(decisions) == 0 {
		t.Fatal("expected runtime decisions feed to include at least one producer record")
	}
	for _, key := range []string{"id", "decisionType", "recommendation", "score", "confidence", "explanation", "timestamp", "status"} {
		if _, ok := decisions[0][key]; !ok {
			t.Fatalf("decision missing %q in %#v", key, decisions[0])
		}
	}

	scalingMetrics := getRuntimeOrchestrationJSON[[]map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/scaling/metrics?range=1h", http.StatusOK)
	if len(scalingMetrics) == 0 {
		t.Fatal("expected scaling metrics feed to include current runtime sample")
	}
	for _, key := range []string{"timestamp", "totalVMs", "cpuUtilization", "memoryUtilization", "requestRate", "responseTime", "throughput", "errorRate", "scalingEvents"} {
		if _, ok := scalingMetrics[0][key]; !ok {
			t.Fatalf("scaling metric missing %q in %#v", key, scalingMetrics[0])
		}
	}

	scalingEvents := getRuntimeOrchestrationJSON[[]map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/scaling/events?limit=5", http.StatusOK)
	if len(scalingEvents) == 0 {
		t.Fatal("expected scaling events feed to include current runtime event")
	}
	for _, key := range []string{"timestamp", "action", "vmId", "beforeCount", "afterCount", "reason", "cpuUtilization", "memoryUtilization", "requestRate", "responseTime"} {
		if _, ok := scalingEvents[0][key]; !ok {
			t.Fatalf("scaling event missing %q in %#v", key, scalingEvents[0])
		}
	}
}

func TestRuntimeOrchestrationMLModelLifecycle(t *testing.T) {
	t.Parallel()

	router := newRuntimeOrchestrationTestRouter(t)

	models := getRuntimeOrchestrationJSON[[]map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/ml-models", http.StatusOK)
	if len(models) == 0 {
		t.Fatal("expected runtime to publish model inventory")
	}
	for _, key := range []string{"modelType", "accuracy", "throughput", "latency", "lastTraining", "version", "status"} {
		if _, ok := models[0][key]; !ok {
			t.Fatalf("model missing %q in %#v", key, models[0])
		}
	}

	retrain := postRuntimeOrchestrationJSON[map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/ml-models/resource-prediction/retrain", map[string]interface{}{
		"reason": "test",
	}, http.StatusAccepted)
	if got, want := retrain["status"], "queued"; got != want {
		t.Fatalf("retrain status = %q, want %q", got, want)
	}
	if retrain["jobId"] == "" {
		t.Fatalf("retrain response missing job id: %#v", retrain)
	}

	download := getRuntimeOrchestrationJSON[map[string]interface{}](t, router, "/internal/runtime/v1/orchestration/ml-models/resource-prediction/download", http.StatusNotImplemented)
	if got, want := download["status"], "unavailable"; got != want {
		t.Fatalf("download status = %q, want %q", got, want)
	}
}

func newRuntimeOrchestrationTestRouter(t *testing.T) http.Handler {
	t.Helper()
	return newRuntimeRouter(defaultRuntimeConfig("node-a", t.TempDir()), nil, nil, nil, nil, nil, nil, nil, nil, nil, nil)
}

func getRuntimeOrchestrationJSON[T any](t *testing.T, router http.Handler, path string, expectedStatus int) T {
	t.Helper()
	return methodRuntimeOrchestrationJSON[T](t, router, http.MethodGet, path, nil, expectedStatus)
}

func postRuntimeOrchestrationJSON[T any](t *testing.T, router http.Handler, path string, payload interface{}, expectedStatus int) T {
	t.Helper()
	return methodRuntimeOrchestrationJSON[T](t, router, http.MethodPost, path, payload, expectedStatus)
}

func methodRuntimeOrchestrationJSON[T any](t *testing.T, router http.Handler, method string, path string, payload interface{}, expectedStatus int) T {
	t.Helper()

	var body *bytes.Reader
	if payload == nil {
		body = bytes.NewReader(nil)
	} else {
		data, err := json.Marshal(payload)
		if err != nil {
			t.Fatalf("marshal payload: %v", err)
		}
		body = bytes.NewReader(data)
	}

	var zero T
	req := httptest.NewRequest(method, path, body)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)
	if rec.Code != expectedStatus {
		t.Fatalf("%s %s status = %d, want %d: %s", method, path, rec.Code, expectedStatus, rec.Body.String())
	}
	if err := json.NewDecoder(rec.Body).Decode(&zero); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return zero
}
