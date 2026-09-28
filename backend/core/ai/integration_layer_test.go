package ai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/uuid"
)

// testConfig keeps every knob explicit so tests do not depend on defaults.
func testConfig() AIConfig {
	return AIConfig{
		Timeout:                 2 * time.Second,
		Retries:                 1,
		MaxConnections:          10,
		CircuitBreakerThreshold: 100,
		CircuitBreakerTimeout:   time.Minute,
		CacheSize:               10,
		CacheTTL:                time.Minute,
	}
}

func writeAIResponse(t *testing.T, w http.ResponseWriter, resp AIResponse) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		t.Errorf("encode response: %v", err)
	}
}

func okData(t *testing.T, w http.ResponseWriter, data map[string]interface{}) {
	t.Helper()
	writeAIResponse(t, w, AIResponse{ID: "resp", Success: true, Data: data})
}

func TestRequestEnvelopePerOperation(t *testing.T) {
	ctx := context.Background()
	cases := []struct {
		name    string
		service string
		method  string
		call    func(*AIIntegrationLayer) error
		check   func(*testing.T, map[string]interface{})
	}{
		{
			name: "PredictResourceDemand", service: "resource_prediction", method: "predict_demand",
			call: func(ai *AIIntegrationLayer) error {
				_, err := ai.PredictResourceDemand(ctx, ResourcePredictionRequest{NodeID: "node-1", ResourceType: "cpu", HorizonMinutes: 30})
				return err
			},
			check: func(t *testing.T, d map[string]interface{}) {
				if d["node_id"] != "node-1" || d["resource_type"] != "cpu" || d["horizon_minutes"] != 30.0 {
					t.Errorf("unexpected data %v", d)
				}
			},
		},
		{
			name: "OptimizePerformance", service: "performance_optimization", method: "optimize_cluster",
			call: func(ai *AIIntegrationLayer) error {
				_, err := ai.OptimizePerformance(ctx, PerformanceOptimizationRequest{ClusterID: "c1", Goals: []string{"latency"}})
				return err
			},
			check: func(t *testing.T, d map[string]interface{}) {
				goals, _ := d["goals"].([]interface{})
				if d["cluster_id"] != "c1" || len(goals) != 1 || goals[0] != "latency" {
					t.Errorf("unexpected data %v", d)
				}
			},
		},
		{
			name: "DetectAnomalies", service: "anomaly_detection", method: "detect",
			call: func(ai *AIIntegrationLayer) error {
				_, err := ai.DetectAnomalies(ctx, AnomalyDetectionRequest{ResourceID: "vm-7", MetricType: "cpu", Sensitivity: 0.9})
				return err
			},
			check: func(t *testing.T, d map[string]interface{}) {
				if d["resource_id"] != "vm-7" || d["metric_type"] != "cpu" || d["sensitivity"] != 0.9 {
					t.Errorf("unexpected data %v", d)
				}
			},
		},
		{
			name: "AnalyzeWorkloadPattern", service: "workload_pattern_recognition", method: "analyze_patterns",
			call: func(ai *AIIntegrationLayer) error {
				_, err := ai.AnalyzeWorkloadPattern(ctx, WorkloadPatternRequest{WorkloadID: "w1", MetricTypes: []string{"cpu"}})
				return err
			},
			check: func(t *testing.T, d map[string]interface{}) {
				if d["workload_id"] != "w1" {
					t.Errorf("unexpected data %v", d)
				}
			},
		},
		{
			name: "PredictScalingNeeds", service: "predictive_scaling", method: "predict_scaling",
			call: func(ai *AIIntegrationLayer) error {
				_, err := ai.PredictScalingNeeds(ctx, map[string]interface{}{"nodes": 3})
				return err
			},
			check: func(t *testing.T, d map[string]interface{}) {
				if d["nodes"] != 3.0 {
					t.Errorf("unexpected data %v", d)
				}
			},
		},
		{
			name: "TrainModel", service: "model_training", method: "train",
			call: func(ai *AIIntegrationLayer) error {
				return ai.TrainModel(ctx, "anomaly", []float64{1, 2})
			},
			check: func(t *testing.T, d map[string]interface{}) {
				samples, _ := d["training_data"].([]interface{})
				if d["model_type"] != "anomaly" || len(samples) != 2 {
					t.Errorf("unexpected data %v", d)
				}
			},
		},
		{
			name: "GetModelInfo", service: "model_management", method: "get_info",
			call: func(ai *AIIntegrationLayer) error {
				_, err := ai.GetModelInfo(ctx, "lstm")
				return err
			},
			check: func(t *testing.T, d map[string]interface{}) {
				if d["model_type"] != "lstm" {
					t.Errorf("unexpected data %v", d)
				}
			},
		},
		{
			name: "HealthCheck", service: "health", method: "check",
			call: func(ai *AIIntegrationLayer) error { return ai.HealthCheck(ctx) },
			check: func(t *testing.T, d map[string]interface{}) {
				if len(d) != 0 {
					t.Errorf("health check should send empty data, got %v", d)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got AIRequest
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/api/v1/process" {
					t.Errorf("request = %s %s, want POST /api/v1/process", r.Method, r.URL.Path)
				}
				if ct := r.Header.Get("Content-Type"); ct != "application/json" {
					t.Errorf("Content-Type = %q", ct)
				}
				if auth := r.Header.Get("Authorization"); auth != "Bearer secret" {
					t.Errorf("Authorization = %q", auth)
				}
				if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
					t.Errorf("decode request: %v", err)
				}
				okData(t, w, map[string]interface{}{})
			}))
			defer srv.Close()

			ai := NewAIIntegrationLayer(srv.URL, "secret", testConfig())
			if err := tc.call(ai); err != nil {
				t.Fatalf("call failed: %v", err)
			}
			if _, err := uuid.Parse(got.ID); err != nil {
				t.Errorf("request id %q is not a UUID: %v", got.ID, err)
			}
			if got.Service != tc.service || got.Method != tc.method {
				t.Errorf("service/method = %s/%s, want %s/%s", got.Service, got.Method, tc.service, tc.method)
			}
			tc.check(t, got.Data)
		})
	}
}

func TestRequestOmitsAuthorizationWithoutAPIKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if auth, ok := r.Header["Authorization"]; ok {
			t.Errorf("unexpected Authorization header %v", auth)
		}
		okData(t, w, nil)
	}))
	defer srv.Close()

	if err := NewAIIntegrationLayer(srv.URL, "", testConfig()).HealthCheck(context.Background()); err != nil {
		t.Fatalf("HealthCheck: %v", err)
	}
}

func TestResponseParsing(t *testing.T) {
	ctx := context.Background()
	trained := time.Date(2026, 1, 2, 3, 4, 5, 0, time.UTC)
	cases := []struct {
		name  string
		data  map[string]interface{}
		call  func(*AIIntegrationLayer) (interface{}, error)
		check func(*testing.T, interface{})
	}{
		{
			name: "resource prediction",
			data: map[string]interface{}{
				"predictions": []float64{1.5, 2.5},
				"confidence":  0.8,
				"model_info":  map[string]interface{}{"name": "lstm", "version": "2.0.0", "accuracy": 0.9, "last_trained": trained.Format(time.RFC3339)},
			},
			call: func(ai *AIIntegrationLayer) (interface{}, error) {
				return ai.PredictResourceDemand(ctx, ResourcePredictionRequest{NodeID: "n"})
			},
			check: func(t *testing.T, v interface{}) {
				r := v.(*ResourcePredictionResponse)
				if len(r.Predictions) != 2 || r.Predictions[1] != 2.5 || r.Confidence != 0.8 {
					t.Errorf("prediction = %+v", r)
				}
				if r.ModelInfo.Name != "lstm" || r.ModelInfo.Accuracy != 0.9 || !r.ModelInfo.LastTrained.Equal(trained) {
					t.Errorf("model info = %+v", r.ModelInfo)
				}
			},
		},
		{
			name: "performance optimization",
			data: map[string]interface{}{
				"recommendations": []map[string]interface{}{{"type": "scale", "target": "node-1", "priority": 2}},
				"expected_gains":  map[string]float64{"cpu": 0.2},
				"risk_assessment": map[string]interface{}{"overall_risk": 0.1, "risk_factors": []string{"churn"}},
				"confidence":      0.7,
			},
			call: func(ai *AIIntegrationLayer) (interface{}, error) {
				return ai.OptimizePerformance(ctx, PerformanceOptimizationRequest{ClusterID: "c"})
			},
			check: func(t *testing.T, v interface{}) {
				r := v.(*PerformanceOptimizationResponse)
				if len(r.Recommendations) != 1 || r.Recommendations[0].Target != "node-1" || r.Recommendations[0].Priority != 2 {
					t.Errorf("recommendations = %+v", r.Recommendations)
				}
				if r.ExpectedGains["cpu"] != 0.2 || r.RiskAssessment.OverallRisk != 0.1 || r.RiskAssessment.RiskFactors[0] != "churn" || r.Confidence != 0.7 {
					t.Errorf("optimization = %+v", r)
				}
			},
		},
		{
			name: "anomaly detection",
			data: map[string]interface{}{
				"anomalies":     []map[string]interface{}{{"anomaly_type": "spike", "severity": "high", "score": 0.95}},
				"overall_score": 0.95,
			},
			call: func(ai *AIIntegrationLayer) (interface{}, error) {
				return ai.DetectAnomalies(ctx, AnomalyDetectionRequest{ResourceID: "r"})
			},
			check: func(t *testing.T, v interface{}) {
				r := v.(*AnomalyDetectionResponse)
				if len(r.Anomalies) != 1 || r.Anomalies[0].AnomalyType != "spike" || r.Anomalies[0].Severity != "high" || r.OverallScore != 0.95 {
					t.Errorf("anomalies = %+v", r)
				}
			},
		},
		{
			name: "workload pattern",
			data: map[string]interface{}{
				"patterns":       []map[string]interface{}{{"type": "diurnal", "intensity": 0.4}},
				"classification": "periodic",
				"confidence":     0.6,
			},
			call: func(ai *AIIntegrationLayer) (interface{}, error) {
				return ai.AnalyzeWorkloadPattern(ctx, WorkloadPatternRequest{WorkloadID: "w"})
			},
			check: func(t *testing.T, v interface{}) {
				r := v.(*WorkloadPatternResponse)
				if len(r.Patterns) != 1 || r.Patterns[0].Type != "diurnal" || r.Classification != "periodic" || r.Confidence != 0.6 {
					t.Errorf("pattern = %+v", r)
				}
			},
		},
		{
			name: "model info",
			data: map[string]interface{}{"name": "iforest", "version": "1.2"},
			call: func(ai *AIIntegrationLayer) (interface{}, error) {
				return ai.GetModelInfo(ctx, "anomaly")
			},
			check: func(t *testing.T, v interface{}) {
				r := v.(*ModelInfo)
				if r.Name != "iforest" || r.Version != "1.2" {
					t.Errorf("model info = %+v", r)
				}
			},
		},
		{
			name: "scaling passthrough",
			data: map[string]interface{}{"scale_up": true, "target_nodes": 5},
			call: func(ai *AIIntegrationLayer) (interface{}, error) {
				return ai.PredictScalingNeeds(ctx, map[string]interface{}{})
			},
			check: func(t *testing.T, v interface{}) {
				r := v.(map[string]interface{})
				if r["scale_up"] != true || r["target_nodes"] != 5.0 {
					t.Errorf("scaling = %v", r)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				okData(t, w, tc.data)
			}))
			defer srv.Close()

			got, err := tc.call(NewAIIntegrationLayer(srv.URL, "", testConfig()))
			if err != nil {
				t.Fatalf("call failed: %v", err)
			}
			tc.check(t, got)
		})
	}
}

func TestErrorHandling(t *testing.T) {
	cases := []struct {
		name       string
		handler    func(*testing.T, http.ResponseWriter)
		wantErr    string
		wantFailed int64
	}{
		{
			name: "non-200 status",
			handler: func(t *testing.T, w http.ResponseWriter) {
				w.WriteHeader(http.StatusServiceUnavailable)
				w.Write([]byte("overloaded"))
			},
			wantErr:    "AI service returned status 503: overloaded",
			wantFailed: 1,
		},
		{
			name:       "malformed envelope",
			handler:    func(t *testing.T, w http.ResponseWriter) { w.Write([]byte("{not json")) },
			wantErr:    "failed to parse AI response",
			wantFailed: 1,
		},
		{
			name: "service reports failure",
			handler: func(t *testing.T, w http.ResponseWriter) {
				writeAIResponse(t, w, AIResponse{Success: false, Error: "model not loaded"})
			},
			wantErr:    "AI processing failed: model not loaded",
			wantFailed: 1,
		},
		{
			name: "payload does not match response type",
			handler: func(t *testing.T, w http.ResponseWriter) {
				okData(t, w, map[string]interface{}{"predictions": "not-a-list"})
			},
			wantErr:    "failed to parse prediction response",
			wantFailed: 0, // transport succeeded; decoding into the typed response failed
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				tc.handler(t, w)
			}))
			defer srv.Close()

			ai := NewAIIntegrationLayer(srv.URL, "", testConfig())
			resp, err := ai.PredictResourceDemand(context.Background(), ResourcePredictionRequest{NodeID: "n"})
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want containing %q", err, tc.wantErr)
			}
			if resp != nil {
				t.Errorf("expected nil response on error, got %+v", resp)
			}
			if got := ai.GetMetrics()["failed_requests"]; got != tc.wantFailed {
				t.Errorf("failed_requests = %v, want %d", got, tc.wantFailed)
			}
		})
	}
}

func TestRetryRecoversFromTransientFailure(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.WriteHeader(http.StatusBadGateway)
			return
		}
		okData(t, w, nil)
	}))
	defer srv.Close()

	cfg := testConfig()
	cfg.Retries = 2
	ai := NewAIIntegrationLayer(srv.URL, "", cfg)
	if err := ai.HealthCheck(context.Background()); err != nil {
		t.Fatalf("HealthCheck after one transient failure: %v", err)
	}
	if hits.Load() != 2 {
		t.Errorf("server hits = %d, want 2", hits.Load())
	}
	m := ai.GetMetrics()
	if m["successful_requests"] != int64(1) || m["failed_requests"] != int64(0) || m["success_rate"] != 1.0 {
		t.Errorf("metrics = %v", m)
	}
}

func TestCancellationDuringBackoffReturnsPromptly(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	cfg := testConfig()
	cfg.Retries = 3
	ai := NewAIIntegrationLayer(srv.URL, "", cfg)

	ctx, cancel := context.WithTimeout(context.Background(), 150*time.Millisecond)
	defer cancel()
	start := time.Now()
	err := ai.HealthCheck(ctx)
	elapsed := time.Since(start)

	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want context.DeadlineExceeded", err)
	}
	// The first backoff is 1s; returning well before it proves the wait
	// observes the caller's deadline.
	if elapsed > 900*time.Millisecond {
		t.Errorf("returned after %v; backoff ignored the context deadline", elapsed)
	}
	if hits.Load() != 1 {
		t.Errorf("server hits = %d, want 1", hits.Load())
	}
	if got := ai.GetMetrics()["failed_requests"]; got != int64(1) {
		t.Errorf("failed_requests = %v, want 1", got)
	}
}

func TestTimeoutsAndCircuitBreaker(t *testing.T) {
	// slow blocks until the client gives up, so only a timeout ends the call.
	// Draining the body lets the server notice the client disconnect.
	slow := func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}

	t.Run("caller deadline does not trip breaker", func(t *testing.T) {
		var fast atomic.Bool
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if fast.Load() {
				okData(t, w, nil)
				return
			}
			slow(w, r)
		}))
		defer srv.Close()

		cfg := testConfig()
		cfg.CircuitBreakerThreshold = 1
		ai := NewAIIntegrationLayer(srv.URL, "", cfg)

		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()
		if err := ai.HealthCheck(ctx); !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("err = %v, want context.DeadlineExceeded", err)
		}
		if state := ai.GetMetrics()["circuit_breaker_state"]; state != "closed" {
			t.Fatalf("breaker state = %v after caller timeout, want closed", state)
		}
		fast.Store(true)
		if err := ai.HealthCheck(context.Background()); err != nil {
			t.Fatalf("follow-up request rejected: %v", err)
		}
	})

	t.Run("client timeout opens breaker", func(t *testing.T) {
		var hits atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			hits.Add(1)
			slow(w, r)
		}))
		defer srv.Close()

		cfg := testConfig()
		cfg.Timeout = 50 * time.Millisecond
		cfg.CircuitBreakerThreshold = 1
		ai := NewAIIntegrationLayer(srv.URL, "", cfg)

		err := ai.HealthCheck(context.Background())
		if err == nil || !strings.Contains(err.Error(), "HTTP request failed (attempt 1)") {
			t.Fatalf("err = %v, want HTTP request failure", err)
		}
		if state := ai.GetMetrics()["circuit_breaker_state"]; state != "open" {
			t.Fatalf("breaker state = %v, want open", state)
		}

		err = ai.HealthCheck(context.Background())
		if err == nil || err.Error() != "circuit breaker is open" {
			t.Fatalf("err = %v, want circuit breaker rejection", err)
		}
		if hits.Load() != 1 {
			t.Errorf("server hits = %d; open breaker must not reach the service", hits.Load())
		}
		if trips := ai.GetMetrics()["circuit_breaker_trips"]; trips != int64(1) {
			t.Errorf("circuit_breaker_trips = %v, want 1", trips)
		}
	})
}

func TestCircuitBreakerHalfOpenProbe(t *testing.T) {
	cases := []struct {
		name        string
		probeFails  bool
		wantState   string
		wantProbeOK bool
	}{
		{name: "successful probe closes", probeFails: false, wantState: "closed", wantProbeOK: true},
		{name: "failed probe reopens", probeFails: true, wantState: "open", wantProbeOK: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var failing atomic.Bool
			failing.Store(true)
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if failing.Load() {
					w.WriteHeader(http.StatusInternalServerError)
					return
				}
				okData(t, w, nil)
			}))
			defer srv.Close()

			cfg := testConfig()
			cfg.CircuitBreakerThreshold = 2
			cfg.CircuitBreakerTimeout = 30 * time.Millisecond
			ai := NewAIIntegrationLayer(srv.URL, "", cfg)
			ctx := context.Background()

			ai.HealthCheck(ctx)
			if state := ai.GetMetrics()["circuit_breaker_state"]; state != "closed" {
				t.Fatalf("state after 1/2 failures = %v, want closed", state)
			}
			ai.HealthCheck(ctx)
			if state := ai.GetMetrics()["circuit_breaker_state"]; state != "open" {
				t.Fatalf("state after 2/2 failures = %v, want open", state)
			}
			if err := ai.HealthCheck(ctx); err == nil || err.Error() != "circuit breaker is open" {
				t.Fatalf("err = %v, want rejection while open", err)
			}

			time.Sleep(60 * time.Millisecond)
			failing.Store(tc.probeFails)
			err := ai.HealthCheck(ctx)
			if (err == nil) != tc.wantProbeOK {
				t.Fatalf("probe err = %v, want ok=%v", err, tc.wantProbeOK)
			}
			if state := ai.GetMetrics()["circuit_breaker_state"]; state != tc.wantState {
				t.Errorf("state after probe = %v, want %s", state, tc.wantState)
			}
		})
	}
}

func TestResourcePredictionCache(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		okData(t, w, map[string]interface{}{"predictions": []float64{42}})
	}))
	defer srv.Close()

	ai := NewAIIntegrationLayer(srv.URL, "", testConfig())
	ctx := context.Background()
	req := ResourcePredictionRequest{NodeID: "n1", ResourceType: "cpu", HorizonMinutes: 15}

	first, err := ai.PredictResourceDemand(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	second, err := ai.PredictResourceDemand(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if hits.Load() != 1 || second.Predictions[0] != first.Predictions[0] {
		t.Fatalf("identical request should be served from cache: hits=%d", hits.Load())
	}

	req.HorizonMinutes = 60
	if _, err := ai.PredictResourceDemand(ctx, req); err != nil {
		t.Fatal(err)
	}
	if hits.Load() != 2 {
		t.Errorf("different horizon must miss the cache: hits=%d", hits.Load())
	}
	m := ai.GetMetrics()
	if m["cache_hits"] != int64(1) || m["cache_misses"] != int64(2) {
		t.Errorf("cache metrics = hits %v misses %v, want 1/2", m["cache_hits"], m["cache_misses"])
	}
}

func TestResponseCacheEviction(t *testing.T) {
	cases := []struct {
		name    string
		prepare func(*ResponseCache)
		evicted string
		kept    []string
	}{
		{
			name: "expired entry evicted before least used",
			prepare: func(c *ResponseCache) {
				c.Set("stale", 1, -time.Second)
				c.Set("fresh", 2, time.Minute)
			},
			evicted: "stale",
			kept:    []string{"fresh", "new"},
		},
		{
			name: "least used entry evicted when none expired",
			prepare: func(c *ResponseCache) {
				c.Set("popular", 1, time.Minute)
				c.Set("cold", 2, time.Minute)
				c.Get("popular")
			},
			evicted: "cold",
			kept:    []string{"popular", "new"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := &ResponseCache{cache: make(map[string]CacheEntry), maxSize: 2}
			tc.prepare(c)
			c.Set("new", 3, time.Minute)

			if c.Get(tc.evicted) != nil {
				t.Errorf("%q should have been evicted", tc.evicted)
			}
			for _, k := range tc.kept {
				if c.Get(k) == nil {
					t.Errorf("%q should still be cached", k)
				}
			}
		})
	}

	t.Run("expired entry is not returned", func(t *testing.T) {
		c := &ResponseCache{cache: make(map[string]CacheEntry), maxSize: 2}
		c.Set("k", "v", -time.Millisecond)
		if got := c.Get("k"); got != nil {
			t.Errorf("Get returned expired value %v", got)
		}
	})
}

func TestIncompleteConfigFallsBackToDefaults(t *testing.T) {
	cases := []struct {
		name string
		cfg  AIConfig
	}{
		{name: "zero value", cfg: AIConfig{}},
		{name: "zero retries", cfg: AIConfig{MaxConnections: 5, Timeout: time.Second}},
		{name: "zero max connections", cfg: AIConfig{Retries: 1}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				okData(t, w, map[string]interface{}{"predictions": []float64{7}})
			}))
			defer srv.Close()

			resp, err := NewAIIntegrationLayer(srv.URL, "", tc.cfg).PredictResourceDemand(context.Background(), ResourcePredictionRequest{NodeID: "n"})
			if err != nil {
				t.Fatalf("PredictResourceDemand: %v", err)
			}
			if len(resp.Predictions) != 1 || resp.Predictions[0] != 7 {
				t.Errorf("predictions = %v", resp.Predictions)
			}
		})
	}
}

func TestMaxConnectionsLimitsInFlightRequests(t *testing.T) {
	const limit, callers = 3, 10
	arrived := make(chan struct{}, callers)
	release := make(chan struct{})
	var served atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		served.Add(1)
		arrived <- struct{}{}
		<-release
		okData(t, w, nil)
	}))
	defer srv.Close()
	defer close(release) // runs before srv.Close so blocked handlers can finish

	cfg := testConfig()
	cfg.MaxConnections = limit
	ai := NewAIIntegrationLayer(srv.URL, "", cfg)

	errs := make(chan error, callers)
	for range callers {
		go func() { errs <- ai.HealthCheck(context.Background()) }()
	}

	timeout := time.After(5 * time.Second)
	for range limit {
		select {
		case <-arrived:
		case <-timeout:
			t.Fatal("admitted requests never reached the server")
		}
	}
	for range callers - limit {
		select {
		case err := <-errs:
			if err == nil || err.Error() != "too many active requests" {
				t.Fatalf("err = %v, want too many active requests", err)
			}
		case <-timeout:
			t.Fatal("excess requests were not rejected")
		}
	}
	if got := served.Load(); got != limit {
		t.Errorf("server saw %d concurrent requests, want %d", got, limit)
	}
	if got := ai.GetMetrics()["active_requests"]; got != int32(limit) {
		t.Errorf("active_requests = %v, want %d", got, limit)
	}
}

func TestAverageResponseTimeIsMeanOfSuccesses(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			time.Sleep(80 * time.Millisecond)
		}
		okData(t, w, nil)
	}))
	defer srv.Close()

	ai := NewAIIntegrationLayer(srv.URL, "", testConfig())
	for range 2 {
		if err := ai.HealthCheck(context.Background()); err != nil {
			t.Fatal(err)
		}
	}
	// Total latency is at least 80ms over two successes, so the mean cannot be
	// below 40ms; the last (fast) request alone would report far less.
	if avg := ai.GetMetrics()["avg_response_time_ms"].(int64); avg < 40 {
		t.Errorf("avg_response_time_ms = %d, want >= 40 (mean, not last sample)", avg)
	}
}

func TestCloseWaitsForInFlightRequests(t *testing.T) {
	arrived := make(chan struct{})
	release := make(chan struct{})
	releaseOnce := sync.OnceFunc(func() { close(release) })
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(arrived)
		<-release
		okData(t, w, nil)
	}))
	defer srv.Close()
	defer releaseOnce() // runs before srv.Close so the blocked handler can finish

	ai := NewAIIntegrationLayer(srv.URL, "", testConfig())
	go ai.HealthCheck(context.Background())
	<-arrived

	closed := make(chan error, 1)
	go func() { closed <- ai.Close() }()

	select {
	case err := <-closed:
		t.Fatalf("Close returned %v while a request was in flight", err)
	case <-time.After(250 * time.Millisecond):
	}
	releaseOnce()
	select {
	case err := <-closed:
		if err != nil {
			t.Errorf("Close: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Close did not return after the request finished")
	}
}
