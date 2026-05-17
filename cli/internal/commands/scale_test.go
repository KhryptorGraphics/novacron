package commands

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestScaleUpdatesAutoscalingTargetReplicas(t *testing.T) {
	withTempHome(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut {
			t.Fatalf("expected PUT, got %s", r.Method)
		}
		if r.URL.Path != "/orchestration/autoscaling/targets/web" {
			t.Fatalf("expected autoscaling target path, got %s", r.URL.Path)
		}
		var req autoscalingTarget
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode scale request: %v", err)
		}
		if req.ID != "web" || req.Type != "vm" || !req.Enabled {
			t.Fatalf("unexpected target identity: %#v", req)
		}
		if req.Thresholds == nil || req.Thresholds.MinReplicas != 3 || req.Thresholds.MaxReplicas != 3 {
			t.Fatalf("unexpected scale thresholds: %#v", req.Thresholds)
		}
		_ = json.NewEncoder(w).Encode(req)
	}))
	defer server.Close()
	addCurrentTestCluster(t, server.URL)

	var output bytes.Buffer
	cmd := NewScaleCommand()
	cmd.SetOut(&output)
	cmd.SetArgs([]string{"web", "--replicas", "3", "--type", "vm"})
	if err := cmd.Execute(); err != nil {
		t.Fatalf("scale failed: %v", err)
	}

	for _, expected := range []string{"id: web", "type: vm", "enabled: true", "min_replicas: 3", "max_replicas: 3"} {
		if !strings.Contains(output.String(), expected) {
			t.Fatalf("expected scale output to contain %q, got:\n%s", expected, output.String())
		}
	}
}

func TestScaleRequiresReplicas(t *testing.T) {
	cmd := NewScaleCommand()
	cmd.SetArgs([]string{"web"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "replicas is required") {
		t.Fatalf("expected replicas validation error, got %v", err)
	}
}

func TestScaleRejectsNegativeReplicas(t *testing.T) {
	cmd := NewScaleCommand()
	cmd.SetArgs([]string{"web", "--replicas", "-1"})

	if err := cmd.Execute(); err == nil || !strings.Contains(err.Error(), "replicas must be non-negative") {
		t.Fatalf("expected replicas validation error, got %v", err)
	}
}
