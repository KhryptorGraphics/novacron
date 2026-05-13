package compression

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestSyncWithNodePushesBaselines(t *testing.T) {
	received := make(chan baselineSyncRequest, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/dwcp/baselines/sync" {
			t.Errorf("path = %s, want /dwcp/baselines/sync", r.URL.Path)
		}
		if r.Header.Get("X-DWCP-Baseline-Sync") != "v1" {
			t.Errorf("missing baseline sync header")
		}
		var req baselineSyncRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request failed: %v", err)
		}
		select {
		case received <- req:
		default:
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	config := DefaultBaselineSyncConfig()
	config.Enabled = true
	config.SyncInterval = 0
	syncer := NewBaselineSynchronizer(config, zap.NewNop())
	defer syncer.Close()
	syncer.SetBaseline("vm-1-memory", &BaselineState{
		Data:       []byte("baseline"),
		Timestamp:  time.Unix(10, 0),
		DeltaCount: 3,
	})
	if err := syncer.RegisterNode("node-2", server.URL); err != nil {
		t.Fatalf("RegisterNode failed: %v", err)
	}

	if err := syncer.SyncWithCluster(context.Background()); err != nil {
		t.Fatalf("SyncWithCluster failed: %v", err)
	}

	select {
	case req := <-received:
		baseline := req.Baselines["vm-1-memory"]
		if baseline == nil {
			t.Fatal("baseline not sent")
		}
		if string(baseline.Data) != "baseline" || baseline.DeltaCount != 3 {
			t.Fatalf("unexpected baseline payload: %+v", baseline)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for baseline sync request")
	}

	syncer.nodesMutex.RLock()
	node := syncer.remoteNodes["node-2"]
	if node.Status != NodeStatusOnline || node.BaselineCount != 1 || node.LastSync.IsZero() {
		t.Fatalf("node status not updated after sync: %+v", node)
	}
	syncer.nodesMutex.RUnlock()
}

func TestSyncWithNodeMarksOfflineOnFailure(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "unavailable", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	config := DefaultBaselineSyncConfig()
	config.Enabled = true
	config.SyncInterval = 0
	syncer := NewBaselineSynchronizer(config, zap.NewNop())
	defer syncer.Close()
	syncer.SetBaseline("vm-1-memory", &BaselineState{Data: []byte("baseline")})
	if err := syncer.RegisterNode("node-2", server.URL); err != nil {
		t.Fatalf("RegisterNode failed: %v", err)
	}

	if err := syncer.SyncWithCluster(context.Background()); err != nil {
		t.Fatalf("SyncWithCluster should log and continue on per-node failures: %v", err)
	}

	syncer.nodesMutex.RLock()
	status := syncer.remoteNodes["node-2"].Status
	syncer.nodesMutex.RUnlock()
	if status != NodeStatusOffline {
		t.Fatalf("node status = %s, want offline", status)
	}
}

func TestBaselineSyncEndpoint(t *testing.T) {
	tests := map[string]string{
		"127.0.0.1:8080":           "http://127.0.0.1:8080/dwcp/baselines/sync",
		"https://node.example.com": "https://node.example.com/dwcp/baselines/sync",
		"http://node/custom":       "http://node/custom",
	}

	for input, want := range tests {
		got, err := baselineSyncEndpoint(input)
		if err != nil {
			t.Fatalf("baselineSyncEndpoint(%q) error: %v", input, err)
		}
		if got != want {
			t.Fatalf("baselineSyncEndpoint(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestMigrateBaselineValidatesVersionDirection(t *testing.T) {
	syncer := NewBaselineSynchronizer(DefaultBaselineSyncConfig(), zap.NewNop())
	syncer.SetBaseline("vm-1-memory", &BaselineState{
		Data:       []byte("baseline"),
		Timestamp:  time.Unix(10, 0),
		DeltaCount: 4,
	})

	if err := syncer.MigrateBaseline("vm-1-memory", 2, 1); err == nil {
		t.Fatal("expected downgrade migration error")
	}
	if err := syncer.MigrateBaseline("vm-1-memory", 1, 2); err != nil {
		t.Fatalf("MigrateBaseline failed: %v", err)
	}

	baseline, ok := syncer.GetBaseline("vm-1-memory")
	if !ok {
		t.Fatal("baseline missing after migration")
	}
	if baseline.DeltaCount != 0 {
		t.Fatalf("DeltaCount = %d, want 0 after migration", baseline.DeltaCount)
	}
	if !baseline.Timestamp.After(time.Unix(10, 0)) {
		t.Fatalf("timestamp was not refreshed: %v", baseline.Timestamp)
	}
}
