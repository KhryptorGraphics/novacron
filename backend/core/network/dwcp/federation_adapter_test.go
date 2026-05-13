package dwcp

import (
	"encoding/json"
	"testing"
	"time"

	"go.uber.org/zap/zaptest"
)

func TestEncodeStateUpdateEnvelope(t *testing.T) {
	update := &StateUpdate{
		UpdateID:        "update-1",
		SourceCluster:   "cluster-a",
		TargetClusters:  []string{"cluster-b"},
		StateData:       []byte("state-data"),
		DeltaOnly:       true,
		BaselineVersion: 7,
		Timestamp:       time.Unix(100, 0).UTC(),
		Priority:        5,
	}

	data := encodeStateUpdate(update)
	var envelope stateUpdateEnvelope
	if err := json.Unmarshal(data, &envelope); err != nil {
		t.Fatalf("encodeStateUpdate produced invalid JSON: %v", err)
	}

	if envelope.Type != "dwcp_state_update" {
		t.Fatalf("Type = %q, want dwcp_state_update", envelope.Type)
	}
	if envelope.Version != 1 {
		t.Fatalf("Version = %d, want 1", envelope.Version)
	}
	if envelope.Update == nil || envelope.Update.UpdateID != update.UpdateID {
		t.Fatalf("Update = %#v, want update %q", envelope.Update, update.UpdateID)
	}
	if string(envelope.Update.StateData) != string(update.StateData) {
		t.Fatalf("StateData = %q, want %q", envelope.Update.StateData, update.StateData)
	}
}

func TestCollectClusterStateSnapshot(t *testing.T) {
	adapter := NewFederationAdapter(zaptest.NewLogger(t), DefaultFederationConfig())

	conn := &ClusterConnection{
		ClusterID:       "cluster-a",
		Region:          "us-west",
		Endpoint:        "cluster-a.example:9443",
		lastSeen:        time.Unix(100, 0).UTC(),
		baselineID:      "baseline-a",
		baselineVersion: 3,
		lastSync:        time.Unix(110, 0).UTC(),
		compressionRate: 2.5,
	}
	conn.connected.Store(true)
	conn.bytesSent.Store(1024)
	conn.bytesReceived.Store(2048)
	conn.messagesCount.Store(9)

	adapter.clusterConnections[conn.ClusterID] = conn
	adapter.regionManagers[conn.Region] = &RegionManager{
		RegionID:      conn.Region,
		Clusters:      map[string]*ClusterConnection{conn.ClusterID: conn},
		Topology:      "mesh",
		LeaderCluster: conn.ClusterID,
	}
	adapter.metrics.TotalBytesSent.Store(1024)
	adapter.metrics.MessageCount.Store(9)

	data := adapter.collectClusterState()
	var snapshot clusterStateSnapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		t.Fatalf("collectClusterState produced invalid JSON: %v", err)
	}

	if snapshot.Type != "dwcp_cluster_state" {
		t.Fatalf("Type = %q, want dwcp_cluster_state", snapshot.Type)
	}
	if len(snapshot.Clusters) != 1 {
		t.Fatalf("Clusters len = %d, want 1", len(snapshot.Clusters))
	}
	if snapshot.Clusters[0].ClusterID != conn.ClusterID || !snapshot.Clusters[0].Connected {
		t.Fatalf("Cluster snapshot = %#v", snapshot.Clusters[0])
	}
	if snapshot.Metrics.TotalBytesSent != 1024 || snapshot.Metrics.MessageCount != 9 {
		t.Fatalf("Metrics snapshot = %#v", snapshot.Metrics)
	}
}

func TestHDEEngineCompressUsesRealCompression(t *testing.T) {
	engine := NewHDEEngine(1024)
	data := make([]byte, 8192)
	for i := range data {
		data[i] = byte(i % 4)
	}

	compressed, ratio := engine.Compress(data, "baseline-a")
	if len(compressed) == 0 {
		t.Fatal("Compress returned empty payload")
	}
	if len(compressed) >= len(data) {
		t.Fatalf("compressed len = %d, want less than %d", len(compressed), len(data))
	}
	if ratio <= 1.0 {
		t.Fatalf("ratio = %f, want > 1", ratio)
	}
}
