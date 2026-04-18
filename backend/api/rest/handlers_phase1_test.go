package rest

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/gorilla/mux"
	corevm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

func newRESTTestRouter(t *testing.T, vmManager *corevm.VMManager) *mux.Router {
	t.Helper()
	router := mux.NewRouter()
	NewAPIHandler(vmManager, nil).RegisterRoutes(router)
	return router
}

func TestCreateVMRejectsMissingOwnership(t *testing.T) {
	manager, err := corevm.NewVMManager(corevm.DefaultVMManagerConfig())
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}
	defer manager.Stop()

	router := newRESTTestRouter(t, manager)
	body, _ := json.Marshal(CreateVMRequest{
		Name:   "missing-ownership",
		Type:   corevm.VMTypeKVM,
		CPU:    1,
		Memory: 512,
		Disk:   10,
	})

	req := httptest.NewRequest(http.MethodPost, "/api/vms", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rec.Code, rec.Body.String())
	}
}

func TestListNodesReturnsSchedulerInventory(t *testing.T) {
	manager, err := corevm.NewVMManager(corevm.DefaultVMManagerConfig())
	if err != nil {
		t.Fatalf("new vm manager: %v", err)
	}
	defer manager.Stop()

	if err := manager.RegisterSchedulerNode(&corevm.NodeResourceInfo{
		NodeID:             "node-a",
		TotalCPU:           8,
		UsedCPU:            3,
		TotalMemoryMB:      16384,
		UsedMemoryMB:       4096,
		TotalDiskGB:        200,
		UsedDiskGB:         50,
		CPUUsagePercent:    37.5,
		MemoryUsagePercent: 25.0,
		DiskUsagePercent:   25.0,
		VMCount:            2,
		Status:             "available",
		Labels:             map[string]string{"zone": "edge-a"},
	}); err != nil {
		t.Fatalf("register scheduler node: %v", err)
	}

	router := newRESTTestRouter(t, manager)
	req := httptest.NewRequest(http.MethodGet, "/api/cluster/nodes", nil)
	rec := httptest.NewRecorder()
	router.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rec.Code, rec.Body.String())
	}

	var nodes []Node
	if err := json.Unmarshal(rec.Body.Bytes(), &nodes); err != nil {
		t.Fatalf("unmarshal nodes: %v", err)
	}
	if len(nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(nodes))
	}
	if got, want := nodes[0].ID, "node-a"; got != want {
		t.Fatalf("node id = %q, want %q", got, want)
	}
	if got, want := nodes[0].RemainingCPU, 5; got != want {
		t.Fatalf("remaining cpu = %d, want %d", got, want)
	}
	if got, want := nodes[0].RemainingMemoryMB, int64(12288); got != want {
		t.Fatalf("remaining memory = %d, want %d", got, want)
	}
	if !nodes[0].Schedulable {
		t.Fatal("expected node to be schedulable")
	}
}
