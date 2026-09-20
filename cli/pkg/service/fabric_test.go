package service

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/novacron/cli/pkg/api"
)

// recordedServiceRequest is the last request a test server received.
type recordedServiceRequest struct {
	Method string
	Path   string
	Body   string
}

// startFabricServiceServer starts an API server that answers every request with
// status/body and records the request it received.
func startFabricServiceServer(t *testing.T, status int, body string) (*httptest.Server, *recordedServiceRequest) {
	t.Helper()

	recorded := &recordedServiceRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}
		recorded.Method, recorded.Path, recorded.Body = r.Method, r.URL.Path, string(payload)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	return server, recorded
}

// newFabricTestService points a fabric service at a test server.
func newFabricTestService(t *testing.T, server *httptest.Server) *FabricService {
	t.Helper()

	client, err := api.NewClient(server.URL)
	if err != nil {
		t.Fatalf("api.NewClient: %v", err)
	}

	return NewFabricService(client)
}

func TestFabricServiceRequests(t *testing.T) {
	tests := []struct {
		name       string
		response   string
		wantMethod string
		wantPath   string
		wantBody   string
		call       func(context.Context, *FabricService) (interface{}, error)
		want       interface{}
	}{
		{
			name:       "list nodes unwraps the nodes array",
			response:   `{"nodes":[{"node_id":"node-a","addr":"127.0.0.1:18090","arch":"arm64","cores":14,"mem_total_mb":125748,"storage_total_gb":3752,"storage_free_gb":912,"vm_count":2,"reachable":true,"link":null},{"node_id":"node-b","reachable":false,"link":{"rtt_ms":0.341,"last_heartbeat":"2026-09-20T11:29:24Z","stale":false}}]}`,
			wantMethod: http.MethodGet,
			wantPath:   "/api/cluster/nodes",
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.ListNodes(ctx)
			},
			want: []api.FabricNode{
				{
					NodeID:         "node-a",
					Addr:           "127.0.0.1:18090",
					Arch:           "arm64",
					Cores:          14,
					MemTotalMB:     125748,
					StorageTotalGB: 3752,
					StorageFreeGB:  912,
					VMCount:        2,
					Reachable:      true,
				},
				{
					NodeID: "node-b",
					Link: &api.FabricLinkProfile{
						RTTMS:         0.341,
						LastHeartbeat: "2026-09-20T11:29:24Z",
					},
				},
			},
		},
		{
			name:       "submit job sends the spec and reads back placement",
			response:   `{"job_id":"job-1","vm_id":"vm-1","node_id":"node-b","status":"running","placement":{"decision":"cost","cost_estimate_s":0.125,"reason":"fewest bytes to move"}}`,
			wantMethod: http.MethodPost,
			wantPath:   "/api/compute/jobs",
			wantBody:   `{"command":"echo","args":["hi"],"env":{"A":"1"},"node_id":"node-b","bytes_to_move":4096,"memory_mb":256,"vcpus":2}`,
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.SubmitJob(ctx, api.FabricJobSpec{
					Command:     "echo",
					Args:        []string{"hi"},
					Env:         map[string]string{"A": "1"},
					NodeID:      "node-b",
					BytesToMove: 4096,
					MemoryMB:    256,
					VCPUs:       2,
				})
			},
			want: &api.FabricJobSubmission{
				JobID:  "job-1",
				VMID:   "vm-1",
				NodeID: "node-b",
				Status: "running",
				Placement: &api.FabricPlacement{
					Decision:      "cost",
					CostEstimateS: float64Ptr(0.125),
					Reason:        "fewest bytes to move",
				},
			},
		},
		{
			name:       "list jobs unwraps the jobs array",
			response:   `{"jobs":[{"job_id":"job-1","name":"demo","command":"echo","status":"completed","node_id":"node-a","vm_id":"vm-1","created_at":"2026-09-20T11:29:24Z"}]}`,
			wantMethod: http.MethodGet,
			wantPath:   "/api/compute/jobs",
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.ListJobs(ctx)
			},
			want: []api.FabricJob{{
				JobID:     "job-1",
				Name:      "demo",
				Command:   "echo",
				Status:    "completed",
				NodeID:    "node-a",
				VMID:      "vm-1",
				CreatedAt: "2026-09-20T11:29:24Z",
			}},
		},
		{
			name:       "get job decodes the log tails",
			response:   `{"job_id":"job-1","command":"echo","status":"completed","node_id":"node-a","vm_id":"vm-1","created_at":"2026-09-20T11:29:24Z","logs":{"stdout":"hello\n","stderr":""}}`,
			wantMethod: http.MethodGet,
			wantPath:   "/api/compute/jobs/job-1",
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.GetJob(ctx, "job-1")
			},
			want: &api.FabricJobDetail{
				FabricJob: api.FabricJob{
					JobID:     "job-1",
					Command:   "echo",
					Status:    "completed",
					NodeID:    "node-a",
					VMID:      "vm-1",
					CreatedAt: "2026-09-20T11:29:24Z",
				},
				Logs: &api.FabricJobLogs{Stdout: "hello\n"},
			},
		},
		{
			name:       "cancel job posts to the cancel endpoint",
			response:   `{"cancelled":false,"status":"failed-to-cancel","error":"pinned node is not reachable"}`,
			wantMethod: http.MethodPost,
			wantPath:   "/api/compute/jobs/job-1/cancel",
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.CancelJob(ctx, "job-1")
			},
			want: &api.FabricJobCancellation{
				Status: "failed-to-cancel",
				Error:  "pinned node is not reachable",
			},
		},
		{
			name:       "list transfers accepts the wrapped shape",
			response:   `{"transfers":[{"transfer_id":"tr-1","status":"running","bytes_total":2048,"bytes_moved":1024,"measured_bps":512,"compression":"zstd-multifd","eta_seconds":4}]}`,
			wantMethod: http.MethodGet,
			wantPath:   "/api/transfers",
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.ListTransfers(ctx)
			},
			want: []api.FabricTransfer{{
				TransferID:  "tr-1",
				Status:      "running",
				BytesTotal:  2048,
				BytesMoved:  1024,
				MeasuredBps: 512,
				Compression: "zstd-multifd",
				ETASeconds:  float64Ptr(4),
			}},
		},
		{
			name:       "list transfers accepts a bare array",
			response:   `[{"transfer_id":"tr-2","status":"queued","compression":"none"}]`,
			wantMethod: http.MethodGet,
			wantPath:   "/api/transfers",
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.ListTransfers(ctx)
			},
			want: []api.FabricTransfer{{TransferID: "tr-2", Status: "queued", Compression: "none"}},
		},
		{
			name:       "get transfer decodes progress and decision inputs",
			response:   `{"transfer_id":"tr-1","status":"running","bytes_total":2048,"bytes_moved":1024,"measured_bps":512,"compression":"zstd-multifd","eta_seconds":4,"decision_inputs":{"link_bps":1000000,"sample_ratio":0.1,"threshold":0.75}}`,
			wantMethod: http.MethodGet,
			wantPath:   "/api/transfers/tr-1",
			call: func(ctx context.Context, s *FabricService) (interface{}, error) {
				return s.GetTransfer(ctx, "tr-1")
			},
			want: &api.FabricTransfer{
				TransferID:  "tr-1",
				Status:      "running",
				BytesTotal:  2048,
				BytesMoved:  1024,
				MeasuredBps: 512,
				Compression: "zstd-multifd",
				ETASeconds:  float64Ptr(4),
				DecisionInputs: map[string]interface{}{
					"link_bps":     float64(1000000),
					"sample_ratio": 0.1,
					"threshold":    0.75,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServiceServer(t, http.StatusOK, tt.response)

			got, err := tt.call(context.Background(), newFabricTestService(t, server))
			if err != nil {
				t.Fatalf("call failed: %v", err)
			}

			if recorded.Method != tt.wantMethod || recorded.Path != tt.wantPath {
				t.Errorf("request = %s %s, want %s %s", recorded.Method, recorded.Path, tt.wantMethod, tt.wantPath)
			}
			if recorded.Body != tt.wantBody {
				t.Errorf("request body = %q, want %q", recorded.Body, tt.wantBody)
			}

			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("result = %#v, want %#v", got, tt.want)
			}
		})
	}
}

func TestFabricServiceSurfacesAPIReason(t *testing.T) {
	server, _ := startFabricServiceServer(t, http.StatusServiceUnavailable,
		`{"error":"no node in the fabric can fit the job"}`)

	_, err := newFabricTestService(t, server).SubmitJob(context.Background(), api.FabricJobSpec{Command: "echo"})
	if err == nil {
		t.Fatal("expected a 503 to fail the submission")
	}
	if !strings.Contains(err.Error(), "failed to submit job") {
		t.Errorf("error = %q, want it to name the failed operation", err)
	}
	if !strings.Contains(err.Error(), "no node in the fabric can fit the job") {
		t.Errorf("error = %q, want the server's reason", err)
	}
}

// float64Ptr takes the address of a value in table literals. The module's go
// directive (1.21) predates the new(expr) form.
func float64Ptr(value float64) *float64 {
	return &value
}
