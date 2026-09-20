package commands

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/novacron/cli/pkg/api"
	"github.com/novacron/cli/pkg/service"
)

// recordedRequest is the last request the test server received.
type recordedRequest struct {
	Method string
	Path   string
	Body   []byte
}

// startFabricServer starts an API test server that answers every request with
// status/body and records the request it received.
func startFabricServer(t *testing.T, status int, body string) (*httptest.Server, *recordedRequest) {
	t.Helper()

	recorded := &recordedRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("failed to read request body: %v", err)
		}
		recorded.Method, recorded.Path, recorded.Body = r.Method, r.URL.Path, payload

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(server.Close)

	return server, recorded
}

// runFabricCommand runs `novacron fabric <args...>` against server and returns
// what the command wrote.
func runFabricCommand(t *testing.T, server *httptest.Server, args ...string) (string, error) {
	t.Helper()

	client, err := api.NewClient(server.URL)
	if err != nil {
		t.Fatalf("api.NewClient: %v", err)
	}

	original := newFabricService
	newFabricService = func() (*service.FabricService, error) {
		return service.NewFabricService(client), nil
	}
	t.Cleanup(func() { newFabricService = original })

	cmd := NewFabricCommand()
	var out bytes.Buffer
	cmd.SetOut(&out)
	cmd.SetErr(&out)
	cmd.SetArgs(args)

	err = cmd.Execute()

	return out.String(), err
}

func TestFabricNodesOutput(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{
			name: "local node without link and probed peer",
			payload: `{"nodes":[
				{"node_id":"node-a","arch":"arm64","cores":14,"mem_total_mb":125748,
				 "mem_allocated_mb":4096,"storage_total_gb":3752,"storage_free_gb":912,
				 "vm_count":1,"reachable":true,"link":null},
				{"node_id":"node-b","addr":"127.0.0.1:18091","arch":"arm64","cores":8,
				 "mem_total_mb":65536,"mem_allocated_mb":0,"storage_total_gb":1000,
				 "storage_free_gb":500,"vm_count":0,"reachable":true,
				 "link":{"rtt_ms":0.341,"last_heartbeat":"2026-09-20T11:29:24Z","stale":false}}
			]}`,
			want: "NODE    ADDR             ARCH   CORES  MEM FREE   STORAGE FREE  REACHABLE  RTT\n" +
				"node-a  -                arm64  14     118.8 GiB  912.0 GiB     yes        -\n" +
				"node-b  127.0.0.1:18091  arm64  8      64.0 GiB   500.0 GiB     yes        0.34ms\n",
		},
		{
			name:    "unreachable peer with no capacity reported",
			payload: `{"nodes":[{"node_id":"node-c","addr":"10.0.0.5:18091","arch":"","cores":0,"mem_total_mb":0,"mem_allocated_mb":0,"storage_total_gb":0,"storage_free_gb":0,"vm_count":0,"reachable":false,"link":{"rtt_ms":12.5,"last_heartbeat":"2026-09-20T10:00:00Z","stale":true}}]}`,
			want: "NODE    ADDR            ARCH  CORES  MEM FREE  STORAGE FREE  REACHABLE  RTT\n" +
				"node-c  10.0.0.5:18091  -     0      0 B       0 B           no         12.50ms (stale)\n",
		},
		{
			name:    "no nodes",
			payload: `{"nodes":[]}`,
			want:    "No fabric nodes found\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServer(t, http.StatusOK, tt.payload)

			out, err := runFabricCommand(t, server, "nodes")
			if err != nil {
				t.Fatalf("fabric nodes: %v", err)
			}

			if recorded.Method != http.MethodGet || recorded.Path != "/api/cluster/nodes" {
				t.Errorf("request = %s %s, want GET /api/cluster/nodes", recorded.Method, recorded.Path)
			}

			if out != tt.want {
				t.Errorf("output =\n%s\nwant\n%s", out, tt.want)
			}
		})
	}
}

func TestFabricJobsOutput(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{
			name: "jobs from several nodes",
			payload: `{"jobs":[
				{"job_id":"job-1","name":"demo","command":"echo","status":"completed",
				 "node_id":"node-a","vm_id":"vm-1","created_at":"2026-09-20T11:29:24Z"},
				{"job_id":"job-2","command":"sleep","status":"running",
				 "node_id":"node-b","vm_id":"vm-2","created_at":"2026-09-20T11:30:00Z",
				 "error":"none"}
			]}`,
			want: "JOB ID  NAME  STATUS     NODE    VM    COMMAND  CREATED\n" +
				"job-1   demo  completed  node-a  vm-1  echo     2026-09-20T11:29:24Z\n" +
				"job-2   -     running    node-b  vm-2  sleep    2026-09-20T11:30:00Z\n",
		},
		{
			name:    "no jobs",
			payload: `{"jobs":[]}`,
			want:    "No fabric jobs found\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServer(t, http.StatusOK, tt.payload)

			out, err := runFabricCommand(t, server, "jobs")
			if err != nil {
				t.Fatalf("fabric jobs: %v", err)
			}

			if recorded.Method != http.MethodGet || recorded.Path != "/api/compute/jobs" {
				t.Errorf("request = %s %s, want GET /api/compute/jobs", recorded.Method, recorded.Path)
			}

			if out != tt.want {
				t.Errorf("output =\n%s\nwant\n%s", out, tt.want)
			}
		})
	}
}

func TestFabricJobSubmitRequest(t *testing.T) {
	const response = `{"job_id":"job-1","vm_id":"vm-1","node_id":"node-b","status":"running",
		"placement":{"decision":"locality","cost_estimate_s":0.125,"reason":"fewest bytes to move"}}`

	tests := []struct {
		name string
		args []string
		want map[string]interface{}
	}{
		{
			name: "command only",
			args: []string{"job", "submit", "--command", "echo"},
			want: map[string]interface{}{"command": "echo"},
		},
		{
			name: "every flag",
			args: []string{
				"job", "submit", "--name", "demo", "--command", "sh",
				"--arg", "-c", "--arg", "echo hi",
				"--env", "A=1", "--env", "B=2",
				"--node", "node-b", "--memory-mb", "256", "--vcpus", "2", "--bytes-to-move", "4096",
			},
			want: map[string]interface{}{
				"name":          "demo",
				"command":       "sh",
				"args":          []interface{}{"-c", "echo hi"},
				"env":           map[string]interface{}{"A": "1", "B": "2"},
				"node_id":       "node-b",
				"memory_mb":     float64(256),
				"vcpus":         float64(2),
				"bytes_to_move": float64(4096),
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServer(t, http.StatusCreated, response)

			out, err := runFabricCommand(t, server, tt.args...)
			if err != nil {
				t.Fatalf("fabric job submit: %v", err)
			}

			if recorded.Method != http.MethodPost || recorded.Path != "/api/compute/jobs" {
				t.Errorf("request = %s %s, want POST /api/compute/jobs", recorded.Method, recorded.Path)
			}

			body := map[string]interface{}{}
			if err := json.Unmarshal(recorded.Body, &body); err != nil {
				t.Fatalf("request body %q: %v", recorded.Body, err)
			}
			if !reflect.DeepEqual(body, tt.want) {
				t.Errorf("request body = %#v, want %#v", body, tt.want)
			}

			want := "Job ID:      job-1\n" +
				"Status:      running\n" +
				"Node:        node-b\n" +
				"VM:          vm-1\n" +
				"Placement:   locality\n" +
				"Reason:      fewest bytes to move\n" +
				"Cost:        0.125s\n"
			if out != want {
				t.Errorf("output =\n%s\nwant\n%s", out, want)
			}
		})
	}
}

func TestFabricJobSubmitRejectsBadInput(t *testing.T) {
	tests := []struct {
		name    string
		args    []string
		wantErr string
	}{
		{
			name:    "missing command flag",
			args:    []string{"job", "submit"},
			wantErr: `required flag(s) "command" not set`,
		},
		{
			name:    "blank command",
			args:    []string{"job", "submit", "--command", "   "},
			wantErr: "--command is required",
		},
		{
			name:    "env without a value",
			args:    []string{"job", "submit", "--command", "echo", "--env", "NOEQUALS"},
			wantErr: `invalid --env "NOEQUALS": expected KEY=VALUE`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServer(t, http.StatusCreated, `{}`)

			_, err := runFabricCommand(t, server, tt.args...)
			if err == nil {
				t.Fatalf("expected an error containing %q", tt.wantErr)
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Errorf("error = %q, want it to contain %q", err, tt.wantErr)
			}
			if recorded.Path != "" {
				t.Errorf("unexpected request %s %s with body %s", recorded.Method, recorded.Path, recorded.Body)
			}
		})
	}
}

func TestFabricJobSubmitSurfacesPlacementFailure(t *testing.T) {
	server, _ := startFabricServer(t, http.StatusServiceUnavailable,
		`{"error":"no node in the fabric can fit the job"}`)

	_, err := runFabricCommand(t, server, "job", "submit", "--command", "echo")
	if err == nil {
		t.Fatal("expected the 503 to fail the command")
	}
	if !strings.Contains(err.Error(), "no node in the fabric can fit the job") {
		t.Errorf("error = %q, want the server's reason", err)
	}
}

func TestFabricJobStatusOutput(t *testing.T) {
	tests := []struct {
		name     string
		payload  string
		wantPath string
		want     string
	}{
		{
			name: "completed job with captured output",
			payload: `{"job_id":"job-1","name":"demo","command":"echo","status":"completed",
				"node_id":"node-a","vm_id":"vm-1","created_at":"2026-09-20T11:29:24Z",
				"logs":{"stdout":"hello\nworld\n","stderr":""}}`,
			wantPath: "/api/compute/jobs/job-1",
			want: "Job ID:      job-1\n" +
				"Name:        demo\n" +
				"Command:     echo\n" +
				"Status:      completed\n" +
				"Node:        node-a\n" +
				"VM:          vm-1\n" +
				"Created:     2026-09-20T11:29:24Z\n" +
				"\n--- stdout (tail) ---\nhello\nworld\n" +
				"\n--- stderr (tail) ---\n(empty)\n",
		},
		{
			name: "failed job with an error and unterminated output",
			payload: `{"job_id":"job-2","command":"false","status":"failed","node_id":"node-a",
				"vm_id":"vm-2","created_at":"2026-09-20T11:30:00Z","error":"exit status 1",
				"logs":{"stdout":"","stderr":"boom"}}`,
			wantPath: "/api/compute/jobs/job-2",
			want: "Job ID:      job-2\n" +
				"Command:     false\n" +
				"Status:      failed\n" +
				"Node:        node-a\n" +
				"VM:          vm-2\n" +
				"Created:     2026-09-20T11:30:00Z\n" +
				"Error:       exit status 1\n" +
				"\n--- stdout (tail) ---\n(empty)\n" +
				"\n--- stderr (tail) ---\nboom\n",
		},
		{
			name: "remote job whose logs could not be fetched",
			payload: `{"job_id":"job-3","command":"sleep","status":"running","node_id":"node-b",
				"vm_id":"vm-3","created_at":"2026-09-20T11:31:00Z","logs_error":"peer unreachable"}`,
			wantPath: "/api/compute/jobs/job-3",
			want: "Job ID:      job-3\n" +
				"Command:     sleep\n" +
				"Status:      running\n" +
				"Node:        node-b\n" +
				"VM:          vm-3\n" +
				"Created:     2026-09-20T11:31:00Z\n" +
				"\nLogs unavailable: peer unreachable\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServer(t, http.StatusOK, tt.payload)

			jobID := strings.TrimPrefix(tt.wantPath, "/api/compute/jobs/")
			out, err := runFabricCommand(t, server, "job", "status", jobID)
			if err != nil {
				t.Fatalf("fabric job status: %v", err)
			}

			if recorded.Method != http.MethodGet || recorded.Path != tt.wantPath {
				t.Errorf("request = %s %s, want GET %s", recorded.Method, recorded.Path, tt.wantPath)
			}

			if out != tt.want {
				t.Errorf("output =\n%s\nwant\n%s", out, tt.want)
			}
		})
	}
}

func TestFabricJobCancelOutput(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
		wantErr string
	}{
		{
			name:    "cancelled",
			payload: `{"cancelled":true,"status":"cancelled"}`,
			want:    "Job job-1 cancelled (status cancelled)\n",
		},
		{
			name:    "failed to cancel",
			payload: `{"cancelled":false,"status":"failed-to-cancel","error":"job's node \"node-b\" is not registered"}`,
			wantErr: `job job-1 was not cancelled: job's node "node-b" is not registered`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServer(t, http.StatusOK, tt.payload)

			out, err := runFabricCommand(t, server, "job", "cancel", "job-1")

			if recorded.Method != http.MethodPost || recorded.Path != "/api/compute/jobs/job-1/cancel" {
				t.Errorf("request = %s %s, want POST /api/compute/jobs/job-1/cancel", recorded.Method, recorded.Path)
			}

			if tt.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
					t.Fatalf("error = %v, want it to contain %q", err, tt.wantErr)
				}
				return
			}

			if err != nil {
				t.Fatalf("fabric job cancel: %v", err)
			}
			if out != tt.want {
				t.Errorf("output = %q, want %q", out, tt.want)
			}
		})
	}
}

func TestFabricTransfersOutput(t *testing.T) {
	tests := []struct {
		name    string
		payload string
		want    string
	}{
		{
			name:    "no transfers",
			payload: `{"transfers":[]}`,
			want:    "No transfers found\n",
		},
		{
			name: "one running transfer",
			payload: `{"transfers":[{"transfer_id":"tr-1","status":"running","bytes_total":2147483648,
				"bytes_moved":1073741824,"measured_bps":12582912,"compression":"zstd-multifd",
				"eta_seconds":85}]}`,
			want: "TRANSFER ID  STATUS   MOVED    TOTAL    RATE        ETA    COMPRESSION\n" +
				"tr-1         running  1.0 GiB  2.0 GiB  12.0 MiB/s  1m25s  zstd-multifd\n",
		},
		{
			name:    "queued transfer with nothing measured yet, bare array",
			payload: `[{"transfer_id":"tr-2","status":"queued","bytes_total":0,"bytes_moved":0,"measured_bps":0,"compression":"none"}]`,
			want: "TRANSFER ID  STATUS  MOVED  TOTAL  RATE  ETA  COMPRESSION\n" +
				"tr-2         queued  0 B    0 B    -     -    none\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			server, recorded := startFabricServer(t, http.StatusOK, tt.payload)

			out, err := runFabricCommand(t, server, "transfers")
			if err != nil {
				t.Fatalf("fabric transfers: %v", err)
			}

			if recorded.Method != http.MethodGet || recorded.Path != "/api/transfers" {
				t.Errorf("request = %s %s, want GET /api/transfers", recorded.Method, recorded.Path)
			}

			if out != tt.want {
				t.Errorf("output =\n%s\nwant\n%s", out, tt.want)
			}
		})
	}
}

func TestFabricTransferDetailOutput(t *testing.T) {
	const payload = `{"transfer_id":"tr-9","status":"running","bytes_total":2147483648,
		"bytes_moved":1073741824,"measured_bps":12582912,"compression":"zstd-multifd",
		"eta_seconds":85,"decision_inputs":{"threshold":0.75,"link_bps":1000000,"sample_ratio":0.1}}`

	server, recorded := startFabricServer(t, http.StatusOK, payload)

	out, err := runFabricCommand(t, server, "transfer", "tr-9")
	if err != nil {
		t.Fatalf("fabric transfer: %v", err)
	}

	if recorded.Method != http.MethodGet || recorded.Path != "/api/transfers/tr-9" {
		t.Errorf("request = %s %s, want GET /api/transfers/tr-9", recorded.Method, recorded.Path)
	}

	want := "Transfer ID: tr-9\n" +
		"Status:      running\n" +
		"Moved:       1.0 GiB / 2.0 GiB\n" +
		"Rate:        12.0 MiB/s\n" +
		"Compression: zstd-multifd\n" +
		"ETA:         1m25s\n" +
		"Decision inputs:\n" +
		"  link_bps: 1000000\n" +
		"  sample_ratio: 0.1\n" +
		"  threshold: 0.75\n"
	if out != want {
		t.Errorf("output =\n%s\nwant\n%s", out, want)
	}
}

func TestFabricTransferNotFoundSurfacesServerReason(t *testing.T) {
	server, _ := startFabricServer(t, http.StatusNotFound, `{"error":"transfer not found"}`)

	_, err := runFabricCommand(t, server, "transfer", "tr-missing")
	if err == nil {
		t.Fatal("expected a 404 to fail the command")
	}
	if !strings.Contains(err.Error(), "transfer not found") {
		t.Errorf("error = %q, want the server's reason", err)
	}
}
