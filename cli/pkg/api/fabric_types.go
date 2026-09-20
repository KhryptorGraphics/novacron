package api

// Types for the bandwidth-aware peer-to-peer compute fabric (API v1):
// node inventory with live link state, compute jobs placed on a node, and bulk
// data transfers between nodes.

// FabricNode is one node's live capacity plus its measured link profile.
type FabricNode struct {
	NodeID         string             `json:"node_id"`
	Addr           string             `json:"addr,omitempty"`
	Arch           string             `json:"arch"`
	Cores          int                `json:"cores"`
	MemTotalMB     int64              `json:"mem_total_mb"`
	MemAllocatedMB int64              `json:"mem_allocated_mb"`
	StorageTotalGB int64              `json:"storage_total_gb"`
	StorageFreeGB  int64              `json:"storage_free_gb"`
	VMCount        int                `json:"vm_count"`
	Reachable      bool               `json:"reachable"`
	Link           *FabricLinkProfile `json:"link"`
}

// FabricLinkProfile is the heartbeat-measured link state to a peer. It is null
// for the local node and for peers that have never been probed.
type FabricLinkProfile struct {
	RTTMS         float64 `json:"rtt_ms"`
	LastHeartbeat string  `json:"last_heartbeat"`
	Stale         bool    `json:"stale"`
}

// FabricJobSpec is the POST /api/compute/jobs request body.
type FabricJobSpec struct {
	Name        string            `json:"name,omitempty"`
	Command     string            `json:"command"`
	Args        []string          `json:"args,omitempty"`
	Env         map[string]string `json:"env,omitempty"`
	NodeID      string            `json:"node_id,omitempty"`
	BytesToMove int               `json:"bytes_to_move,omitempty"`
	MemoryMB    int               `json:"memory_mb,omitempty"`
	VCPUs       int               `json:"vcpus,omitempty"`
}

// FabricPlacement is the placement cost's observable decision.
type FabricPlacement struct {
	Decision      string   `json:"decision"`
	CostEstimateS *float64 `json:"cost_estimate_s,omitempty"`
	Reason        string   `json:"reason"`
}

// FabricJobSubmission is the 201 response to a job submission.
type FabricJobSubmission struct {
	JobID     string           `json:"job_id"`
	VMID      string           `json:"vm_id"`
	NodeID    string           `json:"node_id"`
	Status    string           `json:"status"`
	Placement *FabricPlacement `json:"placement,omitempty"`
}

// FabricJob is one job summary from the job list.
type FabricJob struct {
	JobID     string `json:"job_id"`
	Name      string `json:"name,omitempty"`
	Command   string `json:"command"`
	Status    string `json:"status"`
	NodeID    string `json:"node_id"`
	VMID      string `json:"vm_id"`
	CreatedAt string `json:"created_at"`
	Error     string `json:"error,omitempty"`
}

// FabricJobLogs is the tail of a job's captured output streams.
type FabricJobLogs struct {
	Stdout string `json:"stdout"`
	Stderr string `json:"stderr"`
}

// FabricJobDetail is one job plus its log tails. LogsError carries the reason a
// remote job's logs could not be fetched; Logs is nil when none were returned.
type FabricJobDetail struct {
	FabricJob
	Logs      *FabricJobLogs `json:"logs,omitempty"`
	LogsError string         `json:"logs_error,omitempty"`
}

// FabricJobCancellation is the response to a job cancel.
type FabricJobCancellation struct {
	Cancelled bool   `json:"cancelled"`
	Status    string `json:"status"`
	Error     string `json:"error,omitempty"`
}

// FabricTransfer is one bulk transfer's progress and the inputs its scheduling
// decision was made from.
type FabricTransfer struct {
	TransferID     string                 `json:"transfer_id"`
	Status         string                 `json:"status"`
	BytesTotal     int64                  `json:"bytes_total"`
	BytesMoved     int64                  `json:"bytes_moved"`
	MeasuredBps    int64                  `json:"measured_bps"`
	Compression    string                 `json:"compression"`
	ETASeconds     *float64               `json:"eta_seconds,omitempty"`
	DecisionInputs map[string]interface{} `json:"decision_inputs,omitempty"`
}
