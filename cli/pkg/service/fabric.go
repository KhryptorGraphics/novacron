package service

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	"github.com/novacron/cli/pkg/api"
)

// FabricService talks to the compute fabric API: node inventory with live link
// state, compute jobs, and bulk transfers between nodes.
type FabricService struct {
	client *api.Client
}

// NewFabricService creates a new fabric service
func NewFabricService(client *api.Client) *FabricService {
	return &FabricService{client: client}
}

// ListNodes returns every node in the fabric with its live capacity and link
// profile. The local node and never-probed peers carry no link profile.
func (s *FabricService) ListNodes(ctx context.Context) ([]api.FabricNode, error) {
	var out struct {
		Nodes []api.FabricNode `json:"nodes"`
	}
	if err := s.client.Get(ctx, "/api/cluster/nodes", &out); err != nil {
		return nil, fmt.Errorf("failed to list fabric nodes: %w", err)
	}

	return out.Nodes, nil
}

// SubmitJob places and starts a job, returning its id, executor and the
// placement decision that chose the node.
func (s *FabricService) SubmitJob(ctx context.Context, spec api.FabricJobSpec) (*api.FabricJobSubmission, error) {
	var out api.FabricJobSubmission
	if err := s.client.Post(ctx, "/api/compute/jobs", spec, &out); err != nil {
		return nil, fmt.Errorf("failed to submit job: %w", err)
	}

	return &out, nil
}

// ListJobs returns every fabric job.
func (s *FabricService) ListJobs(ctx context.Context) ([]api.FabricJob, error) {
	var out struct {
		Jobs []api.FabricJob `json:"jobs"`
	}
	if err := s.client.Get(ctx, "/api/compute/jobs", &out); err != nil {
		return nil, fmt.Errorf("failed to list jobs: %w", err)
	}

	return out.Jobs, nil
}

// GetJob returns one job's record plus the tail of its captured streams.
func (s *FabricService) GetJob(ctx context.Context, id string) (*api.FabricJobDetail, error) {
	var out api.FabricJobDetail
	if err := s.client.Get(ctx, fabricJobPath(id), &out); err != nil {
		return nil, fmt.Errorf("failed to get job: %w", err)
	}

	return &out, nil
}

// CancelJob cancels a job by stopping its executor VM.
func (s *FabricService) CancelJob(ctx context.Context, id string) (*api.FabricJobCancellation, error) {
	var out api.FabricJobCancellation
	if err := s.client.Post(ctx, fabricJobPath(id)+"/cancel", nil, &out); err != nil {
		return nil, fmt.Errorf("failed to cancel job: %w", err)
	}

	return &out, nil
}

// ListTransfers returns the fabric's bulk transfers, newest first. The contract
// shape is {"transfers":[...]}; a bare array is accepted too.
func (s *FabricService) ListTransfers(ctx context.Context) ([]api.FabricTransfer, error) {
	resp, err := s.client.Request(ctx, http.MethodGet, "/api/transfers", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to list transfers: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, fmt.Errorf("failed to read transfer list: %w", err)
	}

	var wrapped struct {
		Transfers []api.FabricTransfer `json:"transfers"`
	}
	if err := json.Unmarshal(body, &wrapped); err == nil {
		return wrapped.Transfers, nil
	}

	var bare []api.FabricTransfer
	if err := json.Unmarshal(body, &bare); err != nil {
		return nil, fmt.Errorf("failed to decode transfer list: %w", err)
	}

	return bare, nil
}

// GetTransfer returns one transfer's progress and the inputs its scheduling
// decision was made from.
func (s *FabricService) GetTransfer(ctx context.Context, id string) (*api.FabricTransfer, error) {
	var out api.FabricTransfer
	if err := s.client.Get(ctx, "/api/transfers/"+url.PathEscape(id), &out); err != nil {
		return nil, fmt.Errorf("failed to get transfer: %w", err)
	}

	return &out, nil
}

func fabricJobPath(id string) string {
	return "/api/compute/jobs/" + url.PathEscape(id)
}
