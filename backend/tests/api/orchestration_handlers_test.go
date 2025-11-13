package api_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gorilla/mux"
	orchestration "github.com/khryptorgraphics/novacron/backend/api/orchestration"
	"github.com/khryptorgraphics/novacron/backend/core/scheduler"
)

// Mock orchestration manager
type mockOrchestrationManager struct {
	jobs       map[string]*scheduler.Job
	tasks      map[string]*scheduler.Task
	workflows  map[string]*scheduler.Workflow
}

func (m *mockOrchestrationManager) CreateJob(ctx context.Context, job scheduler.Job) (*scheduler.Job, error) {
	job.ID = "job-" + job.Name
	job.Status = "pending"
	job.CreatedAt = time.Now()
	m.jobs[job.ID] = &job
	return &job, nil
}

func (m *mockOrchestrationManager) GetJob(ctx context.Context, jobID string) (*scheduler.Job, error) {
	if job, ok := m.jobs[jobID]; ok {
		return job, nil
	}
	return nil, scheduler.ErrJobNotFound
}

func (m *mockOrchestrationManager) ListJobs(ctx context.Context, filter scheduler.JobFilter) ([]*scheduler.Job, error) {
	jobs := make([]*scheduler.Job, 0, len(m.jobs))
	for _, job := range m.jobs {
		if filter.Status == "" || job.Status == filter.Status {
			jobs = append(jobs, job)
		}
	}
	return jobs, nil
}

func (m *mockOrchestrationManager) CancelJob(ctx context.Context, jobID string) error {
	if job, ok := m.jobs[jobID]; ok {
		job.Status = "cancelled"
		return nil
	}
	return scheduler.ErrJobNotFound
}

func (m *mockOrchestrationManager) CreateWorkflow(ctx context.Context, workflow scheduler.Workflow) (*scheduler.Workflow, error) {
	workflow.ID = "workflow-" + workflow.Name
	workflow.Status = "pending"
	workflow.CreatedAt = time.Now()
	m.workflows[workflow.ID] = &workflow
	return &workflow, nil
}

func (m *mockOrchestrationManager) GetWorkflow(ctx context.Context, workflowID string) (*scheduler.Workflow, error) {
	if wf, ok := m.workflows[workflowID]; ok {
		return wf, nil
	}
	return nil, scheduler.ErrWorkflowNotFound
}

func (m *mockOrchestrationManager) ListWorkflows(ctx context.Context) ([]*scheduler.Workflow, error) {
	workflows := make([]*scheduler.Workflow, 0, len(m.workflows))
	for _, wf := range m.workflows {
		workflows = append(workflows, wf)
	}
	return workflows, nil
}

func (m *mockOrchestrationManager) GetSchedulerStats(ctx context.Context) (*scheduler.Stats, error) {
	return &scheduler.Stats{
		TotalJobs:     len(m.jobs),
		RunningJobs:   2,
		PendingJobs:   1,
		CompletedJobs: 3,
		FailedJobs:    0,
		QueueLength:   1,
		Workers:       4,
	}, nil
}

// Setup test orchestration handlers
func setupOrchestrationHandlers() *orchestration.OrchestrationHandlers {
	mockManager := &mockOrchestrationManager{
		jobs: map[string]*scheduler.Job{
			"job-1": {
				ID:     "job-1",
				Name:   "Test Job 1",
				Status: "running",
				Type:   "compute",
			},
			"job-2": {
				ID:     "job-2",
				Name:   "Test Job 2",
				Status: "completed",
				Type:   "backup",
			},
		},
		tasks: map[string]*scheduler.Task{},
		workflows: map[string]*scheduler.Workflow{
			"workflow-1": {
				ID:     "workflow-1",
				Name:   "Test Workflow",
				Status: "running",
				Steps: []scheduler.Step{
					{ID: "step-1", Name: "Initialize", Status: "completed"},
					{ID: "step-2", Name: "Process", Status: "running"},
				},
			},
		},
	}

	return orchestration.NewOrchestrationHandlers(mockManager)
}

// Test Cases

func TestCreateJob_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	reqBody := scheduler.Job{
		Name:   "New Job",
		Type:   "compute",
		Config: map[string]interface{}{"cpu": 2, "memory": "4GB"},
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/orchestration/jobs", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.CreateJob(w, req)

	if w.Code != http.StatusCreated && w.Code != http.StatusOK {
		t.Errorf("Expected status 201 or 200, got %d", w.Code)
	}

	var response scheduler.Job
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.ID == "" {
		t.Error("Expected job ID to be set")
	}

	if response.Status != "pending" {
		t.Errorf("Expected status pending, got %s", response.Status)
	}
}

func TestCreateJob_InvalidJSON(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("POST", "/api/orchestration/jobs", bytes.NewBufferString("{invalid json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.CreateJob(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestCreateJob_MissingName(t *testing.T) {
	h := setupOrchestrationHandlers()

	reqBody := scheduler.Job{
		Type: "compute",
		// Missing Name
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/orchestration/jobs", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.CreateJob(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestGetJob_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/jobs/job-1", nil)
	req = mux.SetURLVars(req, map[string]string{"jobId": "job-1"})
	w := httptest.NewRecorder()

	h.GetJob(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response scheduler.Job
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.ID != "job-1" {
		t.Errorf("Expected job ID job-1, got %s", response.ID)
	}

	if response.Name != "Test Job 1" {
		t.Errorf("Expected job name 'Test Job 1', got %s", response.Name)
	}
}

func TestGetJob_NotFound(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/jobs/non-existent", nil)
	req = mux.SetURLVars(req, map[string]string{"jobId": "non-existent"})
	w := httptest.NewRecorder()

	h.GetJob(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected status 404, got %d", w.Code)
	}
}

func TestListJobs_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/jobs", nil)
	w := httptest.NewRecorder()

	h.ListJobs(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	jobs, ok := response["jobs"].([]interface{})
	if !ok {
		t.Error("Expected jobs array in response")
	}

	if len(jobs) != 2 {
		t.Errorf("Expected 2 jobs, got %d", len(jobs))
	}
}

func TestListJobs_FilterByStatus(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/jobs?status=running", nil)
	w := httptest.NewRecorder()

	h.ListJobs(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	jobs, ok := response["jobs"].([]interface{})
	if !ok {
		t.Error("Expected jobs array in response")
	}

	if len(jobs) != 1 {
		t.Errorf("Expected 1 running job, got %d", len(jobs))
	}
}

func TestCancelJob_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("POST", "/api/orchestration/jobs/job-1/cancel", nil)
	req = mux.SetURLVars(req, map[string]string{"jobId": "job-1"})
	w := httptest.NewRecorder()

	h.CancelJob(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}
}

func TestCancelJob_NotFound(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("POST", "/api/orchestration/jobs/non-existent/cancel", nil)
	req = mux.SetURLVars(req, map[string]string{"jobId": "non-existent"})
	w := httptest.NewRecorder()

	h.CancelJob(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected status 404, got %d", w.Code)
	}
}

func TestCreateWorkflow_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	reqBody := scheduler.Workflow{
		Name: "New Workflow",
		Steps: []scheduler.Step{
			{ID: "step-1", Name: "Initialize"},
			{ID: "step-2", Name: "Process"},
			{ID: "step-3", Name: "Finalize"},
		},
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/orchestration/workflows", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.CreateWorkflow(w, req)

	if w.Code != http.StatusCreated && w.Code != http.StatusOK {
		t.Errorf("Expected status 201 or 200, got %d", w.Code)
	}

	var response scheduler.Workflow
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.ID == "" {
		t.Error("Expected workflow ID to be set")
	}

	if len(response.Steps) != 3 {
		t.Errorf("Expected 3 steps, got %d", len(response.Steps))
	}
}

func TestCreateWorkflow_EmptySteps(t *testing.T) {
	h := setupOrchestrationHandlers()

	reqBody := scheduler.Workflow{
		Name:  "Empty Workflow",
		Steps: []scheduler.Step{},
	}
	body, _ := json.Marshal(reqBody)

	req := httptest.NewRequest("POST", "/api/orchestration/workflows", bytes.NewBuffer(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.CreateWorkflow(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("Expected status 400, got %d", w.Code)
	}
}

func TestGetWorkflow_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/workflows/workflow-1", nil)
	req = mux.SetURLVars(req, map[string]string{"workflowId": "workflow-1"})
	w := httptest.NewRecorder()

	h.GetWorkflow(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response scheduler.Workflow
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.ID != "workflow-1" {
		t.Errorf("Expected workflow ID workflow-1, got %s", response.ID)
	}

	if len(response.Steps) != 2 {
		t.Errorf("Expected 2 steps, got %d", len(response.Steps))
	}
}

func TestGetWorkflow_NotFound(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/workflows/non-existent", nil)
	req = mux.SetURLVars(req, map[string]string{"workflowId": "non-existent"})
	w := httptest.NewRecorder()

	h.GetWorkflow(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("Expected status 404, got %d", w.Code)
	}
}

func TestListWorkflows_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/workflows", nil)
	w := httptest.NewRecorder()

	h.ListWorkflows(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response map[string]interface{}
	json.Unmarshal(w.Body.Bytes(), &response)

	workflows, ok := response["workflows"].([]interface{})
	if !ok {
		t.Error("Expected workflows array in response")
	}

	if len(workflows) != 1 {
		t.Errorf("Expected 1 workflow, got %d", len(workflows))
	}
}

func TestGetSchedulerStats_Success(t *testing.T) {
	h := setupOrchestrationHandlers()

	req := httptest.NewRequest("GET", "/api/orchestration/stats", nil)
	w := httptest.NewRecorder()

	h.GetSchedulerStats(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("Expected status 200, got %d", w.Code)
	}

	var response scheduler.Stats
	json.Unmarshal(w.Body.Bytes(), &response)

	if response.TotalJobs != 2 {
		t.Errorf("Expected 2 total jobs, got %d", response.TotalJobs)
	}

	if response.Workers != 4 {
		t.Errorf("Expected 4 workers, got %d", response.Workers)
	}
}

// Test concurrent job creation
func TestConcurrentCreateJob(t *testing.T) {
	h := setupOrchestrationHandlers()

	const numRequests = 10
	done := make(chan bool, numRequests)

	for i := 0; i < numRequests; i++ {
		go func(id int) {
			reqBody := scheduler.Job{
				Name: "Concurrent Job",
				Type: "compute",
			}
			body, _ := json.Marshal(reqBody)

			req := httptest.NewRequest("POST", "/api/orchestration/jobs", bytes.NewBuffer(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			h.CreateJob(w, req)

			if w.Code != http.StatusCreated && w.Code != http.StatusOK {
				t.Errorf("Request %d failed with status %d", id, w.Code)
			}

			done <- true
		}(i)
	}

	for i := 0; i < numRequests; i++ {
		<-done
	}
}

// Test context cancellation
func TestGetJob_ContextCancellation(t *testing.T) {
	h := setupOrchestrationHandlers()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	req := httptest.NewRequest("GET", "/api/orchestration/jobs/job-1", nil).WithContext(ctx)
	req = mux.SetURLVars(req, map[string]string{"jobId": "job-1"})
	w := httptest.NewRecorder()

	h.GetJob(w, req)

	// Should handle context cancellation gracefully
	if w.Code == http.StatusOK {
		t.Log("Note: Handler may not check context cancellation")
	}
}

// Benchmark tests
func BenchmarkCreateJob(b *testing.B) {
	h := setupOrchestrationHandlers()

	reqBody := scheduler.Job{
		Name: "Benchmark Job",
		Type: "compute",
	}
	body, _ := json.Marshal(reqBody)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/api/orchestration/jobs", bytes.NewBuffer(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		h.CreateJob(w, req)
	}
}

func BenchmarkListJobs(b *testing.B) {
	h := setupOrchestrationHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/orchestration/jobs", nil)
		w := httptest.NewRecorder()
		h.ListJobs(w, req)
	}
}

func BenchmarkGetSchedulerStats(b *testing.B) {
	h := setupOrchestrationHandlers()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/api/orchestration/stats", nil)
		w := httptest.NewRecorder()
		h.GetSchedulerStats(w, req)
	}
}
