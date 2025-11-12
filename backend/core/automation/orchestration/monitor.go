// Package orchestration - Workflow Monitor component
package orchestration

import (
	"context"
	"time"

	"github.com/sirupsen/logrus"
)

// WorkflowMonitor monitors workflow executions
type WorkflowMonitor struct {
	engine *WorkflowEngine
	logger *logrus.Logger
	ctx    context.Context
	cancel context.CancelFunc
}

// NewWorkflowMonitor creates a new workflow monitor
func NewWorkflowMonitor(engine *WorkflowEngine, logger *logrus.Logger) *WorkflowMonitor {
	ctx, cancel := context.WithCancel(context.Background())

	monitor := &WorkflowMonitor{
		engine: engine,
		logger: logger,
		ctx:    ctx,
		cancel: cancel,
	}

	go monitor.monitorLoop()

	return monitor
}

// monitorLoop continuously monitors workflow executions
func (wm *WorkflowMonitor) monitorLoop() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-wm.ctx.Done():
			return
		case <-ticker.C:
			wm.checkExecutions()
		}
	}
}

// checkExecutions checks all active executions
func (wm *WorkflowMonitor) checkExecutions() {
	wm.engine.mu.RLock()
	defer wm.engine.mu.RUnlock()

	for _, execution := range wm.engine.executions {
		if execution.Status == StatusRunning {
			wm.checkExecutionHealth(execution)
		}
	}
}

// checkExecutionHealth checks execution health and SLA compliance
func (wm *WorkflowMonitor) checkExecutionHealth(execution *WorkflowExecution) {
	duration := time.Since(execution.StartTime)

	wm.engine.mu.RLock()
	workflow := wm.engine.workflows[execution.WorkflowID]
	wm.engine.mu.RUnlock()

	// Check timeout
	if duration > workflow.Timeout {
		wm.logger.WithFields(logrus.Fields{
			"execution_id": execution.ID,
			"duration":     duration,
			"timeout":      workflow.Timeout,
		}).Warn("Workflow execution timeout exceeded")
	}

	// Check SLA
	if workflow.SLA != nil {
		if duration > workflow.SLA.MaxDuration {
			wm.logger.WithFields(logrus.Fields{
				"execution_id": execution.ID,
				"duration":     duration,
				"sla":          workflow.SLA.MaxDuration,
			}).Warn("Workflow SLA violation")
		}
	}
}

// GetWorkflowMetrics returns metrics for a workflow
func (wm *WorkflowMonitor) GetWorkflowMetrics(workflowID string) *WorkflowMetrics {
	wm.engine.mu.RLock()
	defer wm.engine.mu.RUnlock()

	metrics := &WorkflowMetrics{
		WorkflowID:       workflowID,
		TotalExecutions:  0,
		SuccessfulExecs:  0,
		FailedExecs:      0,
		AverageDuration:  0,
	}

	var totalDuration time.Duration
	for _, exec := range wm.engine.executions {
		if exec.WorkflowID == workflowID {
			metrics.TotalExecutions++
			if exec.Status == StatusSuccess {
				metrics.SuccessfulExecs++
			} else if exec.Status == StatusFailed {
				metrics.FailedExecs++
			}
			if exec.EndTime != nil {
				totalDuration += exec.Duration
			}
		}
	}

	if metrics.TotalExecutions > 0 {
		metrics.AverageDuration = totalDuration / time.Duration(metrics.TotalExecutions)
		metrics.SuccessRate = float64(metrics.SuccessfulExecs) / float64(metrics.TotalExecutions)
	}

	return metrics
}

// WorkflowMetrics contains workflow performance metrics
type WorkflowMetrics struct {
	WorkflowID      string
	TotalExecutions int
	SuccessfulExecs int
	FailedExecs     int
	AverageDuration time.Duration
	SuccessRate     float64
}
