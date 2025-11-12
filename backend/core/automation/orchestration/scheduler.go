// Package orchestration - Scheduler component
package orchestration

import (
	"context"
	"fmt"
	"time"

	"github.com/robfig/cron/v3"
	"github.com/sirupsen/logrus"
)

// Scheduler manages workflow scheduling
type Scheduler struct {
	engine    *WorkflowEngine
	cron      *cron.Cron
	logger    *logrus.Logger
	ctx       context.Context
	cancel    context.CancelFunc
}

// NewScheduler creates a new workflow scheduler
func NewScheduler(engine *WorkflowEngine, logger *logrus.Logger) *Scheduler {
	ctx, cancel := context.WithCancel(context.Background())

	return &Scheduler{
		engine: engine,
		cron:   cron.New(cron.WithSeconds()),
		logger: logger,
		ctx:    ctx,
		cancel: cancel,
	}
}

// Start begins the scheduler
func (s *Scheduler) Start() error {
	s.logger.Info("Starting workflow scheduler")

	// Register all scheduled workflows
	s.engine.mu.RLock()
	for _, workflow := range s.engine.workflows {
		if workflow.Schedule != nil && workflow.Schedule.Enabled {
			if err := s.registerScheduledWorkflow(workflow); err != nil {
				s.logger.WithError(err).Error("Failed to register scheduled workflow")
			}
		}
	}
	s.engine.mu.RUnlock()

	s.cron.Start()
	return nil
}

// registerScheduledWorkflow registers a workflow with the scheduler
func (s *Scheduler) registerScheduledWorkflow(workflow *Workflow) error {
	schedule := workflow.Schedule

	switch schedule.Type {
	case ScheduleTypeCron:
		_, err := s.cron.AddFunc(schedule.Cron, func() {
			s.triggerWorkflow(workflow.ID, "scheduled")
		})
		return err

	case ScheduleTypeInterval:
		_, err := s.cron.AddFunc(fmt.Sprintf("@every %s", schedule.Interval), func() {
			s.triggerWorkflow(workflow.ID, "scheduled")
		})
		return err

	default:
		return fmt.Errorf("unsupported schedule type: %s", schedule.Type)
	}
}

// triggerWorkflow triggers a workflow execution
func (s *Scheduler) triggerWorkflow(workflowID string, trigger string) {
	s.logger.WithFields(logrus.Fields{
		"workflow_id": workflowID,
		"trigger":     trigger,
	}).Info("Triggering scheduled workflow")

	_, err := s.engine.ExecuteWorkflow(workflowID, trigger, nil)
	if err != nil {
		s.logger.WithError(err).Error("Failed to execute scheduled workflow")
	}
}

// Stop gracefully stops the scheduler
func (s *Scheduler) Stop() {
	s.logger.Info("Stopping scheduler")
	s.cancel()
	s.cron.Stop()
}
