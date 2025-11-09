package pipeline

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// PipelineOrchestrator manages ML pipeline execution
type PipelineOrchestrator struct {
	pipelines map[string]*Pipeline
	mu        sync.RWMutex
}

type Pipeline struct {
	Name      string
	Stages    []*Stage
	Status    string
	CreatedAt time.Time
}

type Stage struct {
	Name         string
	Dependencies []string
	Execute      func(context.Context) error
	Status       string
}

func NewPipelineOrchestrator() *PipelineOrchestrator {
	return &PipelineOrchestrator{
		pipelines: make(map[string]*Pipeline),
	}
}

func (po *PipelineOrchestrator) CreatePipeline(name string, stages []*Stage) error {
	po.mu.Lock()
	defer po.mu.Unlock()

	po.pipelines[name] = &Pipeline{
		Name:      name,
		Stages:    stages,
		Status:    "created",
		CreatedAt: time.Now(),
	}
	return nil
}

func (po *PipelineOrchestrator) RunPipeline(ctx context.Context, name string) error {
	po.mu.RLock()
	pipeline, exists := po.pipelines[name]
	po.mu.RUnlock()

	if !exists {
		return fmt.Errorf("pipeline %s not found", name)
	}

	pipeline.Status = "running"
	
	for _, stage := range pipeline.Stages {
		stage.Status = "running"
		if err := stage.Execute(ctx); err != nil {
			stage.Status = "failed"
			return err
		}
		stage.Status = "completed"
	}

	pipeline.Status = "completed"
	return nil
}
