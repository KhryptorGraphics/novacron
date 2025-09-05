package performance

import (
	"context"
	"runtime"
	"sync"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// WorkerPool represents a pool of workers for handling concurrent tasks
type WorkerPool struct {
	workers    chan chan Task
	workerPool chan chan Task
	quit       chan bool
	wg         sync.WaitGroup
	logger     logger.Logger
}

// Task represents a unit of work to be processed
type Task struct {
	ID      string
	Handler func(context.Context) error
	Context context.Context
	Done    chan error
}

// PoolConfig represents the configuration for the worker pool
type PoolConfig struct {
	WorkerCount int           `yaml:"workerCount" json:"workerCount"`
	QueueSize   int           `yaml:"queueSize" json:"queueSize"`
	Timeout     time.Duration `yaml:"timeout" json:"timeout"`
}

// NewWorkerPool creates a new worker pool with the given configuration
func NewWorkerPool(config PoolConfig, logger logger.Logger) *WorkerPool {
	if config.WorkerCount <= 0 {
		config.WorkerCount = runtime.NumCPU() * 2
	}
	if config.QueueSize <= 0 {
		config.QueueSize = config.WorkerCount * 10
	}
	if config.Timeout <= 0 {
		config.Timeout = 30 * time.Second
	}

	pool := &WorkerPool{
		workers:    make(chan chan Task, config.WorkerCount),
		workerPool: make(chan chan Task, config.QueueSize),
		quit:       make(chan bool),
		logger:     logger,
	}

	pool.start(config.WorkerCount, config.Timeout)
	return pool
}

// start initializes and starts the worker pool
func (p *WorkerPool) start(workerCount int, timeout time.Duration) {
	for i := 0; i < workerCount; i++ {
		worker := NewWorker(i, p.workers, timeout, p.logger)
		worker.Start()
		p.wg.Add(1)
	}

	go p.dispatch()
}

// dispatch handles the dispatching of tasks to available workers
func (p *WorkerPool) dispatch() {
	for {
		select {
		case job := <-p.workerPool:
			select {
			case worker := <-p.workers:
				worker <- job
			case <-p.quit:
				return
			}
		case <-p.quit:
			return
		}
	}
}

// Submit submits a task to the worker pool
func (p *WorkerPool) Submit(ctx context.Context, id string, handler func(context.Context) error) <-chan error {
	done := make(chan error, 1)
	
	task := Task{
		ID:      id,
		Handler: handler,
		Context: ctx,
		Done:    done,
	}

	select {
	case p.workerPool <- task:
		return done
	case <-ctx.Done():
		done <- ctx.Err()
		return done
	default:
		done <- ErrPoolFull
		return done
	}
}

// Stop gracefully shuts down the worker pool
func (p *WorkerPool) Stop() {
	close(p.quit)
	p.wg.Wait()
	close(p.workerPool)
	close(p.workers)
}

// Worker represents a single worker in the pool
type Worker struct {
	ID      int
	work    chan Task
	workers chan chan Task
	timeout time.Duration
	logger  logger.Logger
	quit    chan bool
}

// NewWorker creates a new worker
func NewWorker(id int, workers chan chan Task, timeout time.Duration, logger logger.Logger) *Worker {
	return &Worker{
		ID:      id,
		work:    make(chan Task),
		workers: workers,
		timeout: timeout,
		logger:  logger,
		quit:    make(chan bool),
	}
}

// Start starts the worker
func (w *Worker) Start() {
	go func() {
		defer func() {
			if r := recover(); r != nil {
				w.logger.Error("Worker panic recovered", "worker_id", w.ID, "error", r)
			}
		}()

		for {
			w.workers <- w.work
			
			select {
			case task := <-w.work:
				w.processTask(task)
			case <-w.quit:
				return
			}
		}
	}()
}

// processTask processes a single task with timeout
func (w *Worker) processTask(task Task) {
	startTime := time.Now()
	defer func() {
		duration := time.Since(startTime)
		w.logger.Debug("Task completed", 
			"worker_id", w.ID, 
			"task_id", task.ID, 
			"duration", duration,
		)
	}()

	ctx, cancel := context.WithTimeout(task.Context, w.timeout)
	defer cancel()

	done := make(chan error, 1)
	go func() {
		defer func() {
			if r := recover(); r != nil {
				w.logger.Error("Task panic recovered", 
					"worker_id", w.ID, 
					"task_id", task.ID, 
					"error", r,
				)
				done <- ErrTaskPanic
			}
		}()
		done <- task.Handler(ctx)
	}()

	select {
	case err := <-done:
		task.Done <- err
	case <-ctx.Done():
		task.Done <- ctx.Err()
	}
}

// Stop stops the worker
func (w *Worker) Stop() {
	close(w.quit)
}

// Errors
var (
	ErrPoolFull  = fmt.Errorf("worker pool is full")
	ErrTaskPanic = fmt.Errorf("task panicked during execution")
)

// PerformanceMetrics represents performance metrics for the worker pool
type PerformanceMetrics struct {
	ActiveWorkers   int           `json:"activeWorkers"`
	QueuedTasks     int           `json:"queuedTasks"`
	ProcessedTasks  int64         `json:"processedTasks"`
	FailedTasks     int64         `json:"failedTasks"`
	AverageLatency  time.Duration `json:"averageLatency"`
	ThroughputPerSec float64      `json:"throughputPerSec"`
	LastUpdate      time.Time     `json:"lastUpdate"`
}

// GetMetrics returns current performance metrics
func (p *WorkerPool) GetMetrics() PerformanceMetrics {
	return PerformanceMetrics{
		ActiveWorkers:    len(p.workers),
		QueuedTasks:      len(p.workerPool),
		LastUpdate:       time.Now(),
		// Additional metrics would be tracked via counters
	}
}