package edge

import (
	"encoding/json"
	"fmt"
	"log"
	"time"
)

// handleTaskMessage processes task assignment messages from cloud
func handleTaskMessage(agent *EdgeAgent, message *Message) error {
	log.Printf("Received task message: %s", message.ID)

	// Parse task data
	taskData, ok := message.Data["task"]
	if !ok {
		return fmt.Errorf("missing task data in message")
	}

	taskJSON, err := json.Marshal(taskData)
	if err != nil {
		return fmt.Errorf("failed to marshal task data: %v", err)
	}

	var task EdgeTask
	if err := json.Unmarshal(taskJSON, &task); err != nil {
		return fmt.Errorf("failed to unmarshal task: %v", err)
	}

	// Set initial status
	task.Status = TaskStatusQueued
	task.CreatedAt = time.Now()

	// Validate task
	if err := validateTask(&task); err != nil {
		// Send rejection back to cloud
		agent.sendTaskRejection(&task, fmt.Sprintf("Task validation failed: %v", err))
		return err
	}

	// Check if we can handle this task type
	if !agent.taskManager.CanHandle(task.Type) {
		agent.sendTaskRejection(&task, fmt.Sprintf("Unsupported task type: %s", task.Type))
		return fmt.Errorf("unsupported task type: %s", task.Type)
	}

	// Check resource availability
	if !agent.canExecuteTask(&task) {
		// Queue for later or reject based on priority
		if task.Priority < 5 {
			agent.sendTaskRejection(&task, "Insufficient resources and low priority")
			return fmt.Errorf("insufficient resources for low priority task")
		}
		// High priority tasks get queued
		log.Printf("Queueing high priority task %s for later execution", task.ID)
	}

	// Store task
	agent.taskMutex.Lock()
	agent.pendingTasks[task.ID] = &task
	agent.taskMutex.Unlock()

	// Send acknowledgment
	agent.sendTaskAcknowledgment(&task)

	log.Printf("Task %s queued successfully", task.ID)
	return nil
}

// handleConfigUpdateMessage processes configuration update messages
func handleConfigUpdateMessage(agent *EdgeAgent, message *Message) error {
	log.Printf("Received config update message: %s", message.ID)

	configData, ok := message.Data["config"]
	if !ok {
		return fmt.Errorf("missing config data in message")
	}

	configMap, ok := configData.(map[string]interface{})
	if !ok {
		return fmt.Errorf("invalid config data format")
	}

	// Apply configuration updates
	updates := make(map[string]interface{})
	
	// Update metrics interval if provided
	if interval, ok := configMap["metrics_interval"]; ok {
		if intervalStr, ok := interval.(string); ok {
			if duration, err := time.ParseDuration(intervalStr); err == nil {
				agent.config.MetricsInterval = duration
				updates["metrics_interval"] = duration.String()
			}
		}
	}

	// Update heartbeat interval if provided
	if interval, ok := configMap["heartbeat_interval"]; ok {
		if intervalStr, ok := interval.(string); ok {
			if duration, err := time.ParseDuration(intervalStr); err == nil {
				agent.config.HeartbeatInterval = duration
				updates["heartbeat_interval"] = duration.String()
			}
		}
	}

	// Update resource limits if provided
	if limits, ok := configMap["resource_limits"]; ok {
		if limitsMap, ok := limits.(map[string]interface{}); ok {
			if maxCPU, ok := limitsMap["max_cpu_percent"]; ok {
				if cpuFloat, ok := maxCPU.(float64); ok {
					agent.config.MaxCPUPercent = cpuFloat
					updates["max_cpu_percent"] = cpuFloat
				}
			}
			if maxMem, ok := limitsMap["max_memory_percent"]; ok {
				if memFloat, ok := maxMem.(float64); ok {
					agent.config.MaxMemoryPercent = memFloat
					updates["max_memory_percent"] = memFloat
				}
			}
		}
	}

	// Update tags if provided
	if tags, ok := configMap["tags"]; ok {
		if tagsMap, ok := tags.(map[string]interface{}); ok {
			for k, v := range tagsMap {
				if vStr, ok := v.(string); ok {
					agent.config.Tags[k] = vStr
					updates["tags."+k] = vStr
				}
			}
		}
	}

	// Send acknowledgment with applied updates
	responseData := map[string]interface{}{
		"agent_id": agent.config.AgentID,
		"status":   "applied",
		"updates":  updates,
	}

	if err := agent.sendMessage("config_update_ack", responseData); err != nil {
		log.Printf("Failed to acknowledge config update: %v", err)
	}

	log.Printf("Applied %d configuration updates", len(updates))
	return nil
}

// handleResourceRequestMessage processes resource availability requests
func handleResourceRequestMessage(agent *EdgeAgent, message *Message) error {
	log.Printf("Received resource request message: %s", message.ID)

	// Get current resource availability
	metrics := agent.systemMonitor.GetCurrentMetrics()
	availability := agent.resourceManager.GetAvailability()

	resourceData := map[string]interface{}{
		"agent_id":     agent.config.AgentID,
		"timestamp":    time.Now(),
		"request_id":   message.ID,
		"availability": availability,
		"current_usage": map[string]interface{}{
			"cpu_percent":    metrics.CPU.Average,
			"memory_percent": metrics.Memory.UsedPct,
			"disk_usage":     calculateDiskUsage(metrics.Disk),
			"network_usage":  calculateNetworkUsage(metrics.Network),
		},
		"capacity": map[string]interface{}{
			"cpu_cores":      agent.systemMonitor.GetSystemInfo().CPUCores,
			"total_memory":   agent.systemMonitor.GetSystemInfo().TotalMemory,
			"total_storage":  agent.systemMonitor.GetSystemInfo().TotalStorage,
		},
		"active_tasks": len(agent.pendingTasks),
		"queue_length": agent.taskManager.GetQueueLength(),
	}

	if err := agent.sendMessage("resource_response", resourceData); err != nil {
		return fmt.Errorf("failed to send resource response: %v", err)
	}

	return nil
}

// handleCacheInvalidateMessage processes cache invalidation requests
func handleCacheInvalidateMessage(agent *EdgeAgent, message *Message) error {
	log.Printf("Received cache invalidate message: %s", message.ID)

	// Get cache keys to invalidate
	keys, ok := message.Data["keys"]
	if !ok {
		// Invalidate all cache
		agent.localCache.Clear()
		log.Printf("Invalidated entire local cache")
	} else {
		// Invalidate specific keys
		if keysList, ok := keys.([]interface{}); ok {
			invalidatedCount := 0
			for _, key := range keysList {
				if keyStr, ok := key.(string); ok {
					if agent.localCache.Delete(keyStr) {
						invalidatedCount++
					}
				}
			}
			log.Printf("Invalidated %d cache entries", invalidatedCount)
		}
	}

	// Send acknowledgment
	responseData := map[string]interface{}{
		"agent_id":   agent.config.AgentID,
		"request_id": message.ID,
		"status":     "invalidated",
		"timestamp":  time.Now(),
	}

	if err := agent.sendMessage("cache_invalidate_ack", responseData); err != nil {
		return fmt.Errorf("failed to acknowledge cache invalidation: %v", err)
	}

	return nil
}

// handlePingMessage processes ping messages for connectivity testing
func handlePingMessage(agent *EdgeAgent, message *Message) error {
	// Send pong response
	responseData := map[string]interface{}{
		"agent_id":   agent.config.AgentID,
		"request_id": message.ID,
		"timestamp":  time.Now(),
		"uptime":     time.Since(agent.systemMonitor.startTime).Seconds(),
		"status":     agent.getAgentStatus(),
	}

	if err := agent.sendMessage("pong", responseData); err != nil {
		return fmt.Errorf("failed to send pong response: %v", err)
	}

	return nil
}

// handleShutdownMessage processes graceful shutdown requests
func handleShutdownMessage(agent *EdgeAgent, message *Message) error {
	log.Printf("Received shutdown message: %s", message.ID)

	// Parse shutdown parameters
	graceful := true
	timeout := 30 * time.Second

	if params, ok := message.Data["parameters"]; ok {
		if paramsMap, ok := params.(map[string]interface{}); ok {
			if gracefulVal, ok := paramsMap["graceful"]; ok {
				if gracefulBool, ok := gracefulVal.(bool); ok {
					graceful = gracefulBool
				}
			}
			if timeoutVal, ok := paramsMap["timeout"]; ok {
				if timeoutStr, ok := timeoutVal.(string); ok {
					if duration, err := time.ParseDuration(timeoutStr); err == nil {
						timeout = duration
					}
				}
			}
		}
	}

	// Send acknowledgment
	responseData := map[string]interface{}{
		"agent_id":   agent.config.AgentID,
		"request_id": message.ID,
		"status":     "acknowledged",
		"graceful":   graceful,
		"timeout":    timeout.String(),
	}

	if err := agent.sendMessage("shutdown_ack", responseData); err != nil {
		log.Printf("Failed to acknowledge shutdown: %v", err)
	}

	// Initiate shutdown
	go func() {
		if graceful {
			log.Printf("Initiating graceful shutdown with %v timeout", timeout)
			agent.gracefulShutdown(timeout)
		} else {
			log.Printf("Initiating immediate shutdown")
			agent.Stop()
		}
	}()

	return nil
}

// Helper functions

func validateTask(task *EdgeTask) error {
	if task.ID == "" {
		return fmt.Errorf("task ID is required")
	}

	if task.Type == "" {
		return fmt.Errorf("task type is required")
	}

	// Validate resource requirements
	if task.ResourceReqs.CPUCores < 0 {
		return fmt.Errorf("invalid CPU requirement: %f", task.ResourceReqs.CPUCores)
	}

	if task.ResourceReqs.MemoryMB < 0 {
		return fmt.Errorf("invalid memory requirement: %d", task.ResourceReqs.MemoryMB)
	}

	// Validate deadline if specified
	if !task.Deadline.IsZero() && task.Deadline.Before(time.Now()) {
		return fmt.Errorf("task deadline is in the past")
	}

	// Validate max retries
	if task.MaxRetries < 0 {
		task.MaxRetries = 3 // Default
	}

	return nil
}

func calculateDiskUsage(diskMetrics map[string]*DiskMetrics) float64 {
	if len(diskMetrics) == 0 {
		return 0
	}

	var totalUsed, totalCapacity uint64
	for _, disk := range diskMetrics {
		totalUsed += disk.Used
		totalCapacity += disk.Total
	}

	if totalCapacity == 0 {
		return 0
	}

	return float64(totalUsed) / float64(totalCapacity) * 100
}

func calculateNetworkUsage(netMetrics *NetworkMetrics) map[string]interface{} {
	return map[string]interface{}{
		"total_rx_bytes":   netMetrics.TotalRxBytes,
		"total_tx_bytes":   netMetrics.TotalTxBytes,
		"total_rx_packets": netMetrics.TotalRxPackets,
		"total_tx_packets": netMetrics.TotalTxPackets,
		"total_errors":     netMetrics.TotalErrors,
		"total_dropped":    netMetrics.TotalDropped,
	}
}

// sendTaskAcknowledgment sends task acceptance acknowledgment
func (a *EdgeAgent) sendTaskAcknowledgment(task *EdgeTask) {
	ackData := map[string]interface{}{
		"agent_id":     a.config.AgentID,
		"task_id":      task.ID,
		"status":       "accepted",
		"queued_at":    task.CreatedAt,
		"estimated_start": time.Now().Add(a.taskManager.GetEstimatedWaitTime(task)),
	}

	if err := a.sendMessage("task_ack", ackData); err != nil {
		log.Printf("Failed to send task acknowledgment for %s: %v", task.ID, err)
	}
}

// sendTaskRejection sends task rejection with reason
func (a *EdgeAgent) sendTaskRejection(task *EdgeTask, reason string) {
	rejectionData := map[string]interface{}{
		"agent_id": a.config.AgentID,
		"task_id":  task.ID,
		"status":   "rejected",
		"reason":   reason,
		"timestamp": time.Now(),
	}

	if err := a.sendMessage("task_rejection", rejectionData); err != nil {
		log.Printf("Failed to send task rejection for %s: %v", task.ID, err)
	}
}

// gracefulShutdown performs graceful shutdown with timeout
func (a *EdgeAgent) gracefulShutdown(timeout time.Duration) {
	log.Printf("Starting graceful shutdown...")

	// Create timeout context
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Stop accepting new tasks
	a.taskManager.StopAcceptingTasks()

	// Wait for running tasks to complete
	done := make(chan struct{})
	go func() {
		a.taskManager.WaitForCompletion()
		close(done)
	}()

	select {
	case <-done:
		log.Printf("All tasks completed, shutting down gracefully")
	case <-ctx.Done():
		log.Printf("Graceful shutdown timeout reached, forcing shutdown")
		a.taskManager.ForceStop()
	}

	// Final shutdown
	a.Stop()
}