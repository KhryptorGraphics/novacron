package main

import (
	"context"
	"database/sql"
	"fmt"
	"sync"
	"time"

	"github.com/sirupsen/logrus"

	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/autoscaling"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
	"github.com/khryptorgraphics/novacron/backend/core/orchestration/healing"
	core_vm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

// --- bounded in-memory alert store backing GET /api/monitoring/alerts -----

type storedAlert struct {
	Type      string                 `json:"type"`
	Data      map[string]interface{} `json:"data"`
	Timestamp time.Time              `json:"timestamp"`
}

type alertStore struct {
	mu      sync.RWMutex
	items   []storedAlert
	maxKeep int
}

func newAlertStore(maxKeep int) *alertStore { return &alertStore{maxKeep: maxKeep} }

func (s *alertStore) add(a storedAlert) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.items = append(s.items, a)
	if len(s.items) > s.maxKeep {
		s.items = s.items[len(s.items)-s.maxKeep:]
	}
}

// list returns a newest-first snapshot.
func (s *alertStore) list() []storedAlert {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]storedAlert, len(s.items))
	for i, a := range s.items {
		out[len(s.items)-1-i] = a
	}
	return out
}

// recentAlerts is the process-wide registry backing GET /api/monitoring/alerts,
// same pattern as the package-level restartSupervisor/transfers vars.
var recentAlerts = newAlertStore(200)

func publishAndStoreAlert(ws *websocketapi.WebSocketHandler, alertType, severity, title, description, source, status string) {
	data := map[string]interface{}{
		"severity":    severity,
		"title":       title,
		"description": description,
		"source":      source,
		"status":      status,
	}
	ts := time.Now()
	ws.PublishAlert(websocketapi.AlertMessage{Type: alertType, Data: data, Timestamp: ts})
	recentAlerts.add(storedAlert{Type: alertType, Data: data, Timestamp: ts})
}

// --- eventBus -> alerts bridge ---------------------------------------------

// realtimeEventBridge subscribes to real, reachable orchestration events and
// republishes them as alerts-channel messages: healing health/recovery
// events, heartbeat-detected node failure/recovery (peerLiveness), and
// triggered (scale_up/scale_down) autoscaler decisions — no_action decisions
// (decision_made only) are not alerts.
type realtimeEventBridge struct {
	ws     *websocketapi.WebSocketHandler
	logger *logrus.Logger
}

func newRealtimeEventBridge(ws *websocketapi.WebSocketHandler, logger *logrus.Logger) *realtimeEventBridge {
	return &realtimeEventBridge{ws: ws, logger: logger}
}

func (b *realtimeEventBridge) subscribe(ctx context.Context, bus events.EventBus) {
	register := func(t events.EventType, fn func(*events.OrchestrationEvent)) {
		h := events.NewEventHandlerFunc("realtime-"+string(t), string(t), func(_ context.Context, e *events.OrchestrationEvent) error {
			fn(e)
			return nil
		})
		if _, err := bus.Subscribe(ctx, []events.EventType{t}, h); err != nil {
			b.logger.WithError(err).WithField("event_type", t).Warn("realtime: subscribe failed")
		}
	}
	register(events.EventType(healing.EventTypeHealthDegraded), b.handleHealthDegraded)
	register(events.EventType(healing.EventTypeHealthRestored), b.handleHealthRestored)
	register(events.EventType(healing.EventTypeRecoveryStarted), b.handleRecoveryStarted)
	register(events.EventType(healing.EventTypeRecoveryCompleted), b.handleRecoveryCompleted)
	register(events.EventTypeScalingTriggered, b.handleScalingTriggered)
	register(events.EventTypeNodeFailure, b.handleNodeFailure)
	register(events.EventTypeNodeRecovered, b.handleNodeRecovered)
}

func (b *realtimeEventBridge) handleHealthDegraded(e *events.OrchestrationEvent) {
	status, _ := e.Data["status"].(*healing.HealthStatus)
	severity, desc := "medium", fmt.Sprintf("Health degraded for target %s", e.Target)
	if status != nil {
		if status.ConsecutiveFailures >= 10 {
			severity = "critical"
		} else if status.ConsecutiveFailures >= 5 {
			severity = "high"
		}
		if status.FailureReason != "" {
			desc = status.FailureReason
		}
	}
	publishAndStoreAlert(b.ws, "security_alert", severity, fmt.Sprintf("Health degraded: %s", e.Target), desc, "healing-controller", "firing")
}

func (b *realtimeEventBridge) handleHealthRestored(e *events.OrchestrationEvent) {
	publishAndStoreAlert(b.ws, "security_alert", "low", fmt.Sprintf("Health restored: %s", e.Target),
		fmt.Sprintf("Target %s is healthy again", e.Target), "healing-controller", "resolved")
}

func (b *realtimeEventBridge) handleRecoveryStarted(e *events.OrchestrationEvent) {
	failure, _ := e.Data["failure"].(*healing.FailureInfo)
	decision, _ := e.Data["decision"].(*healing.HealingDecision)
	desc, severity, strategy := fmt.Sprintf("Healing triggered for target %s", e.Target), "medium", ""
	if failure != nil {
		desc, severity = failure.Description, string(failure.Severity)
	}
	if decision != nil {
		strategy = decision.Strategy
	}
	publishAndStoreAlert(b.ws, "security_alert", severity, fmt.Sprintf("Healing triggered (%s): %s", strategy, e.Target), desc, "healing-controller", "firing")
}

func (b *realtimeEventBridge) handleRecoveryCompleted(e *events.OrchestrationEvent) {
	result, _ := e.Data["result"].(*healing.HealingResult)
	title, severity, status, desc := fmt.Sprintf("Healing completed: %s", e.Target), "low", "resolved", "Recovery finished successfully"
	if result != nil && !result.Success {
		title, severity, status = fmt.Sprintf("Healing failed: %s", e.Target), "high", "firing"
		desc = "Recovery did not succeed"
		if n := len(result.RecoveryResults); n > 0 && result.RecoveryResults[n-1] != nil {
			desc = result.RecoveryResults[n-1].Message
		}
	}
	publishAndStoreAlert(b.ws, "security_alert", severity, title, desc, "healing-controller", status)
}

func (b *realtimeEventBridge) handleScalingTriggered(e *events.OrchestrationEvent) {
	decision, _ := e.Data["decision"].(*autoscaling.ScalingDecision)
	if decision == nil {
		return
	}
	publishAndStoreAlert(b.ws, "security_alert", "low",
		fmt.Sprintf("Scaling %s: %s -> %d", decision.Action, e.Target, decision.TargetScale),
		decision.Reason, "autoscaler", "firing")
}

func (b *realtimeEventBridge) handleNodeFailure(e *events.OrchestrationEvent) {
	desc := fmt.Sprintf("Node %s missed %v consecutive heartbeats", e.Target, e.Data["consecutive_failures"])
	if reason, _ := e.Data["reason"].(string); reason != "" {
		desc += ": " + reason
	}
	publishAndStoreAlert(b.ws, "security_alert", "high", fmt.Sprintf("Node unreachable: %s", e.Target), desc, "cluster-heartbeat", "firing")
}

func (b *realtimeEventBridge) handleNodeRecovered(e *events.OrchestrationEvent) {
	publishAndStoreAlert(b.ws, "security_alert", "low", fmt.Sprintf("Node reachable: %s", e.Target),
		fmt.Sprintf("Node %s answers heartbeats again", e.Target), "cluster-heartbeat", "resolved")
}

// --- VMManager -> vm_status / VM-error alerts -------------------------------

type realtimeVMListener struct {
	ws   *websocketapi.WebSocketHandler
	mu   sync.Mutex
	last map[string]string // vmID -> last published status
}

func newRealtimeVMListener(ws *websocketapi.WebSocketHandler) *realtimeVMListener {
	return &realtimeVMListener{ws: ws, last: make(map[string]string)}
}

func (l *realtimeVMListener) OnVMEvent(event core_vm.VMEvent) {
	if event.VM == nil {
		return
	}
	id, status := event.VM.ID(), string(event.VM.State())
	l.mu.Lock()
	previous := l.last[id]
	l.last[id] = status
	l.mu.Unlock()

	l.ws.PublishVMStatus(websocketapi.VMStatusMessage{ID: id, Name: event.VM.Name(), Status: status, PreviousStatus: previous})

	if event.Type == core_vm.VMEventError {
		msg := event.Message
		if msg == "" {
			msg = fmt.Sprintf("VM %s reported an error", id)
		}
		publishAndStoreAlert(l.ws, "security_alert", "high", fmt.Sprintf("VM error: %s", event.VM.Name()), msg, "vm-manager", "firing")
	}
}

// --- wiring ------------------------------------------------------------

// wireVMManager finishes realtime wiring that needs the VM manager and DB,
// which aren't constructed yet when initializeCanonicalServices runs. Called
// once from buildCanonicalServer, after vmManager exists.
func (s *canonicalServices) wireVMManager(vmManager *core_vm.VMManager, db *sql.DB, storagePath string) {
	if vmManager == nil {
		return
	}

	s.healingController.SetVMController(&vmHealingController{vmManager: vmManager, supervisor: restartSupervisor})
	s.healingController.SetMigrationTargetSelector(&vmMigrationTargetSelector{
		vmManager:  vmManager,
		candidates: migrationCandidatesFunc(vmManager, storagePath, db),
	})
	s.healingController.SetHealthSource(vmHealthSource(vmManager, restartSupervisor))

	bridge := newRealtimeEventBridge(s.websocketHandler, s.orchLogger)
	bridgeCtx, bridgeCancel := context.WithCancel(context.Background())
	bridge.subscribe(bridgeCtx, s.eventBus)

	listener := newRealtimeVMListener(s.websocketHandler)
	vmManager.AddEventListener(listener)

	prevShutdown := s.shutdown
	s.shutdown = func() {
		bridgeCancel()
		vmManager.RemoveEventListener(listener)
		if prevShutdown != nil {
			prevShutdown()
		}
	}
}
