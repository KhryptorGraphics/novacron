package main

import (
	"fmt"

	websocketapi "github.com/khryptorgraphics/novacron/backend/api/websocket"
	"github.com/khryptorgraphics/novacron/backend/pkg/logger"
)

// wsLogSink streams api-server log entries (pkg/logger's global logger) to
// /api/ws/logs clients as source "system" log messages.
//
// Non-blocking and bounded: WriteEntry runs on the logging goroutine and only
// hands the message to PublishLog, which enqueues on the websocket handler's
// bounded logs broadcast queue and drops when it is full, so a slow consumer
// never stalls a caller of logger.Info/Warn/Error.
//
// No recursion: the sink never logs, and the websocket handler reports its
// own diagnostics (including PublishLog's full-queue drop) through its own
// logrus logger, which carries no sink.
type wsLogSink struct {
	publish  func(websocketapi.LogMessage)
	minLevel logger.Level
}

func newWSLogSink(ws *websocketapi.WebSocketHandler, minLevel logger.Level) *wsLogSink {
	return &wsLogSink{publish: ws.PublishLog, minLevel: minLevel}
}

func (s *wsLogSink) WriteEntry(e logger.Entry) {
	if logger.LevelFromString(e.Level) < s.minLevel {
		return
	}
	s.publish(logMessageFromEntry(e))
}

// logMessageFromEntry formats every field value now, on the logging
// goroutine: the message is marshaled later by the broadcast worker, and a
// caller may mutate a value it logged by reference once logging returns.
func logMessageFromEntry(e logger.Entry) websocketapi.LogMessage {
	msg := websocketapi.LogMessage{
		Type:      "log",
		Source:    "system",
		Level:     e.Level,
		Message:   e.Message,
		Timestamp: e.Timestamp,
		Component: "api-server",
	}
	// Entry.File/Line are not forwarded: for the package-level logger.X
	// helpers api-server calls, they name pkg/logger itself, not the caller.
	var labels map[string]string
	set := func(k, v string) {
		if labels == nil {
			labels = make(map[string]string, len(e.Fields)+1)
		}
		labels[k] = v
	}
	for k, v := range e.Fields {
		set(k, fmt.Sprint(v))
	}
	// api-server logs a VM id under "vm"; lifting it into VMID makes the
	// stream's ?vm_id= filter match those entries.
	if vm, ok := labels["vm"]; ok {
		msg.VMID = vm
	} else if vm, ok := labels["vm_id"]; ok {
		msg.VMID = vm
	}
	for _, kv := range [...]struct{ k, v string }{
		{"error", e.Error},
		{"request_id", e.RequestID},
		{"trace_id", e.TraceID},
		{"user_id", e.UserID},
		{"tenant_id", e.TenantID},
	} {
		if kv.v != "" {
			set(kv.k, kv.v)
		}
	}
	msg.Labels = labels
	return msg
}
