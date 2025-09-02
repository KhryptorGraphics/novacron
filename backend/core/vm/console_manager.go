package vm

import (
	"context"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"
)

// ConsoleType represents the type of console connection
type ConsoleType string

const (
	ConsoleTypeSerial ConsoleType = "serial"
	ConsoleTypeVNC    ConsoleType = "vnc"
	ConsoleTypeSPICE  ConsoleType = "spice"
	ConsoleTypeRDP    ConsoleType = "rdp"
)

// ConsoleSession represents an active console session
type ConsoleSession struct {
	ID         string      `json:"id"`
	VMID       string      `json:"vm_id"`
	Type       ConsoleType `json:"type"`
	UserID     string      `json:"user_id"`
	StartedAt  time.Time   `json:"started_at"`
	LastActive time.Time   `json:"last_active"`
	RemoteAddr string      `json:"remote_addr"`
	conn       net.Conn
	wsConn     *websocket.Conn
	closed     bool
	mutex      sync.RWMutex
}

// ConsoleManager manages VM console connections
type ConsoleManager struct {
	sessions     map[string]*ConsoleSession
	vmSessions   map[string][]*ConsoleSession
	sessionMutex sync.RWMutex
	vmManager    *VMManager
	maxSessions  int
	timeout      time.Duration
}

// NewConsoleManager creates a new console manager
func NewConsoleManager(vmManager *VMManager) *ConsoleManager {
	return &ConsoleManager{
		sessions:    make(map[string]*ConsoleSession),
		vmSessions:  make(map[string][]*ConsoleSession),
		vmManager:   vmManager,
		maxSessions: 100,
		timeout:     30 * time.Minute,
	}
}

// CreateConsoleSession creates a new console session for a VM
func (cm *ConsoleManager) CreateConsoleSession(ctx context.Context, vmID string, consoleType ConsoleType, userID string, remoteAddr string) (*ConsoleSession, error) {
	cm.sessionMutex.Lock()
	defer cm.sessionMutex.Unlock()

	// Check if VM exists
	vm, err := cm.vmManager.GetVM(vmID)
	if err != nil {
		return nil, fmt.Errorf("VM not found: %w", err)
	}

	// Check if VM is running
	if vm.State() != StateRunning {
		return nil, fmt.Errorf("VM is not running")
	}

	// Check session limit
	if len(cm.sessions) >= cm.maxSessions {
		return nil, fmt.Errorf("maximum console sessions reached")
	}

	// Create new session
	session := &ConsoleSession{
		ID:         uuid.New().String(),
		VMID:       vmID,
		Type:       consoleType,
		UserID:     userID,
		StartedAt:  time.Now(),
		LastActive: time.Now(),
		RemoteAddr: remoteAddr,
		closed:     false,
	}

	// Store session
	cm.sessions[session.ID] = session
	
	// Add to VM sessions
	if cm.vmSessions[vmID] == nil {
		cm.vmSessions[vmID] = make([]*ConsoleSession, 0)
	}
	cm.vmSessions[vmID] = append(cm.vmSessions[vmID], session)

	return session, nil
}

// GetConsoleSession retrieves a console session by ID
func (cm *ConsoleManager) GetConsoleSession(sessionID string) (*ConsoleSession, error) {
	cm.sessionMutex.RLock()
	defer cm.sessionMutex.RUnlock()

	session, exists := cm.sessions[sessionID]
	if !exists {
		return nil, fmt.Errorf("console session not found")
	}

	return session, nil
}

// CloseConsoleSession closes a console session
func (cm *ConsoleManager) CloseConsoleSession(sessionID string) error {
	cm.sessionMutex.Lock()
	defer cm.sessionMutex.Unlock()

	session, exists := cm.sessions[sessionID]
	if !exists {
		return fmt.Errorf("console session not found")
	}

	// Mark as closed
	session.mutex.Lock()
	session.closed = true
	session.mutex.Unlock()

	// Close connections
	if session.conn != nil {
		session.conn.Close()
	}
	if session.wsConn != nil {
		session.wsConn.Close()
	}

	// Remove from sessions
	delete(cm.sessions, sessionID)

	// Remove from VM sessions
	if vmSessions, exists := cm.vmSessions[session.VMID]; exists {
		for i, s := range vmSessions {
			if s.ID == sessionID {
				cm.vmSessions[session.VMID] = append(vmSessions[:i], vmSessions[i+1:]...)
				break
			}
		}
	}

	return nil
}

// ListConsoleSessions lists all active console sessions
func (cm *ConsoleManager) ListConsoleSessions() []*ConsoleSession {
	cm.sessionMutex.RLock()
	defer cm.sessionMutex.RUnlock()

	sessions := make([]*ConsoleSession, 0, len(cm.sessions))
	for _, session := range cm.sessions {
		sessions = append(sessions, session)
	}

	return sessions
}

// ListVMConsoleSessions lists console sessions for a specific VM
func (cm *ConsoleManager) ListVMConsoleSessions(vmID string) []*ConsoleSession {
	cm.sessionMutex.RLock()
	defer cm.sessionMutex.RUnlock()

	sessions, exists := cm.vmSessions[vmID]
	if !exists {
		return []*ConsoleSession{}
	}

	return sessions
}

// ConnectWebSocket connects a WebSocket to a console session
func (cm *ConsoleManager) ConnectWebSocket(sessionID string, ws *websocket.Conn) error {
	session, err := cm.GetConsoleSession(sessionID)
	if err != nil {
		return err
	}

	session.mutex.Lock()
	defer session.mutex.Unlock()

	if session.closed {
		return fmt.Errorf("session is closed")
	}

	session.wsConn = ws
	session.LastActive = time.Now()

	// Start proxy between console and WebSocket
	go cm.proxyConsole(session)

	return nil
}

// proxyConsole proxies data between console and WebSocket
func (cm *ConsoleManager) proxyConsole(session *ConsoleSession) {
	// In a real implementation, this would connect to the hypervisor console
	// For now, it's a placeholder
	defer cm.CloseConsoleSession(session.ID)

	// This would typically involve:
	// 1. Connecting to the hypervisor console (e.g., via libvirt)
	// 2. Proxying data between the console and WebSocket
	// 3. Handling resize events
	// 4. Managing authentication and authorization
}

// CleanupInactiveSessions removes inactive console sessions
func (cm *ConsoleManager) CleanupInactiveSessions() {
	cm.sessionMutex.Lock()
	defer cm.sessionMutex.Unlock()

	now := time.Now()
	for sessionID, session := range cm.sessions {
		if now.Sub(session.LastActive) > cm.timeout {
			// Close inactive session
			if session.conn != nil {
				session.conn.Close()
			}
			if session.wsConn != nil {
				session.wsConn.Close()
			}
			delete(cm.sessions, sessionID)

			// Remove from VM sessions
			if vmSessions, exists := cm.vmSessions[session.VMID]; exists {
				for i, s := range vmSessions {
					if s.ID == sessionID {
						cm.vmSessions[session.VMID] = append(vmSessions[:i], vmSessions[i+1:]...)
						break
					}
				}
			}
		}
	}
}

// StartCleanupRoutine starts a background routine to cleanup inactive sessions
func (cm *ConsoleManager) StartCleanupRoutine(ctx context.Context) {
	go func() {
		ticker := time.NewTicker(5 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				cm.CleanupInactiveSessions()
			}
		}
	}()
}

// GetConsoleURL generates a console URL for a VM
func (cm *ConsoleManager) GetConsoleURL(vmID string, consoleType ConsoleType) (string, error) {
	// In a real implementation, this would generate a secure console URL
	// with authentication tokens and appropriate protocols
	
	// Check if VM exists and is running
	vm, err := cm.vmManager.GetVM(vmID)
	if err != nil {
		return "", fmt.Errorf("VM not found: %w", err)
	}

	if vm.State() != StateRunning {
		return "", fmt.Errorf("VM is not running")
	}

	// Generate console URL based on type
	var consoleURL string
	switch consoleType {
	case ConsoleTypeVNC:
		consoleURL = fmt.Sprintf("vnc://console.novacron.local:5900/%s", vmID)
	case ConsoleTypeSPICE:
		consoleURL = fmt.Sprintf("spice://console.novacron.local:5900/%s", vmID)
	case ConsoleTypeRDP:
		consoleURL = fmt.Sprintf("rdp://console.novacron.local:3389/%s", vmID)
	case ConsoleTypeSerial:
		consoleURL = fmt.Sprintf("wss://console.novacron.local/serial/%s", vmID)
	default:
		return "", fmt.Errorf("unsupported console type: %s", consoleType)
	}

	return consoleURL, nil
}

// SendInput sends input to a console session
func (session *ConsoleSession) SendInput(data []byte) error {
	session.mutex.RLock()
	defer session.mutex.RUnlock()

	if session.closed {
		return fmt.Errorf("session is closed")
	}

	if session.conn != nil {
		_, err := session.conn.Write(data)
		return err
	}

	return fmt.Errorf("no console connection")
}

// ReadOutput reads output from a console session
func (session *ConsoleSession) ReadOutput(buffer []byte) (int, error) {
	session.mutex.RLock()
	defer session.mutex.RUnlock()

	if session.closed {
		return 0, io.EOF
	}

	if session.conn != nil {
		return session.conn.Read(buffer)
	}

	return 0, fmt.Errorf("no console connection")
}