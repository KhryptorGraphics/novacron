package security

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/khryptorgraphics/novacron/backend/core/audit"
	"golang.org/x/crypto/curve25519"
	"golang.org/x/crypto/hkdf"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// DistributedSecureMessaging provides secure communication for distributed cluster nodes
type DistributedSecureMessaging struct {
	clusterChannels    map[string]*SecureChannel
	federationManager  FederationManager
	crossClusterRunner CrossClusterRunner
	stateCoordinator   DistributedStateCoordinator
	encryptionManager  *EncryptionManager
	auditLogger        audit.AuditLogger
	tlsConfig          *tls.Config
	nodeID             string
	mu                 sync.RWMutex
}

// SecureChannel represents a secure communication channel between cluster nodes
type SecureChannel struct {
	ChannelID        string           `json:"channel_id"`
	LocalNodeID      string           `json:"local_node_id"`
	RemoteNodeID     string           `json:"remote_node_id"`
	ChannelType      ChannelType      `json:"channel_type"`
	SessionKeys      *SessionKeyPair  `json:"session_keys"`
	Connection       *grpc.ClientConn `json:"-"`
	Status           ChannelStatus    `json:"status"`
	CreatedAt        time.Time        `json:"created_at"`
	LastActivity     time.Time        `json:"last_activity"`
	MetricsCollector *ChannelMetrics  `json:"-"`
}

// DistributedMessageType defines types of distributed messages
type DistributedMessageType string

const (
	DistributedMessageTypeClusterSync     DistributedMessageType = "cluster_sync"
	DistributedMessageTypeFederationEvent DistributedMessageType = "federation_event"
	DistributedMessageTypeStateUpdate     DistributedMessageType = "state_update"
	DistributedMessageTypeMigrationEvent  DistributedMessageType = "migration_event"
	DistributedMessageTypeHeartbeat       DistributedMessageType = "heartbeat"
	DistributedMessageTypeSecurityEvent   DistributedMessageType = "security_event"
	DistributedMessageTypeControlPlane    DistributedMessageType = "control_plane"
	DistributedMessageTypeDataPlane       DistributedMessageType = "data_plane"
)

// ChannelType defines types of communication channels
type ChannelType string

const (
	ChannelTypeIntraCluster ChannelType = "intra_cluster"
	ChannelTypeInterCluster ChannelType = "inter_cluster"
	ChannelTypeFederation   ChannelType = "federation"
	ChannelTypeMigration    ChannelType = "migration"
	ChannelTypeControl      ChannelType = "control"
)

// ChannelStatus represents channel status
type ChannelStatus string

const (
	ChannelStatusActive       ChannelStatus = "active"
	ChannelStatusConnecting   ChannelStatus = "connecting"
	ChannelStatusDisconnected ChannelStatus = "disconnected"
	ChannelStatusError        ChannelStatus = "error"
)

// SecureDeliveryStatus represents message delivery status
type SecureDeliveryStatus string

const (
	SecureStatusSent      SecureDeliveryStatus = "sent"
	SecureStatusDelivered SecureDeliveryStatus = "delivered"
	SecureStatusFailed    SecureDeliveryStatus = "failed"
	SecureStatusPending   SecureDeliveryStatus = "pending"
)

// SessionKeyPair contains encryption keys for a secure session
type SessionKeyPair struct {
	SendKey    []byte    `json:"send_key"`
	ReceiveKey []byte    `json:"receive_key"`
	IV         []byte    `json:"iv"`
	CreatedAt  time.Time `json:"created_at"`
	ExpiresAt  time.Time `json:"expires_at"`
	Counter    uint64    `json:"counter"`
}

// DistributedMessage represents a secure message in distributed communications
type DistributedMessage struct {
	MessageID        string                 `json:"message_id"`
	ChannelID        string                 `json:"channel_id"`
	MessageType      DistributedMessageType `json:"message_type"`
	SourceNodeID     string                 `json:"source_node_id"`
	TargetNodeID     string                 `json:"target_node_id"`
	ClusterID        string                 `json:"cluster_id"`
	Payload          []byte                 `json:"payload"`
	EncryptedPayload []byte                 `json:"encrypted_payload"`
	Signature        []byte                 `json:"signature"`
	Timestamp        time.Time              `json:"timestamp"`
	ExpiresAt        *time.Time             `json:"expires_at,omitempty"`
	Priority         MessagePriority        `json:"priority"`
	Headers          map[string]string      `json:"headers"`
	Metadata         map[string]interface{} `json:"metadata"`
	RetryCount       int                    `json:"retry_count"`
	MaxRetries       int                    `json:"max_retries"`
}

// MessagePriority defines message priority levels
type MessagePriority string

const (
	MessagePriorityLow      MessagePriority = "low"
	MessagePriorityNormal   MessagePriority = "normal"
	MessagePriorityHigh     MessagePriority = "high"
	MessagePriorityCritical MessagePriority = "critical"
)

// ChannelMetrics collects metrics for secure channels
type ChannelMetrics struct {
	MessagesSent     uint64        `json:"messages_sent"`
	MessagesReceived uint64        `json:"messages_received"`
	BytesSent        uint64        `json:"bytes_sent"`
	BytesReceived    uint64        `json:"bytes_received"`
	ErrorCount       uint64        `json:"error_count"`
	LastError        string        `json:"last_error,omitempty"`
	AverageLatency   time.Duration `json:"average_latency"`
	ConnectionUptime time.Duration `json:"connection_uptime"`
}

// FederationManager interface for federation operations
type FederationManager interface {
	GetClusterNodes(clusterID string) ([]string, error)
	RegisterNode(nodeID, clusterID string) error
	HandleFederationEvent(event interface{}) error
}

// CrossClusterRunner interface for cross-cluster operations
type CrossClusterRunner interface {
	ExecuteRemoteOperation(ctx context.Context, targetNode string, operation interface{}) error
	GetRemoteState(ctx context.Context, targetNode string) (interface{}, error)
	SynchronizeState(ctx context.Context, nodes []string) error
}

// DistributedStateCoordinator interface for state coordination
type DistributedStateCoordinator interface {
	UpdateState(key string, value interface{}) error
	GetState(key string) (interface{}, error)
	SynchronizeState(nodes []string) error
	GetDistributedLock(key string) (interface{}, error)
}

// NewDistributedSecureMessaging creates a new distributed secure messaging service
func NewDistributedSecureMessaging(
	nodeID string,
	encMgr *EncryptionManager,
	auditLogger audit.AuditLogger,
	fedMgr FederationManager,
	crossRunner CrossClusterRunner,
	stateCoord DistributedStateCoordinator,
) (*DistributedSecureMessaging, error) {

	tlsConfig, err := encMgr.GetTLSConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get TLS config: %w", err)
	}

	dsm := &DistributedSecureMessaging{
		clusterChannels:    make(map[string]*SecureChannel),
		nodeID:             nodeID,
		encryptionManager:  encMgr,
		auditLogger:        auditLogger,
		federationManager:  fedMgr,
		crossClusterRunner: crossRunner,
		stateCoordinator:   stateCoord,
		tlsConfig:          tlsConfig,
	}

	return dsm, nil
}

// EstablishSecureChannel creates a secure communication channel with another node
func (dsm *DistributedSecureMessaging) EstablishSecureChannel(
	ctx context.Context,
	targetNodeID string,
	channelType ChannelType,
) (*SecureChannel, error) {
	dsm.mu.Lock()
	defer dsm.mu.Unlock()

	channelID := dsm.generateChannelID(dsm.nodeID, targetNodeID)

	// Check if channel already exists
	if existingChannel, exists := dsm.clusterChannels[channelID]; exists {
		if existingChannel.Status == ChannelStatusActive {
			return existingChannel, nil
		}
	}

	// Generate session keys
	sessionKeys, err := dsm.generateSessionKeys()
	if err != nil {
		return nil, fmt.Errorf("failed to generate session keys: %w", err)
	}

	// Establish gRPC connection with mutual TLS
	conn, err := dsm.establishGRPCConnection(ctx, targetNodeID)
	if err != nil {
		return nil, fmt.Errorf("failed to establish gRPC connection: %w", err)
	}

	channel := &SecureChannel{
		ChannelID:        channelID,
		LocalNodeID:      dsm.nodeID,
		RemoteNodeID:     targetNodeID,
		ChannelType:      channelType,
		SessionKeys:      sessionKeys,
		Connection:       conn,
		Status:           ChannelStatusActive,
		CreatedAt:        time.Now(),
		LastActivity:     time.Now(),
		MetricsCollector: &ChannelMetrics{},
	}

	dsm.clusterChannels[channelID] = channel

	// Audit log channel establishment
	dsm.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   dsm.nodeID,
		Resource: fmt.Sprintf("channel_%s", channelID),
		Action:   audit.ActionUpdate,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description":  fmt.Sprintf("Secure channel established between %s and %s", dsm.nodeID, targetNodeID),
			"channel_id":   channelID,
			"channel_type": string(channelType),
			"local_node":   dsm.nodeID,
			"remote_node":  targetNodeID,
			"severity":     "info",
		},
	})

	log.Printf("Established secure channel %s with node %s", channelID, targetNodeID)
	return channel, nil
}

// SendMessage sends a secure message to another node
func (dsm *DistributedSecureMessaging) SendMessage(
	ctx context.Context,
	targetNodeID string,
	messageType DistributedMessageType,
	payload []byte,
	priority MessagePriority,
) (*DistributedMessage, error) {

	// Get or create secure channel
	channel, err := dsm.getOrCreateChannel(ctx, targetNodeID, ChannelTypeIntraCluster)
	if err != nil {
		return nil, fmt.Errorf("failed to get secure channel: %w", err)
	}

	// Create message
	message := &DistributedMessage{
		MessageID:    uuid.New().String(),
		ChannelID:    channel.ChannelID,
		MessageType:  messageType,
		SourceNodeID: dsm.nodeID,
		TargetNodeID: targetNodeID,
		Payload:      payload,
		Timestamp:    time.Now(),
		Priority:     priority,
		Headers:      make(map[string]string),
		Metadata:     make(map[string]interface{}),
		MaxRetries:   3,
	}

	// Set message expiration based on priority
	switch priority {
	case MessagePriorityCritical:
		expiry := time.Now().Add(5 * time.Minute)
		message.ExpiresAt = &expiry
	case MessagePriorityHigh:
		expiry := time.Now().Add(15 * time.Minute)
		message.ExpiresAt = &expiry
	case MessagePriorityNormal:
		expiry := time.Now().Add(1 * time.Hour)
		message.ExpiresAt = &expiry
	}

	// Encrypt payload
	encryptedPayload, err := dsm.encryptPayload(payload, channel.SessionKeys)
	if err != nil {
		return nil, fmt.Errorf("failed to encrypt payload: %w", err)
	}
	message.EncryptedPayload = encryptedPayload

	// Sign message
	signature, err := dsm.signMessage(message)
	if err != nil {
		return nil, fmt.Errorf("failed to sign message: %w", err)
	}
	message.Signature = signature

	// Send message via gRPC or other transport
	if err := dsm.transmitMessage(ctx, channel, message); err != nil {
		return nil, fmt.Errorf("failed to transmit message: %w", err)
	}

	// Update metrics
	channel.MetricsCollector.MessagesSent++
	channel.MetricsCollector.BytesSent += uint64(len(encryptedPayload))
	channel.LastActivity = time.Now()

	// Audit log message sending
	dsm.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   dsm.nodeID,
		Resource: "secure_message",
		Action:   audit.ActionWrite,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description":  fmt.Sprintf("Secure message sent from %s to %s", dsm.nodeID, targetNodeID),
			"message_id":   message.MessageID,
			"channel_id":   channel.ChannelID,
			"message_type": string(messageType),
			"priority":     string(priority),
			"size_bytes":   len(encryptedPayload),
		},
	})

	return message, nil
}

// ReceiveMessage handles incoming secure messages
func (dsm *DistributedSecureMessaging) ReceiveMessage(
	ctx context.Context,
	encryptedMessage *DistributedMessage,
) ([]byte, error) {

	// Verify message signature
	if err := dsm.verifyMessage(encryptedMessage); err != nil {
		return nil, fmt.Errorf("message signature verification failed: %w", err)
	}

	// Check message expiration
	if encryptedMessage.ExpiresAt != nil && encryptedMessage.ExpiresAt.Before(time.Now()) {
		return nil, fmt.Errorf("message has expired")
	}

	// Get channel
	channel, exists := dsm.clusterChannels[encryptedMessage.ChannelID]
	if !exists {
		return nil, fmt.Errorf("channel not found: %s", encryptedMessage.ChannelID)
	}

	// Decrypt payload
	payload, err := dsm.decryptPayload(encryptedMessage.EncryptedPayload, channel.SessionKeys)
	if err != nil {
		return nil, fmt.Errorf("failed to decrypt payload: %w", err)
	}

	// Update metrics
	channel.MetricsCollector.MessagesReceived++
	channel.MetricsCollector.BytesReceived += uint64(len(encryptedMessage.EncryptedPayload))
	channel.LastActivity = time.Now()

	// Process message based on type
	if err := dsm.processMessage(ctx, encryptedMessage, payload); err != nil {
		log.Printf("Error processing message %s: %v", encryptedMessage.MessageID, err)
	}

	// Audit log message reception
	dsm.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   encryptedMessage.SourceNodeID,
		Resource: "secure_message",
		Action:   audit.ActionRead,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description":  fmt.Sprintf("Secure message received from %s to %s", encryptedMessage.SourceNodeID, dsm.nodeID),
			"message_id":   encryptedMessage.MessageID,
			"channel_id":   encryptedMessage.ChannelID,
			"message_type": string(encryptedMessage.MessageType),
			"source_node":  encryptedMessage.SourceNodeID,
		},
	})

	return payload, nil
}

// BroadcastToCluster sends a message to all nodes in a cluster
func (dsm *DistributedSecureMessaging) BroadcastToCluster(
	ctx context.Context,
	clusterID string,
	messageType DistributedMessageType,
	payload []byte,
	priority MessagePriority,
) error {

	// Get cluster nodes from federation manager
	nodes, err := dsm.federationManager.GetClusterNodes(clusterID)
	if err != nil {
		return fmt.Errorf("failed to get cluster nodes: %w", err)
	}

	var errors []error
	for _, nodeID := range nodes {
		if nodeID == dsm.nodeID {
			continue // Skip self
		}

		if _, err := dsm.SendMessage(ctx, nodeID, messageType, payload, priority); err != nil {
			errors = append(errors, fmt.Errorf("failed to send to %s: %w", nodeID, err))
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("broadcast partially failed: %v", errors)
	}

	return nil
}

// Helper methods

func (dsm *DistributedSecureMessaging) generateChannelID(localNodeID, remoteNodeID string) string {
	if localNodeID < remoteNodeID {
		return fmt.Sprintf("%s-%s", localNodeID, remoteNodeID)
	}
	return fmt.Sprintf("%s-%s", remoteNodeID, localNodeID)
}

func (dsm *DistributedSecureMessaging) generateSessionKeys() (*SessionKeyPair, error) {
	sendKey := make([]byte, 32)
	receiveKey := make([]byte, 32)
	iv := make([]byte, 16)

	if _, err := rand.Read(sendKey); err != nil {
		return nil, err
	}
	if _, err := rand.Read(receiveKey); err != nil {
		return nil, err
	}
	if _, err := rand.Read(iv); err != nil {
		return nil, err
	}

	return &SessionKeyPair{
		SendKey:    sendKey,
		ReceiveKey: receiveKey,
		IV:         iv,
		CreatedAt:  time.Now(),
		ExpiresAt:  time.Now().Add(24 * time.Hour), // 24 hour key rotation
		Counter:    0,
	}, nil
}

func (dsm *DistributedSecureMessaging) establishGRPCConnection(ctx context.Context, targetNodeID string) (*grpc.ClientConn, error) {
	// In production, this would resolve the target node's address
	targetAddress := fmt.Sprintf("%s:8443", targetNodeID) // Example

	creds := credentials.NewTLS(dsm.tlsConfig)

	conn, err := grpc.DialContext(ctx, targetAddress,
		grpc.WithTransportCredentials(creds),
		grpc.WithBlock(),
		grpc.WithTimeout(30*time.Second),
	)

	if err != nil {
		return nil, fmt.Errorf("failed to dial %s: %w", targetAddress, err)
	}

	return conn, nil
}

func (dsm *DistributedSecureMessaging) getOrCreateChannel(ctx context.Context, targetNodeID string, channelType ChannelType) (*SecureChannel, error) {
	channelID := dsm.generateChannelID(dsm.nodeID, targetNodeID)

	dsm.mu.RLock()
	if channel, exists := dsm.clusterChannels[channelID]; exists && channel.Status == ChannelStatusActive {
		dsm.mu.RUnlock()
		return channel, nil
	}
	dsm.mu.RUnlock()

	return dsm.EstablishSecureChannel(ctx, targetNodeID, channelType)
}

func (dsm *DistributedSecureMessaging) encryptPayload(payload []byte, keys *SessionKeyPair) ([]byte, error) {
	block, err := aes.NewCipher(keys.SendKey)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	// Use a portion of IV for nonce
	nonce := keys.IV[:gcm.NonceSize()]

	ciphertext := gcm.Seal(nil, nonce, payload, nil)
	return ciphertext, nil
}

func (dsm *DistributedSecureMessaging) decryptPayload(encryptedPayload []byte, keys *SessionKeyPair) ([]byte, error) {
	block, err := aes.NewCipher(keys.ReceiveKey)
	if err != nil {
		return nil, err
	}

	gcm, err := cipher.NewGCM(block)
	if err != nil {
		return nil, err
	}

	nonce := keys.IV[:gcm.NonceSize()]

	plaintext, err := gcm.Open(nil, nonce, encryptedPayload, nil)
	if err != nil {
		return nil, err
	}

	return plaintext, nil
}

func (dsm *DistributedSecureMessaging) signMessage(message *DistributedMessage) ([]byte, error) {
	// Use EncryptionManager to sign the message
	return dsm.encryptionManager.SignMessage(message.EncryptedPayload)
}

func (dsm *DistributedSecureMessaging) verifyMessage(message *DistributedMessage) error {
	// Use EncryptionManager to verify the message signature
	return dsm.encryptionManager.VerifySignature(message.EncryptedPayload, message.Signature)
}

func (dsm *DistributedSecureMessaging) transmitMessage(ctx context.Context, channel *SecureChannel, message *DistributedMessage) error {
	// In a real implementation, this would use gRPC or other transport
	// For now, simulate successful transmission
	log.Printf("Transmitting message %s via channel %s", message.MessageID, channel.ChannelID)
	return nil
}

func (dsm *DistributedSecureMessaging) processMessage(ctx context.Context, message *DistributedMessage, payload []byte) error {
	switch message.MessageType {
	case DistributedMessageTypeClusterSync:
		return dsm.handleClusterSync(ctx, payload)
	case DistributedMessageTypeFederationEvent:
		return dsm.handleFederationEvent(ctx, payload)
	case DistributedMessageTypeStateUpdate:
		return dsm.handleStateUpdate(ctx, payload)
	case DistributedMessageTypeMigrationEvent:
		return dsm.handleMigrationEvent(ctx, payload)
	case DistributedMessageTypeHeartbeat:
		return dsm.handleHeartbeat(ctx, message.SourceNodeID, payload)
	case DistributedMessageTypeSecurityEvent:
		return dsm.handleSecurityEvent(ctx, payload)
	default:
		log.Printf("Unknown message type: %s", message.MessageType)
		return nil
	}
}

func (dsm *DistributedSecureMessaging) handleClusterSync(ctx context.Context, payload []byte) error {
	// Handle cluster synchronization message
	log.Printf("Processing cluster sync message")

	var syncData map[string]interface{}
	if err := json.Unmarshal(payload, &syncData); err != nil {
		return fmt.Errorf("failed to unmarshal sync data: %w", err)
	}

	// Coordinate with state coordinator
	if dsm.stateCoordinator != nil {
		nodes, ok := syncData["nodes"].([]string)
		if ok {
			return dsm.stateCoordinator.SynchronizeState(nodes)
		}
	}

	return nil
}

func (dsm *DistributedSecureMessaging) handleFederationEvent(ctx context.Context, payload []byte) error {
	// Handle federation event message
	log.Printf("Processing federation event message")

	var eventData interface{}
	if err := json.Unmarshal(payload, &eventData); err != nil {
		return fmt.Errorf("failed to unmarshal federation event: %w", err)
	}

	if dsm.federationManager != nil {
		return dsm.federationManager.HandleFederationEvent(eventData)
	}

	return nil
}

func (dsm *DistributedSecureMessaging) handleStateUpdate(ctx context.Context, payload []byte) error {
	// Handle state update message
	log.Printf("Processing state update message")

	var updateData map[string]interface{}
	if err := json.Unmarshal(payload, &updateData); err != nil {
		return fmt.Errorf("failed to unmarshal state update: %w", err)
	}

	if dsm.stateCoordinator != nil {
		for key, value := range updateData {
			if err := dsm.stateCoordinator.UpdateState(key, value); err != nil {
				log.Printf("Failed to update state %s: %v", key, err)
			}
		}
	}

	return nil
}

func (dsm *DistributedSecureMessaging) handleMigrationEvent(ctx context.Context, payload []byte) error {
	// Handle migration event message
	log.Printf("Processing migration event message")

	var migrationData interface{}
	if err := json.Unmarshal(payload, &migrationData); err != nil {
		return fmt.Errorf("failed to unmarshal migration event: %w", err)
	}

	// Coordinate with cross-cluster runner
	// Implementation would depend on migration event structure

	return nil
}

func (dsm *DistributedSecureMessaging) handleHeartbeat(ctx context.Context, sourceNodeID string, payload []byte) error {
	// Handle heartbeat message
	log.Printf("Received heartbeat from node %s", sourceNodeID)

	// Update node status in federation manager
	if dsm.federationManager != nil {
		// Implementation would update node health status
	}

	return nil
}

func (dsm *DistributedSecureMessaging) handleSecurityEvent(ctx context.Context, payload []byte) error {
	// Handle security event message
	log.Printf("Processing security event message")

	var securityEvent SecurityEvent
	if err := json.Unmarshal(payload, &securityEvent); err != nil {
		return fmt.Errorf("failed to unmarshal security event: %w", err)
	}

	// Process security event - could trigger alerts, policy enforcement, etc.
	dsm.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   securityEvent.NodeID,
		Resource: "security_event",
		Action:   audit.ActionRead,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description":    fmt.Sprintf("Security event received: %s", securityEvent.Type),
			"security_event": securityEvent,
		},
	})

	return nil
}

// GetChannelMetrics returns metrics for a specific channel
func (dsm *DistributedSecureMessaging) GetChannelMetrics(channelID string) (*ChannelMetrics, error) {
	dsm.mu.RLock()
	defer dsm.mu.RUnlock()

	channel, exists := dsm.clusterChannels[channelID]
	if !exists {
		return nil, fmt.Errorf("channel not found: %s", channelID)
	}

	// Calculate uptime
	channel.MetricsCollector.ConnectionUptime = time.Since(channel.CreatedAt)

	return channel.MetricsCollector, nil
}

// GetAllChannels returns all active secure channels
func (dsm *DistributedSecureMessaging) GetAllChannels() map[string]*SecureChannel {
	dsm.mu.RLock()
	defer dsm.mu.RUnlock()

	channels := make(map[string]*SecureChannel)
	for id, channel := range dsm.clusterChannels {
		channelCopy := *channel
		channels[id] = &channelCopy
	}

	return channels
}

// CloseChannel closes a secure channel
func (dsm *DistributedSecureMessaging) CloseChannel(channelID string) error {
	dsm.mu.Lock()
	defer dsm.mu.Unlock()

	channel, exists := dsm.clusterChannels[channelID]
	if !exists {
		return fmt.Errorf("channel not found: %s", channelID)
	}

	if channel.Connection != nil {
		channel.Connection.Close()
	}

	channel.Status = ChannelStatusDisconnected
	delete(dsm.clusterChannels, channelID)

	dsm.auditLogger.LogEvent(context.Background(), &audit.AuditEvent{
		UserID:   dsm.nodeID,
		Resource: "secure_channel",
		Action:   audit.ActionDelete,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description": fmt.Sprintf("Secure channel %s closed", channelID),
			"channel_id":  channelID,
			"local_node":  channel.LocalNodeID,
			"remote_node": channel.RemoteNodeID,
		},
	})

	return nil
}

// RotateSessionKeys rotates session keys for a channel
func (dsm *DistributedSecureMessaging) RotateSessionKeys(ctx context.Context, channelID string) error {
	dsm.mu.Lock()
	defer dsm.mu.Unlock()

	channel, exists := dsm.clusterChannels[channelID]
	if !exists {
		return fmt.Errorf("channel not found: %s", channelID)
	}

	newKeys, err := dsm.generateSessionKeys()
	if err != nil {
		return fmt.Errorf("failed to generate new session keys: %w", err)
	}

	oldKeys := channel.SessionKeys
	channel.SessionKeys = newKeys

	dsm.auditLogger.LogEvent(ctx, &audit.AuditEvent{
		UserID:   dsm.nodeID,
		Resource: "session_keys",
		Action:   audit.ActionRotate,
		Result:   audit.ResultSuccess,
		Details: map[string]interface{}{
			"description": fmt.Sprintf("Session keys rotated for channel %s", channelID),
			"channel_id":  channelID,
			"old_key_age": time.Since(oldKeys.CreatedAt).String(),
		},
	})

	return nil
}
