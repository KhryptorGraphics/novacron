//go:build !experimental

package orchestration

import (
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	events "github.com/khryptorgraphics/novacron/backend/core/orchestration/events"
)

// checkOrigin validates the origin of WebSocket connection requests
func checkOrigin(r *http.Request, allowedOrigins []string) bool {
	origin := r.Header.Get("Origin")
	
	// Allow same-origin requests (no Origin header in some cases)
	if origin == "" {
		return true
	}

	// Check against allowed origins list
	for _, allowed := range allowedOrigins {
		if origin == allowed {
			return true
		}
		
		// Handle wildcard origins (be very careful with this in production)
		if strings.HasSuffix(allowed, "*") {
			prefix := strings.TrimSuffix(allowed, "*")
			if strings.HasPrefix(origin, prefix) {
				return true
			}
		}
	}

	return false
}

// getClientIP extracts the real client IP from the request
func getClientIP(r *http.Request) string {
	// Check X-Forwarded-For header (most common for load balancers)
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		// Take the first IP in the chain (original client)
		parts := strings.Split(xff, ",")
		return strings.TrimSpace(parts[0])
	}

	// Check X-Real-IP header (nginx)
	if xri := r.Header.Get("X-Real-IP"); xri != "" {
		return xri
	}

	// Check CF-Connecting-IP header (Cloudflare)
	if cfIP := r.Header.Get("CF-Connecting-IP"); cfIP != "" {
		return cfIP
	}

	// Fall back to RemoteAddr
	ip, _, _ := net.SplitHostPort(r.RemoteAddr)
	return ip
}

// extractAuthToken extracts authentication token from various sources
func extractAuthToken(r *http.Request) string {
	// 1. Check Authorization header (preferred method)
	authHeader := r.Header.Get("Authorization")
	if strings.HasPrefix(authHeader, "Bearer ") {
		return strings.TrimPrefix(authHeader, "Bearer ")
	}

	// 2. Check Sec-WebSocket-Protocol header (WebSocket-specific)
	protocols := r.Header.Get("Sec-WebSocket-Protocol")
	if protocols != "" {
		parts := strings.Split(protocols, ",")
		for _, protocol := range parts {
			protocol = strings.TrimSpace(protocol)
			if strings.HasPrefix(protocol, "access_token.") {
				return strings.TrimPrefix(protocol, "access_token.")
			}
		}
	}

	// 3. Check query parameter (less secure, should be avoided in production)
	if token := r.URL.Query().Get("token"); token != "" {
		return token
	}

	// 4. Check cookie (if using cookie-based auth)
	if cookie, err := r.Cookie("auth_token"); err == nil && cookie.Value != "" {
		return cookie.Value
	}

	return ""
}

// checkConnectionRateLimit checks if connection rate limit is exceeded for an IP
func (wsm *WebSocketManager) checkConnectionRateLimit(clientIP string) bool {
	wsm.rateLimiter.mutex.Lock()
	defer wsm.rateLimiter.mutex.Unlock()

	now := time.Now()
	entry, exists := wsm.rateLimiter.connectionLimits[clientIP]

	if !exists {
		wsm.rateLimiter.connectionLimits[clientIP] = &RateLimitEntry{
			count:     1,
			window:    now,
			lastReset: now,
		}
		return true
	}

	// Reset counter if window has passed
	if now.Sub(entry.window) >= time.Minute {
		entry.count = 1
		entry.window = now
		entry.lastReset = now
		return true
	}

	// Check if limit exceeded
	if entry.count >= wsm.securityConfig.RateLimitConnections {
		return false
	}

	entry.count++
	return true
}

// checkMessageRateLimit checks if message rate limit is exceeded for a client
func (wsm *WebSocketManager) checkMessageRateLimit(clientID string) bool {
	wsm.rateLimiter.mutex.Lock()
	defer wsm.rateLimiter.mutex.Unlock()

	now := time.Now()
	entry, exists := wsm.rateLimiter.messageLimits[clientID]

	if !exists {
		wsm.rateLimiter.messageLimits[clientID] = &RateLimitEntry{
			count:     1,
			window:    now,
			lastReset: now,
		}
		return true
	}

	// Reset counter if window has passed
	if now.Sub(entry.window) >= time.Minute {
		entry.count = 1
		entry.window = now
		entry.lastReset = now
		return true
	}

	// Check if limit exceeded
	if entry.count >= wsm.securityConfig.RateLimitMessages {
		return false
	}

	entry.count++
	return true
}

// cleanupRateLimits removes old rate limit entries
func (wsm *WebSocketManager) cleanupRateLimits() {
	wsm.rateLimiter.mutex.Lock()
	defer wsm.rateLimiter.mutex.Unlock()

	cutoff := time.Now().Add(-5 * time.Minute) // Clean up entries older than 5 minutes

	// Clean connection limits
	for ip, entry := range wsm.rateLimiter.connectionLimits {
		if entry.lastReset.Before(cutoff) {
			delete(wsm.rateLimiter.connectionLimits, ip)
		}
	}

	// Clean message limits
	for clientID, entry := range wsm.rateLimiter.messageLimits {
		if entry.lastReset.Before(cutoff) {
			delete(wsm.rateLimiter.messageLimits, clientID)
		}
	}
}

// hasPermissionForEvent checks if client has permission for specific event
func (c *WebSocketClient) hasPermissionForEvent(event *events.OrchestrationEvent) bool {
	// If authentication is not required, allow all events
	if !c.manager.securityConfig.RequireAuthentication {
		return true
	}

	// If client is not authenticated, deny all events
	if !c.authenticated {
		return false
	}

	// Check tenant isolation - client can only see events from their tenant
	if c.tenantID != "" && event.Target != "" {
		// Parse target to extract tenant info if available
		// This would depend on your event target format
		// For now, we'll allow all events for authenticated users
	}

	// Check if client has required permissions based on event type
	requiredPermission := getRequiredPermissionForEvent(event)
	if requiredPermission == "" {
		return true // No specific permission required
	}

	// Check if client has the required permission
	for _, perm := range c.permissions {
		if perm == requiredPermission || perm == "*:*" || perm == "system:admin" {
			return true
		}
		// Also check for wildcard permissions
		parts := strings.Split(requiredPermission, ":")
		if len(parts) == 2 {
			resource, action := parts[0], parts[1]
			if perm == resource+":*" || perm == "*:"+action {
				return true
			}
		}
	}

	return false
}

// getRequiredPermissionForEvent maps event types to required permissions
func getRequiredPermissionForEvent(event *events.OrchestrationEvent) string {
	switch event.Type {
	case events.EventTypeVMCreated, events.EventTypeVMUpdated, events.EventTypeVMDeleted:
		return "vm:read"
	case events.EventTypeNodeAdded, events.EventTypeNodeRemoved, events.EventTypeNodeUpdated:
		return "node:read"
	case events.EventTypeStorageCreated, events.EventTypeStorageDeleted:
		return "storage:read"
	case events.EventTypeNetworkCreated, events.EventTypeNetworkDeleted:
		return "network:read"
	case events.EventTypeSystemAlert, events.EventTypeSystemHealthUpdate:
		return "system:read"
	default:
		return "system:read" // Default permission for unknown event types
	}
}

// Start periodic cleanup of rate limits
func (wsm *WebSocketManager) startRateLimitCleanup() {
	go func() {
		ticker := time.NewTicker(2 * time.Minute)
		defer ticker.Stop()

		for {
			select {
			case <-wsm.ctx.Done():
				return
			case <-ticker.C:
				wsm.cleanupRateLimits()
			}
		}
	}()
}

// cleanupStaleConnections removes connections that haven't been active
func (wsm *WebSocketManager) cleanupStaleConnections() {
	wsm.clientsMutex.RLock()
	staleClients := make([]*WebSocketClient, 0)
	now := time.Now()
	staleThreshold := 10 * time.Minute // 10 minutes of inactivity

	for client := range wsm.clients {
		// Check for stale connections based on last activity
		if now.Sub(client.lastActivity) > staleThreshold {
			staleClients = append(staleClients, client)
		}
	}
	wsm.clientsMutex.RUnlock()

	// Clean up stale clients
	for _, client := range staleClients {
		wsm.logger.WithFields(logrus.Fields{
			"client_id":     client.id,
			"user_id":       client.userID,
			"last_activity": client.lastActivity,
			"connected_at":  client.connectedAt,
		}).Info("Cleaning up stale WebSocket connection")
		client.conn.Close()
		wsm.unregister <- client
	}

	// Also cleanup rate limits
	wsm.cleanupRateLimits()
}