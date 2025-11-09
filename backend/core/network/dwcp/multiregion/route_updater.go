package multiregion

import (
	"fmt"
	"log"
	"sync"
	"time"
)

// RouteUpdater handles dynamic route updates
type RouteUpdater struct {
	routingTable *RoutingTable
	topology     *GlobalTopology
	engine       *RoutingEngine
	subscribers  []RouteUpdateSubscriber
	mu           sync.RWMutex
}

// RouteUpdateSubscriber receives route update notifications
type RouteUpdateSubscriber interface {
	OnRouteUpdate(update *RouteUpdate)
}

// RouteUpdate represents a routing table update
type RouteUpdate struct {
	Type        UpdateType
	Destination string
	OldRoute    *Route
	NewRoute    *Route
	Reason      string
	Timestamp   time.Time
}

// UpdateType defines the type of route update
type UpdateType int

const (
	UpdateTypeAdd UpdateType = iota
	UpdateTypeModify
	UpdateTypeDelete
	UpdateTypeFailover
)

// RouteUpdateEvent represents an event that triggers route updates
type RouteUpdateEvent struct {
	EventType   EventType
	LinkID      LinkID
	RegionID    string
	Timestamp   time.Time
	Description string
}

// EventType defines types of network events
type EventType int

const (
	EventLinkFailure EventType = iota
	EventLinkRecovery
	EventLinkDegraded
	EventRegionDown
	EventRegionUp
)

// NewRouteUpdater creates a new route updater
func NewRouteUpdater(topology *GlobalTopology, routingTable *RoutingTable) *RouteUpdater {
	return &RouteUpdater{
		routingTable: routingTable,
		topology:     topology,
		engine:       NewRoutingEngine(topology, StrategyBalanced),
		subscribers:  make([]RouteUpdateSubscriber, 0),
	}
}

// Subscribe adds a subscriber to receive route updates
func (ru *RouteUpdater) Subscribe(subscriber RouteUpdateSubscriber) {
	ru.mu.Lock()
	defer ru.mu.Unlock()
	ru.subscribers = append(ru.subscribers, subscriber)
}

// UpdateOnLinkFailure handles link failure events
func (ru *RouteUpdater) UpdateOnLinkFailure(linkID LinkID) error {
	log.Printf("Route updater: handling link failure for %s", linkID)

	// Mark link as down
	if err := ru.topology.UpdateLinkHealth(linkID, HealthDown); err != nil {
		return fmt.Errorf("failed to update link health: %w", err)
	}

	// Find all routes affected by this link
	affectedRoutes := ru.findAffectedRoutes(linkID)
	log.Printf("Route updater: found %d affected routes", len(affectedRoutes))

	// Recompute routes
	for _, route := range affectedRoutes {
		if err := ru.recomputeRoute(route, linkID, "link failure"); err != nil {
			log.Printf("Route updater: failed to recompute route to %s: %v", route.Destination, err)
			continue
		}
	}

	// Notify subscribers
	event := &RouteUpdateEvent{
		EventType:   EventLinkFailure,
		LinkID:      linkID,
		Timestamp:   time.Now(),
		Description: fmt.Sprintf("Link %s failed", linkID),
	}
	ru.notifySubscribers(event)

	return nil
}

// UpdateOnLinkRecovery handles link recovery events
func (ru *RouteUpdater) UpdateOnLinkRecovery(linkID LinkID) error {
	log.Printf("Route updater: handling link recovery for %s", linkID)

	// Mark link as up
	if err := ru.topology.UpdateLinkHealth(linkID, HealthUp); err != nil {
		return fmt.Errorf("failed to update link health: %w", err)
	}

	// Re-optimize routes to potentially use the recovered link
	if err := ru.optimizeAllRoutes(); err != nil {
		log.Printf("Route updater: failed to optimize routes: %v", err)
	}

	// Notify subscribers
	event := &RouteUpdateEvent{
		EventType:   EventLinkRecovery,
		LinkID:      linkID,
		Timestamp:   time.Now(),
		Description: fmt.Sprintf("Link %s recovered", linkID),
	}
	ru.notifySubscribers(event)

	return nil
}

// UpdateOnRegionFailure handles region failure events
func (ru *RouteUpdater) UpdateOnRegionFailure(regionID string) error {
	log.Printf("Route updater: handling region failure for %s", regionID)

	// Mark all links from/to this region as down
	for _, link := range ru.topology.ListLinks() {
		if link.Source == regionID || link.Destination == regionID {
			ru.topology.UpdateLinkHealth(link.ID, HealthDown)
		}
	}

	// Find all routes involving this region
	allRoutes := ru.routingTable.List()
	affectedRoutes := make([]*Route, 0)

	for _, route := range allRoutes {
		for _, hop := range route.Path {
			if hop == regionID {
				affectedRoutes = append(affectedRoutes, route)
				break
			}
		}
	}

	// Recompute affected routes
	for _, route := range affectedRoutes {
		if err := ru.recomputeRoute(route, "", "region failure"); err != nil {
			log.Printf("Route updater: failed to recompute route to %s: %v", route.Destination, err)
			continue
		}
	}

	// Notify subscribers
	event := &RouteUpdateEvent{
		EventType:   EventRegionDown,
		RegionID:    regionID,
		Timestamp:   time.Now(),
		Description: fmt.Sprintf("Region %s failed", regionID),
	}
	ru.notifySubscribers(event)

	return nil
}

// findAffectedRoutes finds all routes using a specific link
func (ru *RouteUpdater) findAffectedRoutes(linkID LinkID) []*Route {
	allRoutes := ru.routingTable.List()
	affected := make([]*Route, 0)

	for _, route := range allRoutes {
		for _, routeLinkID := range route.Links {
			if routeLinkID == linkID {
				affected = append(affected, route)
				break
			}
		}
	}

	return affected
}

// recomputeRoute recomputes a route avoiding a failed link
func (ru *RouteUpdater) recomputeRoute(oldRoute *Route, failedLink LinkID, reason string) error {
	// Extract source from route path
	if len(oldRoute.Path) < 2 {
		return fmt.Errorf("invalid route path")
	}
	source := oldRoute.Path[0]
	destination := oldRoute.Destination

	// Compute new route
	newRoute, err := ru.engine.ComputeRoute(source, destination)
	if err != nil {
		// No alternative path available
		log.Printf("Route updater: no alternative path from %s to %s", source, destination)

		// Delete the route
		ru.routingTable.Delete(source, destination)

		// Notify about deletion
		update := &RouteUpdate{
			Type:        UpdateTypeDelete,
			Destination: destination,
			OldRoute:    oldRoute,
			NewRoute:    nil,
			Reason:      reason,
			Timestamp:   time.Now(),
		}
		ru.broadcastUpdate(update)

		return err
	}

	// Update routing table atomically
	ru.routingTable.Update(source, destination, newRoute)

	// Invalidate cache in routing engine
	ru.engine.InvalidateCache(source, destination)

	// Broadcast update
	update := &RouteUpdate{
		Type:        UpdateTypeFailover,
		Destination: destination,
		OldRoute:    oldRoute,
		NewRoute:    newRoute,
		Reason:      reason,
		Timestamp:   time.Now(),
	}
	ru.broadcastUpdate(update)

	log.Printf("Route updater: updated route to %s (hops: %d -> %d)",
		destination, len(oldRoute.Path), len(newRoute.Path))

	return nil
}

// optimizeAllRoutes re-optimizes all routes
func (ru *RouteUpdater) optimizeAllRoutes() error {
	routes := ru.routingTable.List()

	for _, route := range routes {
		if len(route.Path) < 2 {
			continue
		}

		source := route.Path[0]
		destination := route.Destination

		// Compute potentially better route
		newRoute, err := ru.engine.ComputeRoute(source, destination)
		if err != nil {
			continue
		}

		// Check if new route is significantly better (>10% improvement)
		if ru.isSignificantlyBetter(newRoute, route) {
			ru.routingTable.Update(source, destination, newRoute)

			update := &RouteUpdate{
				Type:        UpdateTypeModify,
				Destination: destination,
				OldRoute:    route,
				NewRoute:    newRoute,
				Reason:      "optimization",
				Timestamp:   time.Now(),
			}
			ru.broadcastUpdate(update)
		}
	}

	return nil
}

// isSignificantlyBetter checks if a new route is significantly better than the old one
func (ru *RouteUpdater) isSignificantlyBetter(newRoute, oldRoute *Route) bool {
	// Compare latency
	latencyImprovement := float64(oldRoute.Metric.Latency-newRoute.Metric.Latency) / float64(oldRoute.Metric.Latency)
	if latencyImprovement > 0.1 { // 10% improvement
		return true
	}

	// Compare cost
	costImprovement := (oldRoute.Metric.Cost - newRoute.Metric.Cost) / oldRoute.Metric.Cost
	if costImprovement > 0.1 {
		return true
	}

	// Compare bandwidth
	bandwidthImprovement := float64(newRoute.Metric.Bandwidth-oldRoute.Metric.Bandwidth) / float64(oldRoute.Metric.Bandwidth)
	if bandwidthImprovement > 0.1 {
		return true
	}

	return false
}

// broadcastUpdate broadcasts a route update to all subscribers
func (ru *RouteUpdater) broadcastUpdate(update *RouteUpdate) {
	ru.mu.RLock()
	subscribers := make([]RouteUpdateSubscriber, len(ru.subscribers))
	copy(subscribers, ru.subscribers)
	ru.mu.RUnlock()

	for _, subscriber := range subscribers {
		go subscriber.OnRouteUpdate(update)
	}
}

// notifySubscribers notifies subscribers about network events
func (ru *RouteUpdater) notifySubscribers(event *RouteUpdateEvent) {
	log.Printf("Route updater: event %s at %v", event.Description, event.Timestamp)
}

// PeriodicOptimization runs periodic route optimization
func (ru *RouteUpdater) PeriodicOptimization(interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for range ticker.C {
		if err := ru.optimizeAllRoutes(); err != nil {
			log.Printf("Route updater: periodic optimization failed: %v", err)
		}
	}
}

// GetUpdateHistory returns recent route updates (would need to be tracked)
func (ru *RouteUpdater) GetUpdateHistory(limit int) []*RouteUpdate {
	// In production, maintain a circular buffer of updates
	return make([]*RouteUpdate, 0)
}

// String methods

func (ut UpdateType) String() string {
	switch ut {
	case UpdateTypeAdd:
		return "ADD"
	case UpdateTypeModify:
		return "MODIFY"
	case UpdateTypeDelete:
		return "DELETE"
	case UpdateTypeFailover:
		return "FAILOVER"
	default:
		return "UNKNOWN"
	}
}

func (et EventType) String() string {
	switch et {
	case EventLinkFailure:
		return "LINK_FAILURE"
	case EventLinkRecovery:
		return "LINK_RECOVERY"
	case EventLinkDegraded:
		return "LINK_DEGRADED"
	case EventRegionDown:
		return "REGION_DOWN"
	case EventRegionUp:
		return "REGION_UP"
	default:
		return "UNKNOWN"
	}
}
