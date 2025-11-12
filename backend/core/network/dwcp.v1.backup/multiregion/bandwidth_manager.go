package multiregion

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// BandwidthManager manages bandwidth reservations
type BandwidthManager struct {
	reservations map[LinkID]*Reservation
	topology     *GlobalTopology
	mu           sync.RWMutex
}

// Reservation represents a bandwidth reservation
type Reservation struct {
	ID        string
	FlowID    string
	LinkID    LinkID
	Bandwidth int64 // Mbps
	Priority  int
	StartTime time.Time
	Duration  time.Duration
	ExpiresAt time.Time
	Active    bool
}

// ReservationRequest represents a request for bandwidth reservation
type ReservationRequest struct {
	FlowID    string
	Path      *Route
	Bandwidth int64
	Priority  int
	Duration  time.Duration
}

// ReservationStats tracks reservation statistics
type ReservationStats struct {
	TotalReservations    int
	ActiveReservations   int
	TotalBandwidthMbps   int64
	ReservedBandwidthMbps int64
	AvailableBandwidthMbps int64
}

var (
	ErrInsufficientBandwidth = errors.New("insufficient bandwidth available")
	ErrReservationNotFound   = errors.New("reservation not found")
	ErrReservationExpired    = errors.New("reservation expired")
)

// NewBandwidthManager creates a new bandwidth manager
func NewBandwidthManager(topology *GlobalTopology) *BandwidthManager {
	bm := &BandwidthManager{
		reservations: make(map[LinkID]*Reservation),
		topology:     topology,
	}

	// Start cleanup goroutine for expired reservations
	go bm.cleanupExpiredReservations()

	return bm
}

// ReserveBandwidth reserves bandwidth on a path
func (bm *BandwidthManager) ReserveBandwidth(req *ReservationRequest) (string, error) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	// Check availability on all links in path
	for _, linkID := range req.Path.Links {
		link, err := bm.topology.GetLink(linkID)
		if err != nil {
			return "", fmt.Errorf("failed to get link %s: %w", linkID, err)
		}

		available := bm.getAvailableBandwidth(link)
		if available < req.Bandwidth {
			return "", fmt.Errorf("%w: link %s has %d Mbps available, need %d Mbps",
				ErrInsufficientBandwidth, linkID, available, req.Bandwidth)
		}
	}

	// Create reservations for all links
	reservationID := fmt.Sprintf("rsv-%s-%d", req.FlowID, time.Now().UnixNano())

	for _, linkID := range req.Path.Links {
		reservation := &Reservation{
			ID:        reservationID,
			FlowID:    req.FlowID,
			LinkID:    linkID,
			Bandwidth: req.Bandwidth,
			Priority:  req.Priority,
			StartTime: time.Now(),
			Duration:  req.Duration,
			ExpiresAt: time.Now().Add(req.Duration),
			Active:    true,
		}

		// Store reservation (keyed by link to allow multiple reservations per link)
		key := LinkID(fmt.Sprintf("%s-%s", linkID, reservationID))
		bm.reservations[key] = reservation
	}

	return reservationID, nil
}

// ReleaseBandwidth releases a bandwidth reservation
func (bm *BandwidthManager) ReleaseBandwidth(reservationID string) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	found := false
	for key, reservation := range bm.reservations {
		if reservation.ID == reservationID {
			reservation.Active = false
			delete(bm.reservations, key)
			found = true
		}
	}

	if !found {
		return ErrReservationNotFound
	}

	return nil
}

// GetReservation retrieves a reservation by ID
func (bm *BandwidthManager) GetReservation(reservationID string) (*Reservation, error) {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	for _, reservation := range bm.reservations {
		if reservation.ID == reservationID {
			if time.Now().After(reservation.ExpiresAt) {
				return nil, ErrReservationExpired
			}
			return reservation, nil
		}
	}

	return nil, ErrReservationNotFound
}

// ListReservations returns all active reservations
func (bm *BandwidthManager) ListReservations() []*Reservation {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	reservations := make([]*Reservation, 0, len(bm.reservations))
	for _, reservation := range bm.reservations {
		if reservation.Active && time.Now().Before(reservation.ExpiresAt) {
			reservations = append(reservations, reservation)
		}
	}

	return reservations
}

// GetLinkReservations returns all reservations for a specific link
func (bm *BandwidthManager) GetLinkReservations(linkID LinkID) []*Reservation {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	reservations := make([]*Reservation, 0)
	for _, reservation := range bm.reservations {
		if reservation.LinkID == linkID && reservation.Active && time.Now().Before(reservation.ExpiresAt) {
			reservations = append(reservations, reservation)
		}
	}

	return reservations
}

// getReservedBandwidth calculates total reserved bandwidth on a link
func (bm *BandwidthManager) getReservedBandwidth(linkID LinkID) int64 {
	var total int64

	for _, reservation := range bm.reservations {
		if reservation.LinkID == linkID && reservation.Active && time.Now().Before(reservation.ExpiresAt) {
			total += reservation.Bandwidth
		}
	}

	return total
}

// getAvailableBandwidth calculates available bandwidth on a link
func (bm *BandwidthManager) getAvailableBandwidth(link *InterRegionLink) int64 {
	reserved := bm.getReservedBandwidth(link.ID)

	// Account for current utilization
	link.mu.RLock()
	utilized := link.Bandwidth * int64(link.Utilization) / 100
	link.mu.RUnlock()

	available := link.Bandwidth - reserved - utilized
	if available < 0 {
		available = 0
	}

	return available
}

// GetStats returns bandwidth reservation statistics
func (bm *BandwidthManager) GetStats() *ReservationStats {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	stats := &ReservationStats{}

	// Count active reservations
	activeReservations := make(map[string]bool)
	for _, reservation := range bm.reservations {
		if reservation.Active && time.Now().Before(reservation.ExpiresAt) {
			activeReservations[reservation.ID] = true
			stats.ReservedBandwidthMbps += reservation.Bandwidth
		}
	}
	stats.ActiveReservations = len(activeReservations)

	// Calculate total bandwidth across all links
	for _, link := range bm.topology.ListLinks() {
		stats.TotalBandwidthMbps += link.Bandwidth
	}

	stats.AvailableBandwidthMbps = stats.TotalBandwidthMbps - stats.ReservedBandwidthMbps

	return stats
}

// ExtendReservation extends the duration of an existing reservation
func (bm *BandwidthManager) ExtendReservation(reservationID string, additionalDuration time.Duration) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	found := false
	for _, reservation := range bm.reservations {
		if reservation.ID == reservationID {
			if !reservation.Active {
				return errors.New("reservation is not active")
			}
			if time.Now().After(reservation.ExpiresAt) {
				return ErrReservationExpired
			}

			reservation.Duration += additionalDuration
			reservation.ExpiresAt = reservation.ExpiresAt.Add(additionalDuration)
			found = true
		}
	}

	if !found {
		return ErrReservationNotFound
	}

	return nil
}

// UpdateReservationPriority updates the priority of a reservation
func (bm *BandwidthManager) UpdateReservationPriority(reservationID string, priority int) error {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	found := false
	for _, reservation := range bm.reservations {
		if reservation.ID == reservationID {
			reservation.Priority = priority
			found = true
		}
	}

	if !found {
		return ErrReservationNotFound
	}

	return nil
}

// PreemptLowPriorityReservations attempts to free bandwidth by preempting low priority reservations
func (bm *BandwidthManager) PreemptLowPriorityReservations(linkID LinkID, requiredBandwidth int64, minPriority int) (int64, error) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	// Find reservations on this link with lower priority
	candidates := make([]*Reservation, 0)
	for _, reservation := range bm.reservations {
		if reservation.LinkID == linkID &&
		   reservation.Priority < minPriority &&
		   reservation.Active &&
		   time.Now().Before(reservation.ExpiresAt) {
			candidates = append(candidates, reservation)
		}
	}

	// Sort by priority (lowest first)
	// In production, use proper sorting

	var freed int64
	for _, reservation := range candidates {
		if freed >= requiredBandwidth {
			break
		}

		// Preempt this reservation
		reservation.Active = false
		freed += reservation.Bandwidth
	}

	return freed, nil
}

// cleanupExpiredReservations periodically removes expired reservations
func (bm *BandwidthManager) cleanupExpiredReservations() {
	ticker := time.NewTicker(1 * time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		bm.mu.Lock()

		for key, reservation := range bm.reservations {
			if time.Now().After(reservation.ExpiresAt) {
				delete(bm.reservations, key)
			}
		}

		bm.mu.Unlock()
	}
}

// GetLinkUtilization returns current and projected utilization for a link
func (bm *BandwidthManager) GetLinkUtilization(linkID LinkID) (current float64, projected float64, err error) {
	link, err := bm.topology.GetLink(linkID)
	if err != nil {
		return 0, 0, err
	}

	bm.mu.RLock()
	reserved := bm.getReservedBandwidth(linkID)
	bm.mu.RUnlock()

	link.mu.RLock()
	current = link.Utilization
	link.mu.RUnlock()

	// Projected utilization includes reservations
	projected = (float64(reserved) / float64(link.Bandwidth)) * 100.0

	return current, projected, nil
}

// CanAccommodate checks if a path can accommodate a bandwidth request
func (bm *BandwidthManager) CanAccommodate(path *Route, bandwidth int64) bool {
	bm.mu.RLock()
	defer bm.mu.RUnlock()

	for _, linkID := range path.Links {
		link, err := bm.topology.GetLink(linkID)
		if err != nil {
			return false
		}

		available := bm.getAvailableBandwidth(link)
		if available < bandwidth {
			return false
		}
	}

	return true
}
