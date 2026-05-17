package websocket

import (
	"context"
	"fmt"

	corevm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

type guestAgentSocketResolver interface {
	GuestAgentSocketPath(ctx context.Context, vmID string) (string, error)
}

type QGAGuestFileClientResolver struct {
	socketResolver guestAgentSocketResolver
}

func NewQGAGuestFileClientResolver(socketResolver guestAgentSocketResolver) *QGAGuestFileClientResolver {
	return &QGAGuestFileClientResolver{socketResolver: socketResolver}
}

func (r *QGAGuestFileClientResolver) ResolveGuestFileClient(ctx context.Context, vmID string) (VMGuestFileClient, error) {
	if r == nil || r.socketResolver == nil {
		return nil, fmt.Errorf("guest agent socket resolver is required")
	}

	socketPath, err := r.socketResolver.GuestAgentSocketPath(ctx, vmID)
	if err != nil {
		return nil, err
	}
	return corevm.NewQGAClient(socketPath), nil
}
