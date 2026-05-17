package websocket

import (
	"context"
	"path/filepath"
	"testing"

	corevm "github.com/khryptorgraphics/novacron/backend/core/vm"
)

func TestQGAGuestFileClientResolverCreatesClientFromSocketPath(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "qga.sock")
	resolver := NewQGAGuestFileClientResolver(staticGuestAgentSocketResolver{socketPath: socketPath})

	client, err := resolver.ResolveGuestFileClient(context.Background(), "vm-1")
	if err != nil {
		t.Fatalf("resolve guest file client: %v", err)
	}

	qgaClient, ok := client.(*corevm.QGAClient)
	if !ok {
		t.Fatalf("expected *vm.QGAClient, got %T", client)
	}
	if qgaClient == nil {
		t.Fatalf("expected non-nil qga client")
	}
}

type staticGuestAgentSocketResolver struct {
	socketPath string
}

func (r staticGuestAgentSocketResolver) GuestAgentSocketPath(context.Context, string) (string, error) {
	return r.socketPath, nil
}
