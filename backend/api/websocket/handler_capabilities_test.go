package websocket

import (
	"context"
	"testing"

	"github.com/gorilla/websocket"
)

type capabilityCopyService struct{}

func (capabilityCopyService) HandleVMCopy(context.Context, string, VMCopyOptions, *websocket.Conn) error {
	return nil
}

type capabilityPortForwardService struct{}

func (capabilityPortForwardService) HandleVMPortForward(context.Context, string, VMPortForwardOptions, *websocket.Conn) error {
	return nil
}

func TestWebSocketHandlerReportsVMIOCapabilities(t *testing.T) {
	emptyHandler := NewWebSocketHandler(nil, nil)
	defer emptyHandler.Shutdown()

	if emptyHandler.SupportsVMCopy() {
		t.Fatal("expected handler without copy service to report VM copy unsupported")
	}
	if emptyHandler.SupportsVMPortForward() {
		t.Fatal("expected handler without port-forward service to report VM port-forward unsupported")
	}

	fullHandler := NewWebSocketHandler(nil, nil, capabilityCopyService{}, capabilityPortForwardService{})
	defer fullHandler.Shutdown()

	if !fullHandler.SupportsVMCopy() {
		t.Fatal("expected handler with copy service to report VM copy supported")
	}
	if !fullHandler.SupportsVMPortForward() {
		t.Fatal("expected handler with port-forward service to report VM port-forward supported")
	}
}
