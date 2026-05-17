package vm

import (
	"context"
	"fmt"
)

type GuestAgentSocketProvider interface {
	GuestAgentSocketPath(ctx context.Context, vmID string) (string, error)
}

func (d *KVMDriverEnhanced) GuestAgentSocketPath(ctx context.Context, vmID string) (string, error) {
	d.vmLock.RLock()
	defer d.vmLock.RUnlock()

	vmInfo, exists := d.vms[vmID]
	if !exists {
		return "", fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.AgentSockPath == "" {
		return "", fmt.Errorf("VM %s guest agent socket is not configured", vmID)
	}
	return vmInfo.AgentSockPath, nil
}

func (m *VMManager) GuestAgentSocketPath(ctx context.Context, vmID string) (string, error) {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return "", err
	}

	driver, err := m.getDriver(vm.Config())
	if err != nil {
		return "", fmt.Errorf("failed to get VM driver: %w", err)
	}

	provider, ok := driver.(GuestAgentSocketProvider)
	if !ok {
		return "", fmt.Errorf("VM driver for %s does not expose a guest agent socket", vmID)
	}
	return provider.GuestAgentSocketPath(ctx, vmID)
}
