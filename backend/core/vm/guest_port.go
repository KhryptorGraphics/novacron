package vm

import (
	"context"
	"fmt"
	"net"
	"strconv"
	"sync"
	"time"
)

const (
	kvmUserNetID            = "net0"
	kvmHostForwardBind      = "127.0.0.1"
	kvmMonitorCommandTimout = 2 * time.Second
)

type GuestPortDialerProvider interface {
	DialGuestPort(ctx context.Context, vmID string, port int) (net.Conn, error)
}

func (d *KVMDriverEnhanced) DialGuestPort(ctx context.Context, vmID string, port int) (net.Conn, error) {
	if port < 1 || port > 65535 {
		return nil, fmt.Errorf("guest port must be between 1 and 65535")
	}

	d.vmLock.RLock()
	vmInfo, exists := d.vms[vmID]
	d.vmLock.RUnlock()
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}
	if vmInfo.State != StateRunning {
		return nil, fmt.Errorf("VM %s is not running", vmID)
	}
	if vmInfo.MonitorPath == "" {
		return nil, fmt.Errorf("VM %s monitor socket is not configured", vmID)
	}

	localPort, err := reserveLocalTCPPort()
	if err != nil {
		return nil, err
	}

	addCommand := fmt.Sprintf("hostfwd_add %s tcp:%s:%d-:%d", kvmUserNetID, kvmHostForwardBind, localPort, port)
	if err := sendKVMMonitorCommand(ctx, vmInfo.MonitorPath, addCommand); err != nil {
		return nil, fmt.Errorf("add QEMU host forward: %w", err)
	}

	conn, err := dialKVMHostForward(ctx, localPort)
	if err != nil {
		_ = removeKVMHostForward(context.Background(), vmInfo.MonitorPath, localPort)
		return nil, fmt.Errorf("dial QEMU host forward: %w", err)
	}

	return &kvmHostForwardConn{
		Conn:        conn,
		monitorPath: vmInfo.MonitorPath,
		localPort:   localPort,
	}, nil
}

func (m *VMManager) DialGuestPort(ctx context.Context, vmID string, port int) (net.Conn, error) {
	vm, err := m.GetVM(vmID)
	if err != nil {
		return nil, err
	}

	driver, err := m.getDriver(vm.Config())
	if err != nil {
		return nil, fmt.Errorf("failed to get VM driver: %w", err)
	}

	provider, ok := driver.(GuestPortDialerProvider)
	if !ok {
		return nil, fmt.Errorf("VM driver for %s does not expose a guest port dialer", vmID)
	}
	return provider.DialGuestPort(ctx, vmID, port)
}

type kvmHostForwardConn struct {
	net.Conn
	monitorPath string
	localPort   int
	closeOnce   sync.Once
	closeErr    error
}

func (c *kvmHostForwardConn) Close() error {
	c.closeOnce.Do(func() {
		connErr := c.Conn.Close()
		removeErr := removeKVMHostForward(context.Background(), c.monitorPath, c.localPort)
		if connErr != nil {
			c.closeErr = connErr
			return
		}
		c.closeErr = removeErr
	})
	return c.closeErr
}

func reserveLocalTCPPort() (int, error) {
	listener, err := net.Listen("tcp", net.JoinHostPort(kvmHostForwardBind, "0"))
	if err != nil {
		return 0, fmt.Errorf("reserve local TCP port: %w", err)
	}
	defer listener.Close()

	addr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		return 0, fmt.Errorf("reserved listener returned non-TCP address %s", listener.Addr())
	}
	return addr.Port, nil
}

func dialKVMHostForward(ctx context.Context, localPort int) (net.Conn, error) {
	target := net.JoinHostPort(kvmHostForwardBind, strconv.Itoa(localPort))
	var dialer net.Dialer
	var lastErr error
	for attempt := 0; attempt < 20; attempt++ {
		conn, err := dialer.DialContext(ctx, "tcp", target)
		if err == nil {
			return conn, nil
		}
		lastErr = err
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(25 * time.Millisecond):
		}
	}
	return nil, lastErr
}

func removeKVMHostForward(ctx context.Context, monitorPath string, localPort int) error {
	command := fmt.Sprintf("hostfwd_remove %s tcp:%s:%d", kvmUserNetID, kvmHostForwardBind, localPort)
	return sendKVMMonitorCommand(ctx, monitorPath, command)
}

func sendKVMMonitorCommand(ctx context.Context, monitorPath string, command string) error {
	dialer := net.Dialer{Timeout: kvmMonitorCommandTimout}
	conn, err := dialer.DialContext(ctx, "unix", monitorPath)
	if err != nil {
		return err
	}
	defer conn.Close()

	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(kvmMonitorCommandTimout)
	}
	_ = conn.SetDeadline(deadline)

	_, err = fmt.Fprintf(conn, "%s\n", command)
	return err
}
