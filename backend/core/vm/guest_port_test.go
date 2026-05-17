package vm

import (
	"bufio"
	"context"
	"net"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"
)

func TestKVMDriverEnhancedDialGuestPortUsesQEMUUserNetHostForward(t *testing.T) {
	monitorPath := filepath.Join(t.TempDir(), "monitor.sock")
	monitorListener, err := net.Listen("unix", monitorPath)
	if err != nil {
		t.Fatalf("listen on monitor socket: %v", err)
	}
	defer monitorListener.Close()

	commands := make(chan string, 2)
	tcpReady := make(chan struct{})
	tcpAccepted := make(chan net.Conn, 1)
	go serveKVMHostForwardMonitor(t, monitorListener, commands, tcpReady, tcpAccepted)

	driver := &KVMDriverEnhanced{
		vms: map[string]*KVMVMInfo{
			"vm-1": {
				ID:          "vm-1",
				State:       StateRunning,
				MonitorPath: monitorPath,
			},
		},
	}

	conn, err := driver.DialGuestPort(context.Background(), "vm-1", 8080)
	if err != nil {
		t.Fatalf("dial guest port: %v", err)
	}

	addCommand := <-commands
	if !strings.HasPrefix(addCommand, "hostfwd_add net0 tcp:127.0.0.1:") || !strings.HasSuffix(addCommand, "-:8080") {
		t.Fatalf("unexpected hostfwd_add command %q", addCommand)
	}

	<-tcpReady
	guestConn := <-tcpAccepted
	defer guestConn.Close()

	if _, err := conn.Write([]byte("ping")); err != nil {
		t.Fatalf("write through forwarded conn: %v", err)
	}
	buffer := make([]byte, 4)
	if _, err := guestConn.Read(buffer); err != nil {
		t.Fatalf("read forwarded payload: %v", err)
	}
	if string(buffer) != "ping" {
		t.Fatalf("expected forwarded payload ping, got %q", buffer)
	}

	if err := conn.Close(); err != nil {
		t.Fatalf("close forwarded conn: %v", err)
	}
	removeCommand := <-commands
	if !strings.HasPrefix(removeCommand, "hostfwd_remove net0 tcp:127.0.0.1:") {
		t.Fatalf("unexpected hostfwd_remove command %q", removeCommand)
	}
}

func serveKVMHostForwardMonitor(t *testing.T, listener net.Listener, commands chan<- string, tcpReady chan<- struct{}, tcpAccepted chan<- net.Conn) {
	t.Helper()
	for i := 0; i < 2; i++ {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		command, err := bufio.NewReader(conn).ReadString('\n')
		if err == nil {
			command = strings.TrimSpace(command)
			commands <- command
			if strings.HasPrefix(command, "hostfwd_add ") {
				startForwardedTCPListener(t, command, tcpReady, tcpAccepted)
			}
		}
		_, _ = conn.Write([]byte("(qemu) "))
		_ = conn.Close()
	}
}

func startForwardedTCPListener(t *testing.T, command string, tcpReady chan<- struct{}, tcpAccepted chan<- net.Conn) {
	t.Helper()
	matches := regexp.MustCompile(`127\.0\.0\.1:(\d+)-`).FindStringSubmatch(command)
	if len(matches) != 2 {
		t.Fatalf("could not parse local port from command %q", command)
	}
	port, err := strconv.Atoi(matches[1])
	if err != nil {
		t.Fatalf("parse local port: %v", err)
	}

	listener, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
	if err != nil {
		t.Fatalf("listen on forwarded TCP port: %v", err)
	}
	close(tcpReady)
	go func() {
		defer listener.Close()
		conn, err := listener.Accept()
		if err == nil {
			tcpAccepted <- conn
		}
	}()
}
