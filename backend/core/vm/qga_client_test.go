package vm

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"net"
	"path/filepath"
	"testing"
)

func TestQGAClientExecuteSendsCommandAndDecodesReturn(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "qga.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen on unix socket: %v", err)
	}
	defer listener.Close()

	requests := make(chan map[string]interface{}, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()

		line, err := bufio.NewReader(conn).ReadBytes('\n')
		if err != nil {
			return
		}

		var request map[string]interface{}
		if err := json.Unmarshal(line, &request); err != nil {
			return
		}
		requests <- request
		_, _ = conn.Write([]byte(`{"return":{"version":"8.2.0"}}` + "\n"))
	}()

	client := NewQGAClient(socketPath)
	var result struct {
		Version string `json:"version"`
	}
	if err := client.Execute(context.Background(), "guest-info", nil, &result); err != nil {
		t.Fatalf("execute guest-info: %v", err)
	}

	if result.Version != "8.2.0" {
		t.Fatalf("expected version 8.2.0, got %s", result.Version)
	}

	request := <-requests
	if request["execute"] != "guest-info" {
		t.Fatalf("expected guest-info command, got %#v", request["execute"])
	}
	if _, ok := request["arguments"]; ok {
		t.Fatalf("nil arguments should be omitted, got %#v", request["arguments"])
	}
}

func TestQGAClientExecuteReturnsGuestAgentError(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "qga.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen on unix socket: %v", err)
	}
	defer listener.Close()

	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		_, _ = bufio.NewReader(conn).ReadBytes('\n')
		_, _ = conn.Write([]byte(`{"error":{"class":"GenericError","desc":"guest denied request"}}` + "\n"))
	}()

	client := NewQGAClient(socketPath)
	err = client.Execute(context.Background(), "guest-file-open", map[string]interface{}{
		"path": "/root/secret",
	}, nil)
	if err == nil {
		t.Fatalf("expected guest agent error")
	}
}

func TestKVMDriverEnhancedAddsGuestAgentChannel(t *testing.T) {
	vmInfo := &KVMVMInfo{
		ID:            "vm-qga",
		Config:        VMConfig{MemoryMB: 512, CPUShares: 2},
		DiskPath:      filepath.Join(t.TempDir(), "disk.qcow2"),
		AgentSockPath: filepath.Join(t.TempDir(), "qga.sock"),
		VNCPort:       5901,
	}
	driver := &KVMDriverEnhanced{}

	args := driver.buildQEMUArgs(vmInfo)
	assertArgPair(t, args, "-chardev", "socket,path="+vmInfo.AgentSockPath+",server=on,wait=off,id=qga0")
	assertArgPair(t, args, "-device", "virtio-serial-pci")
	assertArgPair(t, args, "-device", "virtserialport,chardev=qga0,name=org.qemu.guest_agent.0")
}

func TestQGAClientFileOperations(t *testing.T) {
	socketPath := filepath.Join(t.TempDir(), "qga.sock")
	listener, err := net.Listen("unix", socketPath)
	if err != nil {
		t.Fatalf("listen on unix socket: %v", err)
	}
	defer listener.Close()

	responses := []string{
		`{"return":7}` + "\n",
		`{"return":{"count":5,"buf-b64":"` + base64.StdEncoding.EncodeToString([]byte("hello")) + `","eof":false}}` + "\n",
		`{"return":{"count":5,"eof":false}}` + "\n",
		`{"return":{}}` + "\n",
		`{"return":{}}` + "\n",
	}
	requests := make(chan map[string]interface{}, len(responses))
	go serveQGATestResponses(listener, requests, responses)

	client := NewQGAClient(socketPath)
	handle, err := client.FileOpen(context.Background(), "/tmp/file", "w+")
	if err != nil {
		t.Fatalf("file open: %v", err)
	}
	if handle != 7 {
		t.Fatalf("expected handle 7, got %d", handle)
	}

	readData, eof, err := client.FileRead(context.Background(), handle, 4096)
	if err != nil {
		t.Fatalf("file read: %v", err)
	}
	if string(readData) != "hello" || eof {
		t.Fatalf("unexpected read result data=%q eof=%v", string(readData), eof)
	}

	written, eof, err := client.FileWrite(context.Background(), handle, []byte("world"))
	if err != nil {
		t.Fatalf("file write: %v", err)
	}
	if written != 5 || eof {
		t.Fatalf("unexpected write result count=%d eof=%v", written, eof)
	}

	if err := client.FileFlush(context.Background(), handle); err != nil {
		t.Fatalf("file flush: %v", err)
	}
	if err := client.FileClose(context.Background(), handle); err != nil {
		t.Fatalf("file close: %v", err)
	}

	wantCommands := []string{"guest-file-open", "guest-file-read", "guest-file-write", "guest-file-flush", "guest-file-close"}
	for _, want := range wantCommands {
		request := <-requests
		if request["execute"] != want {
			t.Fatalf("expected command %s, got %#v", want, request["execute"])
		}
	}
}

func assertArgPair(t *testing.T, args []string, flag, value string) {
	t.Helper()
	for i := 0; i < len(args)-1; i++ {
		if args[i] == flag && args[i+1] == value {
			return
		}
	}
	t.Fatalf("expected args to include %s %s, got %#v", flag, value, args)
}

func serveQGATestResponses(listener net.Listener, requests chan<- map[string]interface{}, responses []string) {
	for _, response := range responses {
		conn, err := listener.Accept()
		if err != nil {
			return
		}

		line, err := bufio.NewReader(conn).ReadBytes('\n')
		if err == nil {
			var request map[string]interface{}
			if json.Unmarshal(line, &request) == nil {
				requests <- request
			}
		}
		_, _ = conn.Write([]byte(response))
		_ = conn.Close()
	}
}
