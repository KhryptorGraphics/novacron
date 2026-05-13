package optimization

import (
	"net"
	"testing"
	"time"
)

func TestBatchSenderFlushUsesSendmmsg(t *testing.T) {
	receiver, err := net.ListenUDP("udp4", &net.UDPAddr{IP: net.ParseIP("127.0.0.1")})
	if err != nil {
		t.Fatalf("ListenUDP receiver failed: %v", err)
	}
	defer receiver.Close()

	senderConn, err := net.ListenUDP("udp4", nil)
	if err != nil {
		t.Fatalf("ListenUDP sender failed: %v", err)
	}
	defer senderConn.Close()

	sender, err := NewBatchSender(senderConn, 2)
	if err != nil {
		t.Fatalf("NewBatchSender failed: %v", err)
	}

	addr := receiver.LocalAddr().(*net.UDPAddr)
	if err := sender.Send([]byte("one"), addr); err != nil {
		t.Fatalf("first Send failed: %v", err)
	}
	if err := sender.Send([]byte("two"), addr); err != nil {
		t.Fatalf("second Send/Flush failed: %v", err)
	}

	_ = receiver.SetReadDeadline(time.Now().Add(time.Second))
	seen := map[string]bool{}
	for i := 0; i < 2; i++ {
		buf := make([]byte, 16)
		n, _, err := receiver.ReadFromUDP(buf)
		if err != nil {
			t.Fatalf("ReadFromUDP failed: %v", err)
		}
		seen[string(buf[:n])] = true
	}
	if !seen["one"] || !seen["two"] {
		t.Fatalf("missing batched datagrams: %v", seen)
	}
}

func TestBatchSenderRejectsInvalidAddress(t *testing.T) {
	conn, err := net.ListenUDP("udp4", nil)
	if err != nil {
		t.Fatalf("ListenUDP failed: %v", err)
	}
	defer conn.Close()

	sender, err := NewBatchSender(conn, 1)
	if err != nil {
		t.Fatalf("NewBatchSender failed: %v", err)
	}
	if err := sender.Send([]byte("data"), nil); err == nil {
		t.Fatal("expected nil address error")
	}
	if err := sender.Send([]byte("data"), &net.UDPAddr{IP: net.ParseIP("::1"), Port: 1}); err == nil {
		t.Fatal("expected IPv6 address error")
	}
}
