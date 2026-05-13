package optimization

import (
	"io"
	"net"
	"os"
	"testing"
	"time"
)

func TestSendFileToTCP(t *testing.T) {
	file, err := os.CreateTemp(t.TempDir(), "sendfile-*")
	if err != nil {
		t.Fatalf("CreateTemp failed: %v", err)
	}
	if _, err := file.WriteString("abcdefghij"); err != nil {
		t.Fatalf("write temp file failed: %v", err)
	}
	if _, err := file.Seek(0, io.SeekStart); err != nil {
		t.Fatalf("seek temp file failed: %v", err)
	}
	defer file.Close()

	listener, err := net.Listen("tcp4", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("Listen failed: %v", err)
	}
	defer listener.Close()

	received := make(chan []byte, 1)
	errs := make(chan error, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			errs <- err
			return
		}
		defer conn.Close()
		buf := make([]byte, 5)
		_, err = io.ReadFull(conn, buf)
		if err != nil {
			errs <- err
			return
		}
		received <- buf
	}()

	client, err := net.DialTimeout("tcp4", listener.Addr().String(), time.Second)
	if err != nil {
		t.Fatalf("DialTimeout failed: %v", err)
	}
	defer client.Close()

	n, err := sendFileToTCP(client.(*net.TCPConn), file, 2, 5)
	if err != nil {
		t.Fatalf("sendFileToTCP failed: %v", err)
	}
	if n != 5 {
		t.Fatalf("sendFileToTCP wrote %d bytes, want 5", n)
	}

	select {
	case err := <-errs:
		t.Fatalf("server read failed: %v", err)
	case data := <-received:
		if string(data) != "cdefg" {
			t.Fatalf("received %q, want cdefg", data)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for sendfile data")
	}
}

func TestSendFileToTCPRejectsNegativeCount(t *testing.T) {
	if _, err := sendFileToTCP(nil, nil, 0, -1); err == nil {
		t.Fatal("expected negative count error")
	}
}
