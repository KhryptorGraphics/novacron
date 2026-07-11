package dwcp

import (
	"bytes"
	"context"
	"crypto/rand"
	"net"
	"testing"
	"time"
)

// TestAMSTTransferRate_MeasuredFromRealTransfer proves transferRate is a real
// measurement, not the permanent-zero self-referential no-op it was before
// this fix (novacron-77u).
//
// Before the fix, transferRate was written exactly once per AMST instance -
// by Connect() reading amst.transferRate.Load() and immediately storing that
// same value back via UpdateMetrics (see the call site there) - so it could
// never become anything other than its zero value. Transfer()/Receive()
// tracked bytesTransferred in real time but never converted it into a rate
// or fed it back into UpdateMetrics. Net effect: optimize()'s
// `currentRate := amst.transferRate.Load(); if currentRate == 0 { return }`
// guard fired on every single tick, for every AMST instance with
// EnableAdaptive=true, regardless of caller - confirmed independently by
// v3/tests/benchmark_amst_bandwidth_test.go, which already had to work
// around this exact gap with a hardcoded fake UpdateMetrics call to make its
// adaptive-vs-static comparison mean anything.
//
// This test drives one real AMST->AMST transfer over a loopback TCP
// connection - real Connect() dial, real Transfer() chunking/writes, real
// Receive() reads - and asserts both sides' transferRate becomes nonzero and
// plausible for the payload size and elapsed wall-clock time, without
// pinning an exact value (real timing is inherently noisy).
func TestAMSTTransferRate_MeasuredFromRealTransfer(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen: %v", err)
	}
	defer ln.Close()
	addr := ln.Addr().(*net.TCPAddr)

	receiver, err := NewAMST(AMSTConfig{MinStreams: 1, MaxStreams: 1, InitialStreams: 1})
	if err != nil {
		t.Fatalf("NewAMST(receiver) failed: %v", err)
	}
	defer receiver.Close()

	// AMST.Connect only dials; there is no exported server-side accept path,
	// so the receiver's single stream is wired up manually here exactly the
	// way Connect() wires up a dialed stream (id/conn/active/lastActive set,
	// appended to amst.streams, activeStreams incremented).
	acceptErr := make(chan error, 1)
	go func() {
		conn, err := ln.Accept()
		if err != nil {
			acceptErr <- err
			return
		}
		receiver.mu.Lock()
		receiver.streams = append(receiver.streams, &Stream{
			id:         "receiver-stream-0",
			conn:       conn,
			amst:       receiver,
			active:     true,
			lastActive: time.Now(),
		})
		receiver.mu.Unlock()
		receiver.activeStreams.Add(1)
		acceptErr <- nil
	}()

	sender, err := NewAMST(AMSTConfig{MinStreams: 1, MaxStreams: 1, InitialStreams: 1})
	if err != nil {
		t.Fatalf("NewAMST(sender) failed: %v", err)
	}
	defer sender.Close()

	if err := sender.Connect(context.Background(), "127.0.0.1", addr.Port); err != nil {
		t.Fatalf("sender.Connect failed: %v", err)
	}
	if err := <-acceptErr; err != nil {
		t.Fatalf("listener accept failed: %v", err)
	}

	// Before Transfer/Receive run, transferRate must still read exactly the
	// pre-fix permanent value (0) - Connect() only ever touches latency for
	// real; this pins down that the nonzero assertion below is actually
	// caused by Transfer/Receive, not some other path.
	if rate := sender.transferRate.Load(); rate != 0 {
		t.Fatalf("sender.transferRate = %d before any Transfer(); want 0", rate)
	}

	payload := make([]byte, 2*1024*1024) // 2MB: several chunks at the default 64KB chunk size
	if _, err := rand.Read(payload); err != nil {
		t.Fatalf("rand.Read failed: %v", err)
	}

	type recvResult struct {
		data []byte
		err  error
	}
	recvCh := make(chan recvResult, 1)
	go func() {
		data, err := receiver.Receive(context.Background(), nil)
		recvCh <- recvResult{data: data, err: err}
	}()

	if err := sender.Transfer(context.Background(), payload, nil); err != nil {
		t.Fatalf("sender.Transfer failed: %v", err)
	}

	var recv recvResult
	select {
	case recv = <-recvCh:
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for receiver.Receive to complete")
	}
	if recv.err != nil {
		t.Fatalf("receiver.Receive failed: %v", recv.err)
	}
	if !bytes.Equal(recv.data, payload) {
		t.Fatalf("received data does not match sent payload (got %d bytes, want %d)", len(recv.data), len(payload))
	}

	// The core assertion: both the send side (Transfer) and receive side
	// (Receive) must have measured and stored a real, positive rate. Loopback
	// throughput varies a lot by machine, so this only checks the value is
	// positive and within a generous sanity ceiling (100 GB/s) that would
	// only be crossed by a broken computation (e.g. a unit error), not by
	// real variance.
	const saneCeiling = 100 * 1024 * 1024 * 1024 // 100 GB/s
	senderRate := sender.transferRate.Load()
	receiverRate := receiver.transferRate.Load()

	if senderRate <= 0 {
		t.Errorf("sender.transferRate = %d after Transfer(); want > 0", senderRate)
	}
	if senderRate > saneCeiling {
		t.Errorf("sender.transferRate = %d after Transfer(); want <= %d (sane ceiling)", senderRate, saneCeiling)
	}
	if receiverRate <= 0 {
		t.Errorf("receiver.transferRate = %d after Receive(); want > 0", receiverRate)
	}
	if receiverRate > saneCeiling {
		t.Errorf("receiver.transferRate = %d after Receive(); want <= %d (sane ceiling)", receiverRate, saneCeiling)
	}

	// GetMetrics is the exported surface most callers actually read; prove
	// the fix is visible there too, not just on the internal atomic.
	if got := sender.GetMetrics()["transfer_rate"].(int64); got != senderRate {
		t.Errorf("GetMetrics()[\"transfer_rate\"] = %d, want %d (transferRate)", got, senderRate)
	}
}
