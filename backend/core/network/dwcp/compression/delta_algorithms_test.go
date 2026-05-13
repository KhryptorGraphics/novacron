package compression

import (
	"bytes"
	"testing"
)

func TestBSDiffDeltaComputerRoundTripWithCopyAndLiteral(t *testing.T) {
	baseline := bytes.Repeat([]byte("0123456789abcdef"), 8)
	current := append([]byte("prefix-"), baseline[16:96]...)
	current = append(current, []byte("-suffix")...)

	computer := &BSDiffDeltaComputer{}
	delta, err := computer.ComputeDelta(baseline, current)
	if err != nil {
		t.Fatalf("ComputeDelta failed: %v", err)
	}
	if len(delta) >= len(current) {
		t.Fatalf("delta should be smaller than current for reused binary regions: delta=%d current=%d", len(delta), len(current))
	}

	reconstructed, err := computer.ApplyDelta(baseline, delta)
	if err != nil {
		t.Fatalf("ApplyDelta failed: %v", err)
	}
	if !bytes.Equal(reconstructed, current) {
		t.Fatal("reconstructed data does not match current data")
	}
}

func TestBSDiffDeltaComputerRejectsMalformedDelta(t *testing.T) {
	computer := &BSDiffDeltaComputer{}
	if _, err := computer.ApplyDelta([]byte("baseline"), []byte("not-a-delta")); err == nil {
		t.Fatal("expected malformed delta error")
	}
	if _, err := computer.ApplyDelta([]byte("baseline"), []byte(bsdiffMagic+"\x01\x00")); err == nil {
		t.Fatal("expected truncated copy command error")
	}
}
