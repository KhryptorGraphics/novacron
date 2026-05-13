package optimization

import (
	"bytes"
	"compress/gzip"
	"io"
	"testing"
)

func TestCompressionContextCompressesData(t *testing.T) {
	ctx := NewCompressionContext(gzip.BestSpeed)
	data := bytes.Repeat([]byte("dwcp-v3-compression-context"), 128)

	compressed, err := ctx.Compress(data)
	if err != nil {
		t.Fatalf("Compress failed: %v", err)
	}
	reader, err := gzip.NewReader(bytes.NewReader(compressed))
	if err != nil {
		t.Fatalf("NewReader failed: %v", err)
	}
	defer reader.Close()
	decoded, err := io.ReadAll(reader)
	if err != nil {
		t.Fatalf("ReadAll failed: %v", err)
	}
	if !bytes.Equal(decoded, data) {
		t.Fatal("decoded compressed data does not match original")
	}
}

func TestSignatureContextDigest(t *testing.T) {
	ctx := NewSignatureContext()
	first := ctx.Digest([]byte("message"))
	second := ctx.Digest([]byte("message"))
	if !bytes.Equal(first, second) {
		t.Fatal("digest should be deterministic")
	}
	ctx.Reset()
	zero := make([]byte, len(first))
	if bytes.Equal(ctx.lastDigest[:], first) || !bytes.Equal(ctx.lastDigest[:], zero) {
		t.Fatal("Reset did not clear last digest")
	}
}
