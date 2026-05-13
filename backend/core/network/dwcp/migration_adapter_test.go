package dwcp

import (
	"encoding/binary"
	"net"
	"testing"
	"time"
)

func TestMigrationAdapterReceiveMemoryStoresBaseline(t *testing.T) {
	adapter, err := NewMigrationAdapter(MigrationAdapterConfig{
		EnableDWCP:      false,
		MetricsInterval: time.Hour,
		MaxMemoryUsage:  1024 * 1024,
	})
	if err != nil {
		t.Fatalf("NewMigrationAdapter() error = %v", err)
	}
	defer adapter.Close()

	server, client := net.Pipe()
	done := make(chan struct{})
	go func() {
		defer close(done)
		adapter.handleIncomingMigration(server)
	}()

	memory := []byte("guest-memory-page")
	if err := writeMigrationHeader(client, migrationTypeMemory, "vm-123"); err != nil {
		t.Fatalf("writeMigrationHeader() error = %v", err)
	}
	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(len(memory)))
	if err := writeAll(client, header); err != nil {
		t.Fatalf("write memory size error = %v", err)
	}
	if err := writeAll(client, memory); err != nil {
		t.Fatalf("write memory data error = %v", err)
	}
	_ = client.Close()
	<-done

	baseline := adapter.vmBaselines["vm-123"]
	if baseline == nil {
		t.Fatal("memory baseline was not stored")
	}
	if string(baseline.MemoryBaseline) != string(memory) {
		t.Fatalf("memory baseline = %q, want %q", baseline.MemoryBaseline, memory)
	}
	if got := adapter.migrationsCompleted.Load(); got != 1 {
		t.Fatalf("migrationsCompleted = %d, want 1", got)
	}
}

func TestMigrationAdapterReceiveDiskStoresBaselines(t *testing.T) {
	adapter, err := NewMigrationAdapter(MigrationAdapterConfig{
		EnableDWCP:      false,
		MetricsInterval: time.Hour,
		MaxMemoryUsage:  1024 * 1024,
	})
	if err != nil {
		t.Fatalf("NewMigrationAdapter() error = %v", err)
	}
	defer adapter.Close()

	server, client := net.Pipe()
	done := make(chan struct{})
	go func() {
		defer close(done)
		adapter.handleIncomingMigration(server)
	}()

	blocks := map[int][]byte{
		7: []byte("disk-block-7"),
		9: []byte("disk-block-9"),
	}
	totalSize := 0
	for _, block := range blocks {
		totalSize += len(block)
	}

	if err := writeMigrationHeader(client, migrationTypeDisk, "vm-disk"); err != nil {
		t.Fatalf("writeMigrationHeader() error = %v", err)
	}
	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, uint64(totalSize))
	if err := writeAll(client, header); err != nil {
		t.Fatalf("write disk size error = %v", err)
	}
	for blockID, blockData := range blocks {
		blockHeader := make([]byte, 8)
		binary.BigEndian.PutUint32(blockHeader[0:4], uint32(blockID))
		binary.BigEndian.PutUint32(blockHeader[4:8], uint32(len(blockData)))
		if err := writeAll(client, blockHeader); err != nil {
			t.Fatalf("write disk block header error = %v", err)
		}
		if err := writeAll(client, blockData); err != nil {
			t.Fatalf("write disk block data error = %v", err)
		}
	}
	_ = client.Close()
	<-done

	baseline := adapter.vmBaselines["vm-disk"]
	if baseline == nil {
		t.Fatal("disk baseline was not stored")
	}
	for blockID, blockData := range blocks {
		if string(baseline.DiskBaselines[blockID]) != string(blockData) {
			t.Fatalf("disk baseline block %d = %q, want %q", blockID, baseline.DiskBaselines[blockID], blockData)
		}
	}
	if got := adapter.migrationsCompleted.Load(); got != 1 {
		t.Fatalf("migrationsCompleted = %d, want 1", got)
	}
}

func TestMigrationAdapterRejectsOversizedMemoryReceive(t *testing.T) {
	adapter, err := NewMigrationAdapter(MigrationAdapterConfig{
		EnableDWCP:      false,
		MetricsInterval: time.Hour,
		MaxMemoryUsage:  4,
	})
	if err != nil {
		t.Fatalf("NewMigrationAdapter() error = %v", err)
	}
	defer adapter.Close()

	server, client := net.Pipe()
	done := make(chan struct{})
	go func() {
		defer close(done)
		adapter.handleIncomingMigration(server)
	}()

	if err := writeMigrationHeader(client, migrationTypeMemory, "vm-big"); err != nil {
		t.Fatalf("writeMigrationHeader() error = %v", err)
	}
	header := make([]byte, 8)
	binary.BigEndian.PutUint64(header, 5)
	if err := writeAll(client, header); err != nil {
		t.Fatalf("write memory size error = %v", err)
	}
	_ = client.Close()
	<-done

	if _, exists := adapter.vmBaselines["vm-big"]; exists {
		t.Fatal("oversized memory migration stored a baseline")
	}
	if got := adapter.migrationsFailed.Load(); got != 1 {
		t.Fatalf("migrationsFailed = %d, want 1", got)
	}
}
