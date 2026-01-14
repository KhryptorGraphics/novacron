package storage

import (
	"context"
	"fmt"
	"io"
	"strings"
	"testing"
	"time"
)

func TestBaseStorageService_CreateVolume(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024 * 1024, // 1GB
	}

	ctx := context.Background()
	volume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	if volume.Name != opts.Name {
		t.Errorf("Expected volume name %s, got %s", opts.Name, volume.Name)
	}

	if volume.Type != opts.Type {
		t.Errorf("Expected volume type %s, got %s", opts.Type, volume.Type)
	}

	if volume.Size != opts.Size {
		t.Errorf("Expected volume size %d, got %d", opts.Size, volume.Size)
	}

	if volume.State != VolumeStateCreating {
		t.Errorf("Expected volume state %s, got %s", VolumeStateCreating, volume.State)
	}
}

func TestBaseStorageService_GetVolume(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024 * 1024,
	}

	ctx := context.Background()
	createdVolume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	// Test getting existing volume
	retrievedVolume, err := service.GetVolume(ctx, createdVolume.ID)
	if err != nil {
		t.Fatalf("Failed to get volume: %v", err)
	}

	if retrievedVolume.ID != createdVolume.ID {
		t.Errorf("Expected volume ID %s, got %s", createdVolume.ID, retrievedVolume.ID)
	}

	// Test getting non-existent volume
	_, err = service.GetVolume(ctx, "non-existent")
	if err != ErrVolumeNotFound {
		t.Errorf("Expected ErrVolumeNotFound, got %v", err)
	}
}

func TestBaseStorageService_ListVolumes(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	ctx := context.Background()

	// Initially should have no volumes
	volumes, err := service.ListVolumes(ctx)
	if err != nil {
		t.Fatalf("Failed to list volumes: %v", err)
	}
	if len(volumes) != 0 {
		t.Errorf("Expected 0 volumes, got %d", len(volumes))
	}

	// Create some volumes
	for i := 0; i < 3; i++ {
		opts := VolumeCreateOptions{
			Name: fmt.Sprintf("test-volume-%d", i),
			Type: VolumeTypeFile,
			Size: 1024 * 1024 * 1024,
		}
		_, err := service.CreateVolume(ctx, opts)
		if err != nil {
			t.Fatalf("Failed to create volume %d: %v", i, err)
		}
	}

	// Should now have 3 volumes
	volumes, err = service.ListVolumes(ctx)
	if err != nil {
		t.Fatalf("Failed to list volumes: %v", err)
	}
	if len(volumes) != 3 {
		t.Errorf("Expected 3 volumes, got %d", len(volumes))
	}
}

func TestBaseStorageService_AttachDetachVolume(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024 * 1024,
	}

	ctx := context.Background()
	volume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	// Wait for volume to become available
	time.Sleep(3 * time.Second)

	// Test attach
	attachOpts := VolumeAttachOptions{
		VMID: "test-vm",
	}
	err = service.AttachVolume(ctx, volume.ID, attachOpts)
	if err != nil {
		t.Fatalf("Failed to attach volume: %v", err)
	}

	// Wait for attachment to complete
	time.Sleep(1 * time.Second)

	// Verify volume is attached
	attachedVolume, err := service.GetVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get volume after attach: %v", err)
	}
	if attachedVolume.State != VolumeStateAttached {
		t.Errorf("Expected volume state %s, got %s", VolumeStateAttached, attachedVolume.State)
	}
	if attachedVolume.AttachedToVM != attachOpts.VMID {
		t.Errorf("Expected attached VM %s, got %s", attachOpts.VMID, attachedVolume.AttachedToVM)
	}

	// Test detach
	detachOpts := VolumeDetachOptions{}
	err = service.DetachVolume(ctx, volume.ID, detachOpts)
	if err != nil {
		t.Fatalf("Failed to detach volume: %v", err)
	}

	// Wait for detachment to complete
	time.Sleep(1 * time.Second)

	// Verify volume is detached
	detachedVolume, err := service.GetVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get volume after detach: %v", err)
	}
	if detachedVolume.State != VolumeStateAvailable {
		t.Errorf("Expected volume state %s, got %s", VolumeStateAvailable, detachedVolume.State)
	}
	if detachedVolume.AttachedToVM != "" {
		t.Errorf("Expected no attached VM, got %s", detachedVolume.AttachedToVM)
	}
}

func TestBaseStorageService_ResizeVolume(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024 * 1024, // 1GB
	}

	ctx := context.Background()
	volume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	// Wait for volume to become available
	time.Sleep(3 * time.Second)

	// Test resize
	newSize := int64(2 * 1024 * 1024 * 1024) // 2GB
	resizeOpts := VolumeResizeOptions{
		NewSize: newSize,
	}
	err = service.ResizeVolume(ctx, volume.ID, resizeOpts)
	if err != nil {
		t.Fatalf("Failed to resize volume: %v", err)
	}

	// Verify new size
	resizedVolume, err := service.GetVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get volume after resize: %v", err)
	}
	if resizedVolume.Size != newSize {
		t.Errorf("Expected volume size %d, got %d", newSize, resizedVolume.Size)
	}

	// Test invalid resize (smaller size)
	smallerSize := int64(512 * 1024 * 1024) // 512MB
	invalidResizeOpts := VolumeResizeOptions{
		NewSize: smallerSize,
	}
	err = service.ResizeVolume(ctx, volume.ID, invalidResizeOpts)
	if err == nil {
		t.Error("Expected error when resizing to smaller size")
	}
}

func TestBaseStorageService_DeleteVolume(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024 * 1024,
	}

	ctx := context.Background()
	volume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	// Wait for volume to become available
	time.Sleep(3 * time.Second)

	// Test delete
	err = service.DeleteVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to delete volume: %v", err)
	}

	// Wait for deletion to complete
	time.Sleep(2 * time.Second)

	// Verify volume is deleted
	_, err = service.GetVolume(ctx, volume.ID)
	if err != ErrVolumeNotFound {
		t.Errorf("Expected ErrVolumeNotFound after deletion, got %v", err)
	}
}

func TestBaseStorageService_OpenVolume(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024, // 1MB
	}

	ctx := context.Background()
	volume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	// Wait for volume to become available
	time.Sleep(3 * time.Second)

	// Test open volume
	handle, err := service.OpenVolume(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to open volume: %v", err)
	}
	defer handle.Close()

	// Test write
	testData := []byte("Hello, World!")
	n, err := handle.Write(testData)
	if err != nil {
		t.Fatalf("Failed to write to volume: %v", err)
	}
	if n != len(testData) {
		t.Errorf("Expected to write %d bytes, wrote %d", len(testData), n)
	}

	// Test read
	readBuffer := make([]byte, len(testData))
	// Reset to beginning (simulate seek)
	volumeHandle := handle.(*volumeHandle)
	volumeHandle.offset = 0

	n, err = handle.Read(readBuffer)
	if err != nil {
		t.Fatalf("Failed to read from volume: %v", err)
	}
	if n != len(testData) {
		t.Errorf("Expected to read %d bytes, read %d", len(testData), n)
	}
	if string(readBuffer) != string(testData) {
		t.Errorf("Expected to read %s, got %s", string(testData), string(readBuffer))
	}
}

func TestBaseStorageService_GetVolumeStats(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024 * 1024,
	}

	ctx := context.Background()
	volume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	// Wait for volume to become available
	time.Sleep(3 * time.Second)

	// Test get stats
	stats, err := service.GetVolumeStats(ctx, volume.ID)
	if err != nil {
		t.Fatalf("Failed to get volume stats: %v", err)
	}

	// Verify stats contain expected fields
	expectedFields := []string{"id", "name", "type", "state", "size_bytes", "created_at", "updated_at"}
	for _, field := range expectedFields {
		if _, exists := stats[field]; !exists {
			t.Errorf("Expected stats to contain field %s", field)
		}
	}

	// Verify specific values
	if stats["id"] != volume.ID {
		t.Errorf("Expected stats ID %s, got %v", volume.ID, stats["id"])
	}
	if stats["name"] != volume.Name {
		t.Errorf("Expected stats name %s, got %v", volume.Name, stats["name"])
	}
}

func TestBaseStorageService_Events(t *testing.T) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)

	err := service.Start()
	if err != nil {
		t.Fatalf("Failed to start service: %v", err)
	}
	defer service.Stop()

	// Set up event listener
	events := make(chan VolumeEvent, 10)
	listener := func(event VolumeEvent) {
		events <- event
	}
	service.AddVolumeEventListener(listener)

	opts := VolumeCreateOptions{
		Name: "test-volume",
		Type: VolumeTypeFile,
		Size: 1024 * 1024 * 1024,
	}

	ctx := context.Background()
	volume, err := service.CreateVolume(ctx, opts)
	if err != nil {
		t.Fatalf("Failed to create volume: %v", err)
	}

	// Wait for creation event
	select {
	case event := <-events:
		if event.Type != VolumeEventCreated {
			t.Errorf("Expected event type %s, got %s", VolumeEventCreated, event.Type)
		}
		if event.VolumeID != volume.ID {
			t.Errorf("Expected event volume ID %s, got %s", volume.ID, event.VolumeID)
		}
	case <-time.After(5 * time.Second):
		t.Error("Timeout waiting for volume created event")
	}

	// Remove event listener
	service.RemoveVolumeEventListener(listener)
}

func TestVolumeHandle_ReadWrite(t *testing.T) {
	handle := &volumeHandle{
		volumeID: "test",
		buffer:   make([]byte, 1024),
		offset:   0,
	}

	// Test write
	testData := []byte("Hello, World!")
	n, err := handle.Write(testData)
	if err != nil {
		t.Fatalf("Failed to write: %v", err)
	}
	if n != len(testData) {
		t.Errorf("Expected to write %d bytes, wrote %d", len(testData), n)
	}

	// Reset offset for reading
	handle.offset = 0

	// Test read
	readBuffer := make([]byte, len(testData))
	n, err = handle.Read(readBuffer)
	if err != nil {
		t.Fatalf("Failed to read: %v", err)
	}
	if n != len(testData) {
		t.Errorf("Expected to read %d bytes, read %d", len(testData), n)
	}
	if string(readBuffer) != string(testData) {
		t.Errorf("Expected to read %s, got %s", string(testData), string(readBuffer))
	}

	// Test EOF
	handle.offset = int64(len(handle.buffer))
	n, err = handle.Read(readBuffer)
	if err != io.EOF {
		t.Errorf("Expected EOF at end of buffer, got %v", err)
	}
	if n != 0 {
		t.Errorf("Expected to read 0 bytes at EOF, read %d", n)
	}
}

func TestGenerateVolumeID(t *testing.T) {
	id1 := generateVolumeID()
	id2 := generateVolumeID()

	// IDs should be different
	if id1 == id2 {
		t.Error("Expected different volume IDs")
	}

	// IDs should start with "vol-"
	if !strings.HasPrefix(id1, "vol-") {
		t.Errorf("Expected volume ID to start with 'vol-', got %s", id1)
	}
	if !strings.HasPrefix(id2, "vol-") {
		t.Errorf("Expected volume ID to start with 'vol-', got %s", id2)
	}
}


// Benchmark tests
func BenchmarkBaseStorageService_CreateVolume(b *testing.B) {
	config := DefaultStorageConfig()
	service := NewBaseStorageService(config)
	service.Start()
	defer service.Stop()

	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		opts := VolumeCreateOptions{
			Name: fmt.Sprintf("bench-volume-%d", i),
			Type: VolumeTypeFile,
			Size: 1024 * 1024, // 1MB
		}
		_, err := service.CreateVolume(ctx, opts)
		if err != nil {
			b.Fatalf("Failed to create volume: %v", err)
		}
	}
}

func BenchmarkVolumeHandle_Write(b *testing.B) {
	handle := &volumeHandle{
		volumeID: "bench-test",
		buffer:   make([]byte, 1024*1024), // 1MB buffer
		offset:   0,
	}

	testData := make([]byte, 1024) // 1KB data
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle.offset = 0 // Reset for each iteration
		_, err := handle.Write(testData)
		if err != nil {
			b.Fatalf("Failed to write: %v", err)
		}
	}
}

func BenchmarkVolumeHandle_Read(b *testing.B) {
	handle := &volumeHandle{
		volumeID: "bench-test",
		buffer:   make([]byte, 1024*1024), // 1MB buffer
		offset:   0,
	}

	// Fill buffer with test data
	for i := range handle.buffer {
		handle.buffer[i] = byte(i % 256)
	}

	readBuffer := make([]byte, 1024) // 1KB read

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		handle.offset = 0 // Reset for each iteration
		_, err := handle.Read(readBuffer)
		if err != nil {
			b.Fatalf("Failed to read: %v", err)
		}
	}
}