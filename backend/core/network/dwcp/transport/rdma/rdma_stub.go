//go:build !cgo || !linux

// Package rdma provides stub types for platforms without libibverbs (e.g. Jetson Thor aarch64).
// On platforms with CGO + Linux + libibverbs, rdma_cgo.go is used instead.
package rdma

import "fmt"

// DeviceInfo represents RDMA device information (stub).
type DeviceInfo struct {
	Name               string
	GUID               string
	NumPorts           int
	MaxMRSize          uint64
	MaxQP              uint32
	MaxCQ              uint32
	MaxCQE             uint32
	SupportsRC         bool
	SupportsUD         bool
	SupportsRDMAWrite  bool
	SupportsRDMARead   bool
	SupportsAtomic     bool
}

// ConnInfo represents RDMA connection information (stub).
type ConnInfo struct {
	LID   uint16
	QPNum uint32
	PSN   uint32
	GID   [16]byte
}

// Stats represents RDMA statistics (stub).
type Stats struct {
	SendCompletions uint64
	RecvCompletions uint64
	SendErrors      uint64
	RecvErrors      uint64
	BytesSent       uint64
	BytesReceived   uint64
}

// Context represents an RDMA context (stub).
type Context struct{}

// CheckAvailability always returns false on unsupported platforms.
func CheckAvailability() bool { return false }

// GetDeviceList returns an empty list on unsupported platforms.
func GetDeviceList() ([]DeviceInfo, error) {
	return nil, fmt.Errorf("RDMA not supported on this platform (requires CGO + Linux + libibverbs)")
}

// Initialize returns an error on unsupported platforms.
func Initialize(deviceName string, port int, useEventChannel bool) (*Context, error) {
	return nil, fmt.Errorf("RDMA not supported on this platform (requires CGO + Linux + libibverbs)")
}

// Close is a no-op stub.
func (c *Context) Close() {}

// RegisterMemory returns an error stub.
func (c *Context) RegisterMemory(buf []byte) error {
	return fmt.Errorf("RDMA not supported on this platform")
}

// UnregisterMemory returns an error stub.
func (c *Context) UnregisterMemory() error {
	return fmt.Errorf("RDMA not supported on this platform")
}

// GetConnInfo returns an error stub.
func (c *Context) GetConnInfo() (ConnInfo, error) {
	return ConnInfo{}, fmt.Errorf("RDMA not supported on this platform")
}

// Connect returns an error stub.
func (c *Context) Connect(remoteInfo ConnInfo) error {
	return fmt.Errorf("RDMA not supported on this platform")
}

// PostSend returns an error stub.
func (c *Context) PostSend(buf []byte, wrID uint64) error {
	return fmt.Errorf("RDMA not supported on this platform")
}

// PostRecv returns an error stub.
func (c *Context) PostRecv(buf []byte, wrID uint64) error {
	return fmt.Errorf("RDMA not supported on this platform")
}

// PostWrite returns an error stub.
func (c *Context) PostWrite(localBuf []byte, remoteAddr uint64, rkey uint32, wrID uint64) error {
	return fmt.Errorf("RDMA not supported on this platform")
}

// PostRead returns an error stub.
func (c *Context) PostRead(localBuf []byte, remoteAddr uint64, rkey uint32, wrID uint64) error {
	return fmt.Errorf("RDMA not supported on this platform")
}

// PollCompletion returns a stub result.
func (c *Context) PollCompletion(isSend bool) (bool, uint64, int, error) {
	return false, 0, 0, fmt.Errorf("RDMA not supported on this platform")
}

// WaitCompletion returns a stub result.
func (c *Context) WaitCompletion(isSend bool) (uint64, int, error) {
	return 0, 0, fmt.Errorf("RDMA not supported on this platform")
}

// GetStats returns empty stats stub.
func (c *Context) GetStats() (Stats, error) {
	return Stats{}, fmt.Errorf("RDMA not supported on this platform")
}

// GetBufferAddr returns 0 stub.
func (c *Context) GetBufferAddr() uint64 { return 0 }

// GetRKey returns 0 stub.
func (c *Context) GetRKey() uint32 { return 0 }

// IsConnected always returns false on unsupported platforms.
func (c *Context) IsConnected() bool { return false }
