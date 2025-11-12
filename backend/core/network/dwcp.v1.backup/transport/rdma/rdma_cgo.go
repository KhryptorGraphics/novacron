package rdma

/*
#cgo LDFLAGS: -libverbs
#include "rdma_native.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

// DeviceInfo represents RDMA device information
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

// ConnInfo represents RDMA connection information for exchange
type ConnInfo struct {
	LID   uint16
	QPNum uint32
	PSN   uint32
	GID   [16]byte
}

// Stats represents RDMA statistics
type Stats struct {
	SendCompletions  uint64
	RecvCompletions  uint64
	SendErrors       uint64
	RecvErrors       uint64
	BytesSent        uint64
	BytesReceived    uint64
}

// Context wraps the C RDMA context
type Context struct {
	ctx *C.rdma_context_t
}

// CheckAvailability checks if RDMA is available on the system
func CheckAvailability() bool {
	return int(C.rdma_check_availability()) > 0
}

// GetDeviceList returns a list of available RDMA devices
func GetDeviceList() ([]DeviceInfo, error) {
	var cDevices *C.rdma_device_info_t
	count := int(C.rdma_get_device_list(&cDevices))

	if count < 0 {
		return nil, fmt.Errorf("failed to get device list: %s", C.GoString(C.rdma_get_error_string()))
	}

	if count == 0 {
		return []DeviceInfo{}, nil
	}

	defer C.rdma_free_device_list(cDevices, C.int(count))

	// Convert C array to Go slice
	devices := make([]DeviceInfo, count)
	cDeviceSlice := (*[1 << 30]C.rdma_device_info_t)(unsafe.Pointer(cDevices))[:count:count]

	for i := 0; i < count; i++ {
		cDev := &cDeviceSlice[i]
		devices[i] = DeviceInfo{
			Name:              C.GoString(&cDev.name[0]),
			GUID:              C.GoString(&cDev.guid[0]),
			NumPorts:          int(cDev.num_ports),
			MaxMRSize:         uint64(cDev.max_mr_size),
			MaxQP:             uint32(cDev.max_qp),
			MaxCQ:             uint32(cDev.max_cq),
			MaxCQE:            uint32(cDev.max_cqe),
			SupportsRC:        cDev.supports_rc != 0,
			SupportsUD:        cDev.supports_ud != 0,
			SupportsRDMAWrite: cDev.supports_rdma_write != 0,
			SupportsRDMARead:  cDev.supports_rdma_read != 0,
			SupportsAtomic:    cDev.supports_atomic != 0,
		}
	}

	return devices, nil
}

// Initialize creates a new RDMA context
func Initialize(deviceName string, port int, useEventChannel bool) (*Context, error) {
	var cDeviceName *C.char
	if deviceName != "" {
		cDeviceName = C.CString(deviceName)
		defer C.free(unsafe.Pointer(cDeviceName))
	}

	useEvent := C.int(0)
	if useEventChannel {
		useEvent = C.int(1)
	}

	ctx := C.rdma_init(cDeviceName, C.int(port), useEvent)
	if ctx == nil {
		return nil, fmt.Errorf("RDMA initialization failed: %s", C.GoString(C.rdma_get_error_string()))
	}

	return &Context{ctx: ctx}, nil
}

// Close cleans up the RDMA context
func (c *Context) Close() {
	if c.ctx != nil {
		C.rdma_cleanup(c.ctx)
		c.ctx = nil
	}
}

// RegisterMemory registers a memory region for RDMA operations
func (c *Context) RegisterMemory(buf []byte) error {
	if len(buf) == 0 {
		return fmt.Errorf("buffer cannot be empty")
	}

	ret := C.rdma_register_memory(c.ctx, unsafe.Pointer(&buf[0]), C.size_t(len(buf)))
	if ret != 0 {
		return fmt.Errorf("failed to register memory: %s", C.GoString(C.rdma_get_error_string()))
	}

	return nil
}

// UnregisterMemory unregisters the memory region
func (c *Context) UnregisterMemory() error {
	ret := C.rdma_unregister_memory(c.ctx)
	if ret != 0 {
		return fmt.Errorf("failed to unregister memory: %s", C.GoString(C.rdma_get_error_string()))
	}

	return nil
}

// GetConnInfo returns connection information for exchange with peer
func (c *Context) GetConnInfo() (ConnInfo, error) {
	var cInfo C.rdma_conn_info_t

	ret := C.rdma_get_conn_info(c.ctx, &cInfo)
	if ret != 0 {
		return ConnInfo{}, fmt.Errorf("failed to get connection info: %s", C.GoString(C.rdma_get_error_string()))
	}

	info := ConnInfo{
		LID:   uint16(cInfo.lid),
		QPNum: uint32(cInfo.qp_num),
		PSN:   uint32(cInfo.psn),
	}

	for i := 0; i < 16; i++ {
		info.GID[i] = byte(cInfo.gid[i])
	}

	return info, nil
}

// Connect establishes RDMA connection with remote peer
func (c *Context) Connect(remoteInfo ConnInfo) error {
	var cInfo C.rdma_conn_info_t
	cInfo.lid = C.uint16_t(remoteInfo.LID)
	cInfo.qp_num = C.uint32_t(remoteInfo.QPNum)
	cInfo.psn = C.uint32_t(remoteInfo.PSN)

	for i := 0; i < 16; i++ {
		cInfo.gid[i] = C.uint8_t(remoteInfo.GID[i])
	}

	ret := C.rdma_connect(c.ctx, &cInfo)
	if ret != 0 {
		return fmt.Errorf("failed to connect: %s", C.GoString(C.rdma_get_error_string()))
	}

	return nil
}

// PostSend posts a send work request
func (c *Context) PostSend(buf []byte, wrID uint64) error {
	if len(buf) == 0 {
		return fmt.Errorf("buffer cannot be empty")
	}

	ret := C.rdma_post_send(c.ctx, unsafe.Pointer(&buf[0]), C.size_t(len(buf)), C.uint64_t(wrID))
	if ret != 0 {
		return fmt.Errorf("failed to post send: %s", C.GoString(C.rdma_get_error_string()))
	}

	return nil
}

// PostRecv posts a receive work request
func (c *Context) PostRecv(buf []byte, wrID uint64) error {
	if len(buf) == 0 {
		return fmt.Errorf("buffer cannot be empty")
	}

	ret := C.rdma_post_recv(c.ctx, unsafe.Pointer(&buf[0]), C.size_t(len(buf)), C.uint64_t(wrID))
	if ret != 0 {
		return fmt.Errorf("failed to post recv: %s", C.GoString(C.rdma_get_error_string()))
	}

	return nil
}

// PostWrite posts an RDMA write operation (one-sided)
func (c *Context) PostWrite(localBuf []byte, remoteAddr uint64, rkey uint32, wrID uint64) error {
	if len(localBuf) == 0 {
		return fmt.Errorf("buffer cannot be empty")
	}

	ret := C.rdma_post_write(c.ctx, unsafe.Pointer(&localBuf[0]), C.size_t(len(localBuf)),
		C.uint64_t(remoteAddr), C.uint32_t(rkey), C.uint64_t(wrID))
	if ret != 0 {
		return fmt.Errorf("failed to post write: %s", C.GoString(C.rdma_get_error_string()))
	}

	return nil
}

// PostRead posts an RDMA read operation (one-sided)
func (c *Context) PostRead(localBuf []byte, remoteAddr uint64, rkey uint32, wrID uint64) error {
	if len(localBuf) == 0 {
		return fmt.Errorf("buffer cannot be empty")
	}

	ret := C.rdma_post_read(c.ctx, unsafe.Pointer(&localBuf[0]), C.size_t(len(localBuf)),
		C.uint64_t(remoteAddr), C.uint32_t(rkey), C.uint64_t(wrID))
	if ret != 0 {
		return fmt.Errorf("failed to post read: %s", C.GoString(C.rdma_get_error_string()))
	}

	return nil
}

// PollCompletion polls for a completion event
// Returns: completed (bool), wrID, length, error
func (c *Context) PollCompletion(isSend bool) (bool, uint64, int, error) {
	var wrID C.uint64_t
	var length C.size_t

	cIsSend := C.int(0)
	if isSend {
		cIsSend = C.int(1)
	}

	ret := C.rdma_poll_completion(c.ctx, cIsSend, &wrID, &length)
	if ret < 0 {
		return false, 0, 0, fmt.Errorf("poll failed: %s", C.GoString(C.rdma_get_error_string()))
	}

	if ret == 0 {
		return false, 0, 0, nil // No completion
	}

	return true, uint64(wrID), int(length), nil
}

// WaitCompletion waits for a completion event (blocking)
func (c *Context) WaitCompletion(isSend bool) (uint64, int, error) {
	var wrID C.uint64_t
	var length C.size_t

	cIsSend := C.int(0)
	if isSend {
		cIsSend = C.int(1)
	}

	ret := C.rdma_wait_completion(c.ctx, cIsSend, &wrID, &length)
	if ret <= 0 {
		return 0, 0, fmt.Errorf("wait completion failed: %s", C.GoString(C.rdma_get_error_string()))
	}

	return uint64(wrID), int(length), nil
}

// GetStats returns RDMA statistics
func (c *Context) GetStats() (Stats, error) {
	var cStats C.rdma_stats_t

	ret := C.rdma_get_stats(c.ctx, &cStats)
	if ret != 0 {
		return Stats{}, fmt.Errorf("failed to get stats: %s", C.GoString(C.rdma_get_error_string()))
	}

	return Stats{
		SendCompletions: uint64(cStats.send_completions),
		RecvCompletions: uint64(cStats.recv_completions),
		SendErrors:      uint64(cStats.send_errors),
		RecvErrors:      uint64(cStats.recv_errors),
		BytesSent:       uint64(cStats.bytes_sent),
		BytesReceived:   uint64(cStats.bytes_received),
	}, nil
}

// GetBufferAddr returns the registered buffer address (for one-sided operations)
func (c *Context) GetBufferAddr() uint64 {
	return uint64(C.rdma_get_buffer_addr(c.ctx))
}

// GetRKey returns the remote key for the registered buffer
func (c *Context) GetRKey() uint32 {
	return uint32(C.rdma_get_rkey(c.ctx))
}

// IsConnected returns whether the RDMA connection is established
func (c *Context) IsConnected() bool {
	return C.rdma_is_connected(c.ctx) != 0
}
