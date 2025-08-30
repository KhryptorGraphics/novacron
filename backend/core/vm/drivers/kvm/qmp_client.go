package kvm

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"
)

// QMPClient represents a QEMU Machine Protocol client
type QMPClient struct {
	conn        net.Conn
	encoder     *json.Encoder
	decoder     *json.Decoder
	scanner     *bufio.Scanner
	mu          sync.Mutex
	capabilities []string
	eventChan   chan QMPEvent
	closed      bool
}

// QMPMessage represents a QMP message
type QMPMessage struct {
	Execute   string                 `json:"execute,omitempty"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
	ID        string                 `json:"id,omitempty"`
	Return    interface{}            `json:"return,omitempty"`
	Error     *QMPError              `json:"error,omitempty"`
	Event     string                 `json:"event,omitempty"`
	Data      map[string]interface{} `json:"data,omitempty"`
	Timestamp QMPTimestamp           `json:"timestamp,omitempty"`
}

// QMPError represents a QMP error
type QMPError struct {
	Class       string `json:"class"`
	Description string `json:"desc"`
}

// QMPTimestamp represents a QMP timestamp
type QMPTimestamp struct {
	Seconds      int64 `json:"seconds"`
	Microseconds int64 `json:"microseconds"`
}

// QMPEvent represents a QMP event
type QMPEvent struct {
	Event     string                 `json:"event"`
	Data      map[string]interface{} `json:"data"`
	Timestamp QMPTimestamp           `json:"timestamp"`
}

// QMPGreeting represents the QMP greeting message
type QMPGreeting struct {
	QMP QMPInfo `json:"QMP"`
}

// QMPInfo contains QMP version and capabilities
type QMPInfo struct {
	Version      QMPVersion `json:"version"`
	Capabilities []string   `json:"capabilities"`
}

// QMPVersion contains QEMU version information
type QMPVersion struct {
	QEMU    QEMUVersion `json:"qemu"`
	Package string      `json:"package"`
}

// QEMUVersion contains QEMU version details
type QEMUVersion struct {
	Major int `json:"major"`
	Minor int `json:"minor"`
	Micro int `json:"micro"`
}

// QMPBlockStats represents block device statistics
type QMPBlockStats struct {
	Device    string `json:"device"`
	Stats     BlockDeviceStats `json:"stats"`
	Parent    *QMPBlockStats   `json:"parent,omitempty"`
	Backing   *QMPBlockStats   `json:"backing,omitempty"`
}

// BlockDeviceStats contains block device statistics
type BlockDeviceStats struct {
	ReadBytes      int64 `json:"rd_bytes"`
	WriteBytes     int64 `json:"wr_bytes"`
	ReadOps        int64 `json:"rd_operations"`
	WriteOps       int64 `json:"wr_operations"`
	FlushOps       int64 `json:"flush_operations"`
	ReadTotalTime  int64 `json:"rd_total_time_ns"`
	WriteTotalTime int64 `json:"wr_total_time_ns"`
	FlushTotalTime int64 `json:"flush_total_time_ns"`
}

// QMPCPUStats represents CPU statistics
type QMPCPUStats struct {
	CPU       int   `json:"CPU"`
	Current   bool  `json:"current"`
	Halted    bool  `json:"halted"`
	PCValue   int64 `json:"pc"`
	ThreadID  int   `json:"thread_id"`
}

// QMPMemoryInfo represents memory information
type QMPMemoryInfo struct {
	BaseMemory int64 `json:"base-memory"`
	PluggedMemory int64 `json:"plugged-memory,omitempty"`
}

// QMPMachineInfo represents machine information
type QMPMachineInfo struct {
	Name        string `json:"name"`
	Alias       string `json:"alias,omitempty"`
	IsDefault   bool   `json:"is-default,omitempty"`
	CPUMax      int    `json:"cpu-max"`
	HotpluggableCPUs bool `json:"hotpluggable-cpus,omitempty"`
	NumaMemSupported bool `json:"numa-mem-supported,omitempty"`
}

// QMPDeviceInfo represents device information
type QMPDeviceInfo struct {
	Name        string   `json:"name"`
	Parent      string   `json:"parent,omitempty"`
	Bus         string   `json:"bus,omitempty"`
	Type        string   `json:"type"`
	ID          string   `json:"id,omitempty"`
	Description string   `json:"desc,omitempty"`
	HotPluggable bool    `json:"hotpluggable,omitempty"`
}

// NewQMPClient creates a new QMP client
func NewQMPClient(socketPath string) (*QMPClient, error) {
	conn, err := net.DialTimeout("unix", socketPath, 10*time.Second)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to QMP socket %s: %w", socketPath, err)
	}

	client := &QMPClient{
		conn:      conn,
		encoder:   json.NewEncoder(conn),
		decoder:   json.NewDecoder(conn),
		scanner:   bufio.NewScanner(conn),
		eventChan: make(chan QMPEvent, 100),
	}

	// Handle greeting
	if err := client.handleGreeting(); err != nil {
		conn.Close()
		return nil, fmt.Errorf("failed to handle QMP greeting: %w", err)
	}

	// Start event loop
	go client.eventLoop()

	return client, nil
}

// handleGreeting handles the initial QMP greeting
func (c *QMPClient) handleGreeting() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Read greeting message
	var greeting QMPGreeting
	if err := c.decoder.Decode(&greeting); err != nil {
		return fmt.Errorf("failed to decode greeting: %w", err)
	}

	c.capabilities = greeting.QMP.Capabilities

	// Send qmp_capabilities command
	capabilitiesCmd := QMPMessage{
		Execute: "qmp_capabilities",
		ID:      "capabilities",
	}

	if err := c.encoder.Encode(capabilitiesCmd); err != nil {
		return fmt.Errorf("failed to send capabilities command: %w", err)
	}

	// Read capabilities response
	var response QMPMessage
	if err := c.decoder.Decode(&response); err != nil {
		return fmt.Errorf("failed to decode capabilities response: %w", err)
	}

	if response.Error != nil {
		return fmt.Errorf("capabilities command failed: %s", response.Error.Description)
	}

	return nil
}

// eventLoop processes QMP events
func (c *QMPClient) eventLoop() {
	defer close(c.eventChan)

	for c.scanner.Scan() {
		line := c.scanner.Text()
		if line == "" {
			continue
		}

		var message QMPMessage
		if err := json.Unmarshal([]byte(line), &message); err != nil {
			continue // Skip malformed messages
		}

		// Handle events
		if message.Event != "" {
			event := QMPEvent{
				Event:     message.Event,
				Data:      message.Data,
				Timestamp: message.Timestamp,
			}

			select {
			case c.eventChan <- event:
			default:
				// Channel full, drop oldest event
				select {
				case <-c.eventChan:
					c.eventChan <- event
				default:
				}
			}
		}
	}
}

// Execute executes a QMP command
func (c *QMPClient) Execute(ctx context.Context, command string, args map[string]interface{}) (*QMPMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, fmt.Errorf("QMP client is closed")
	}

	// Generate unique ID
	id := fmt.Sprintf("%s-%d", command, time.Now().UnixNano())

	message := QMPMessage{
		Execute:   command,
		Arguments: args,
		ID:        id,
	}

	// Send command
	if err := c.encoder.Encode(message); err != nil {
		return nil, fmt.Errorf("failed to send command: %w", err)
	}

	// Read response with timeout
	responseChan := make(chan QMPMessage, 1)
	errorChan := make(chan error, 1)

	go func() {
		var response QMPMessage
		if err := c.decoder.Decode(&response); err != nil {
			errorChan <- err
			return
		}
		responseChan <- response
	}()

	select {
	case response := <-responseChan:
		if response.Error != nil {
			return &response, fmt.Errorf("QMP command failed: %s", response.Error.Description)
		}
		return &response, nil
	case err := <-errorChan:
		return nil, fmt.Errorf("failed to read response: %w", err)
	case <-ctx.Done():
		return nil, fmt.Errorf("command timeout: %w", ctx.Err())
	}
}

// GetBlockStats gets block device statistics
func (c *QMPClient) GetBlockStats(ctx context.Context) ([]QMPBlockStats, error) {
	response, err := c.Execute(ctx, "query-blockstats", nil)
	if err != nil {
		return nil, err
	}

	var stats []QMPBlockStats
	if statsData, ok := response.Return.([]interface{}); ok {
		for _, statInterface := range statsData {
			if statMap, ok := statInterface.(map[string]interface{}); ok {
				statJSON, _ := json.Marshal(statMap)
				var stat QMPBlockStats
				if err := json.Unmarshal(statJSON, &stat); err == nil {
					stats = append(stats, stat)
				}
			}
		}
	}

	return stats, nil
}

// GetCPUStats gets CPU statistics
func (c *QMPClient) GetCPUStats(ctx context.Context) ([]QMPCPUStats, error) {
	response, err := c.Execute(ctx, "query-cpus-fast", nil)
	if err != nil {
		return nil, err
	}

	var stats []QMPCPUStats
	if statsData, ok := response.Return.([]interface{}); ok {
		for _, statInterface := range statsData {
			if statMap, ok := statInterface.(map[string]interface{}); ok {
				statJSON, _ := json.Marshal(statMap)
				var stat QMPCPUStats
				if err := json.Unmarshal(statJSON, &stat); err == nil {
					stats = append(stats, stat)
				}
			}
		}
	}

	return stats, nil
}

// GetMemoryInfo gets memory information
func (c *QMPClient) GetMemoryInfo(ctx context.Context) (*QMPMemoryInfo, error) {
	response, err := c.Execute(ctx, "query-memory-size-summary", nil)
	if err != nil {
		return nil, err
	}

	var memInfo QMPMemoryInfo
	if memData, ok := response.Return.(map[string]interface{}); ok {
		memJSON, _ := json.Marshal(memData)
		if err := json.Unmarshal(memJSON, &memInfo); err != nil {
			return nil, fmt.Errorf("failed to parse memory info: %w", err)
		}
	}

	return &memInfo, nil
}

// GetMachineInfo gets machine information
func (c *QMPClient) GetMachineInfo(ctx context.Context) ([]QMPMachineInfo, error) {
	response, err := c.Execute(ctx, "query-machines", nil)
	if err != nil {
		return nil, err
	}

	var machines []QMPMachineInfo
	if machineData, ok := response.Return.([]interface{}); ok {
		for _, machineInterface := range machineData {
			if machineMap, ok := machineInterface.(map[string]interface{}); ok {
				machineJSON, _ := json.Marshal(machineMap)
				var machine QMPMachineInfo
				if err := json.Unmarshal(machineJSON, &machine); err == nil {
					machines = append(machines, machine)
				}
			}
		}
	}

	return machines, nil
}

// GetDevices gets device information
func (c *QMPClient) GetDevices(ctx context.Context) ([]QMPDeviceInfo, error) {
	response, err := c.Execute(ctx, "query-hotpluggable-cpus", nil)
	if err != nil {
		return nil, err
	}

	var devices []QMPDeviceInfo
	if deviceData, ok := response.Return.([]interface{}); ok {
		for _, deviceInterface := range deviceData {
			if deviceMap, ok := deviceInterface.(map[string]interface{}); ok {
				deviceJSON, _ := json.Marshal(deviceMap)
				var device QMPDeviceInfo
				if err := json.Unmarshal(deviceJSON, &device); err == nil {
					devices = append(devices, device)
				}
			}
		}
	}

	return devices, nil
}

// HotAddCPU adds a CPU to the running VM
func (c *QMPClient) HotAddCPU(ctx context.Context, cpuID string, socketID int, coreID int, threadID int) error {
	args := map[string]interface{}{
		"id": cpuID,
		"socket-id": socketID,
		"core-id": coreID,
		"thread-id": threadID,
	}

	_, err := c.Execute(ctx, "cpu-add", args)
	if err != nil {
		return fmt.Errorf("failed to hot-add CPU: %w", err)
	}

	return nil
}

// HotAddMemory adds memory to the running VM
func (c *QMPClient) HotAddMemory(ctx context.Context, memoryID string, sizeMB int64, nodeID int) error {
	args := map[string]interface{}{
		"id": memoryID,
		"size": sizeMB * 1024 * 1024, // Convert to bytes
		"node": nodeID,
	}

	_, err := c.Execute(ctx, "object-add", args)
	if err != nil {
		return fmt.Errorf("failed to hot-add memory: %w", err)
	}

	return nil
}

// HotAddDisk adds a disk to the running VM
func (c *QMPClient) HotAddDisk(ctx context.Context, driveID string, diskPath string, format string) error {
	// First add the drive
	driveArgs := map[string]interface{}{
		"id":     driveID,
		"file":   diskPath,
		"format": format,
		"if":     "none",
	}

	if _, err := c.Execute(ctx, "drive_add", driveArgs); err != nil {
		return fmt.Errorf("failed to add drive: %w", err)
	}

	// Then add the device
	deviceArgs := map[string]interface{}{
		"driver": "virtio-blk-pci",
		"drive":  driveID,
		"id":     driveID + "-device",
	}

	if _, err := c.Execute(ctx, "device_add", deviceArgs); err != nil {
		return fmt.Errorf("failed to add disk device: %w", err)
	}

	return nil
}

// HotRemoveDisk removes a disk from the running VM
func (c *QMPClient) HotRemoveDisk(ctx context.Context, deviceID string) error {
	args := map[string]interface{}{
		"id": deviceID,
	}

	_, err := c.Execute(ctx, "device_del", args)
	if err != nil {
		return fmt.Errorf("failed to remove disk: %w", err)
	}

	return nil
}

// HotAddNetworkInterface adds a network interface to the running VM
func (c *QMPClient) HotAddNetworkInterface(ctx context.Context, netdevID string, deviceID string, netdevType string, ifname string) error {
	// First add the netdev
	netdevArgs := map[string]interface{}{
		"type": netdevType,
		"id":   netdevID,
	}

	if ifname != "" {
		netdevArgs["ifname"] = ifname
	}

	if _, err := c.Execute(ctx, "netdev_add", netdevArgs); err != nil {
		return fmt.Errorf("failed to add netdev: %w", err)
	}

	// Then add the device
	deviceArgs := map[string]interface{}{
		"driver": "virtio-net-pci",
		"netdev": netdevID,
		"id":     deviceID,
	}

	if _, err := c.Execute(ctx, "device_add", deviceArgs); err != nil {
		return fmt.Errorf("failed to add network device: %w", err)
	}

	return nil
}

// HotRemoveNetworkInterface removes a network interface from the running VM
func (c *QMPClient) HotRemoveNetworkInterface(ctx context.Context, deviceID string, netdevID string) error {
	// Remove the device first
	deviceArgs := map[string]interface{}{
		"id": deviceID,
	}

	if _, err := c.Execute(ctx, "device_del", deviceArgs); err != nil {
		return fmt.Errorf("failed to remove network device: %w", err)
	}

	// Then remove the netdev
	netdevArgs := map[string]interface{}{
		"id": netdevID,
	}

	if _, err := c.Execute(ctx, "netdev_del", netdevArgs); err != nil {
		return fmt.Errorf("failed to remove netdev: %w", err)
	}

	return nil
}

// SetCPUAffinity sets CPU affinity for the VM
func (c *QMPClient) SetCPUAffinity(ctx context.Context, vcpuID int, cpuSet []int) error {
	args := map[string]interface{}{
		"vcpu":    vcpuID,
		"cpu-set": cpuSet,
	}

	_, err := c.Execute(ctx, "set-cpu-affinity", args)
	if err != nil {
		return fmt.Errorf("failed to set CPU affinity: %w", err)
	}

	return nil
}

// SaveVMState saves the VM state to a file
func (c *QMPClient) SaveVMState(ctx context.Context, filename string) error {
	args := map[string]interface{}{
		"filename": filename,
	}

	_, err := c.Execute(ctx, "migrate", args)
	if err != nil {
		return fmt.Errorf("failed to save VM state: %w", err)
	}

	return nil
}

// LoadVMState loads the VM state from a file
func (c *QMPClient) LoadVMState(ctx context.Context, filename string) error {
	args := map[string]interface{}{
		"uri": "exec:cat " + filename,
	}

	_, err := c.Execute(ctx, "migrate-incoming", args)
	if err != nil {
		return fmt.Errorf("failed to load VM state: %w", err)
	}

	return nil
}

// CreateSnapshot creates a VM snapshot
func (c *QMPClient) CreateSnapshot(ctx context.Context, snapshotName string, device string) error {
	args := map[string]interface{}{
		"job-id":        "snapshot-" + snapshotName,
		"device":        device,
		"snapshot-file": snapshotName + ".qcow2",
		"format":        "qcow2",
		"mode":          "absolute-paths",
	}

	_, err := c.Execute(ctx, "blockdev-snapshot-sync", args)
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %w", err)
	}

	return nil
}

// DeleteSnapshot deletes a VM snapshot
func (c *QMPClient) DeleteSnapshot(ctx context.Context, device string, snapshotID string) error {
	args := map[string]interface{}{
		"device":     device,
		"snapshot-id": snapshotID,
	}

	_, err := c.Execute(ctx, "blockdev-snapshot-delete-internal-sync", args)
	if err != nil {
		return fmt.Errorf("failed to delete snapshot: %w", err)
	}

	return nil
}

// GetMigrationStatus gets migration status
func (c *QMPClient) GetMigrationStatus(ctx context.Context) (map[string]interface{}, error) {
	response, err := c.Execute(ctx, "query-migrate", nil)
	if err != nil {
		return nil, err
	}

	if statusData, ok := response.Return.(map[string]interface{}); ok {
		return statusData, nil
	}

	return nil, fmt.Errorf("invalid migration status response")
}

// StartMigration starts VM migration
func (c *QMPClient) StartMigration(ctx context.Context, uri string, blk bool, inc bool) error {
	args := map[string]interface{}{
		"uri": uri,
	}

	if blk {
		args["blk"] = true
	}
	if inc {
		args["inc"] = true
	}

	_, err := c.Execute(ctx, "migrate", args)
	if err != nil {
		return fmt.Errorf("failed to start migration: %w", err)
	}

	return nil
}

// CancelMigration cancels ongoing migration
func (c *QMPClient) CancelMigration(ctx context.Context) error {
	_, err := c.Execute(ctx, "migrate_cancel", nil)
	if err != nil {
		return fmt.Errorf("failed to cancel migration: %w", err)
	}

	return nil
}

// SetMigrationSpeed sets migration speed limit
func (c *QMPClient) SetMigrationSpeed(ctx context.Context, speedMBps int64) error {
	args := map[string]interface{}{
		"value": speedMBps * 1024 * 1024, // Convert to bytes per second
	}

	_, err := c.Execute(ctx, "migrate_set_speed", args)
	if err != nil {
		return fmt.Errorf("failed to set migration speed: %w", err)
	}

	return nil
}

// GetEvents returns the event channel
func (c *QMPClient) GetEvents() <-chan QMPEvent {
	return c.eventChan
}

// GetCapabilities returns QMP capabilities
func (c *QMPClient) GetCapabilities() []string {
	return c.capabilities
}

// Close closes the QMP client
func (c *QMPClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	c.closed = true
	return c.conn.Close()
}

// IsConnected returns whether the client is connected
func (c *QMPClient) IsConnected() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return !c.closed && c.conn != nil
}

// Ping pings the QMP server
func (c *QMPClient) Ping(ctx context.Context) error {
	_, err := c.Execute(ctx, "query-status", nil)
	return err
}