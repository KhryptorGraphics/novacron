package api

import (
	"time"
)

// VirtualMachine represents a VM resource
type VirtualMachine struct {
	ID          string            `json:"id"`
	Name        string            `json:"name"`
	Namespace   string            `json:"namespace"`
	Status      VMStatus          `json:"status"`
	Spec        VMSpec            `json:"spec"`
	Metadata    VMMetadata        `json:"metadata"`
	CreatedAt   time.Time         `json:"createdAt"`
	UpdatedAt   time.Time         `json:"updatedAt"`
}

// VMStatus represents the status of a VM
type VMStatus struct {
	Phase       string          `json:"phase"`
	Conditions  []VMCondition   `json:"conditions"`
	NodeName    string          `json:"nodeName"`
	IPAddresses []string        `json:"ipAddresses"`
	Resources   ResourceUsage   `json:"resources"`
	Migration   *MigrationStatus `json:"migration,omitempty"`
}

// VMSpec represents the specification of a VM
type VMSpec struct {
	Running         bool             `json:"running"`
	Template        VMTemplate       `json:"template"`
	MigrationPolicy *MigrationPolicy `json:"migrationPolicy,omitempty"`
	SnapshotPolicy  *SnapshotPolicy  `json:"snapshotPolicy,omitempty"`
}

// VMTemplate represents the VM template
type VMTemplate struct {
	Spec VMTemplateSpec `json:"spec"`
}

// VMTemplateSpec represents the VM template specification
type VMTemplateSpec struct {
	Resources    Resources         `json:"resources"`
	Image        VMImage          `json:"image"`
	Networks     []NetworkInterface `json:"networks"`
	Volumes      []Volume          `json:"volumes,omitempty"`
	Devices      []Device          `json:"devices,omitempty"`
	UserData     string           `json:"userData,omitempty"`
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
}

// Resources represents compute resources
type Resources struct {
	CPU    int    `json:"cpu"`
	Memory string `json:"memory"`
	Disk   string `json:"disk"`
}

// ResourceUsage represents resource usage statistics
type ResourceUsage struct {
	CPU    ResourceMetric `json:"cpu"`
	Memory ResourceMetric `json:"memory"`
	Disk   ResourceMetric `json:"disk"`
}

// ResourceMetric represents a resource metric
type ResourceMetric struct {
	Used       string  `json:"used"`
	Available  string  `json:"available"`
	Percentage float64 `json:"percentage"`
}

// VMImage represents the VM image
type VMImage struct {
	Source     string `json:"source"`
	PullPolicy string `json:"pullPolicy,omitempty"`
}

// NetworkInterface represents a network interface
type NetworkInterface struct {
	Name string     `json:"name"`
	Type string     `json:"type"`
	IPv4 *IPConfig  `json:"ipv4,omitempty"`
	IPv6 *IPConfig  `json:"ipv6,omitempty"`
}

// IPConfig represents IP configuration
type IPConfig struct {
	Method  string   `json:"method"`
	Address string   `json:"address,omitempty"`
	Gateway string   `json:"gateway,omitempty"`
	DNS     []string `json:"dns,omitempty"`
}

// Volume represents a storage volume
type Volume struct {
	Name         string `json:"name"`
	Size         string `json:"size"`
	StorageClass string `json:"storageClass,omitempty"`
	AccessMode   string `json:"accessMode,omitempty"`
}

// Device represents a hardware device
type Device struct {
	Type   string `json:"type"`
	Vendor string `json:"vendor,omitempty"`
	Model  string `json:"model,omitempty"`
	Count  int    `json:"count,omitempty"`
}

// MigrationPolicy represents migration policy
type MigrationPolicy struct {
	AllowLiveMigration bool   `json:"allowLiveMigration"`
	Compression        bool   `json:"compression"`
	Encrypted          bool   `json:"encrypted"`
	Bandwidth          string `json:"bandwidth,omitempty"`
}

// SnapshotPolicy represents snapshot policy
type SnapshotPolicy struct {
	Schedule  string `json:"schedule"`
	Retention int    `json:"retention"`
}

// VMMetadata represents VM metadata
type VMMetadata struct {
	Labels      map[string]string `json:"labels,omitempty"`
	Annotations map[string]string `json:"annotations,omitempty"`
}

// VMCondition represents a VM condition
type VMCondition struct {
	Type               string    `json:"type"`
	Status             string    `json:"status"`
	LastTransitionTime time.Time `json:"lastTransitionTime"`
	Reason             string    `json:"reason,omitempty"`
	Message            string    `json:"message,omitempty"`
}

// MigrationStatus represents migration status
type MigrationStatus struct {
	State      string    `json:"state"`
	TargetNode string    `json:"targetNode"`
	Progress   int       `json:"progress"`
	StartTime  time.Time `json:"startTime"`
}

// Node represents a cluster node
type Node struct {
	Name      string         `json:"name"`
	Status    NodeStatus     `json:"status"`
	Capacity  Resources      `json:"capacity"`
	Available Resources      `json:"available"`
	Labels    map[string]string `json:"labels"`
	CreatedAt time.Time      `json:"createdAt"`
}

// NodeStatus represents node status
type NodeStatus struct {
	Phase      string         `json:"phase"`
	Conditions []NodeCondition `json:"conditions"`
	Addresses  []NodeAddress   `json:"addresses"`
}

// NodeCondition represents a node condition
type NodeCondition struct {
	Type    string    `json:"type"`
	Status  string    `json:"status"`
	Reason  string    `json:"reason,omitempty"`
	Message string    `json:"message,omitempty"`
	LastHeartbeatTime time.Time `json:"lastHeartbeatTime"`
}

// NodeAddress represents a node address
type NodeAddress struct {
	Type    string `json:"type"`
	Address string `json:"address"`
}

// Snapshot represents a VM snapshot
type Snapshot struct {
	ID        string            `json:"id"`
	Name      string            `json:"name"`
	VMID      string            `json:"vmId"`
	Size      string            `json:"size"`
	Status    string            `json:"status"`
	CreatedAt time.Time         `json:"createdAt"`
	Labels    map[string]string `json:"labels,omitempty"`
}

// Cluster represents cluster information
type Cluster struct {
	Name       string         `json:"name"`
	Version    string         `json:"version"`
	Status     ClusterStatus  `json:"status"`
	Nodes      int            `json:"nodes"`
	VMs        int            `json:"vms"`
	Capacity   Resources      `json:"capacity"`
	Used       Resources      `json:"used"`
}

// ClusterStatus represents cluster status
type ClusterStatus struct {
	Phase   string `json:"phase"`
	Message string `json:"message,omitempty"`
}

// Event represents a cluster event
type Event struct {
	ID        string    `json:"id"`
	Type      string    `json:"type"`
	Reason    string    `json:"reason"`
	Message   string    `json:"message"`
	Object    string    `json:"object"`
	Timestamp time.Time `json:"timestamp"`
}

// Metric represents a performance metric
type Metric struct {
	Name      string    `json:"name"`
	Value     float64   `json:"value"`
	Unit      string    `json:"unit"`
	Timestamp time.Time `json:"timestamp"`
	Labels    map[string]string `json:"labels,omitempty"`
}