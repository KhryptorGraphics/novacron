/*
Copyright 2024 NovaCron.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
*/

package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// VirtualMachineSpec defines the desired state of VirtualMachine
type VirtualMachineSpec struct {
	// Running controls whether the VM should be running
	Running bool `json:"running,omitempty"`

	// Template defines the VM template
	Template VMTemplate `json:"template"`

	// MigrationPolicy defines the migration behavior
	MigrationPolicy *MigrationPolicy `json:"migrationPolicy,omitempty"`

	// SnapshotPolicy defines the snapshot behavior
	SnapshotPolicy *SnapshotPolicy `json:"snapshotPolicy,omitempty"`

	// UpdateStrategy defines how updates are applied
	UpdateStrategy UpdateStrategy `json:"updateStrategy,omitempty"`
}

// VMTemplate defines the VM configuration template
type VMTemplate struct {
	// Metadata for the VM
	Metadata VMMetadata `json:"metadata,omitempty"`

	// Spec defines the VM specifications
	Spec VMSpec `json:"spec"`
}

// VMMetadata contains VM metadata
type VMMetadata struct {
	// Labels to apply to the VM
	Labels map[string]string `json:"labels,omitempty"`

	// Annotations to apply to the VM
	Annotations map[string]string `json:"annotations,omitempty"`
}

// VMSpec defines the VM specifications
type VMSpec struct {
	// Resources defines compute resources
	Resources Resources `json:"resources"`

	// Image defines the VM image
	Image VMImage `json:"image"`

	// Networks defines network configuration
	Networks []NetworkInterface `json:"networks,omitempty"`

	// Volumes defines storage volumes
	Volumes []Volume `json:"volumes,omitempty"`

	// Devices defines hardware devices (GPU, etc)
	Devices []Device `json:"devices,omitempty"`

	// UserData for cloud-init
	UserData string `json:"userData,omitempty"`

	// NodeSelector for placement constraints
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Affinity rules
	Affinity *corev1.Affinity `json:"affinity,omitempty"`

	// Lifecycle hooks
	Lifecycle *Lifecycle `json:"lifecycle,omitempty"`
}

// Resources defines compute resources
type Resources struct {
	// CPU cores
	CPU int32 `json:"cpu"`

	// Memory amount
	Memory resource.Quantity `json:"memory"`

	// Disk size
	Disk resource.Quantity `json:"disk"`
}

// VMImage defines the VM image source
type VMImage struct {
	// Source of the image (URL, marketplace, PVC, etc)
	Source string `json:"source"`

	// PullPolicy defines when to pull the image
	PullPolicy string `json:"pullPolicy,omitempty"`
}

// NetworkInterface defines a network interface
type NetworkInterface struct {
	// Name of the network
	Name string `json:"name"`

	// Type of network (bridge, ovs, sr-iov, macvlan)
	Type string `json:"type"`

	// IPv4 configuration
	IPv4 *IPConfig `json:"ipv4,omitempty"`

	// IPv6 configuration
	IPv6 *IPConfig `json:"ipv6,omitempty"`
}

// IPConfig defines IP configuration
type IPConfig struct {
	// Method (dhcp, static)
	Method string `json:"method"`

	// Address (for static method)
	Address string `json:"address,omitempty"`

	// Gateway (for static method)
	Gateway string `json:"gateway,omitempty"`

	// DNS servers
	DNS []string `json:"dns,omitempty"`
}

// Volume defines a storage volume
type Volume struct {
	// Name of the volume
	Name string `json:"name"`

	// Size of the volume
	Size resource.Quantity `json:"size"`

	// StorageClass to use
	StorageClass string `json:"storageClass,omitempty"`

	// AccessMode (ReadWriteOnce, ReadWriteMany, etc)
	AccessMode string `json:"accessMode,omitempty"`
}

// Device defines a hardware device
type Device struct {
	// Type of device (gpu, fpga, etc)
	Type string `json:"type"`

	// Vendor of the device
	Vendor string `json:"vendor,omitempty"`

	// Model of the device
	Model string `json:"model,omitempty"`

	// Count of devices
	Count int32 `json:"count,omitempty"`
}

// Lifecycle defines lifecycle hooks
type Lifecycle struct {
	// PreStart hook
	PreStart *LifecycleHandler `json:"preStart,omitempty"`

	// PreStop hook
	PreStop *LifecycleHandler `json:"preStop,omitempty"`
}

// LifecycleHandler defines a lifecycle hook handler
type LifecycleHandler struct {
	// Exec command
	Exec *ExecAction `json:"exec,omitempty"`

	// HTTP GET action
	HTTPGet *HTTPGetAction `json:"httpGet,omitempty"`

	// Timeout in seconds
	TimeoutSeconds int32 `json:"timeoutSeconds,omitempty"`
}

// ExecAction defines an exec hook
type ExecAction struct {
	// Command to execute
	Command []string `json:"command"`
}

// HTTPGetAction defines an HTTP GET hook
type HTTPGetAction struct {
	// Path to GET
	Path string `json:"path"`

	// Port to connect to
	Port int32 `json:"port"`

	// Host (defaults to VM IP)
	Host string `json:"host,omitempty"`

	// Scheme (HTTP or HTTPS)
	Scheme string `json:"scheme,omitempty"`
}

// MigrationPolicy defines migration behavior
type MigrationPolicy struct {
	// AllowLiveMigration enables live migration
	AllowLiveMigration bool `json:"allowLiveMigration,omitempty"`

	// Compression enables compression during migration
	Compression bool `json:"compression,omitempty"`

	// Encrypted enables encryption during migration
	Encrypted bool `json:"encrypted,omitempty"`

	// Bandwidth limit for migration
	Bandwidth string `json:"bandwidth,omitempty"`
}

// SnapshotPolicy defines snapshot behavior
type SnapshotPolicy struct {
	// Schedule in cron format
	Schedule string `json:"schedule,omitempty"`

	// Retention count
	Retention int32 `json:"retention,omitempty"`
}

// UpdateStrategy defines update behavior
type UpdateStrategy struct {
	// Type (RollingUpdate or Recreate)
	Type string `json:"type"`

	// RollingUpdate configuration
	RollingUpdate *RollingUpdateStrategy `json:"rollingUpdate,omitempty"`
}

// RollingUpdateStrategy defines rolling update configuration
type RollingUpdateStrategy struct {
	// MaxUnavailable VMs during update
	MaxUnavailable int32 `json:"maxUnavailable,omitempty"`
}

// VirtualMachineStatus defines the observed state of VirtualMachine
type VirtualMachineStatus struct {
	// Phase of the VM (Pending, Running, Migrating, Stopped, Failed)
	Phase VMPhase `json:"phase,omitempty"`

	// NodeName where the VM is running
	NodeName string `json:"nodeName,omitempty"`

	// IPAddresses assigned to the VM
	IPAddresses []string `json:"ipAddresses,omitempty"`

	// VMID is the underlying VM identifier
	VMID string `json:"vmID,omitempty"`

	// Resources usage
	ResourceUsage ResourceUsage `json:"resources,omitempty"`

	// Conditions represent the latest available observations
	Conditions []VMCondition `json:"conditions,omitempty"`

	// Migration status
	Migration *MigrationStatus `json:"migration,omitempty"`

	// Snapshots list
	Snapshots []SnapshotStatus `json:"snapshots,omitempty"`

	// LastUpdated timestamp
	LastUpdated metav1.Time `json:"lastUpdated,omitempty"`
}

// VMPhase represents the phase of a VM
type VMPhase string

const (
	// VMPending means the VM is being created
	VMPending VMPhase = "Pending"
	
	// VMRunning means the VM is running
	VMRunning VMPhase = "Running"
	
	// VMMigrating means the VM is being migrated
	VMMigrating VMPhase = "Migrating"
	
	// VMStopped means the VM is stopped
	VMStopped VMPhase = "Stopped"
	
	// VMFailed means the VM is in a failed state
	VMFailed VMPhase = "Failed"
)

// ResourceUsage defines resource usage
type ResourceUsage struct {
	// CPU usage
	CPU ResourceMetric `json:"cpu,omitempty"`

	// Memory usage
	Memory ResourceMetric `json:"memory,omitempty"`

	// Disk usage
	Disk ResourceMetric `json:"disk,omitempty"`
}

// ResourceMetric defines a resource metric
type ResourceMetric struct {
	// Used amount
	Used string `json:"used,omitempty"`

	// Available amount
	Available string `json:"available,omitempty"`

	// Percentage used
	Percentage float64 `json:"percentage,omitempty"`
}

// VMCondition describes the state of a VM at a certain point
type VMCondition struct {
	// Type of condition
	Type VMConditionType `json:"type"`

	// Status of the condition (True, False, Unknown)
	Status corev1.ConditionStatus `json:"status"`

	// LastProbeTime is the last time the condition was probed
	LastProbeTime metav1.Time `json:"lastProbeTime,omitempty"`

	// LastTransitionTime is the last time the condition transitioned
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty"`

	// Reason for the condition's last transition
	Reason string `json:"reason,omitempty"`

	// Message is a human-readable message
	Message string `json:"message,omitempty"`
}

// VMConditionType is a valid condition type for a VM
type VMConditionType string

const (
	// VMReady means the VM is ready to accept traffic
	VMReady VMConditionType = "Ready"

	// VMLiveMigratable means the VM can be live migrated
	VMLiveMigratable VMConditionType = "LiveMigratable"

	// VMPaused means the VM is paused
	VMPaused VMConditionType = "Paused"

	// VMEvacuating means the VM is being evacuated from a node
	VMEvacuating VMConditionType = "Evacuating"
)

// MigrationStatus defines migration status
type MigrationStatus struct {
	// TargetNode for migration
	TargetNode string `json:"targetNode,omitempty"`

	// StartTime of migration
	StartTime *metav1.Time `json:"startTime,omitempty"`

	// Progress percentage
	Progress int32 `json:"progress,omitempty"`

	// State of migration
	State string `json:"state,omitempty"`
}

// SnapshotStatus defines snapshot status
type SnapshotStatus struct {
	// Name of the snapshot
	Name string `json:"name"`

	// CreatedAt timestamp
	CreatedAt metav1.Time `json:"createdAt"`

	// Size of the snapshot
	Size resource.Quantity `json:"size,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=vm;vms
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase",description="VM phase"
// +kubebuilder:printcolumn:name="Node",type="string",JSONPath=".status.nodeName",description="Node name"
// +kubebuilder:printcolumn:name="IP",type="string",JSONPath=".status.ipAddresses[0]",description="IP address"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// VirtualMachine is the Schema for the virtualmachines API
type VirtualMachine struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VirtualMachineSpec   `json:"spec,omitempty"`
	Status VirtualMachineStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// VirtualMachineList contains a list of VirtualMachine
type VirtualMachineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VirtualMachine `json:"items"`
}

func init() {
	SchemeBuilder.Register(&VirtualMachine{}, &VirtualMachineList{})
}