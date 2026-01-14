package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// VirtualMachineSpec defines the desired state of VirtualMachine
type VirtualMachineSpec struct {
	// Name of the virtual machine
	Name string `json:"name"`

	// Template reference for VM creation
	Template *VMTemplateRef `json:"template,omitempty"`

	// Inline VM configuration (alternative to template)
	Config *VMConfig `json:"config,omitempty"`

	// Node selector for VM placement
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`

	// Affinity rules for VM scheduling
	Affinity *VMAffinity `json:"affinity,omitempty"`

	// Tolerations for VM scheduling
	Tolerations []VMToleration `json:"tolerations,omitempty"`

	// Restart policy for the VM
	RestartPolicy VMRestartPolicy `json:"restartPolicy,omitempty"`

	// Migration policy
	MigrationPolicy *VMMigrationPolicy `json:"migrationPolicy,omitempty"`
}

// VirtualMachineStatus defines the observed state of VirtualMachine
type VirtualMachineStatus struct {
	// Current phase of the VM
	Phase VMPhase `json:"phase,omitempty"`

	// Current state of the VM
	State string `json:"state,omitempty"`

	// Node where the VM is running
	NodeID string `json:"nodeId,omitempty"`

	// VM ID assigned by NovaCron backend
	VMID string `json:"vmId,omitempty"`

	// IP address of the VM
	IPAddress string `json:"ipAddress,omitempty"`

	// Resource usage statistics
	ResourceUsage *ResourceUsage `json:"resourceUsage,omitempty"`

	// Conditions represent the current conditions
	Conditions []VirtualMachineCondition `json:"conditions,omitempty"`

	// Migration status if VM is being migrated
	Migration *MigrationStatus `json:"migration,omitempty"`

	// Last observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// VMTemplateSpec defines the desired state of VMTemplate
type VMTemplateSpec struct {
	// Template metadata
	Description string `json:"description,omitempty"`

	// VM configuration template
	Config VMConfig `json:"config"`

	// Default node selector
	DefaultNodeSelector map[string]string `json:"defaultNodeSelector,omitempty"`

	// Default affinity rules
	DefaultAffinity *VMAffinity `json:"defaultAffinity,omitempty"`

	// Template parameters
	Parameters []TemplateParameter `json:"parameters,omitempty"`
}

// VMTemplateStatus defines the observed state of VMTemplate
type VMTemplateStatus struct {
	// Number of VMs created from this template
	VMCount int32 `json:"vmCount,omitempty"`

	// Validation status
	Valid bool `json:"valid,omitempty"`

	// Validation errors
	ValidationErrors []string `json:"validationErrors,omitempty"`

	// Last update time
	LastUpdated *metav1.Time `json:"lastUpdated,omitempty"`
}

// VMClusterSpec defines the desired state of VMCluster
type VMClusterSpec struct {
	// Number of VM replicas
	Replicas *int32 `json:"replicas,omitempty"`

	// Template for VMs in the cluster
	Template VMTemplateRef `json:"template"`

	// Update strategy
	UpdateStrategy VMClusterUpdateStrategy `json:"updateStrategy,omitempty"`

	// Load balancer configuration
	LoadBalancer *LoadBalancerConfig `json:"loadBalancer,omitempty"`

	// Auto scaling configuration
	AutoScaling *AutoScalingConfig `json:"autoScaling,omitempty"`
}

// VMClusterStatus defines the observed state of VMCluster
type VMClusterStatus struct {
	// Current number of replicas
	Replicas int32 `json:"replicas"`

	// Number of ready replicas
	ReadyReplicas int32 `json:"readyReplicas"`

	// Number of available replicas
	AvailableReplicas int32 `json:"availableReplicas"`

	// Observed generation
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions
	Conditions []VMClusterCondition `json:"conditions,omitempty"`

	// Load balancer status
	LoadBalancerStatus *LoadBalancerStatus `json:"loadBalancerStatus,omitempty"`
}

// Supporting types

type VMTemplateRef struct {
	Name      string `json:"name"`
	Namespace string `json:"namespace,omitempty"`
}

type VMConfig struct {
	// VM resources
	Resources VMResources `json:"resources"`

	// Container/VM image
	Image string `json:"image"`

	// Command to run
	Command []string `json:"command,omitempty"`

	// Arguments
	Args []string `json:"args,omitempty"`

	// Environment variables
	Env []EnvVar `json:"env,omitempty"`

	// Working directory
	WorkingDir string `json:"workingDir,omitempty"`

	// Network configuration
	Network *NetworkConfig `json:"network,omitempty"`

	// Storage configuration
	Storage *StorageConfig `json:"storage,omitempty"`

	// Security context
	SecurityContext *SecurityContext `json:"securityContext,omitempty"`
}

type VMResources struct {
	// CPU request and limit
	CPU ResourceQuantity `json:"cpu"`

	// Memory request and limit
	Memory ResourceQuantity `json:"memory"`

	// Disk size
	Disk ResourceQuantity `json:"disk,omitempty"`
}

type ResourceQuantity struct {
	Request string `json:"request,omitempty"`
	Limit   string `json:"limit,omitempty"`
}

type EnvVar struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

type NetworkConfig struct {
	// Network mode (bridge, host, none)
	Mode string `json:"mode,omitempty"`

	// Port mappings
	Ports []PortMapping `json:"ports,omitempty"`

	// DNS configuration
	DNS *DNSConfig `json:"dns,omitempty"`
}

type PortMapping struct {
	Name          string `json:"name,omitempty"`
	ContainerPort int32  `json:"containerPort"`
	HostPort      int32  `json:"hostPort,omitempty"`
	Protocol      string `json:"protocol,omitempty"`
}

type DNSConfig struct {
	Nameservers []string `json:"nameservers,omitempty"`
	Search      []string `json:"search,omitempty"`
}

type StorageConfig struct {
	// Volume mounts
	Volumes []VolumeMount `json:"volumes,omitempty"`
}

type VolumeMount struct {
	Name      string `json:"name"`
	MountPath string `json:"mountPath"`
	ReadOnly  bool   `json:"readOnly,omitempty"`
}

type SecurityContext struct {
	// Run as user
	RunAsUser *int64 `json:"runAsUser,omitempty"`

	// Run as group
	RunAsGroup *int64 `json:"runAsGroup,omitempty"`

	// Privileged mode
	Privileged *bool `json:"privileged,omitempty"`

	// Capabilities
	Capabilities *Capabilities `json:"capabilities,omitempty"`
}

type Capabilities struct {
	Add  []string `json:"add,omitempty"`
	Drop []string `json:"drop,omitempty"`
}

type VMAffinity struct {
	// Node affinity
	NodeAffinity *NodeAffinity `json:"nodeAffinity,omitempty"`

	// VM affinity
	VMAffinity *PodAffinity `json:"vmAffinity,omitempty"`

	// VM anti-affinity
	VMAntiAffinity *PodAntiAffinity `json:"vmAntiAffinity,omitempty"`
}

type NodeAffinity struct {
	RequiredDuringSchedulingIgnoredDuringExecution  *NodeSelector `json:"requiredDuringSchedulingIgnoredDuringExecution,omitempty"`
	PreferredDuringSchedulingIgnoredDuringExecution []PreferredSchedulingTerm `json:"preferredDuringSchedulingIgnoredDuringExecution,omitempty"`
}

type NodeSelector struct {
	NodeSelectorTerms []NodeSelectorTerm `json:"nodeSelectorTerms"`
}

type NodeSelectorTerm struct {
	MatchExpressions []NodeSelectorRequirement `json:"matchExpressions,omitempty"`
	MatchFields      []NodeSelectorRequirement `json:"matchFields,omitempty"`
}

type NodeSelectorRequirement struct {
	Key      string   `json:"key"`
	Operator string   `json:"operator"`
	Values   []string `json:"values,omitempty"`
}

type PreferredSchedulingTerm struct {
	Weight     int32            `json:"weight"`
	Preference NodeSelectorTerm `json:"preference"`
}

type PodAffinity struct {
	RequiredDuringSchedulingIgnoredDuringExecution  []PodAffinityTerm         `json:"requiredDuringSchedulingIgnoredDuringExecution,omitempty"`
	PreferredDuringSchedulingIgnoredDuringExecution []WeightedPodAffinityTerm `json:"preferredDuringSchedulingIgnoredDuringExecution,omitempty"`
}

type PodAntiAffinity struct {
	RequiredDuringSchedulingIgnoredDuringExecution  []PodAffinityTerm         `json:"requiredDuringSchedulingIgnoredDuringExecution,omitempty"`
	PreferredDuringSchedulingIgnoredDuringExecution []WeightedPodAffinityTerm `json:"preferredDuringSchedulingIgnoredDuringExecution,omitempty"`
}

type PodAffinityTerm struct {
	LabelSelector *metav1.LabelSelector `json:"labelSelector,omitempty"`
	Namespaces    []string              `json:"namespaces,omitempty"`
	TopologyKey   string                `json:"topologyKey"`
}

type WeightedPodAffinityTerm struct {
	Weight          int32           `json:"weight"`
	PodAffinityTerm PodAffinityTerm `json:"podAffinityTerm"`
}

type VMToleration struct {
	Key               string             `json:"key,omitempty"`
	Operator          TolerationOperator `json:"operator,omitempty"`
	Value             string             `json:"value,omitempty"`
	Effect            TaintEffect        `json:"effect,omitempty"`
	TolerationSeconds *int64             `json:"tolerationSeconds,omitempty"`
}

type VMMigrationPolicy struct {
	// Migration type (live, warm, cold)
	Type string `json:"type,omitempty"`

	// Enable automatic migration
	AutoMigrate bool `json:"autoMigrate,omitempty"`

	// Migration triggers
	Triggers []MigrationTrigger `json:"triggers,omitempty"`
}

type MigrationTrigger struct {
	Type      string `json:"type"`
	Threshold string `json:"threshold,omitempty"`
}

type ResourceUsage struct {
	CPU    string `json:"cpu,omitempty"`
	Memory string `json:"memory,omitempty"`
	Disk   string `json:"disk,omitempty"`
}

type MigrationStatus struct {
	Type             string      `json:"type"`
	SourceNode       string      `json:"sourceNode"`
	TargetNode       string      `json:"targetNode"`
	Progress         float64     `json:"progress"`
	StartTime        *metav1.Time `json:"startTime,omitempty"`
	CompletionTime   *metav1.Time `json:"completionTime,omitempty"`
	EstimatedEndTime *metav1.Time `json:"estimatedEndTime,omitempty"`
}

type TemplateParameter struct {
	Name         string      `json:"name"`
	Type         string      `json:"type"`
	Description  string      `json:"description,omitempty"`
	Required     bool        `json:"required,omitempty"`
	DefaultValue interface{} `json:"defaultValue,omitempty"`
}

type VMClusterUpdateStrategy struct {
	Type          string                 `json:"type,omitempty"`
	RollingUpdate *RollingUpdateStrategy `json:"rollingUpdate,omitempty"`
}

type RollingUpdateStrategy struct {
	MaxUnavailable *int32 `json:"maxUnavailable,omitempty"`
	MaxSurge       *int32 `json:"maxSurge,omitempty"`
}

type LoadBalancerConfig struct {
	Type     string            `json:"type"`
	Ports    []LoadBalancerPort `json:"ports"`
	Selector map[string]string `json:"selector,omitempty"`
}

type LoadBalancerPort struct {
	Name       string `json:"name,omitempty"`
	Port       int32  `json:"port"`
	TargetPort int32  `json:"targetPort"`
	Protocol   string `json:"protocol,omitempty"`
}

type LoadBalancerStatus struct {
	IP       string `json:"ip,omitempty"`
	Hostname string `json:"hostname,omitempty"`
	Ready    bool   `json:"ready"`
}

type AutoScalingConfig struct {
	Enabled    bool  `json:"enabled"`
	MinReplicas int32 `json:"minReplicas"`
	MaxReplicas int32 `json:"maxReplicas"`
	
	// CPU utilization threshold for scaling
	TargetCPUUtilization *int32 `json:"targetCPUUtilization,omitempty"`
	
	// Memory utilization threshold for scaling
	TargetMemoryUtilization *int32 `json:"targetMemoryUtilization,omitempty"`
}

// Condition types

type VirtualMachineCondition struct {
	Type               VirtualMachineConditionType `json:"type"`
	Status             metav1.ConditionStatus      `json:"status"`
	LastTransitionTime metav1.Time                 `json:"lastTransitionTime"`
	Reason             string                      `json:"reason,omitempty"`
	Message            string                      `json:"message,omitempty"`
}

type VMClusterCondition struct {
	Type               VMClusterConditionType `json:"type"`
	Status             metav1.ConditionStatus `json:"status"`
	LastTransitionTime metav1.Time            `json:"lastTransitionTime"`
	Reason             string                 `json:"reason,omitempty"`
	Message            string                 `json:"message,omitempty"`
}

// Enums

type VMPhase string

const (
	VMPhasePending   VMPhase = "Pending"
	VMPhaseRunning   VMPhase = "Running"
	VMPhaseFailed    VMPhase = "Failed"
	VMPhaseSucceeded VMPhase = "Succeeded"
	VMPhaseUnknown   VMPhase = "Unknown"
)

type VMRestartPolicy string

const (
	VMRestartPolicyAlways    VMRestartPolicy = "Always"
	VMRestartPolicyOnFailure VMRestartPolicy = "OnFailure"
	VMRestartPolicyNever     VMRestartPolicy = "Never"
)

type VirtualMachineConditionType string

const (
	VirtualMachineReady       VirtualMachineConditionType = "Ready"
	VirtualMachineScheduled   VirtualMachineConditionType = "Scheduled"
	VirtualMachineInitialized VirtualMachineConditionType = "Initialized"
	VirtualMachineMigrating   VirtualMachineConditionType = "Migrating"
)

type VMClusterConditionType string

const (
	VMClusterReady       VMClusterConditionType = "Ready"
	VMClusterProgressing VMClusterConditionType = "Progressing"
	VMClusterReplicaFailure VMClusterConditionType = "ReplicaFailure"
)

type TolerationOperator string

const (
	TolerationOpExists TolerationOperator = "Exists"
	TolerationOpEqual  TolerationOperator = "Equal"
)

type TaintEffect string

const (
	TaintEffectNoSchedule       TaintEffect = "NoSchedule"
	TaintEffectPreferNoSchedule TaintEffect = "PreferNoSchedule"
	TaintEffectNoExecute        TaintEffect = "NoExecute"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="State",type="string",JSONPath=".status.state"
// +kubebuilder:printcolumn:name="Node",type="string",JSONPath=".status.nodeId"
// +kubebuilder:printcolumn:name="IP",type="string",JSONPath=".status.ipAddress"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// VirtualMachine represents a NovaCron VM in Kubernetes
type VirtualMachine struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VirtualMachineSpec   `json:"spec,omitempty"`
	Status VirtualMachineStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// VirtualMachineList contains a list of VirtualMachine
type VirtualMachineList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VirtualMachine `json:"items"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// VMTemplate represents a VM template in Kubernetes
type VMTemplate struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VMTemplateSpec   `json:"spec,omitempty"`
	Status VMTemplateStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// VMTemplateList contains a list of VMTemplate
type VMTemplateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VMTemplate `json:"items"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:subresource:scale:specpath=.spec.replicas,statuspath=.status.replicas
// +kubebuilder:printcolumn:name="Desired",type="integer",JSONPath=".spec.replicas"
// +kubebuilder:printcolumn:name="Current",type="integer",JSONPath=".status.replicas"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyReplicas"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"

// VMCluster represents a cluster of VMs in Kubernetes
type VMCluster struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   VMClusterSpec   `json:"spec,omitempty"`
	Status VMClusterStatus `json:"status,omitempty"`
}

//+kubebuilder:object:root=true

// VMClusterList contains a list of VMCluster
type VMClusterList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []VMCluster `json:"items"`
}

