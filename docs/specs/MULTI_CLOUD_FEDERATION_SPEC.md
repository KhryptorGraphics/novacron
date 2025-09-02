# Multi-Cloud Federation Specification

## Overview

NovaCron's Multi-Cloud Federation enables seamless management of virtual machines across multiple cloud providers (AWS, Azure, GCP, Oracle Cloud) and on-premises infrastructure through a unified control plane.

## Architecture

### Federation Control Plane

```
┌──────────────────────────────────────────────────┐
│              NovaCron Federation Manager          │
├──────────────────────────────────────────────────┤
│  Policy Engine │ Cost Optimizer │ Load Balancer   │
├──────────────────────────────────────────────────┤
│            Provider Abstraction Layer             │
├───────────┬───────────┬───────────┬─────────────┤
│    AWS    │   Azure   │    GCP    │  On-Prem    │
│  Provider │ Provider  │ Provider  │  Provider   │
└───────────┴───────────┴───────────┴─────────────┘
         │           │           │           │
    ┌────▼───┐  ┌───▼────┐ ┌───▼───┐  ┌───▼────┐
    │  EC2   │  │  Azure │ │  GCE  │  │  KVM/  │
    │        │  │   VM   │ │       │  │ VMware │
    └────────┘  └────────┘ └───────┘  └────────┘
```

## Provider Implementations

### AWS Provider

```go
// pkg/providers/aws/provider.go
package aws

import (
    "context"
    "fmt"
    
    "github.com/aws/aws-sdk-go-v2/config"
    "github.com/aws/aws-sdk-go-v2/service/ec2"
    "github.com/aws/aws-sdk-go-v2/service/ec2/types"
    "github.com/novacron/core/provider"
)

type AWSProvider struct {
    client     *ec2.Client
    region     string
    credentials AWSCredentials
}

type AWSCredentials struct {
    AccessKeyID     string
    SecretAccessKey string
    SessionToken    string
    AssumeRole      string
}

func NewAWSProvider(ctx context.Context, cfg ProviderConfig) (*AWSProvider, error) {
    awsCfg, err := config.LoadDefaultConfig(ctx,
        config.WithRegion(cfg.Region),
        config.WithCredentialsProvider(cfg.Credentials),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to load AWS config: %w", err)
    }
    
    return &AWSProvider{
        client: ec2.NewFromConfig(awsCfg),
        region: cfg.Region,
    }, nil
}

// CreateVM creates an EC2 instance
func (p *AWSProvider) CreateVM(ctx context.Context, spec provider.VMSpec) (*provider.VM, error) {
    // Map generic spec to EC2 parameters
    instanceType := p.mapInstanceType(spec.CPU, spec.Memory)
    
    input := &ec2.RunInstancesInput{
        ImageId:      aws.String(p.mapImage(spec.Image)),
        InstanceType: instanceType,
        MinCount:     aws.Int32(1),
        MaxCount:     aws.Int32(1),
        
        // Network configuration
        NetworkInterfaces: []types.InstanceNetworkInterfaceSpecification{
            {
                DeviceIndex:              aws.Int32(0),
                SubnetId:                 aws.String(spec.Network.SubnetID),
                AssociatePublicIpAddress: aws.Bool(spec.Network.PublicIP),
                SecurityGroups:           spec.Network.SecurityGroups,
            },
        },
        
        // Storage configuration
        BlockDeviceMappings: []types.BlockDeviceMapping{
            {
                DeviceName: aws.String("/dev/sda1"),
                Ebs: &types.EbsBlockDevice{
                    VolumeSize:          aws.Int32(int32(spec.Disk.Size)),
                    VolumeType:          types.VolumeType(spec.Disk.Type),
                    Encrypted:           aws.Bool(spec.Disk.Encrypted),
                    DeleteOnTermination: aws.Bool(true),
                },
            },
        },
        
        // User data for cloud-init
        UserData: aws.String(base64.StdEncoding.EncodeToString([]byte(spec.UserData))),
        
        // Tags
        TagSpecifications: []types.TagSpecification{
            {
                ResourceType: types.ResourceTypeInstance,
                Tags: p.mapTags(spec.Labels),
            },
        },
        
        // IAM role
        IamInstanceProfile: &types.IamInstanceProfileSpecification{
            Arn: aws.String(spec.IAMRole),
        },
    }
    
    // Handle spot instances
    if spec.Spot {
        input.InstanceMarketOptions = &types.InstanceMarketOptionsRequest{
            MarketType: types.MarketTypeSpot,
            SpotOptions: &types.SpotMarketOptions{
                MaxPrice:         aws.String(fmt.Sprintf("%.4f", spec.SpotMaxPrice)),
                SpotInstanceType: types.SpotInstanceTypePersistent,
            },
        }
    }
    
    result, err := p.client.RunInstances(ctx, input)
    if err != nil {
        return nil, fmt.Errorf("failed to create EC2 instance: %w", err)
    }
    
    instance := result.Instances[0]
    
    // Wait for instance to be running
    waiter := ec2.NewInstanceRunningWaiter(p.client)
    err = waiter.Wait(ctx, &ec2.DescribeInstancesInput{
        InstanceIds: []string{*instance.InstanceId},
    }, 5*time.Minute)
    if err != nil {
        return nil, fmt.Errorf("timeout waiting for instance: %w", err)
    }
    
    return p.instanceToVM(instance), nil
}

// ListVMs lists all EC2 instances
func (p *AWSProvider) ListVMs(ctx context.Context, filters provider.FilterOptions) ([]*provider.VM, error) {
    input := &ec2.DescribeInstancesInput{}
    
    // Apply filters
    if len(filters.Labels) > 0 {
        input.Filters = p.buildFilters(filters)
    }
    
    var vms []*provider.VM
    paginator := ec2.NewDescribeInstancesPaginator(p.client, input)
    
    for paginator.HasMorePages() {
        output, err := paginator.NextPage(ctx)
        if err != nil {
            return nil, fmt.Errorf("failed to list instances: %w", err)
        }
        
        for _, reservation := range output.Reservations {
            for _, instance := range reservation.Instances {
                // Skip terminated instances
                if instance.State.Name == types.InstanceStateNameTerminated {
                    continue
                }
                vms = append(vms, p.instanceToVM(&instance))
            }
        }
    }
    
    return vms, nil
}

// MigrateVM migrates VM to another region/zone
func (p *AWSProvider) MigrateVM(ctx context.Context, vmID string, target provider.MigrationTarget) error {
    // Create AMI from instance
    createImageInput := &ec2.CreateImageInput{
        InstanceId:  aws.String(vmID),
        Name:        aws.String(fmt.Sprintf("migration-%s-%d", vmID, time.Now().Unix())),
        Description: aws.String("Migration image"),
        NoReboot:    aws.Bool(target.LiveMigration),
    }
    
    imageResult, err := p.client.CreateImage(ctx, createImageInput)
    if err != nil {
        return fmt.Errorf("failed to create AMI: %w", err)
    }
    
    // Wait for AMI to be available
    waiter := ec2.NewImageAvailableWaiter(p.client)
    err = waiter.Wait(ctx, &ec2.DescribeImagesInput{
        ImageIds: []string{*imageResult.ImageId},
    }, 10*time.Minute)
    if err != nil {
        return fmt.Errorf("timeout waiting for AMI: %w", err)
    }
    
    // Copy AMI to target region if different
    if target.Region != p.region {
        targetClient := p.getRegionalClient(target.Region)
        copyResult, err := targetClient.CopyImage(ctx, &ec2.CopyImageInput{
            SourceImageId: imageResult.ImageId,
            SourceRegion:  aws.String(p.region),
            Name:          aws.String(fmt.Sprintf("migration-%s", vmID)),
        })
        if err != nil {
            return fmt.Errorf("failed to copy AMI: %w", err)
        }
        imageResult.ImageId = copyResult.ImageId
    }
    
    // Launch instance in target location
    // ... (similar to CreateVM)
    
    // Clean up source instance if not live migration
    if !target.LiveMigration {
        _, err = p.client.TerminateInstances(ctx, &ec2.TerminateInstancesInput{
            InstanceIds: []string{vmID},
        })
        if err != nil {
            return fmt.Errorf("failed to terminate source instance: %w", err)
        }
    }
    
    return nil
}

// Cost estimation
func (p *AWSProvider) EstimateCost(ctx context.Context, spec provider.VMSpec) (*provider.CostEstimate, error) {
    // Use AWS Pricing API
    instanceType := p.mapInstanceType(spec.CPU, spec.Memory)
    
    // Simplified cost calculation
    hourlyCost := p.getInstancePrice(instanceType)
    storageCost := float64(spec.Disk.Size) * 0.10 / 730 // $0.10 per GB per month
    
    return &provider.CostEstimate{
        Hourly:  hourlyCost + storageCost,
        Monthly: (hourlyCost + storageCost) * 730,
        Breakdown: map[string]float64{
            "compute": hourlyCost * 730,
            "storage": storageCost * 730,
            "network": 0.09 * float64(spec.NetworkBandwidth), // $0.09 per GB
        },
    }, nil
}
```

### Azure Provider

```go
// pkg/providers/azure/provider.go
package azure

import (
    "context"
    "fmt"
    
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
    "github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/compute/armcompute"
    "github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/network/armnetwork"
)

type AzureProvider struct {
    vmClient      *armcompute.VirtualMachinesClient
    networkClient *armnetwork.InterfacesClient
    subscription  string
    resourceGroup string
}

func NewAzureProvider(ctx context.Context, cfg ProviderConfig) (*AzureProvider, error) {
    cred, err := azidentity.NewDefaultAzureCredential(nil)
    if err != nil {
        return nil, fmt.Errorf("failed to create Azure credential: %w", err)
    }
    
    vmClient, err := armcompute.NewVirtualMachinesClient(cfg.SubscriptionID, cred, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to create VM client: %w", err)
    }
    
    networkClient, err := armnetwork.NewInterfacesClient(cfg.SubscriptionID, cred, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to create network client: %w", err)
    }
    
    return &AzureProvider{
        vmClient:      vmClient,
        networkClient: networkClient,
        subscription:  cfg.SubscriptionID,
        resourceGroup: cfg.ResourceGroup,
    }, nil
}

func (p *AzureProvider) CreateVM(ctx context.Context, spec provider.VMSpec) (*provider.VM, error) {
    // Create network interface
    nicName := fmt.Sprintf("%s-nic", spec.Name)
    nic, err := p.createNetworkInterface(ctx, nicName, spec.Network)
    if err != nil {
        return nil, fmt.Errorf("failed to create network interface: %w", err)
    }
    
    // Prepare VM parameters
    vmSize := p.mapVMSize(spec.CPU, spec.Memory)
    
    parameters := armcompute.VirtualMachine{
        Location: to.Ptr(spec.Location),
        Properties: &armcompute.VirtualMachineProperties{
            HardwareProfile: &armcompute.HardwareProfile{
                VMSize: (*armcompute.VirtualMachineSizeTypes)(to.Ptr(vmSize)),
            },
            StorageProfile: &armcompute.StorageProfile{
                ImageReference: p.getImageReference(spec.Image),
                OSDisk: &armcompute.OSDisk{
                    CreateOption: to.Ptr(armcompute.DiskCreateOptionTypesFromImage),
                    ManagedDisk: &armcompute.ManagedDiskParameters{
                        StorageAccountType: to.Ptr(armcompute.StorageAccountTypesPremiumLRS),
                    },
                    DiskSizeGB: to.Ptr(int32(spec.Disk.Size)),
                },
            },
            OSProfile: &armcompute.OSProfile{
                ComputerName:  to.Ptr(spec.Name),
                AdminUsername: to.Ptr("azureuser"),
                CustomData:    to.Ptr(base64.StdEncoding.EncodeToString([]byte(spec.UserData))),
            },
            NetworkProfile: &armcompute.NetworkProfile{
                NetworkInterfaces: []*armcompute.NetworkInterfaceReference{
                    {
                        ID: nic.ID,
                        Properties: &armcompute.NetworkInterfaceReferenceProperties{
                            Primary: to.Ptr(true),
                        },
                    },
                },
            },
        },
        Tags: p.mapTags(spec.Labels),
    }
    
    // Handle spot instances
    if spec.Spot {
        parameters.Properties.Priority = to.Ptr(armcompute.VirtualMachinePriorityTypesSpot)
        parameters.Properties.EvictionPolicy = to.Ptr(armcompute.VirtualMachineEvictionPolicyTypesDeallocate)
        parameters.Properties.BillingProfile = &armcompute.BillingProfile{
            MaxPrice: to.Ptr(spec.SpotMaxPrice),
        }
    }
    
    // Create VM
    poller, err := p.vmClient.BeginCreateOrUpdate(ctx, p.resourceGroup, spec.Name, parameters, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to create VM: %w", err)
    }
    
    resp, err := poller.PollUntilDone(ctx, nil)
    if err != nil {
        return nil, fmt.Errorf("failed to wait for VM creation: %w", err)
    }
    
    return p.azureVMToVM(&resp.VirtualMachine), nil
}
```

### GCP Provider

```go
// pkg/providers/gcp/provider.go
package gcp

import (
    "context"
    "fmt"
    
    compute "cloud.google.com/go/compute/apiv1"
    "google.golang.org/api/option"
)

type GCPProvider struct {
    client    *compute.InstancesClient
    project   string
    zone      string
}

func NewGCPProvider(ctx context.Context, cfg ProviderConfig) (*GCPProvider, error) {
    client, err := compute.NewInstancesRESTClient(ctx, option.WithCredentialsFile(cfg.CredentialsPath))
    if err != nil {
        return nil, fmt.Errorf("failed to create GCP client: %w", err)
    }
    
    return &GCPProvider{
        client:  client,
        project: cfg.ProjectID,
        zone:    cfg.Zone,
    }, nil
}

func (p *GCPProvider) CreateVM(ctx context.Context, spec provider.VMSpec) (*provider.VM, error) {
    machineType := p.mapMachineType(spec.CPU, spec.Memory)
    
    instance := &computepb.Instance{
        Name:        proto.String(spec.Name),
        MachineType: proto.String(fmt.Sprintf("zones/%s/machineTypes/%s", p.zone, machineType)),
        
        Disks: []*computepb.AttachedDisk{
            {
                Boot:       proto.Bool(true),
                AutoDelete: proto.Bool(true),
                InitializeParams: &computepb.AttachedDiskInitializeParams{
                    SourceImage: proto.String(p.getImageURL(spec.Image)),
                    DiskSizeGb:  proto.Int64(spec.Disk.Size),
                    DiskType:    proto.String(fmt.Sprintf("zones/%s/diskTypes/%s", p.zone, spec.Disk.Type)),
                },
            },
        },
        
        NetworkInterfaces: []*computepb.NetworkInterface{
            {
                Network: proto.String(fmt.Sprintf("projects/%s/global/networks/%s", p.project, spec.Network.Name)),
                AccessConfigs: []*computepb.AccessConfig{
                    {
                        Type: proto.String("ONE_TO_ONE_NAT"),
                        Name: proto.String("External NAT"),
                    },
                },
            },
        },
        
        Metadata: &computepb.Metadata{
            Items: []*computepb.Items{
                {
                    Key:   proto.String("user-data"),
                    Value: proto.String(spec.UserData),
                },
            },
        },
        
        Labels: p.mapLabels(spec.Labels),
        
        Scheduling: &computepb.Scheduling{
            Preemptible:       proto.Bool(spec.Spot),
            OnHostMaintenance: proto.String("MIGRATE"),
            AutomaticRestart:  proto.Bool(!spec.Spot),
        },
    }
    
    // GPU support
    if spec.GPU.Count > 0 {
        instance.GuestAccelerators = []*computepb.AcceleratorConfig{
            {
                AcceleratorType:  proto.String(p.getAcceleratorType(spec.GPU)),
                AcceleratorCount: proto.Int32(int32(spec.GPU.Count)),
            },
        }
    }
    
    req := &computepb.InsertInstanceRequest{
        Project:          p.project,
        Zone:             p.zone,
        InstanceResource: instance,
    }
    
    op, err := p.client.Insert(ctx, req)
    if err != nil {
        return nil, fmt.Errorf("failed to create instance: %w", err)
    }
    
    // Wait for operation to complete
    if err := p.waitForOperation(ctx, op); err != nil {
        return nil, fmt.Errorf("operation failed: %w", err)
    }
    
    // Get created instance
    getReq := &computepb.GetInstanceRequest{
        Project:  p.project,
        Zone:     p.zone,
        Instance: spec.Name,
    }
    
    created, err := p.client.Get(ctx, getReq)
    if err != nil {
        return nil, fmt.Errorf("failed to get created instance: %w", err)
    }
    
    return p.gcpInstanceToVM(created), nil
}
```

## Federation Manager

```go
// pkg/federation/manager.go
package federation

import (
    "context"
    "fmt"
    "sync"
    
    "github.com/novacron/core/provider"
)

type FederationManager struct {
    providers map[string]provider.Provider
    policies  []Policy
    optimizer *CostOptimizer
    mu        sync.RWMutex
}

type Policy struct {
    Name      string
    Priority  int
    Condition PolicyCondition
    Action    PolicyAction
}

type PolicyCondition struct {
    Type     string // cost, performance, compliance, location
    Operator string // <, >, ==, contains
    Value    interface{}
}

type PolicyAction struct {
    Type   string // place, migrate, scale
    Target string // provider, region, zone
}

func NewFederationManager() *FederationManager {
    return &FederationManager{
        providers: make(map[string]provider.Provider),
        optimizer: NewCostOptimizer(),
    }
}

// RegisterProvider adds a cloud provider
func (fm *FederationManager) RegisterProvider(name string, p provider.Provider) {
    fm.mu.Lock()
    defer fm.mu.Unlock()
    fm.providers[name] = p
}

// CreateVM with intelligent placement
func (fm *FederationManager) CreateVM(ctx context.Context, spec provider.VMSpec) (*provider.VM, error) {
    // Select best provider based on policies and optimization
    provider, err := fm.selectProvider(ctx, spec)
    if err != nil {
        return nil, fmt.Errorf("failed to select provider: %w", err)
    }
    
    // Create VM on selected provider
    vm, err := provider.CreateVM(ctx, spec)
    if err != nil {
        // Fallback to next best provider
        fallback, err := fm.selectFallbackProvider(ctx, spec, provider)
        if err != nil {
            return nil, fmt.Errorf("all providers failed: %w", err)
        }
        return fallback.CreateVM(ctx, spec)
    }
    
    // Register VM in federation database
    fm.registerVM(vm)
    
    return vm, nil
}

// selectProvider chooses optimal provider
func (fm *FederationManager) selectProvider(ctx context.Context, spec provider.VMSpec) (provider.Provider, error) {
    candidates := make([]ProviderScore, 0)
    
    fm.mu.RLock()
    defer fm.mu.RUnlock()
    
    for name, p := range fm.providers {
        score := fm.scoreProvider(ctx, p, spec)
        candidates = append(candidates, ProviderScore{
            Provider: p,
            Name:     name,
            Score:    score,
        })
    }
    
    // Sort by score
    sort.Slice(candidates, func(i, j int) bool {
        return candidates[i].Score > candidates[j].Score
    })
    
    if len(candidates) == 0 {
        return nil, fmt.Errorf("no providers available")
    }
    
    return candidates[0].Provider, nil
}

// scoreProvider calculates provider fitness score
func (fm *FederationManager) scoreProvider(ctx context.Context, p provider.Provider, spec provider.VMSpec) float64 {
    score := 100.0
    
    // Cost factor (40% weight)
    cost, _ := p.EstimateCost(ctx, spec)
    costScore := 100 - (cost.Hourly * 10) // Lower cost = higher score
    score = score*0.6 + costScore*0.4
    
    // Performance factor (30% weight)
    perfScore := fm.estimatePerformance(p, spec)
    score = score*0.7 + perfScore*0.3
    
    // Compliance factor (20% weight)
    compScore := fm.checkCompliance(p, spec)
    score = score*0.8 + compScore*0.2
    
    // Location factor (10% weight)
    locScore := fm.scoreLocation(p, spec)
    score = score*0.9 + locScore*0.1
    
    // Apply policy adjustments
    for _, policy := range fm.policies {
        if fm.matchesPolicy(policy, p, spec) {
            score *= policy.ScoreMultiplier
        }
    }
    
    return score
}

// MigrateVM across providers
func (fm *FederationManager) MigrateVM(ctx context.Context, vmID string, targetProvider string) error {
    // Get current VM details
    vm, currentProvider, err := fm.findVM(vmID)
    if err != nil {
        return fmt.Errorf("VM not found: %w", err)
    }
    
    target := fm.providers[targetProvider]
    if target == nil {
        return fmt.Errorf("target provider not found")
    }
    
    // Export VM from source
    exportData, err := currentProvider.ExportVM(ctx, vmID)
    if err != nil {
        return fmt.Errorf("failed to export VM: %w", err)
    }
    
    // Import to target
    newVM, err := target.ImportVM(ctx, exportData)
    if err != nil {
        return fmt.Errorf("failed to import VM: %w", err)
    }
    
    // Update DNS/Load balancer
    if err := fm.updateNetworking(vm, newVM); err != nil {
        return fmt.Errorf("failed to update networking: %w", err)
    }
    
    // Delete source VM after verification
    go func() {
        time.Sleep(5 * time.Minute) // Grace period
        currentProvider.DeleteVM(context.Background(), vmID)
    }()
    
    return nil
}

// OptimizePlacement rebalances VMs for cost/performance
func (fm *FederationManager) OptimizePlacement(ctx context.Context) (*OptimizationPlan, error) {
    vms := fm.listAllVMs()
    
    plan := &OptimizationPlan{
        Migrations: make([]Migration, 0),
        EstimatedSavings: 0,
    }
    
    for _, vm := range vms {
        // Calculate optimal placement
        optimalProvider := fm.findOptimalProvider(ctx, vm)
        
        if optimalProvider != vm.CurrentProvider {
            migration := Migration{
                VM:           vm,
                Source:       vm.CurrentProvider,
                Destination:  optimalProvider,
                EstimatedSavings: fm.calculateSavings(vm, optimalProvider),
            }
            plan.Migrations = append(plan.Migrations, migration)
            plan.EstimatedSavings += migration.EstimatedSavings
        }
    }
    
    return plan, nil
}
```

## Cost Optimization Engine

```go
// pkg/federation/optimizer.go
package federation

type CostOptimizer struct {
    pricing     map[string]PricingModel
    predictions *MLPredictor
}

type PricingModel struct {
    OnDemand map[string]float64
    Spot     map[string]SpotPricing
    Reserved map[string]ReservedPricing
}

func (co *CostOptimizer) RecommendInstanceType(requirements Requirements) []Recommendation {
    recommendations := make([]Recommendation, 0)
    
    for provider, pricing := range co.pricing {
        // Find matching instance types
        instances := co.findMatchingInstances(provider, requirements)
        
        for _, instance := range instances {
            rec := Recommendation{
                Provider:     provider,
                InstanceType: instance,
                Cost:         pricing.OnDemand[instance],
                Savings:      0,
            }
            
            // Check for spot availability
            if spotPrice, available := pricing.Spot[instance]; available {
                rec.SpotOption = &SpotOption{
                    Price:       spotPrice.Current,
                    Savings:     pricing.OnDemand[instance] - spotPrice.Current,
                    Probability: spotPrice.InterruptionProbability,
                }
            }
            
            // Check for reserved instance options
            if reserved, available := pricing.Reserved[instance]; available {
                rec.ReservedOption = &ReservedOption{
                    Term:        reserved.Term,
                    HourlyPrice: reserved.HourlyPrice,
                    Savings:     pricing.OnDemand[instance] - reserved.HourlyPrice,
                }
            }
            
            recommendations = append(recommendations, rec)
        }
    }
    
    // Sort by cost-effectiveness
    sort.Slice(recommendations, func(i, j int) bool {
        return recommendations[i].Cost < recommendations[j].Cost
    })
    
    return recommendations
}

// PredictCost uses ML to forecast costs
func (co *CostOptimizer) PredictCost(vm *VM, duration time.Duration) (*CostPrediction, error) {
    // Get historical usage data
    usage := co.getHistoricalUsage(vm)
    
    // Predict future usage patterns
    prediction := co.predictions.Predict(usage, duration)
    
    // Calculate costs across providers
    costs := make(map[string]float64)
    for provider, pricing := range co.pricing {
        instanceType := co.mapInstanceType(provider, vm.Spec)
        hourlyRate := pricing.OnDemand[instanceType]
        
        // Apply predicted usage patterns
        totalCost := 0.0
        for _, hour := range prediction.HourlyUsage {
            if hour.Active {
                totalCost += hourlyRate
            }
        }
        
        costs[provider] = totalCost
    }
    
    return &CostPrediction{
        Duration:      duration,
        ProviderCosts: costs,
        Confidence:    prediction.Confidence,
        OptimalProvider: co.findLowestCost(costs),
    }, nil
}
```

## Networking Across Clouds

```go
// pkg/federation/network.go
package federation

type CrossCloudNetwork struct {
    overlay   *OverlayNetwork
    vpns      map[string]*VPNConnection
    peerings  map[string]*Peering
}

// EstablishConnectivity sets up network between clouds
func (n *CrossCloudNetwork) EstablishConnectivity(source, target CloudProvider) error {
    // Option 1: VPN connection
    if n.supportsVPN(source, target) {
        vpn, err := n.createVPN(source, target)
        if err != nil {
            return fmt.Errorf("failed to create VPN: %w", err)
        }
        n.vpns[fmt.Sprintf("%s-%s", source.Name(), target.Name())] = vpn
        return nil
    }
    
    // Option 2: Direct peering
    if n.supportsPeering(source, target) {
        peering, err := n.createPeering(source, target)
        if err != nil {
            return fmt.Errorf("failed to create peering: %w", err)
        }
        n.peerings[fmt.Sprintf("%s-%s", source.Name(), target.Name())] = peering
        return nil
    }
    
    // Option 3: Overlay network
    return n.overlay.Connect(source, target)
}

// OverlayNetwork using WireGuard
type OverlayNetwork struct {
    nodes map[string]*WireGuardNode
}

func (o *OverlayNetwork) Connect(source, target CloudProvider) error {
    // Deploy WireGuard nodes
    sourceNode, err := o.deployNode(source)
    if err != nil {
        return fmt.Errorf("failed to deploy source node: %w", err)
    }
    
    targetNode, err := o.deployNode(target)
    if err != nil {
        return fmt.Errorf("failed to deploy target node: %w", err)
    }
    
    // Exchange keys and establish tunnel
    if err := o.establishTunnel(sourceNode, targetNode); err != nil {
        return fmt.Errorf("failed to establish tunnel: %w", err)
    }
    
    // Configure routing
    if err := o.configureRouting(sourceNode, targetNode); err != nil {
        return fmt.Errorf("failed to configure routing: %w", err)
    }
    
    return nil
}
```

## Configuration Examples

### Federation Configuration

```yaml
# federation.yaml
apiVersion: novacron.io/v1alpha1
kind: FederationConfig
metadata:
  name: production-federation
spec:
  providers:
    - name: aws-us-east
      type: aws
      config:
        region: us-east-1
        credentials:
          secret: aws-credentials
        defaults:
          instanceFamily: t3
          spotEnabled: true
          
    - name: azure-west
      type: azure
      config:
        location: westus2
        subscription: ${AZURE_SUBSCRIPTION_ID}
        resourceGroup: novacron-prod
        
    - name: gcp-central
      type: gcp
      config:
        project: novacron-prod
        region: us-central1
        zone: us-central1-a
        
    - name: on-prem
      type: openstack
      config:
        authUrl: https://openstack.internal:5000/v3
        projectId: novacron
        
  policies:
    - name: data-residency
      priority: 100
      condition:
        type: label
        key: data-residency
        value: us
      action:
        type: restrict
        providers: [aws-us-east, on-prem]
        
    - name: cost-optimization
      priority: 50
      condition:
        type: workload
        value: batch
      action:
        type: prefer
        providers: [spot-instances]
        
    - name: high-availability
      priority: 90
      condition:
        type: label
        key: tier
        value: critical
      action:
        type: distribute
        minProviders: 2
        
  networking:
    mode: overlay  # overlay, vpn, peering
    overlay:
      type: wireguard
      mesh: true
      encryption: chacha20poly1305
      
  optimization:
    enabled: true
    schedule: "0 2 * * *"  # Daily at 2 AM
    strategies:
      - cost
      - performance
      - carbon-footprint
```

### VM Placement Example

```yaml
apiVersion: novacron.io/v1alpha1
kind: VirtualMachine
metadata:
  name: multi-cloud-app
spec:
  federation:
    enabled: true
    placement:
      strategy: cost-optimized  # cost-optimized, performance, balanced
      constraints:
        - type: provider
          operator: in
          values: [aws, azure]  # Only AWS or Azure
        - type: region
          operator: near
          value: us-east-1
          maxLatency: 10ms
        - type: cost
          operator: "<"
          value: 0.10  # Max $0.10/hour
          
    failover:
      enabled: true
      priority:
        - aws-us-east
        - azure-west
        - gcp-central
        
  template:
    spec:
      resources:
        cpu: 4
        memory: 8Gi
        disk: 100Gi
```

## Monitoring & Observability

```go
// pkg/federation/metrics.go
package federation

var (
    VMsByProvider = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "novacron_federation_vms_total",
            Help: "Total VMs by provider",
        },
        []string{"provider", "region"},
    )
    
    CrossCloudMigrations = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "novacron_federation_migrations_total",
            Help: "Total cross-cloud migrations",
        },
        []string{"source", "destination", "status"},
    )
    
    ProviderCosts = prometheus.NewGaugeVec(
        prometheus.GaugeOpts{
            Name: "novacron_federation_provider_costs_dollars",
            Help: "Current costs by provider",
        },
        []string{"provider", "type"}, // type: compute, storage, network
    )
    
    NetworkLatency = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name: "novacron_federation_network_latency_ms",
            Help: "Cross-cloud network latency",
            Buckets: prometheus.ExponentialBuckets(1, 2, 10),
        },
        []string{"source", "destination"},
    )
)
```