// Package terraform provides Terraform provider for NovaCron resources
package terraform

import (
	"context"
	"fmt"
	"time"

	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/plugin"
	"github.com/sirupsen/logrus"
)

// Provider returns the NovaCron Terraform provider
func Provider() *schema.Provider {
	return &schema.Provider{
		Schema: map[string]*schema.Schema{
			"endpoint": {
				Type:        schema.TypeString,
				Optional:    true,
				DefaultFunc: schema.EnvDefaultFunc("NOVACRON_ENDPOINT", "http://localhost:8080"),
				Description: "NovaCron API endpoint",
			},
			"api_key": {
				Type:        schema.TypeString,
				Optional:    true,
				Sensitive:   true,
				DefaultFunc: schema.EnvDefaultFunc("NOVACRON_API_KEY", ""),
				Description: "NovaCron API key",
			},
			"timeout": {
				Type:        schema.TypeInt,
				Optional:    true,
				Default:     300,
				Description: "API request timeout in seconds",
			},
		},
		ResourcesMap: map[string]*schema.Resource{
			"novacron_vm":      resourceVM(),
			"novacron_network": resourceNetwork(),
			"novacron_storage": resourceStorage(),
			"novacron_policy":  resourcePolicy(),
		},
		DataSourcesMap: map[string]*schema.Resource{
			"novacron_vm":       dataSourceVM(),
			"novacron_network":  dataSourceNetwork(),
			"novacron_template": dataSourceTemplate(),
		},
		ConfigureContextFunc: providerConfigure,
	}
}

// ProviderConfig holds the provider configuration
type ProviderConfig struct {
	Endpoint string
	APIKey   string
	Timeout  time.Duration
	Logger   *logrus.Logger
}

func providerConfigure(ctx context.Context, d *schema.ResourceData) (interface{}, diag.Diagnostics) {
	var diags diag.Diagnostics

	endpoint := d.Get("endpoint").(string)
	apiKey := d.Get("api_key").(string)
	timeout := d.Get("timeout").(int)

	if apiKey == "" {
		diags = append(diags, diag.Diagnostic{
			Severity: diag.Warning,
			Summary:  "Missing API key",
			Detail:   "API key should be set via api_key or NOVACRON_API_KEY environment variable",
		})
	}

	logger := logrus.New()
	logger.SetLevel(logrus.InfoLevel)

	config := &ProviderConfig{
		Endpoint: endpoint,
		APIKey:   apiKey,
		Timeout:  time.Duration(timeout) * time.Second,
		Logger:   logger,
	}

	logger.WithFields(logrus.Fields{
		"endpoint": endpoint,
		"timeout":  timeout,
	}).Info("NovaCron provider configured")

	return config, diags
}

// resourceVM defines the VM resource
func resourceVM() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceVMCreate,
		ReadContext:   resourceVMRead,
		UpdateContext: resourceVMUpdate,
		DeleteContext: resourceVMDelete,
		Importer: &schema.ResourceImporter{
			StateContext: resourceVMImport,
		},
		Timeouts: &schema.ResourceTimeout{
			Create: schema.DefaultTimeout(30 * time.Minute),
			Update: schema.DefaultTimeout(30 * time.Minute),
			Delete: schema.DefaultTimeout(10 * time.Minute),
		},
		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "VM name",
			},
			"cpu_cores": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Number of CPU cores",
			},
			"memory_gb": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Memory in GB",
			},
			"disk_gb": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Disk size in GB",
			},
			"network_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Network ID",
			},
			"template_id": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "VM template ID",
			},
			"metadata": {
				Type:        schema.TypeMap,
				Optional:    true,
				Description: "VM metadata",
				Elem: &schema.Schema{
					Type: schema.TypeString,
				},
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "VM status",
			},
			"ip_address": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "VM IP address",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Creation timestamp",
			},
		},
	}
}

func resourceVMCreate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	vmSpec := map[string]interface{}{
		"name":      d.Get("name").(string),
		"cpu_cores": d.Get("cpu_cores").(int),
		"memory_gb": d.Get("memory_gb").(int),
		"disk_gb":   d.Get("disk_gb").(int),
	}

	if networkID, ok := d.GetOk("network_id"); ok {
		vmSpec["network_id"] = networkID.(string)
	}

	if templateID, ok := d.GetOk("template_id"); ok {
		vmSpec["template_id"] = templateID.(string)
	}

	if metadata, ok := d.GetOk("metadata"); ok {
		vmSpec["metadata"] = metadata
	}

	config.Logger.WithFields(logrus.Fields{
		"name":      vmSpec["name"],
		"cpu_cores": vmSpec["cpu_cores"],
	}).Info("Creating VM")

	// Call NovaCron API to create VM
	vmID, err := createVM(ctx, config, vmSpec)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(vmID)

	// Wait for VM to be ready
	if err := waitForVMReady(ctx, config, vmID, d.Timeout(schema.TimeoutCreate)); err != nil {
		return diag.FromErr(err)
	}

	return append(diags, resourceVMRead(ctx, d, meta)...)
}

func resourceVMRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	vmID := d.Id()

	config.Logger.WithField("vm_id", vmID).Debug("Reading VM")

	// Call NovaCron API to get VM details
	vm, err := getVM(ctx, config, vmID)
	if err != nil {
		if isNotFoundError(err) {
			d.SetId("")
			return diags
		}
		return diag.FromErr(err)
	}

	// Set computed attributes
	d.Set("status", vm["status"])
	d.Set("ip_address", vm["ip_address"])
	d.Set("created_at", vm["created_at"])

	return diags
}

func resourceVMUpdate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)

	vmID := d.Id()

	config.Logger.WithField("vm_id", vmID).Info("Updating VM")

	updates := make(map[string]interface{})

	if d.HasChange("cpu_cores") {
		updates["cpu_cores"] = d.Get("cpu_cores").(int)
	}

	if d.HasChange("memory_gb") {
		updates["memory_gb"] = d.Get("memory_gb").(int)
	}

	if d.HasChange("disk_gb") {
		updates["disk_gb"] = d.Get("disk_gb").(int)
	}

	if d.HasChange("metadata") {
		updates["metadata"] = d.Get("metadata")
	}

	if len(updates) > 0 {
		if err := updateVM(ctx, config, vmID, updates); err != nil {
			return diag.FromErr(err)
		}

		// Wait for update to complete
		if err := waitForVMReady(ctx, config, vmID, d.Timeout(schema.TimeoutUpdate)); err != nil {
			return diag.FromErr(err)
		}
	}

	return resourceVMRead(ctx, d, meta)
}

func resourceVMDelete(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	vmID := d.Id()

	config.Logger.WithField("vm_id", vmID).Info("Deleting VM")

	if err := deleteVM(ctx, config, vmID); err != nil {
		return diag.FromErr(err)
	}

	// Wait for deletion to complete
	if err := waitForVMDeleted(ctx, config, vmID, d.Timeout(schema.TimeoutDelete)); err != nil {
		return diag.FromErr(err)
	}

	d.SetId("")

	return diags
}

func resourceVMImport(ctx context.Context, d *schema.ResourceData, meta interface{}) ([]*schema.ResourceData, error) {
	config := meta.(*ProviderConfig)

	vmID := d.Id()

	config.Logger.WithField("vm_id", vmID).Info("Importing VM")

	vm, err := getVM(ctx, config, vmID)
	if err != nil {
		return nil, err
	}

	d.Set("name", vm["name"])
	d.Set("cpu_cores", vm["cpu_cores"])
	d.Set("memory_gb", vm["memory_gb"])
	d.Set("disk_gb", vm["disk_gb"])
	d.Set("network_id", vm["network_id"])
	d.Set("status", vm["status"])
	d.Set("ip_address", vm["ip_address"])
	d.Set("created_at", vm["created_at"])

	return []*schema.ResourceData{d}, nil
}

// resourceNetwork defines the network resource
func resourceNetwork() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceNetworkCreate,
		ReadContext:   resourceNetworkRead,
		UpdateContext: resourceNetworkUpdate,
		DeleteContext: resourceNetworkDelete,
		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Network name",
			},
			"cidr": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Network CIDR block",
			},
			"gateway": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "Gateway IP",
			},
			"dns_servers": {
				Type:     schema.TypeList,
				Optional: true,
				Elem: &schema.Schema{
					Type: schema.TypeString,
				},
				Description: "DNS servers",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Network status",
			},
		},
	}
}

func resourceNetworkCreate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)

	networkSpec := map[string]interface{}{
		"name": d.Get("name").(string),
		"cidr": d.Get("cidr").(string),
	}

	if gateway, ok := d.GetOk("gateway"); ok {
		networkSpec["gateway"] = gateway.(string)
	}

	if dnsServers, ok := d.GetOk("dns_servers"); ok {
		networkSpec["dns_servers"] = dnsServers
	}

	config.Logger.Info("Creating network")

	networkID, err := createNetwork(ctx, config, networkSpec)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(networkID)

	return resourceNetworkRead(ctx, d, meta)
}

func resourceNetworkRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	networkID := d.Id()

	network, err := getNetwork(ctx, config, networkID)
	if err != nil {
		if isNotFoundError(err) {
			d.SetId("")
			return diags
		}
		return diag.FromErr(err)
	}

	d.Set("status", network["status"])

	return diags
}

func resourceNetworkUpdate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)

	networkID := d.Id()

	updates := make(map[string]interface{})

	if d.HasChange("name") {
		updates["name"] = d.Get("name").(string)
	}

	if d.HasChange("dns_servers") {
		updates["dns_servers"] = d.Get("dns_servers")
	}

	if len(updates) > 0 {
		if err := updateNetwork(ctx, config, networkID, updates); err != nil {
			return diag.FromErr(err)
		}
	}

	return resourceNetworkRead(ctx, d, meta)
}

func resourceNetworkDelete(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	networkID := d.Id()

	config.Logger.Info("Deleting network")

	if err := deleteNetwork(ctx, config, networkID); err != nil {
		return diag.FromErr(err)
	}

	d.SetId("")

	return diags
}

// resourceStorage defines the storage resource
func resourceStorage() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceStorageCreate,
		ReadContext:   resourceStorageRead,
		UpdateContext: resourceStorageUpdate,
		DeleteContext: resourceStorageDelete,
		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Storage volume name",
			},
			"size_gb": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Volume size in GB",
			},
			"type": {
				Type:        schema.TypeString,
				Optional:    true,
				Default:     "ssd",
				Description: "Storage type (ssd, hdd)",
			},
			"attached_to": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "VM ID to attach to",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Storage status",
			},
		},
	}
}

func resourceStorageCreate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)

	storageSpec := map[string]interface{}{
		"name":    d.Get("name").(string),
		"size_gb": d.Get("size_gb").(int),
		"type":    d.Get("type").(string),
	}

	config.Logger.Info("Creating storage volume")

	storageID, err := createStorage(ctx, config, storageSpec)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(storageID)

	// Attach if VM specified
	if attachedTo, ok := d.GetOk("attached_to"); ok {
		if err := attachStorage(ctx, config, storageID, attachedTo.(string)); err != nil {
			return diag.FromErr(err)
		}
	}

	return resourceStorageRead(ctx, d, meta)
}

func resourceStorageRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	storageID := d.Id()

	storage, err := getStorage(ctx, config, storageID)
	if err != nil {
		if isNotFoundError(err) {
			d.SetId("")
			return diags
		}
		return diag.FromErr(err)
	}

	d.Set("status", storage["status"])
	d.Set("attached_to", storage["attached_to"])

	return diags
}

func resourceStorageUpdate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)

	storageID := d.Id()

	if d.HasChange("size_gb") {
		newSize := d.Get("size_gb").(int)
		if err := resizeStorage(ctx, config, storageID, newSize); err != nil {
			return diag.FromErr(err)
		}
	}

	if d.HasChange("attached_to") {
		oldAttached, newAttached := d.GetChange("attached_to")

		if oldAttached != "" {
			if err := detachStorage(ctx, config, storageID); err != nil {
				return diag.FromErr(err)
			}
		}

		if newAttached != "" {
			if err := attachStorage(ctx, config, storageID, newAttached.(string)); err != nil {
				return diag.FromErr(err)
			}
		}
	}

	return resourceStorageRead(ctx, d, meta)
}

func resourceStorageDelete(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	storageID := d.Id()

	config.Logger.Info("Deleting storage volume")

	// Detach if attached
	if attachedTo, ok := d.GetOk("attached_to"); ok && attachedTo != "" {
		if err := detachStorage(ctx, config, storageID); err != nil {
			return diag.FromErr(err)
		}
	}

	if err := deleteStorage(ctx, config, storageID); err != nil {
		return diag.FromErr(err)
	}

	d.SetId("")

	return diags
}

// resourcePolicy defines the policy resource
func resourcePolicy() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourcePolicyCreate,
		ReadContext:   resourcePolicyRead,
		UpdateContext: resourcePolicyUpdate,
		DeleteContext: resourcePolicyDelete,
		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Policy name",
			},
			"type": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Policy type",
			},
			"rule": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Rego policy rule",
			},
			"enabled": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     true,
				Description: "Policy enabled",
			},
			"enforcement_mode": {
				Type:        schema.TypeString,
				Optional:    true,
				Default:     "block",
				Description: "Enforcement mode (block, warn, audit)",
			},
		},
	}
}

func resourcePolicyCreate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)

	policySpec := map[string]interface{}{
		"name":             d.Get("name").(string),
		"type":             d.Get("type").(string),
		"rule":             d.Get("rule").(string),
		"enabled":          d.Get("enabled").(bool),
		"enforcement_mode": d.Get("enforcement_mode").(string),
	}

	config.Logger.Info("Creating policy")

	policyID, err := createPolicy(ctx, config, policySpec)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(policyID)

	return resourcePolicyRead(ctx, d, meta)
}

func resourcePolicyRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	policyID := d.Id()

	policy, err := getPolicy(ctx, config, policyID)
	if err != nil {
		if isNotFoundError(err) {
			d.SetId("")
			return diags
		}
		return diag.FromErr(err)
	}

	d.Set("enabled", policy["enabled"])

	return diags
}

func resourcePolicyUpdate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)

	policyID := d.Id()

	updates := make(map[string]interface{})

	if d.HasChange("rule") {
		updates["rule"] = d.Get("rule").(string)
	}

	if d.HasChange("enabled") {
		updates["enabled"] = d.Get("enabled").(bool)
	}

	if d.HasChange("enforcement_mode") {
		updates["enforcement_mode"] = d.Get("enforcement_mode").(string)
	}

	if len(updates) > 0 {
		if err := updatePolicy(ctx, config, policyID, updates); err != nil {
			return diag.FromErr(err)
		}
	}

	return resourcePolicyRead(ctx, d, meta)
}

func resourcePolicyDelete(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	policyID := d.Id()

	config.Logger.Info("Deleting policy")

	if err := deletePolicy(ctx, config, policyID); err != nil {
		return diag.FromErr(err)
	}

	d.SetId("")

	return diags
}

// Data source implementations
func dataSourceVM() *schema.Resource {
	return &schema.Resource{
		ReadContext: dataSourceVMRead,
		Schema: map[string]*schema.Schema{
			"id": {
				Type:        schema.TypeString,
				Optional:    true,
				Computed:    true,
				Description: "VM ID",
			},
			"name": {
				Type:        schema.TypeString,
				Optional:    true,
				Computed:    true,
				Description: "VM name",
			},
			"cpu_cores": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Number of CPU cores",
			},
			"memory_gb": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Memory in GB",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "VM status",
			},
			"ip_address": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "VM IP address",
			},
		},
	}
}

func dataSourceVMRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	vmID := d.Get("id").(string)
	vmName := d.Get("name").(string)

	var vm map[string]interface{}
	var err error

	if vmID != "" {
		vm, err = getVM(ctx, config, vmID)
	} else if vmName != "" {
		vm, err = getVMByName(ctx, config, vmName)
	} else {
		return diag.Errorf("Either id or name must be specified")
	}

	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(vm["id"].(string))
	d.Set("name", vm["name"])
	d.Set("cpu_cores", vm["cpu_cores"])
	d.Set("memory_gb", vm["memory_gb"])
	d.Set("status", vm["status"])
	d.Set("ip_address", vm["ip_address"])

	return diags
}

func dataSourceNetwork() *schema.Resource {
	return &schema.Resource{
		ReadContext: dataSourceNetworkRead,
		Schema: map[string]*schema.Schema{
			"id": {
				Type:        schema.TypeString,
				Optional:    true,
				Computed:    true,
				Description: "Network ID",
			},
			"name": {
				Type:        schema.TypeString,
				Optional:    true,
				Computed:    true,
				Description: "Network name",
			},
			"cidr": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Network CIDR",
			},
			"status": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Network status",
			},
		},
	}
}

func dataSourceNetworkRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	networkID := d.Get("id").(string)

	network, err := getNetwork(ctx, config, networkID)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(network["id"].(string))
	d.Set("name", network["name"])
	d.Set("cidr", network["cidr"])
	d.Set("status", network["status"])

	return diags
}

func dataSourceTemplate() *schema.Resource {
	return &schema.Resource{
		ReadContext: dataSourceTemplateRead,
		Schema: map[string]*schema.Schema{
			"id": {
				Type:        schema.TypeString,
				Optional:    true,
				Computed:    true,
				Description: "Template ID",
			},
			"name": {
				Type:        schema.TypeString,
				Optional:    true,
				Computed:    true,
				Description: "Template name",
			},
			"os_type": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "OS type",
			},
			"version": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Template version",
			},
		},
	}
}

func dataSourceTemplateRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	config := meta.(*ProviderConfig)
	var diags diag.Diagnostics

	templateID := d.Get("id").(string)

	template, err := getTemplate(ctx, config, templateID)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(template["id"].(string))
	d.Set("name", template["name"])
	d.Set("os_type", template["os_type"])
	d.Set("version", template["version"])

	return diags
}

// Helper functions (these would call actual NovaCron API)
func createVM(ctx context.Context, config *ProviderConfig, spec map[string]interface{}) (string, error) {
	// Placeholder - would make actual API call
	return "vm-" + fmt.Sprintf("%d", time.Now().Unix()), nil
}

func getVM(ctx context.Context, config *ProviderConfig, id string) (map[string]interface{}, error) {
	// Placeholder
	return map[string]interface{}{
		"id":         id,
		"name":       "test-vm",
		"cpu_cores":  4,
		"memory_gb":  8,
		"status":     "running",
		"ip_address": "192.168.1.10",
		"created_at": time.Now().Format(time.RFC3339),
	}, nil
}

func getVMByName(ctx context.Context, config *ProviderConfig, name string) (map[string]interface{}, error) {
	return getVM(ctx, config, "vm-"+name)
}

func updateVM(ctx context.Context, config *ProviderConfig, id string, updates map[string]interface{}) error {
	return nil
}

func deleteVM(ctx context.Context, config *ProviderConfig, id string) error {
	return nil
}

func waitForVMReady(ctx context.Context, config *ProviderConfig, id string, timeout time.Duration) error {
	return nil
}

func waitForVMDeleted(ctx context.Context, config *ProviderConfig, id string, timeout time.Duration) error {
	return nil
}

func createNetwork(ctx context.Context, config *ProviderConfig, spec map[string]interface{}) (string, error) {
	return "net-" + fmt.Sprintf("%d", time.Now().Unix()), nil
}

func getNetwork(ctx context.Context, config *ProviderConfig, id string) (map[string]interface{}, error) {
	return map[string]interface{}{
		"id":     id,
		"name":   "test-network",
		"cidr":   "10.0.0.0/24",
		"status": "active",
	}, nil
}

func updateNetwork(ctx context.Context, config *ProviderConfig, id string, updates map[string]interface{}) error {
	return nil
}

func deleteNetwork(ctx context.Context, config *ProviderConfig, id string) error {
	return nil
}

func createStorage(ctx context.Context, config *ProviderConfig, spec map[string]interface{}) (string, error) {
	return "storage-" + fmt.Sprintf("%d", time.Now().Unix()), nil
}

func getStorage(ctx context.Context, config *ProviderConfig, id string) (map[string]interface{}, error) {
	return map[string]interface{}{
		"id":          id,
		"name":        "test-storage",
		"size_gb":     100,
		"type":        "ssd",
		"status":      "available",
		"attached_to": "",
	}, nil
}

func resizeStorage(ctx context.Context, config *ProviderConfig, id string, newSize int) error {
	return nil
}

func attachStorage(ctx context.Context, config *ProviderConfig, storageID, vmID string) error {
	return nil
}

func detachStorage(ctx context.Context, config *ProviderConfig, storageID string) error {
	return nil
}

func deleteStorage(ctx context.Context, config *ProviderConfig, id string) error {
	return nil
}

func createPolicy(ctx context.Context, config *ProviderConfig, spec map[string]interface{}) (string, error) {
	return "policy-" + fmt.Sprintf("%d", time.Now().Unix()), nil
}

func getPolicy(ctx context.Context, config *ProviderConfig, id string) (map[string]interface{}, error) {
	return map[string]interface{}{
		"id":              id,
		"name":            "test-policy",
		"type":            "access",
		"enabled":         true,
		"enforcement_mode": "block",
	}, nil
}

func updatePolicy(ctx context.Context, config *ProviderConfig, id string, updates map[string]interface{}) error {
	return nil
}

func deletePolicy(ctx context.Context, config *ProviderConfig, id string) error {
	return nil
}

func getTemplate(ctx context.Context, config *ProviderConfig, id string) (map[string]interface{}, error) {
	return map[string]interface{}{
		"id":      id,
		"name":    "ubuntu-22.04",
		"os_type": "linux",
		"version": "22.04",
	}, nil
}

func isNotFoundError(err error) bool {
	// Placeholder - would check for 404 error
	return false
}

// Serve starts the Terraform provider plugin
func Serve() {
	plugin.Serve(&plugin.ServeOpts{
		ProviderFunc: Provider,
	})
}
