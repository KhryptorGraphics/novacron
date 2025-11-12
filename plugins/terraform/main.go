package main

import (
	"context"
	"fmt"

	"github.com/hashicorp/terraform-plugin-sdk/v2/diag"
	"github.com/hashicorp/terraform-plugin-sdk/v2/helper/schema"
	"github.com/hashicorp/terraform-plugin-sdk/v2/plugin"
	"github.com/novacron/dwcp-sdk-go"
)

const (
	providerVersion = "3.0.0"
)

func main() {
	plugin.Serve(&plugin.ServeOpts{
		ProviderFunc: Provider,
	})
}

// Provider returns the NovaCron DWCP Terraform provider
func Provider() *schema.Provider {
	return &schema.Provider{
		Schema: map[string]*schema.Schema{
			"address": {
				Type:        schema.TypeString,
				Required:    true,
				DefaultFunc: schema.EnvDefaultFunc("DWCP_ADDRESS", "localhost"),
				Description: "DWCP server address",
			},
			"port": {
				Type:        schema.TypeInt,
				Optional:    true,
				DefaultFunc: schema.EnvDefaultFunc("DWCP_PORT", 9000),
				Description: "DWCP server port",
			},
			"api_key": {
				Type:        schema.TypeString,
				Required:    true,
				Sensitive:   true,
				DefaultFunc: schema.EnvDefaultFunc("DWCP_API_KEY", nil),
				Description: "API key for authentication",
			},
			"tls_enabled": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     true,
				Description: "Enable TLS connection",
			},
		},

		ResourcesMap: map[string]*schema.Resource{
			"dwcp_vm":       resourceVM(),
			"dwcp_snapshot": resourceSnapshot(),
			"dwcp_network":  resourceNetwork(),
		},

		DataSourcesMap: map[string]*schema.Resource{
			"dwcp_vm":       dataSourceVM(),
			"dwcp_node":     dataSourceNode(),
			"dwcp_template": dataSourceTemplate(),
		},

		ConfigureContextFunc: providerConfigure,
	}
}

func providerConfigure(ctx context.Context, d *schema.ResourceData) (interface{}, diag.Diagnostics) {
	var diags diag.Diagnostics

	config := dwcp.DefaultConfig()
	config.Address = d.Get("address").(string)
	config.Port = d.Get("port").(int)
	config.APIKey = d.Get("api_key").(string)
	config.TLSEnabled = d.Get("tls_enabled").(bool)

	client, err := dwcp.NewClient(config)
	if err != nil {
		return nil, diag.FromErr(err)
	}

	if err := client.Connect(ctx); err != nil {
		return nil, diag.FromErr(err)
	}

	return client, diags
}

func resourceVM() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceVMCreate,
		ReadContext:   resourceVMRead,
		UpdateContext: resourceVMUpdate,
		DeleteContext: resourceVMDelete,

		Schema: map[string]*schema.Schema{
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "VM name",
			},
			"memory": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Memory in bytes",
			},
			"cpus": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Number of CPUs",
			},
			"disk": {
				Type:        schema.TypeInt,
				Required:    true,
				Description: "Disk size in bytes",
			},
			"image": {
				Type:        schema.TypeString,
				Required:    true,
				Description: "Base image",
			},
			"state": {
				Type:        schema.TypeString,
				Optional:    true,
				Default:     "running",
				Description: "Desired VM state (running, stopped)",
			},
			"node": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Node where VM is running",
			},
			"network": {
				Type:     schema.TypeList,
				Optional: true,
				MaxItems: 1,
				Elem: &schema.Resource{
					Schema: map[string]*schema.Schema{
						"mode": {
							Type:        schema.TypeString,
							Optional:    true,
							Default:     "bridge",
							Description: "Network mode",
						},
						"interfaces": {
							Type:     schema.TypeList,
							Optional: true,
							Elem: &schema.Resource{
								Schema: map[string]*schema.Schema{
									"name": {
										Type:     schema.TypeString,
										Required: true,
									},
									"type": {
										Type:     schema.TypeString,
										Optional: true,
										Default:  "virtio",
									},
									"mac": {
										Type:     schema.TypeString,
										Optional: true,
										Computed: true,
									},
									"bridge": {
										Type:     schema.TypeString,
										Optional: true,
									},
									"ip_address": {
										Type:     schema.TypeString,
										Optional: true,
									},
								},
							},
						},
					},
				},
			},
			"labels": {
				Type:        schema.TypeMap,
				Optional:    true,
				Description: "Labels for the VM",
				Elem: &schema.Schema{
					Type: schema.TypeString,
				},
			},
			"enable_gpu": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     false,
				Description: "Enable GPU passthrough",
			},
			"gpu_type": {
				Type:        schema.TypeString,
				Optional:    true,
				Description: "GPU type (nvidia, amd, intel)",
			},
			"enable_tpm": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     false,
				Description: "Enable TPM",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Creation timestamp",
			},
			"updated_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Last update timestamp",
			},
		},

		Importer: &schema.ResourceImporter{
			StateContext: schema.ImportStatePassthroughContext,
		},
	}
}

func resourceVMCreate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	client := meta.(*dwcp.Client)

	config := dwcp.VMConfig{
		Name:   d.Get("name").(string),
		Memory: uint64(d.Get("memory").(int)),
		CPUs:   uint32(d.Get("cpus").(int)),
		Disk:   uint64(d.Get("disk").(int)),
		Image:  d.Get("image").(string),
		Network: dwcp.NetworkConfig{
			Mode: "bridge",
			Interfaces: []dwcp.NetIf{
				{
					Name: "eth0",
					Type: "virtio",
				},
			},
		},
	}

	// Add labels
	if v, ok := d.GetOk("labels"); ok {
		config.Labels = make(map[string]string)
		for key, value := range v.(map[string]interface{}) {
			config.Labels[key] = value.(string)
		}
	}

	// Advanced features
	if v, ok := d.GetOk("enable_gpu"); ok {
		config.EnableGPU = v.(bool)
	}
	if v, ok := d.GetOk("gpu_type"); ok {
		config.GPUType = v.(string)
	}
	if v, ok := d.GetOk("enable_tpm"); ok {
		config.EnableTPM = v.(bool)
	}

	// Create VM
	vmClient := client.VM()
	vm, err := vmClient.Create(ctx, config)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(vm.ID)

	// Start VM if desired state is running
	if d.Get("state").(string) == "running" {
		if err := vmClient.Start(ctx, vm.ID); err != nil {
			return diag.FromErr(err)
		}
	}

	return resourceVMRead(ctx, d, meta)
}

func resourceVMRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	var diags diag.Diagnostics
	client := meta.(*dwcp.Client)

	vmClient := client.VM()
	vm, err := vmClient.Get(ctx, d.Id())
	if err != nil {
		return diag.FromErr(err)
	}

	d.Set("name", vm.Name)
	d.Set("memory", vm.Config.Memory)
	d.Set("cpus", vm.Config.CPUs)
	d.Set("disk", vm.Config.Disk)
	d.Set("image", vm.Config.Image)
	d.Set("node", vm.Node)
	d.Set("state", string(vm.State))
	d.Set("labels", vm.Labels)
	d.Set("created_at", vm.CreatedAt.Format("2006-01-02T15:04:05Z"))
	d.Set("updated_at", vm.UpdatedAt.Format("2006-01-02T15:04:05Z"))

	return diags
}

func resourceVMUpdate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	client := meta.(*dwcp.Client)
	vmClient := client.VM()

	// Handle state changes
	if d.HasChange("state") {
		old, new := d.GetChange("state")
		oldState := old.(string)
		newState := new.(string)

		if oldState == "stopped" && newState == "running" {
			if err := vmClient.Start(ctx, d.Id()); err != nil {
				return diag.FromErr(err)
			}
		} else if oldState == "running" && newState == "stopped" {
			if err := vmClient.Stop(ctx, d.Id(), false); err != nil {
				return diag.FromErr(err)
			}
		}
	}

	return resourceVMRead(ctx, d, meta)
}

func resourceVMDelete(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	var diags diag.Diagnostics
	client := meta.(*dwcp.Client)

	vmClient := client.VM()

	// Stop VM if running
	vm, err := vmClient.Get(ctx, d.Id())
	if err != nil {
		return diag.FromErr(err)
	}

	if vm.State == dwcp.VMStateRunning {
		if err := vmClient.Stop(ctx, d.Id(), false); err != nil {
			return diag.FromErr(err)
		}
	}

	// Destroy VM
	if err := vmClient.Destroy(ctx, d.Id()); err != nil {
		return diag.FromErr(err)
	}

	d.SetId("")
	return diags
}

func resourceSnapshot() *schema.Resource {
	return &schema.Resource{
		CreateContext: resourceSnapshotCreate,
		ReadContext:   resourceSnapshotRead,
		DeleteContext: resourceSnapshotDelete,

		Schema: map[string]*schema.Schema{
			"vm_id": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "VM ID",
			},
			"name": {
				Type:        schema.TypeString,
				Required:    true,
				ForceNew:    true,
				Description: "Snapshot name",
			},
			"description": {
				Type:        schema.TypeString,
				Optional:    true,
				ForceNew:    true,
				Description: "Snapshot description",
			},
			"include_memory": {
				Type:        schema.TypeBool,
				Optional:    true,
				Default:     true,
				ForceNew:    true,
				Description: "Include memory state",
			},
			"size": {
				Type:        schema.TypeInt,
				Computed:    true,
				Description: "Snapshot size in bytes",
			},
			"created_at": {
				Type:        schema.TypeString,
				Computed:    true,
				Description: "Creation timestamp",
			},
		},
	}
}

func resourceSnapshotCreate(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	client := meta.(*dwcp.Client)

	vmID := d.Get("vm_id").(string)
	name := d.Get("name").(string)

	options := dwcp.SnapshotOptions{
		IncludeMemory: d.Get("include_memory").(bool),
		Description:   d.Get("description").(string),
		Quiesce:       true,
	}

	vmClient := client.VM()
	snapshot, err := vmClient.Snapshot(ctx, vmID, name, options)
	if err != nil {
		return diag.FromErr(err)
	}

	d.SetId(snapshot.ID)

	return resourceSnapshotRead(ctx, d, meta)
}

func resourceSnapshotRead(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	// Snapshot read implementation
	return nil
}

func resourceSnapshotDelete(ctx context.Context, d *schema.ResourceData, meta interface{}) diag.Diagnostics {
	var diags diag.Diagnostics
	client := meta.(*dwcp.Client)

	vmClient := client.VM()
	if err := vmClient.DeleteSnapshot(ctx, d.Id()); err != nil {
		return diag.FromErr(err)
	}

	d.SetId("")
	return diags
}

func resourceNetwork() *schema.Resource {
	// Network resource implementation
	return &schema.Resource{}
}

func dataSourceVM() *schema.Resource {
	// VM data source implementation
	return &schema.Resource{}
}

func dataSourceNode() *schema.Resource {
	// Node data source implementation
	return &schema.Resource{}
}

func dataSourceTemplate() *schema.Resource {
	// Template data source implementation
	return &schema.Resource{}
}
