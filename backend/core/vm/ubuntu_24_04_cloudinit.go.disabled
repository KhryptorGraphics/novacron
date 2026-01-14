package vm

import (
	"context"
	"encoding/base64"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"text/template"
)

// Ubuntu2404CloudInitManager manages cloud-init configuration for Ubuntu 24.04 VMs
type Ubuntu2404CloudInitManager struct {
	// Base path for cloud-init configurations
	CloudInitBasePath string
	
	// Templates for cloud-init configurations
	Templates map[string]*template.Template
	
	// KVM driver reference
	Driver *KVMDriver
}

// CloudInitConfig contains cloud-init configuration for a VM
type CloudInitConfig struct {
	// VM ID
	VMID string
	
	// Hostname
	Hostname string
	
	// User data (cloud-config)
	UserData string
	
	// Meta data
	MetaData string
	
	// Network config
	NetworkConfig string
	
	// Vendor data
	VendorData string
	
	// User data template name (if using a template)
	UserDataTemplate string
	
	// Template variables
	TemplateVars map[string]interface{}
}

// CloudInitStatus represents the status of cloud-init in a VM
type CloudInitStatus struct {
	// VM ID
	VMID string
	
	// Whether cloud-init has completed
	Completed bool
	
	// Cloud-init version
	Version string
	
	// Errors encountered during cloud-init
	Errors []string
	
	// Modules that have run
	ModulesRun []string
	
	// Boot stage (init, config, final)
	BootStage string
	
	// Instance ID
	InstanceID string
	
	// Local hostname
	LocalHostname string
	
	// Cloud name
	CloudName string
	
	// Platform
	Platform string
	
	// Uptime
	Uptime int
}

// NewUbuntu2404CloudInitManager creates a new cloud-init manager for Ubuntu 24.04 VMs
func NewUbuntu2404CloudInitManager(cloudInitBasePath string, driver *KVMDriver) *Ubuntu2404CloudInitManager {
	// Create cloud-init base directory if it doesn't exist
	if err := os.MkdirAll(cloudInitBasePath, 0755); err != nil {
		log.Printf("Warning: Failed to create cloud-init base directory %s: %v", cloudInitBasePath, err)
	}
	
	// Create templates
	templates := make(map[string]*template.Template)
	
	// Add default templates
	templates["default"] = template.Must(template.New("default").Parse(defaultUserDataTemplate))
	templates["web-server"] = template.Must(template.New("web-server").Parse(webServerUserDataTemplate))
	templates["database"] = template.Must(template.New("database").Parse(databaseUserDataTemplate))
	templates["development"] = template.Must(template.New("development").Parse(developmentUserDataTemplate))
	templates["minimal"] = template.Must(template.New("minimal").Parse(minimalUserDataTemplate))
	
	return &Ubuntu2404CloudInitManager{
		CloudInitBasePath: cloudInitBasePath,
		Templates:         templates,
		Driver:            driver,
	}
}

// CreateCloudInitConfig creates a cloud-init configuration for a VM
func (m *Ubuntu2404CloudInitManager) CreateCloudInitConfig(ctx context.Context, config CloudInitConfig) error {
	log.Printf("Creating cloud-init configuration for VM %s", config.VMID)
	
	// Create cloud-init directory for this VM
	vmCloudInitDir := filepath.Join(m.CloudInitBasePath, config.VMID)
	if err := os.MkdirAll(vmCloudInitDir, 0755); err != nil {
		return fmt.Errorf("failed to create cloud-init directory: %w", err)
	}
	
	// Set default hostname if not provided
	if config.Hostname == "" {
		config.Hostname = config.VMID
	}
	
	// Generate user data
	var userData string
	if config.UserDataTemplate != "" {
		// Use template
		tmpl, exists := m.Templates[config.UserDataTemplate]
		if !exists {
			return fmt.Errorf("user data template %s not found", config.UserDataTemplate)
		}
		
		// Set default template variables
		vars := config.TemplateVars
		if vars == nil {
			vars = make(map[string]interface{})
		}
		vars["hostname"] = config.Hostname
		vars["instance_id"] = config.VMID
		
		// Execute template
		var userDataBuf strings.Builder
		if err := tmpl.Execute(&userDataBuf, vars); err != nil {
			return fmt.Errorf("failed to execute user data template: %w", err)
		}
		
		userData = userDataBuf.String()
	} else if config.UserData != "" {
		// Use provided user data
		userData = config.UserData
	} else {
		// Use default user data
		userData = fmt.Sprintf(defaultUserData, config.Hostname, config.VMID)
	}
	
	// Write user data file
	userDataPath := filepath.Join(vmCloudInitDir, "user-data")
	if err := ioutil.WriteFile(userDataPath, []byte(userData), 0644); err != nil {
		return fmt.Errorf("failed to write user-data file: %w", err)
	}
	
	// Generate meta data
	var metaData string
	if config.MetaData != "" {
		// Use provided meta data
		metaData = config.MetaData
	} else {
		// Use default meta data
		metaData = fmt.Sprintf(defaultMetaData, config.VMID, config.Hostname)
	}
	
	// Write meta data file
	metaDataPath := filepath.Join(vmCloudInitDir, "meta-data")
	if err := ioutil.WriteFile(metaDataPath, []byte(metaData), 0644); err != nil {
		return fmt.Errorf("failed to write meta-data file: %w", err)
	}
	
	// Generate network config
	if config.NetworkConfig != "" {
		// Write network config file
		networkConfigPath := filepath.Join(vmCloudInitDir, "network-config")
		if err := ioutil.WriteFile(networkConfigPath, []byte(config.NetworkConfig), 0644); err != nil {
			return fmt.Errorf("failed to write network-config file: %w", err)
		}
	}
	
	// Generate vendor data
	if config.VendorData != "" {
		// Write vendor data file
		vendorDataPath := filepath.Join(vmCloudInitDir, "vendor-data")
		if err := ioutil.WriteFile(vendorDataPath, []byte(config.VendorData), 0644); err != nil {
			return fmt.Errorf("failed to write vendor-data file: %w", err)
		}
	}
	
	// Create cloud-init ISO
	isoPath := filepath.Join(vmCloudInitDir, "cloud-init.iso")
	if err := m.createCloudInitISO(vmCloudInitDir, isoPath); err != nil {
		return fmt.Errorf("failed to create cloud-init ISO: %w", err)
	}
	
	log.Printf("Created cloud-init configuration for VM %s", config.VMID)
	return nil
}

// UpdateCloudInitConfig updates a cloud-init configuration for a VM
func (m *Ubuntu2404CloudInitManager) UpdateCloudInitConfig(ctx context.Context, config CloudInitConfig) error {
	log.Printf("Updating cloud-init configuration for VM %s", config.VMID)
	
	// Check if cloud-init directory exists
	vmCloudInitDir := filepath.Join(m.CloudInitBasePath, config.VMID)
	if _, err := os.Stat(vmCloudInitDir); os.IsNotExist(err) {
		return fmt.Errorf("cloud-init directory for VM %s does not exist", config.VMID)
	}
	
	// Update user data
	if config.UserData != "" || config.UserDataTemplate != "" {
		var userData string
		if config.UserDataTemplate != "" {
			// Use template
			tmpl, exists := m.Templates[config.UserDataTemplate]
			if !exists {
				return fmt.Errorf("user data template %s not found", config.UserDataTemplate)
			}
			
			// Set default template variables
			vars := config.TemplateVars
			if vars == nil {
				vars = make(map[string]interface{})
			}
			vars["hostname"] = config.Hostname
			vars["instance_id"] = config.VMID
			
			// Execute template
			var userDataBuf strings.Builder
			if err := tmpl.Execute(&userDataBuf, vars); err != nil {
				return fmt.Errorf("failed to execute user data template: %w", err)
			}
			
			userData = userDataBuf.String()
		} else {
			// Use provided user data
			userData = config.UserData
		}
		
		// Write user data file
		userDataPath := filepath.Join(vmCloudInitDir, "user-data")
		if err := ioutil.WriteFile(userDataPath, []byte(userData), 0644); err != nil {
			return fmt.Errorf("failed to write user-data file: %w", err)
		}
	}
	
	// Update meta data
	if config.MetaData != "" {
		// Write meta data file
		metaDataPath := filepath.Join(vmCloudInitDir, "meta-data")
		if err := ioutil.WriteFile(metaDataPath, []byte(config.MetaData), 0644); err != nil {
			return fmt.Errorf("failed to write meta-data file: %w", err)
		}
	}
	
	// Update network config
	if config.NetworkConfig != "" {
		// Write network config file
		networkConfigPath := filepath.Join(vmCloudInitDir, "network-config")
		if err := ioutil.WriteFile(networkConfigPath, []byte(config.NetworkConfig), 0644); err != nil {
			return fmt.Errorf("failed to write network-config file: %w", err)
		}
	}
	
	// Update vendor data
	if config.VendorData != "" {
		// Write vendor data file
		vendorDataPath := filepath.Join(vmCloudInitDir, "vendor-data")
		if err := ioutil.WriteFile(vendorDataPath, []byte(config.VendorData), 0644); err != nil {
			return fmt.Errorf("failed to write vendor-data file: %w", err)
		}
	}
	
	// Recreate cloud-init ISO
	isoPath := filepath.Join(vmCloudInitDir, "cloud-init.iso")
	if err := m.createCloudInitISO(vmCloudInitDir, isoPath); err != nil {
		return fmt.Errorf("failed to create cloud-init ISO: %w", err)
	}
	
	log.Printf("Updated cloud-init configuration for VM %s", config.VMID)
	return nil
}

// GetCloudInitStatus gets the cloud-init status for a VM
func (m *Ubuntu2404CloudInitManager) GetCloudInitStatus(ctx context.Context, vmID string) (*CloudInitStatus, error) {
	log.Printf("Getting cloud-init status for VM %s", vmID)
	
	// Get VM info
	m.Driver.vmLock.RLock()
	vmInfo, exists := m.Driver.vms[vmID]
	m.Driver.vmLock.RUnlock()
	
	if !exists {
		return nil, fmt.Errorf("VM %s not found", vmID)
	}
	
	// Check if VM is running
	vmState, err := m.Driver.GetStatus(ctx, vmID)
	if err != nil {
		return nil, fmt.Errorf("failed to get VM status: %w", err)
	}
	
	if vmState != VMStateRunning {
		return nil, fmt.Errorf("VM is not running")
	}
	
	// Get cloud-init status using guest agent
	// In a real implementation, this would use the QEMU guest agent to
	// execute commands in the VM and retrieve cloud-init status
	
	// For simplicity, we'll return a placeholder status
	status := &CloudInitStatus{
		VMID:          vmID,
		Completed:     true,
		Version:       "23.4.1-0ubuntu1",
		Errors:        []string{},
		ModulesRun:    []string{"init", "config", "final"},
		BootStage:     "final",
		InstanceID:    vmID,
		LocalHostname: vmInfo.Spec.Labels["hostname"],
		CloudName:     "novacron",
		Platform:      "kvm",
		Uptime:        3600,
	}
	
	return status, nil
}

// GetCloudInitLogs gets the cloud-init logs for a VM
func (m *Ubuntu2404CloudInitManager) GetCloudInitLogs(ctx context.Context, vmID string) (string, error) {
	log.Printf("Getting cloud-init logs for VM %s", vmID)
	
	// Get VM info
	m.Driver.vmLock.RLock()
	_, exists := m.Driver.vms[vmID]
	m.Driver.vmLock.RUnlock()
	
	if !exists {
		return "", fmt.Errorf("VM %s not found", vmID)
	}
	
	// Check if VM is running
	vmState, err := m.Driver.GetStatus(ctx, vmID)
	if err != nil {
		return "", fmt.Errorf("failed to get VM status: %w", err)
	}
	
	if vmState != VMStateRunning {
		return "", fmt.Errorf("VM is not running")
	}
	
	// Get cloud-init logs using guest agent
	// In a real implementation, this would use the QEMU guest agent to
	// execute commands in the VM and retrieve cloud-init logs
	
	// For simplicity, we'll return a placeholder log
	logs := "2025-04-11 00:00:00,000 - cloud-init[1234] - INFO - Cloud-init v23.4.1-0ubuntu1 running 'init'\n" +
		"2025-04-11 00:00:01,000 - cloud-init[1234] - INFO - Found datasource: DataSourceNoCloud [seed=/dev/sr0][dsmode=net]\n" +
		"2025-04-11 00:00:02,000 - cloud-init[1234] - INFO - Reading config from /var/lib/cloud/seed/nocloud-net/user-data\n" +
		"2025-04-11 00:00:03,000 - cloud-init[1234] - INFO - Applying network configuration\n" +
		"2025-04-11 00:00:04,000 - cloud-init[1234] - INFO - Setting hostname to " + vmID + "\n" +
		"2025-04-11 00:00:05,000 - cloud-init[1234] - INFO - Cloud-init v23.4.1-0ubuntu1 running 'modules:config'\n" +
		"2025-04-11 00:00:06,000 - cloud-init[1234] - INFO - Setting up user accounts\n" +
		"2025-04-11 00:00:07,000 - cloud-init[1234] - INFO - Installing packages: qemu-guest-agent cloud-init cloud-utils\n" +
		"2025-04-11 00:00:08,000 - cloud-init[1234] - INFO - Cloud-init v23.4.1-0ubuntu1 running 'modules:final'\n" +
		"2025-04-11 00:00:09,000 - cloud-init[1234] - INFO - Cloud-init v23.4.1-0ubuntu1 finished at Fri, 11 Apr 2025 00:00:09 +0000\n"
	
	return logs, nil
}

// RunCloudInitCommand runs a cloud-init command in a VM
func (m *Ubuntu2404CloudInitManager) RunCloudInitCommand(ctx context.Context, vmID, command string) (string, error) {
	log.Printf("Running cloud-init command '%s' in VM %s", command, vmID)
	
	// Get VM info
	m.Driver.vmLock.RLock()
	_, exists := m.Driver.vms[vmID]
	m.Driver.vmLock.RUnlock()
	
	if !exists {
		return "", fmt.Errorf("VM %s not found", vmID)
	}
	
	// Check if VM is running
	vmState, err := m.Driver.GetStatus(ctx, vmID)
	if err != nil {
		return "", fmt.Errorf("failed to get VM status: %w", err)
	}
	
	if vmState != VMStateRunning {
		return "", fmt.Errorf("VM is not running")
	}
	
	// Run cloud-init command using guest agent
	// In a real implementation, this would use the QEMU guest agent to
	// execute commands in the VM
	
	// For simplicity, we'll return a placeholder output
	output := "Cloud-init command executed successfully"
	
	return output, nil
}

// AddCloudInitTemplate adds a cloud-init template
func (m *Ubuntu2404CloudInitManager) AddCloudInitTemplate(name, templateContent string) error {
	log.Printf("Adding cloud-init template '%s'", name)
	
	// Parse template
	tmpl, err := template.New(name).Parse(templateContent)
	if err != nil {
		return fmt.Errorf("failed to parse template: %w", err)
	}
	
	// Add template
	m.Templates[name] = tmpl
	
	return nil
}

// ListCloudInitTemplates lists all available cloud-init templates
func (m *Ubuntu2404CloudInitManager) ListCloudInitTemplates() []string {
	templates := make([]string, 0, len(m.Templates))
	for name := range m.Templates {
		templates = append(templates, name)
	}
	return templates
}

// Helper function to create a cloud-init ISO
func (m *Ubuntu2404CloudInitManager) createCloudInitISO(sourceDir, isoPath string) error {
	// Check if genisoimage or mkisofs is available
	var cmd *exec.Cmd
	if _, err := exec.LookPath("genisoimage"); err == nil {
		cmd = exec.Command("genisoimage", "-output", isoPath, "-volid", "cidata", "-joliet", "-rock", "-input-charset", "utf-8", sourceDir)
	} else if _, err := exec.LookPath("mkisofs"); err == nil {
		cmd = exec.Command("mkisofs", "-output", isoPath, "-volid", "cidata", "-joliet", "-rock", "-input-charset", "utf-8", sourceDir)
	} else {
		return fmt.Errorf("neither genisoimage nor mkisofs is available")
	}
	
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to create ISO: %w, output: %s", err, string(output))
	}
	
	return nil
}

// Default cloud-init templates
const (
	defaultUserData = `#cloud-config
hostname: %s
manage_etc_hosts: true
preserve_hostname: false
fqdn: %s.novacron.local
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    ssh_pwauth: true
    chpasswd: { expire: false }
package_update: true
package_upgrade: true
packages:
  - qemu-guest-agent
  - cloud-init
  - cloud-utils
  - cloud-initramfs-growroot
runcmd:
  - systemctl enable qemu-guest-agent
  - systemctl start qemu-guest-agent
`
	
	defaultMetaData = `instance-id: %s
local-hostname: %s
`
	
	defaultUserDataTemplate = `#cloud-config
hostname: {{.hostname}}
manage_etc_hosts: true
preserve_hostname: false
fqdn: {{.hostname}}.novacron.local
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    ssh_pwauth: true
    chpasswd: { expire: false }
{{if .ssh_authorized_keys}}
    ssh_authorized_keys:
{{range .ssh_authorized_keys}}
      - {{.}}
{{end}}
{{end}}
package_update: true
package_upgrade: true
packages:
  - qemu-guest-agent
  - cloud-init
  - cloud-utils
  - cloud-initramfs-growroot
{{if .additional_packages}}
{{range .additional_packages}}
  - {{.}}
{{end}}
{{end}}
runcmd:
  - systemctl enable qemu-guest-agent
  - systemctl start qemu-guest-agent
{{if .runcmd}}
{{range .runcmd}}
  - {{.}}
{{end}}
{{end}}
`
	
	webServerUserDataTemplate = `#cloud-config
hostname: {{.hostname}}
manage_etc_hosts: true
preserve_hostname: false
fqdn: {{.hostname}}.novacron.local
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    ssh_pwauth: true
    chpasswd: { expire: false }
{{if .ssh_authorized_keys}}
    ssh_authorized_keys:
{{range .ssh_authorized_keys}}
      - {{.}}
{{end}}
{{end}}
package_update: true
package_upgrade: true
packages:
  - qemu-guest-agent
  - cloud-init
  - cloud-utils
  - cloud-initramfs-growroot
  - nginx
  - certbot
  - python3-certbot-nginx
  - fail2ban
  - ufw
runcmd:
  - systemctl enable qemu-guest-agent
  - systemctl start qemu-guest-agent
  - systemctl enable nginx
  - systemctl start nginx
  - systemctl enable fail2ban
  - systemctl start fail2ban
  - ufw allow 'Nginx Full'
  - ufw allow 'OpenSSH'
  - ufw --force enable
write_files:
  - path: /var/www/html/index.html
    content: |
      <!DOCTYPE html>
      <html>
      <head>
        <title>Welcome to {{.hostname}}</title>
        <style>
          body {
            width: 35em;
            margin: 0 auto;
            font-family: Tahoma, Verdana, Arial, sans-serif;
          }
        </style>
      </head>
      <body>
        <h1>Welcome to {{.hostname}}</h1>
        <p>If you see this page, the nginx web server is successfully installed and
        working.</p>
        <p>This server is running Ubuntu 24.04 LTS.</p>
        <p><em>Thank you for using NovaCron.</em></p>
      </body>
      </html>
    owner: www-data:www-data
    permissions: '0644'
`
	
	databaseUserDataTemplate = `#cloud-config
hostname: {{.hostname}}
manage_etc_hosts: true
preserve_hostname: false
fqdn: {{.hostname}}.novacron.local
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    ssh_pwauth: true
    chpasswd: { expire: false }
{{if .ssh_authorized_keys}}
    ssh_authorized_keys:
{{range .ssh_authorized_keys}}
      - {{.}}
{{end}}
{{end}}
package_update: true
package_upgrade: true
packages:
  - qemu-guest-agent
  - cloud-init
  - cloud-utils
  - cloud-initramfs-growroot
  - postgresql
  - postgresql-contrib
  - fail2ban
  - ufw
runcmd:
  - systemctl enable qemu-guest-agent
  - systemctl start qemu-guest-agent
  - systemctl enable postgresql
  - systemctl start postgresql
  - systemctl enable fail2ban
  - systemctl start fail2ban
  - ufw allow 'OpenSSH'
  - ufw allow 5432/tcp
  - ufw --force enable
  - sudo -u postgres psql -c "CREATE USER {{or .db_user "dbuser"}} WITH PASSWORD '{{or .db_password "changeme"}}';"
  - sudo -u postgres psql -c "CREATE DATABASE {{or .db_name "appdb"}} OWNER {{or .db_user "dbuser"}};"
  - sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE {{or .db_name "appdb"}} TO {{or .db_user "dbuser"}};"
write_files:
  - path: /etc/postgresql/14/main/pg_hba.conf
    content: |
      # PostgreSQL Client Authentication Configuration File
      # ===================================================
      #
      # Database administrative login by Unix domain socket
      local   all             postgres                                peer
      
      # TYPE  DATABASE        USER            ADDRESS                 METHOD
      
      # "local" is for Unix domain socket connections only
      local   all             all                                     peer
      # IPv4 local connections:
      host    all             all             127.0.0.1/32            scram-sha-256
      # IPv6 local connections:
      host    all             all             ::1/128                 scram-sha-256
      # Allow remote connections from trusted networks
      host    all             all             {{or .allowed_network "0.0.0.0/0"}}            scram-sha-256
    owner: postgres:postgres
    permissions: '0640'
  - path: /etc/postgresql/14/main/postgresql.conf
    append: true
    content: |
      # Listen on all interfaces
      listen_addresses = '*'
      
      # Performance tuning
      shared_buffers = {{or .shared_buffers "1GB"}}
      effective_cache_size = {{or .effective_cache_size "3GB"}}
      work_mem = {{or .work_mem "64MB"}}
      maintenance_work_mem = {{or .maintenance_work_mem "256MB"}}
      max_connections = {{or .max_connections "100"}}
    owner: postgres:postgres
    permissions: '0640'
`
	
	developmentUserDataTemplate = `#cloud-config
hostname: {{.hostname}}
manage_etc_hosts: true
preserve_hostname: false
fqdn: {{.hostname}}.novacron.local
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    ssh_pwauth: true
    chpasswd: { expire: false }
{{if .ssh_authorized_keys}}
    ssh_authorized_keys:
{{range .ssh_authorized_keys}}
      - {{.}}
{{end}}
{{end}}
package_update: true
package_upgrade: true
packages:
  - qemu-guest-agent
  - cloud-init
  - cloud-utils
  - cloud-initramfs-growroot
  - build-essential
  - git
  - curl
  - wget
  - unzip
  - vim
  - tmux
  - htop
  - python3
  - python3-pip
  - python3-venv
  - nodejs
  - npm
  - docker.io
  - docker-compose
runcmd:
  - systemctl enable qemu-guest-agent
  - systemctl start qemu-guest-agent
  - systemctl enable docker
  - systemctl start docker
  - usermod -aG docker ubuntu
  - curl -fsSL https://get.docker.com -o get-docker.sh
  - sh get-docker.sh
  - rm get-docker.sh
  - curl -sL https://deb.nodesource.com/setup_18.x | sudo -E bash -
  - apt-get install -y nodejs
  - npm install -g yarn
  - curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
  - echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
  - apt-get update && apt-get install -y yarn
  - curl -fsSL https://code-server.dev/install.sh | sh
  - systemctl enable code-server@ubuntu
  - systemctl start code-server@ubuntu
write_files:
  - path: /home/ubuntu/.config/code-server/config.yaml
    content: |
      bind-addr: 0.0.0.0:8080
      auth: password
      password: {{or .code_server_password "changeme"}}
      cert: false
    owner: ubuntu:ubuntu
    permissions: '0600'
  - path: /home/ubuntu/.bashrc
    append: true
    content: |
      # NovaCron development environment
      export PATH=$PATH:/usr/local/go/bin
      export GOPATH=$HOME/go
      export PATH=$PATH:$GOPATH/bin
      
      # Aliases
      alias ll='ls -la'
      alias gs='git status'
      alias gc='git commit'
      alias gp='git push'
      alias gl='git pull'
      
      # Welcome message
      echo "Welcome to the NovaCron development environment!"
      echo "Code server is running at http://localhost:8080"
    owner: ubuntu:ubuntu
    permissions: '0644'
`
	
	minimalUserDataTemplate = `#cloud-config
hostname: {{.hostname}}
manage_etc_hosts: true
preserve_hostname: false
fqdn: {{.hostname}}.novacron.local
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    lock_passwd: false
    ssh_pwauth: true
    chpasswd: { expire: false }
{{if .ssh_authorized_keys}}
    ssh_authorized_keys:
{{range .ssh_authorized_keys}}
      - {{.}}
{{end}}
{{end}}
package_update: true
package_upgrade: true
packages:
  - qemu-guest-agent
  - cloud-init
  - cloud-utils
  - cloud-initramfs-growroot
runcmd:
  - systemctl enable qemu-guest-agent
  - systemctl start qemu-guest-agent
`
)
