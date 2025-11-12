// Package azure provides Azure Resource Manager (ARM) templates for deployment
package azure

import (
	"encoding/json"
	"fmt"
)

// ARMTemplate represents a complete ARM template
type ARMTemplate struct {
	Schema         string                 `json:"$schema"`
	ContentVersion string                 `json:"contentVersion"`
	Parameters     map[string]Parameter   `json:"parameters,omitempty"`
	Variables      map[string]interface{} `json:"variables,omitempty"`
	Resources      []Resource             `json:"resources"`
	Outputs        map[string]Output      `json:"outputs,omitempty"`
}

// Parameter represents an ARM template parameter
type Parameter struct {
	Type          string      `json:"type"`
	DefaultValue  interface{} `json:"defaultValue,omitempty"`
	AllowedValues []string    `json:"allowedValues,omitempty"`
	Metadata      *Metadata   `json:"metadata,omitempty"`
	MinValue      *int        `json:"minValue,omitempty"`
	MaxValue      *int        `json:"maxValue,omitempty"`
	MinLength     *int        `json:"minLength,omitempty"`
	MaxLength     *int        `json:"maxLength,omitempty"`
}

// Metadata represents parameter metadata
type Metadata struct {
	Description string `json:"description,omitempty"`
}

// Resource represents an ARM resource
type Resource struct {
	Type       string                 `json:"type"`
	APIVersion string                 `json:"apiVersion"`
	Name       interface{}            `json:"name"`
	Location   interface{}            `json:"location,omitempty"`
	Properties map[string]interface{} `json:"properties,omitempty"`
	DependsOn  []string               `json:"dependsOn,omitempty"`
	Tags       map[string]string      `json:"tags,omitempty"`
	SKU        *SKU                   `json:"sku,omitempty"`
	Kind       string                 `json:"kind,omitempty"`
	Resources  []Resource             `json:"resources,omitempty"`
}

// SKU represents a resource SKU
type SKU struct {
	Name     string `json:"name"`
	Tier     string `json:"tier,omitempty"`
	Size     string `json:"size,omitempty"`
	Family   string `json:"family,omitempty"`
	Capacity *int   `json:"capacity,omitempty"`
}

// Output represents an ARM template output
type Output struct {
	Type  string      `json:"type"`
	Value interface{} `json:"value"`
}

// GenerateNovaCronTemplate generates ARM template for NovaCron deployment
func GenerateNovaCronTemplate() (*ARMTemplate, error) {
	template := &ARMTemplate{
		Schema:         "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
		ContentVersion: "1.0.0.0",
		Parameters: map[string]Parameter{
			"location": {
				Type:         "string",
				DefaultValue: "eastus",
				AllowedValues: []string{
					"eastus", "eastus2", "westus", "westus2", "centralus",
					"northeurope", "westeurope", "southeastasia", "japaneast",
				},
				Metadata: &Metadata{
					Description: "Location for all resources",
				},
			},
			"vmSize": {
				Type:         "string",
				DefaultValue: "Standard_D4s_v3",
				AllowedValues: []string{
					"Standard_D2s_v3", "Standard_D4s_v3", "Standard_D8s_v3",
					"Standard_E4s_v3", "Standard_E8s_v3", "Standard_F4s_v2",
				},
				Metadata: &Metadata{
					Description: "VM size for NovaCron nodes",
				},
			},
			"adminUsername": {
				Type: "string",
				Metadata: &Metadata{
					Description: "Admin username for VMs",
				},
			},
			"adminPassword": {
				Type: "securestring",
				Metadata: &Metadata{
					Description: "Admin password for VMs",
				},
			},
			"nodeCount": {
				Type:         "int",
				DefaultValue: 3,
				MinValue:     ptrInt(1),
				MaxValue:     ptrInt(100),
				Metadata: &Metadata{
					Description: "Number of NovaCron nodes",
				},
			},
			"enableHA": {
				Type:         "bool",
				DefaultValue: true,
				Metadata: &Metadata{
					Description: "Enable high availability configuration",
				},
			},
			"databaseSKU": {
				Type:         "string",
				DefaultValue: "GP_Gen5_4",
				AllowedValues: []string{
					"B_Gen5_1", "B_Gen5_2", "GP_Gen5_2", "GP_Gen5_4",
					"GP_Gen5_8", "MO_Gen5_4", "MO_Gen5_8",
				},
				Metadata: &Metadata{
					Description: "Azure Database for PostgreSQL SKU",
				},
			},
			"storageAccountType": {
				Type:         "string",
				DefaultValue: "Premium_LRS",
				AllowedValues: []string{
					"Standard_LRS", "Standard_GRS", "Premium_LRS",
				},
				Metadata: &Metadata{
					Description: "Storage account type for VM disks",
				},
			},
		},
		Variables: map[string]interface{}{
			"vnetName":              "novacron-vnet",
			"vnetAddressPrefix":     "10.0.0.0/16",
			"subnetName":            "novacron-subnet",
			"subnetAddressPrefix":   "10.0.1.0/24",
			"nsgName":               "novacron-nsg",
			"publicIPName":          "novacron-pip",
			"loadBalancerName":      "novacron-lb",
			"availabilitySetName":   "novacron-avset",
			"vmssName":              "novacron-vmss",
			"storageAccountName":    "[concat('novacron', uniqueString(resourceGroup().id))]",
			"databaseServerName":    "[concat('novacron-db-', uniqueString(resourceGroup().id))]",
			"databaseName":          "novacrondb",
			"applicationGatewayName": "novacron-appgw",
		},
		Resources: generateARMResources(),
		Outputs:   generateARMOutputs(),
	}

	return template, nil
}

// generateARMResources generates all ARM resources
func generateARMResources() []Resource {
	resources := []Resource{}

	// Virtual Network
	resources = append(resources, Resource{
		Type:       "Microsoft.Network/virtualNetworks",
		APIVersion: "2021-02-01",
		Name:       "[variables('vnetName')]",
		Location:   "[parameters('location')]",
		Properties: map[string]interface{}{
			"addressSpace": map[string]interface{}{
				"addressPrefixes": []string{"[variables('vnetAddressPrefix')]"},
			},
			"subnets": []map[string]interface{}{
				{
					"name": "[variables('subnetName')]",
					"properties": map[string]interface{}{
						"addressPrefix": "[variables('subnetAddressPrefix')]",
						"networkSecurityGroup": map[string]interface{}{
							"id": "[resourceId('Microsoft.Network/networkSecurityGroups', variables('nsgName'))]",
						},
					},
				},
			},
		},
		DependsOn: []string{
			"[resourceId('Microsoft.Network/networkSecurityGroups', variables('nsgName'))]",
		},
		Tags: map[string]string{
			"Product": "NovaCron",
		},
	})

	// Network Security Group
	resources = append(resources, Resource{
		Type:       "Microsoft.Network/networkSecurityGroups",
		APIVersion: "2021-02-01",
		Name:       "[variables('nsgName')]",
		Location:   "[parameters('location')]",
		Properties: map[string]interface{}{
			"securityRules": []map[string]interface{}{
				{
					"name": "allow-https",
					"properties": map[string]interface{}{
						"priority":                 1000,
						"access":                   "Allow",
						"direction":                "Inbound",
						"protocol":                 "Tcp",
						"sourcePortRange":          "*",
						"destinationPortRange":     "443",
						"sourceAddressPrefix":      "*",
						"destinationAddressPrefix": "*",
					},
				},
				{
					"name": "allow-http",
					"properties": map[string]interface{}{
						"priority":                 1010,
						"access":                   "Allow",
						"direction":                "Inbound",
						"protocol":                 "Tcp",
						"sourcePortRange":          "*",
						"destinationPortRange":     "80",
						"destinationAddressPrefix": "*",
						"sourceAddressPrefix":      "*",
					},
				},
				{
					"name": "allow-ssh",
					"properties": map[string]interface{}{
						"priority":                 1020,
						"access":                   "Allow",
						"direction":                "Inbound",
						"protocol":                 "Tcp",
						"sourcePortRange":          "*",
						"destinationPortRange":     "22",
						"sourceAddressPrefix":      "*",
						"destinationAddressPrefix": "*",
					},
				},
			},
		},
		Tags: map[string]string{
			"Product": "NovaCron",
		},
	})

	// Public IP Address
	resources = append(resources, Resource{
		Type:       "Microsoft.Network/publicIPAddresses",
		APIVersion: "2021-02-01",
		Name:       "[variables('publicIPName')]",
		Location:   "[parameters('location')]",
		SKU: &SKU{
			Name: "Standard",
		},
		Properties: map[string]interface{}{
			"publicIPAllocationMethod": "Static",
			"dnsSettings": map[string]interface{}{
				"domainNameLabel": "[concat('novacron-', uniqueString(resourceGroup().id))]",
			},
		},
		Tags: map[string]string{
			"Product": "NovaCron",
		},
	})

	// Load Balancer
	resources = append(resources, Resource{
		Type:       "Microsoft.Network/loadBalancers",
		APIVersion: "2021-02-01",
		Name:       "[variables('loadBalancerName')]",
		Location:   "[parameters('location')]",
		SKU: &SKU{
			Name: "Standard",
		},
		Properties: map[string]interface{}{
			"frontendIPConfigurations": []map[string]interface{}{
				{
					"name": "LoadBalancerFrontend",
					"properties": map[string]interface{}{
						"publicIPAddress": map[string]interface{}{
							"id": "[resourceId('Microsoft.Network/publicIPAddresses', variables('publicIPName'))]",
						},
					},
				},
			},
			"backendAddressPools": []map[string]interface{}{
				{
					"name": "LoadBalancerBackend",
				},
			},
			"probes": []map[string]interface{}{
				{
					"name": "HealthProbe",
					"properties": map[string]interface{}{
						"protocol":          "Tcp",
						"port":              443,
						"intervalInSeconds": 15,
						"numberOfProbes":    2,
					},
				},
			},
			"loadBalancingRules": []map[string]interface{}{
				{
					"name": "HTTPSRule",
					"properties": map[string]interface{}{
						"frontendIPConfiguration": map[string]interface{}{
							"id": "[resourceId('Microsoft.Network/loadBalancers/frontendIPConfigurations', variables('loadBalancerName'), 'LoadBalancerFrontend')]",
						},
						"backendAddressPool": map[string]interface{}{
							"id": "[resourceId('Microsoft.Network/loadBalancers/backendAddressPools', variables('loadBalancerName'), 'LoadBalancerBackend')]",
						},
						"probe": map[string]interface{}{
							"id": "[resourceId('Microsoft.Network/loadBalancers/probes', variables('loadBalancerName'), 'HealthProbe')]",
						},
						"protocol":             "Tcp",
						"frontendPort":         443,
						"backendPort":          443,
						"enableFloatingIP":     false,
						"idleTimeoutInMinutes": 5,
					},
				},
			},
		},
		DependsOn: []string{
			"[resourceId('Microsoft.Network/publicIPAddresses', variables('publicIPName'))]",
		},
		Tags: map[string]string{
			"Product": "NovaCron",
		},
	})

	// Storage Account
	resources = append(resources, Resource{
		Type:       "Microsoft.Storage/storageAccounts",
		APIVersion: "2021-04-01",
		Name:       "[variables('storageAccountName')]",
		Location:   "[parameters('location')]",
		SKU: &SKU{
			Name: "[parameters('storageAccountType')]",
		},
		Kind: "StorageV2",
		Properties: map[string]interface{}{
			"encryption": map[string]interface{}{
				"services": map[string]interface{}{
					"blob": map[string]interface{}{
						"enabled": true,
					},
					"file": map[string]interface{}{
						"enabled": true,
					},
				},
				"keySource": "Microsoft.Storage",
			},
			"supportsHttpsTrafficOnly": true,
		},
		Tags: map[string]string{
			"Product": "NovaCron",
		},
	})

	// Azure Database for PostgreSQL
	resources = append(resources, Resource{
		Type:       "Microsoft.DBforPostgreSQL/servers",
		APIVersion: "2017-12-01",
		Name:       "[variables('databaseServerName')]",
		Location:   "[parameters('location')]",
		SKU: &SKU{
			Name: "[parameters('databaseSKU')]",
		},
		Properties: map[string]interface{}{
			"version":                "11",
			"administratorLogin":     "[parameters('adminUsername')]",
			"administratorLoginPassword": "[parameters('adminPassword')]",
			"storageProfile": map[string]interface{}{
				"storageMB":           102400,
				"backupRetentionDays": 7,
				"geoRedundantBackup":  "Enabled",
			},
			"sslEnforcement": "Enabled",
		},
		Tags: map[string]string{
			"Product": "NovaCron",
		},
		Resources: []Resource{
			{
				Type:       "databases",
				APIVersion: "2017-12-01",
				Name:       "[variables('databaseName')]",
				Properties: map[string]interface{}{
					"charset":   "UTF8",
					"collation": "English_United States.1252",
				},
				DependsOn: []string{
					"[resourceId('Microsoft.DBforPostgreSQL/servers', variables('databaseServerName'))]",
				},
			},
		},
	})

	// Virtual Machine Scale Set
	resources = append(resources, Resource{
		Type:       "Microsoft.Compute/virtualMachineScaleSets",
		APIVersion: "2021-03-01",
		Name:       "[variables('vmssName')]",
		Location:   "[parameters('location')]",
		SKU: &SKU{
			Name:     "[parameters('vmSize')]",
			Tier:     "Standard",
			Capacity: ptrInt(3),
		},
		Properties: map[string]interface{}{
			"overprovision": false,
			"upgradePolicy": map[string]interface{}{
				"mode": "Manual",
			},
			"virtualMachineProfile": map[string]interface{}{
				"storageProfile": map[string]interface{}{
					"imageReference": map[string]interface{}{
						"publisher": "Canonical",
						"offer":     "0001-com-ubuntu-server-jammy",
						"sku":       "22_04-lts-gen2",
						"version":   "latest",
					},
					"osDisk": map[string]interface{}{
						"createOption": "FromImage",
						"caching":      "ReadWrite",
						"managedDisk": map[string]interface{}{
							"storageAccountType": "[parameters('storageAccountType')]",
						},
					},
				},
				"osProfile": map[string]interface{}{
					"computerNamePrefix": "novacron-",
					"adminUsername":      "[parameters('adminUsername')]",
					"adminPassword":      "[parameters('adminPassword')]",
				},
				"networkProfile": map[string]interface{}{
					"networkInterfaceConfigurations": []map[string]interface{}{
						{
							"name": "nic-config",
							"properties": map[string]interface{}{
								"primary": true,
								"ipConfigurations": []map[string]interface{}{
									{
										"name": "ipconfig",
										"properties": map[string]interface{}{
											"subnet": map[string]interface{}{
												"id": "[resourceId('Microsoft.Network/virtualNetworks/subnets', variables('vnetName'), variables('subnetName'))]",
											},
											"loadBalancerBackendAddressPools": []map[string]interface{}{
												{
													"id": "[resourceId('Microsoft.Network/loadBalancers/backendAddressPools', variables('loadBalancerName'), 'LoadBalancerBackend')]",
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		},
		DependsOn: []string{
			"[resourceId('Microsoft.Network/virtualNetworks', variables('vnetName'))]",
			"[resourceId('Microsoft.Network/loadBalancers', variables('loadBalancerName'))]",
		},
		Tags: map[string]string{
			"Product": "NovaCron",
		},
	})

	return resources
}

// generateARMOutputs generates ARM template outputs
func generateARMOutputs() map[string]Output {
	return map[string]Output{
		"loadBalancerIP": {
			Type:  "string",
			Value: "[reference(resourceId('Microsoft.Network/publicIPAddresses', variables('publicIPName'))).ipAddress]",
		},
		"loadBalancerFQDN": {
			Type:  "string",
			Value: "[reference(resourceId('Microsoft.Network/publicIPAddresses', variables('publicIPName'))).dnsSettings.fqdn]",
		},
		"databaseServerName": {
			Type:  "string",
			Value: "[variables('databaseServerName')]",
		},
		"databaseConnectionString": {
			Type:  "string",
			Value: "[concat('Server=', reference(resourceId('Microsoft.DBforPostgreSQL/servers', variables('databaseServerName'))).fullyQualifiedDomainName, ';Database=', variables('databaseName'), ';Port=5432;User Id=', parameters('adminUsername'), '@', variables('databaseServerName'), ';Password=', parameters('adminPassword'), ';Ssl Mode=Require;')]",
		},
	}
}

// ToJSON converts ARM template to JSON string
func (t *ARMTemplate) ToJSON() (string, error) {
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return "", fmt.Errorf("failed to marshal template: %w", err)
	}
	return string(data), nil
}

// ptrInt returns pointer to int
func ptrInt(i int) *int {
	return &i
}
