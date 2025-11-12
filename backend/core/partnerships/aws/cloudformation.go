// Package aws provides CloudFormation templates for one-click deployment
package aws

import (
	"encoding/json"
	"fmt"
)

// CloudFormationTemplate represents a complete CloudFormation template
type CloudFormationTemplate struct {
	AWSTemplateFormatVersion string                 `json:"AWSTemplateFormatVersion"`
	Description              string                 `json:"Description"`
	Metadata                 map[string]interface{} `json:"Metadata,omitempty"`
	Parameters               map[string]Parameter   `json:"Parameters,omitempty"`
	Mappings                 map[string]interface{} `json:"Mappings,omitempty"`
	Conditions               map[string]interface{} `json:"Conditions,omitempty"`
	Resources                map[string]Resource    `json:"Resources"`
	Outputs                  map[string]Output      `json:"Outputs,omitempty"`
}

// Parameter represents a CloudFormation parameter
type Parameter struct {
	Type          string      `json:"Type"`
	Description   string      `json:"Description,omitempty"`
	Default       interface{} `json:"Default,omitempty"`
	AllowedValues []string    `json:"AllowedValues,omitempty"`
	MinLength     *int        `json:"MinLength,omitempty"`
	MaxLength     *int        `json:"MaxLength,omitempty"`
	MinValue      *int        `json:"MinValue,omitempty"`
	MaxValue      *int        `json:"MaxValue,omitempty"`
	NoEcho        bool        `json:"NoEcho,omitempty"`
}

// Resource represents a CloudFormation resource
type Resource struct {
	Type       string                 `json:"Type"`
	Properties map[string]interface{} `json:"Properties,omitempty"`
	DependsOn  interface{}            `json:"DependsOn,omitempty"`
	Metadata   map[string]interface{} `json:"Metadata,omitempty"`
}

// Output represents a CloudFormation output
type Output struct {
	Description string      `json:"Description,omitempty"`
	Value       interface{} `json:"Value"`
	Export      *Export     `json:"Export,omitempty"`
}

// Export represents a CloudFormation export
type Export struct {
	Name string `json:"Name"`
}

// GenerateNovaCronTemplate generates CloudFormation template for NovaCron deployment
func GenerateNovaCronTemplate() (*CloudFormationTemplate, error) {
	template := &CloudFormationTemplate{
		AWSTemplateFormatVersion: "2010-09-09",
		Description:              "NovaCron DWCP v3 - Distributed VM Management Platform",
		Metadata: map[string]interface{}{
			"AWS::CloudFormation::Interface": map[string]interface{}{
				"ParameterGroups": []map[string]interface{}{
					{
						"Label": map[string]string{"default": "Network Configuration"},
						"Parameters": []string{"VpcCIDR", "PublicSubnet1CIDR", "PublicSubnet2CIDR",
							"PrivateSubnet1CIDR", "PrivateSubnet2CIDR"},
					},
					{
						"Label": map[string]string{"default": "NovaCron Configuration"},
						"Parameters": []string{"InstanceType", "KeyName", "AdminEmail",
							"EnableHA", "BackupRetentionDays"},
					},
					{
						"Label": map[string]string{"default": "Database Configuration"},
						"Parameters": []string{"DBInstanceClass", "DBAllocatedStorage",
							"DBMasterUsername", "DBMasterPassword"},
					},
				},
				"ParameterLabels": map[string]map[string]string{
					"VpcCIDR":          {"default": "VPC CIDR Block"},
					"InstanceType":     {"default": "EC2 Instance Type"},
					"EnableHA":         {"default": "Enable High Availability"},
					"DBInstanceClass":  {"default": "RDS Instance Class"},
				},
			},
		},
		Parameters: map[string]Parameter{
			"VpcCIDR": {
				Type:        "String",
				Description: "CIDR block for the VPC",
				Default:     "10.0.0.0/16",
			},
			"PublicSubnet1CIDR": {
				Type:        "String",
				Description: "CIDR block for public subnet 1",
				Default:     "10.0.1.0/24",
			},
			"PublicSubnet2CIDR": {
				Type:        "String",
				Description: "CIDR block for public subnet 2",
				Default:     "10.0.2.0/24",
			},
			"PrivateSubnet1CIDR": {
				Type:        "String",
				Description: "CIDR block for private subnet 1",
				Default:     "10.0.10.0/24",
			},
			"PrivateSubnet2CIDR": {
				Type:        "String",
				Description: "CIDR block for private subnet 2",
				Default:     "10.0.11.0/24",
			},
			"InstanceType": {
				Type:        "String",
				Description: "EC2 instance type for NovaCron nodes",
				Default:     "m5.xlarge",
				AllowedValues: []string{
					"t3.large", "t3.xlarge", "m5.large", "m5.xlarge",
					"m5.2xlarge", "m5.4xlarge", "c5.xlarge", "c5.2xlarge",
				},
			},
			"KeyName": {
				Type:        "AWS::EC2::KeyPair::KeyName",
				Description: "EC2 key pair for SSH access",
			},
			"AdminEmail": {
				Type:        "String",
				Description: "Email address for admin notifications",
			},
			"EnableHA": {
				Type:        "String",
				Description: "Enable high availability with multi-AZ deployment",
				Default:     "true",
				AllowedValues: []string{"true", "false"},
			},
			"BackupRetentionDays": {
				Type:        "Number",
				Description: "Number of days to retain automated backups",
				Default:     7,
				MinValue:    ptrInt(1),
				MaxValue:    ptrInt(35),
			},
			"DBInstanceClass": {
				Type:        "String",
				Description: "RDS instance class for database",
				Default:     "db.r5.large",
				AllowedValues: []string{
					"db.t3.medium", "db.r5.large", "db.r5.xlarge",
					"db.r5.2xlarge", "db.r5.4xlarge",
				},
			},
			"DBAllocatedStorage": {
				Type:        "Number",
				Description: "Database allocated storage in GB",
				Default:     100,
				MinValue:    ptrInt(100),
				MaxValue:    ptrInt(65536),
			},
			"DBMasterUsername": {
				Type:        "String",
				Description: "Master username for database",
				Default:     "novacronadmin",
			},
			"DBMasterPassword": {
				Type:        "String",
				Description: "Master password for database (min 8 characters)",
				NoEcho:      true,
				MinLength:   ptrInt(8),
				MaxLength:   ptrInt(41),
			},
		},
		Resources: generateResources(),
		Outputs:   generateOutputs(),
	}

	return template, nil
}

// generateResources generates all CloudFormation resources
func generateResources() map[string]Resource {
	resources := make(map[string]Resource)

	// VPC
	resources["VPC"] = Resource{
		Type: "AWS::EC2::VPC",
		Properties: map[string]interface{}{
			"CidrBlock":          map[string]interface{}{"Ref": "VpcCIDR"},
			"EnableDnsHostnames": true,
			"EnableDnsSupport":   true,
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-VPC"}},
			},
		},
	}

	// Internet Gateway
	resources["InternetGateway"] = Resource{
		Type: "AWS::EC2::InternetGateway",
		Properties: map[string]interface{}{
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-IGW"}},
			},
		},
	}

	resources["AttachGateway"] = Resource{
		Type: "AWS::EC2::VPCGatewayAttachment",
		Properties: map[string]interface{}{
			"VpcId":             map[string]interface{}{"Ref": "VPC"},
			"InternetGatewayId": map[string]interface{}{"Ref": "InternetGateway"},
		},
	}

	// Public Subnets
	resources["PublicSubnet1"] = Resource{
		Type: "AWS::EC2::Subnet",
		Properties: map[string]interface{}{
			"VpcId":                map[string]interface{}{"Ref": "VPC"},
			"CidrBlock":            map[string]interface{}{"Ref": "PublicSubnet1CIDR"},
			"AvailabilityZone":     map[string]interface{}{"Fn::Select": []interface{}{0, map[string]interface{}{"Fn::GetAZs": ""}}},
			"MapPublicIpOnLaunch":  true,
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-PublicSubnet1"}},
			},
		},
	}

	resources["PublicSubnet2"] = Resource{
		Type: "AWS::EC2::Subnet",
		Properties: map[string]interface{}{
			"VpcId":                map[string]interface{}{"Ref": "VPC"},
			"CidrBlock":            map[string]interface{}{"Ref": "PublicSubnet2CIDR"},
			"AvailabilityZone":     map[string]interface{}{"Fn::Select": []interface{}{1, map[string]interface{}{"Fn::GetAZs": ""}}},
			"MapPublicIpOnLaunch":  true,
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-PublicSubnet2"}},
			},
		},
	}

	// Private Subnets
	resources["PrivateSubnet1"] = Resource{
		Type: "AWS::EC2::Subnet",
		Properties: map[string]interface{}{
			"VpcId":            map[string]interface{}{"Ref": "VPC"},
			"CidrBlock":        map[string]interface{}{"Ref": "PrivateSubnet1CIDR"},
			"AvailabilityZone": map[string]interface{}{"Fn::Select": []interface{}{0, map[string]interface{}{"Fn::GetAZs": ""}}},
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-PrivateSubnet1"}},
			},
		},
	}

	resources["PrivateSubnet2"] = Resource{
		Type: "AWS::EC2::Subnet",
		Properties: map[string]interface{}{
			"VpcId":            map[string]interface{}{"Ref": "VPC"},
			"CidrBlock":        map[string]interface{}{"Ref": "PrivateSubnet2CIDR"},
			"AvailabilityZone": map[string]interface{}{"Fn::Select": []interface{}{1, map[string]interface{}{"Fn::GetAZs": ""}}},
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-PrivateSubnet2"}},
			},
		},
	}

	// NAT Gateways
	resources["NATGateway1EIP"] = Resource{
		Type: "AWS::EC2::EIP",
		Properties: map[string]interface{}{
			"Domain": "vpc",
		},
		DependsOn: "AttachGateway",
	}

	resources["NATGateway1"] = Resource{
		Type: "AWS::EC2::NatGateway",
		Properties: map[string]interface{}{
			"AllocationId": map[string]interface{}{"Fn::GetAtt": []string{"NATGateway1EIP", "AllocationId"}},
			"SubnetId":     map[string]interface{}{"Ref": "PublicSubnet1"},
		},
	}

	// Route Tables
	resources["PublicRouteTable"] = Resource{
		Type: "AWS::EC2::RouteTable",
		Properties: map[string]interface{}{
			"VpcId": map[string]interface{}{"Ref": "VPC"},
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-PublicRT"}},
			},
		},
	}

	resources["DefaultPublicRoute"] = Resource{
		Type: "AWS::EC2::Route",
		Properties: map[string]interface{}{
			"RouteTableId":         map[string]interface{}{"Ref": "PublicRouteTable"},
			"DestinationCidrBlock": "0.0.0.0/0",
			"GatewayId":            map[string]interface{}{"Ref": "InternetGateway"},
		},
		DependsOn: "AttachGateway",
	}

	// Application Load Balancer
	resources["ApplicationLoadBalancer"] = Resource{
		Type: "AWS::ElasticLoadBalancingV2::LoadBalancer",
		Properties: map[string]interface{}{
			"Scheme": "internet-facing",
			"Type":   "application",
			"Subnets": []interface{}{
				map[string]interface{}{"Ref": "PublicSubnet1"},
				map[string]interface{}{"Ref": "PublicSubnet2"},
			},
			"SecurityGroups": []interface{}{
				map[string]interface{}{"Ref": "LoadBalancerSecurityGroup"},
			},
			"Tags": []map[string]interface{}{
				{"Key": "Name", "Value": map[string]interface{}{"Fn::Sub": "${AWS::StackName}-ALB"}},
			},
		},
	}

	// Security Groups
	resources["LoadBalancerSecurityGroup"] = Resource{
		Type: "AWS::EC2::SecurityGroup",
		Properties: map[string]interface{}{
			"GroupDescription": "Security group for load balancer",
			"VpcId":            map[string]interface{}{"Ref": "VPC"},
			"SecurityGroupIngress": []map[string]interface{}{
				{
					"IpProtocol": "tcp",
					"FromPort":   443,
					"ToPort":     443,
					"CidrIp":     "0.0.0.0/0",
				},
				{
					"IpProtocol": "tcp",
					"FromPort":   80,
					"ToPort":     80,
					"CidrIp":     "0.0.0.0/0",
				},
			},
		},
	}

	// RDS Database
	resources["DBSubnetGroup"] = Resource{
		Type: "AWS::RDS::DBSubnetGroup",
		Properties: map[string]interface{}{
			"DBSubnetGroupDescription": "Subnet group for NovaCron database",
			"SubnetIds": []interface{}{
				map[string]interface{}{"Ref": "PrivateSubnet1"},
				map[string]interface{}{"Ref": "PrivateSubnet2"},
			},
		},
	}

	resources["DatabaseInstance"] = Resource{
		Type: "AWS::RDS::DBInstance",
		Properties: map[string]interface{}{
			"DBInstanceClass":         map[string]interface{}{"Ref": "DBInstanceClass"},
			"AllocatedStorage":        map[string]interface{}{"Ref": "DBAllocatedStorage"},
			"Engine":                  "postgres",
			"EngineVersion":           "15.4",
			"MasterUsername":          map[string]interface{}{"Ref": "DBMasterUsername"},
			"MasterUserPassword":      map[string]interface{}{"Ref": "DBMasterPassword"},
			"DBSubnetGroupName":       map[string]interface{}{"Ref": "DBSubnetGroup"},
			"MultiAZ":                 map[string]interface{}{"Ref": "EnableHA"},
			"BackupRetentionPeriod":   map[string]interface{}{"Ref": "BackupRetentionDays"},
			"StorageEncrypted":        true,
			"EnablePerformanceInsights": true,
		},
	}

	return resources
}

// generateOutputs generates CloudFormation outputs
func generateOutputs() map[string]Output {
	return map[string]Output{
		"LoadBalancerURL": {
			Description: "URL of the Application Load Balancer",
			Value: map[string]interface{}{
				"Fn::GetAtt": []string{"ApplicationLoadBalancer", "DNSName"},
			},
			Export: &Export{
				Name: map[string]interface{}{"Fn::Sub": "${AWS::StackName}-LoadBalancerURL"}.(string),
			},
		},
		"DatabaseEndpoint": {
			Description: "Database endpoint",
			Value: map[string]interface{}{
				"Fn::GetAtt": []string{"DatabaseInstance", "Endpoint.Address"},
			},
		},
		"VpcId": {
			Description: "VPC ID",
			Value:       map[string]interface{}{"Ref": "VPC"},
			Export: &Export{
				Name: map[string]interface{}{"Fn::Sub": "${AWS::StackName}-VpcId"}.(string),
			},
		},
	}
}

// ToJSON converts template to JSON string
func (t *CloudFormationTemplate) ToJSON() (string, error) {
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
