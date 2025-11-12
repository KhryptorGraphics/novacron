# NovaCron AWS Partnership Integration Guide

## Table of Contents
1. [Overview](#overview)
2. [AWS Marketplace Integration](#aws-marketplace-integration)
3. [AWS Partner Network (APN)](#aws-partner-network-apn)
4. [CloudFormation Templates](#cloudformation-templates)
5. [Service Catalog Integration](#service-catalog-integration)
6. [AWS Organizations Support](#aws-organizations-support)
7. [Control Tower Integration](#control-tower-integration)
8. [Best Practices](#best-practices)

## Overview

NovaCron DWCP v3 provides comprehensive AWS integration enabling one-click deployment, marketplace listing, and deep AWS service integration. This guide covers all aspects of the AWS partnership.

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Marketplace                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Listing     │  │  Metering    │  │  Procurement │     │
│  │  Management  │  │  Service     │  │  API         │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              NovaCron Marketplace Manager                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Usage       │  │  Customer    │  │  Subscription│     │
│  │  Reporting   │  │  Resolution  │  │  Management  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   AWS Infrastructure                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ CloudFormation│  │  Service     │  │ Organizations│     │
│  │              │  │  Catalog     │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## AWS Marketplace Integration

### Marketplace Metering Service

NovaCron uses AWS Marketplace Metering Service for usage-based billing.

#### Configuration

```go
import (
    "github.com/novacron/backend/core/partnerships/aws"
)

func main() {
    cfg := aws.MarketplaceConfig{
        ProductCode:      "novacron-dwcp-v3",
        PublicKeyVersion: 1,
        Region:           "us-east-1",
    }

    mgr, err := aws.NewMarketplaceManager(ctx, cfg)
    if err != nil {
        log.Fatal(err)
    }

    // Meter usage
    record := aws.UsageRecord{
        CustomerIdentifier: customerID,
        Dimension:          "vm-hours",
        Quantity:           10,
        Timestamp:          time.Now(),
    }

    err = mgr.MeterUsage(ctx, record)
}
```

#### Billing Dimensions

NovaCron supports the following billing dimensions:

| Dimension | Unit | Description |
|-----------|------|-------------|
| vm-hours | Hours | VM runtime hours |
| storage-gb | GB-Hours | Storage capacity used |
| transfer-gb | GB | Data transfer volume |
| snapshots | Count | Number of snapshots |
| migrations | Count | Number of migrations performed |

#### Usage Allocation

Support for cost allocation using tags:

```go
record := aws.UsageRecord{
    CustomerIdentifier: customerID,
    Dimension:          "vm-hours",
    Quantity:           100,
    UsageAllocations: []aws.UsageAllocation{
        {
            AllocatedUsageQuantity: 60,
            Tags: []aws.Tag{
                {Key: "Department", Value: "Engineering"},
                {Key: "Project", Value: "WebApp"},
            },
        },
        {
            AllocatedUsageQuantity: 40,
            Tags: []aws.Tag{
                {Key: "Department", Value: "Analytics"},
                {Key: "Project", Value: "DataPipeline"},
            },
        },
    },
}
```

### Customer Resolution

Resolve marketplace customers from registration tokens:

```go
subscription, err := mgr.ResolveCustomer(ctx, registrationToken)
if err != nil {
    log.Fatal(err)
}

fmt.Printf("Customer: %s\n", subscription.CustomerID)
fmt.Printf("Product: %s\n", subscription.ProductCode)
```

### Listing Management

Create and update marketplace listings:

```go
listing := aws.ListingMetadata{
    ProductID:   "prod-12345",
    ProductCode: "novacron-dwcp-v3",
    Name:        "NovaCron DWCP v3",
    Description: "Enterprise distributed VM management",
    Version:     "3.0.0",
    Category:    "Infrastructure Software",
    PricingModel: aws.PricingModel{
        Type: "usage-based",
        Dimensions: []aws.PricingDimension{
            {
                Key:          "vm-hours",
                Name:         "VM Hours",
                Unit:         "hours",
                PricePerUnit: 0.15,
            },
        },
    },
}

err := mgr.CreateListing(ctx, listing)
```

## AWS Partner Network (APN)

### Partner Tiers

NovaCron qualifies for **AWS Advanced Technology Partner** status with:
- Migration Competency
- DevOps Competency
- SaaS Competency

### Service Catalog Integration

Deploy NovaCron through AWS Service Catalog:

```go
import "github.com/novacron/backend/core/partnerships/aws"

func deployServiceCatalog() {
    apn, _ := aws.NewAPNManager(ctx, aws.APNConfig{
        PartnerID: "novacron-partner-id",
        Region:    "us-east-1",
    })

    product := aws.ServiceCatalogProduct{
        Name:        "NovaCron DWCP v3",
        Description: "Distributed VM Management Platform",
        Owner:       "NovaCron Technologies",
        Version:     "3.0.0",
        TemplateURL: "https://s3.amazonaws.com/novacron-templates/cfn-main.yaml",
        Parameters: []aws.ProductParameter{
            {
                Key:          "InstanceType",
                Type:         "String",
                DefaultValue: "m5.xlarge",
                Required:     true,
            },
        },
    }

    productID, err := apn.CreateServiceCatalogProduct(ctx, product)
}
```

### AWS Organizations Deployment

Deploy across an entire AWS Organization:

```go
deployment := aws.OrganizationDeployment{
    OrganizationID: "o-12345",
    Accounts:       []string{"123456789012", "210987654321"},
    Regions:        []string{"us-east-1", "eu-west-1"},
    Status:         "pending",
}

err := apn.DeployToOrganization(ctx, deployment)
```

### AWS RAM (Resource Access Manager)

Share NovaCron resources across accounts:

```go
share := aws.ResourceShare{
    Name: "NovaCron-Shared-Resources",
    Resources: []string{
        "arn:aws:novacron:us-east-1:123456789012:cluster/prod",
    },
    Principals: []string{
        "arn:aws:organizations::123456789012:organization/o-12345",
    },
    AllowExternal: false,
}

shareID, err := apn.CreateResourceShare(ctx, share)
```

## CloudFormation Templates

### One-Click Deployment

NovaCron provides comprehensive CloudFormation templates:

#### Main Template Structure

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'NovaCron DWCP v3 - Complete Infrastructure'

Parameters:
  VpcCIDR:
    Type: String
    Default: 10.0.0.0/16
  InstanceType:
    Type: String
    Default: m5.xlarge
  EnableHA:
    Type: String
    Default: 'true'
    AllowedValues: ['true', 'false']

Resources:
  VPC:
    Type: AWS::EC2::VPC
    Properties:
      CidrBlock: !Ref VpcCIDR
      EnableDnsHostnames: true

  LoadBalancer:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Type: application
      Subnets:
        - !Ref PublicSubnet1
        - !Ref PublicSubnet2

  DatabaseInstance:
    Type: AWS::RDS::DBInstance
    Properties:
      DBInstanceClass: !Ref DBInstanceClass
      Engine: postgres
      MultiAZ: !Ref EnableHA

Outputs:
  LoadBalancerURL:
    Description: Application Load Balancer URL
    Value: !GetAtt LoadBalancer.DNSName
```

### Nested Stacks

NovaCron uses nested stacks for modularity:

```
novacron-main.yaml
├── network.yaml (VPC, Subnets, Security Groups)
├── compute.yaml (EC2, Auto Scaling)
├── database.yaml (RDS, ElastiCache)
├── storage.yaml (S3, EBS, EFS)
└── monitoring.yaml (CloudWatch, SNS)
```

### Parameter Groups

Organized parameter grouping for better UX:

```yaml
Metadata:
  AWS::CloudFormation::Interface:
    ParameterGroups:
      - Label:
          default: "Network Configuration"
        Parameters:
          - VpcCIDR
          - PublicSubnet1CIDR
          - PrivateSubnet1CIDR
      - Label:
          default: "Compute Configuration"
        Parameters:
          - InstanceType
          - KeyName
      - Label:
          default: "High Availability"
        Parameters:
          - EnableHA
          - BackupRetentionDays
```

### Stack Updates

Support for zero-downtime stack updates:

```bash
# Update stack with change sets
aws cloudformation create-change-set \
  --stack-name novacron-prod \
  --template-url https://s3.amazonaws.com/novacron-templates/main.yaml \
  --change-set-name update-v3-1 \
  --parameters ParameterKey=InstanceType,ParameterValue=m5.2xlarge

# Review changes
aws cloudformation describe-change-set \
  --stack-name novacron-prod \
  --change-set-name update-v3-1

# Execute change set
aws cloudformation execute-change-set \
  --stack-name novacron-prod \
  --change-set-name update-v3-1
```

## Service Catalog Integration

### Portfolio Management

Create Service Catalog portfolios:

```go
portfolio := aws.ServiceCatalogPortfolio{
    DisplayName: "NovaCron Solutions",
    Description: "Enterprise VM Management Solutions",
    ProviderName: "NovaCron Technologies",
}

portfolioID, err := apn.CreatePortfolio(ctx, portfolio)
```

### Product Versions

Manage multiple product versions:

```go
// Add new version
err := apn.CreateProvisioningArtifact(ctx, productID, "3.1.0",
    "https://s3.amazonaws.com/novacron-templates/v3.1.0/main.yaml")
```

### Launch Constraints

Define launch constraints for security:

```yaml
LaunchRole:
  Type: AWS::IAM::Role
  Properties:
    AssumeRolePolicyDocument:
      Statement:
        - Effect: Allow
          Principal:
            Service: servicecatalog.amazonaws.com
          Action: sts:AssumeRole
    Policies:
      - PolicyName: NovaCronLaunchPolicy
        PolicyDocument:
          Statement:
            - Effect: Allow
              Action:
                - ec2:*
                - rds:*
                - elasticloadbalancing:*
              Resource: '*'
```

## AWS Organizations Support

### Multi-Account Strategy

NovaCron supports AWS multi-account best practices:

```
Management Account
├── Security OU
│   ├── Security Tooling Account
│   └── Log Archive Account
├── Infrastructure OU
│   ├── Shared Services Account
│   └── Network Account
└── Workloads OU
    ├── Development Account
    ├── Staging Account
    └── Production Account
```

### StackSets Deployment

Deploy across accounts and regions:

```bash
aws cloudformation create-stack-set \
  --stack-set-name novacron-global \
  --template-url https://s3.amazonaws.com/novacron-templates/global.yaml \
  --capabilities CAPABILITY_IAM \
  --permission-model SERVICE_MANAGED \
  --auto-deployment Enabled=true

aws cloudformation create-stack-instances \
  --stack-set-name novacron-global \
  --deployment-targets OrganizationalUnitIds=ou-12345 \
  --regions us-east-1 eu-west-1 ap-southeast-1
```

## Control Tower Integration

### Landing Zone Integration

NovaCron integrates with AWS Control Tower:

1. **Account Factory Customization**
   - Automatic NovaCron deployment on new accounts
   - Baseline configuration enforcement

2. **Guardrails Compliance**
   - Mandatory guardrails: Fully compliant
   - Strongly recommended: Fully compliant
   - Elective: Configurable

3. **Customizations for Control Tower (CfCT)**

```yaml
# manifest.yaml
resources:
  - name: NovaCronDeployment
    description: Deploy NovaCron to new accounts
    resource_file: novacron-baseline.yaml
    deploy_method: stack_set
    deployment_targets:
      organizational_units:
        - Workloads
```

## Best Practices

### 1. Cost Optimization

```go
// Use savings plans
savingsPlans := []string{"compute-sp-1year-no-upfront"}

// Implement tagging strategy
tags := map[string]string{
    "Product":     "NovaCron",
    "Environment": "Production",
    "CostCenter":  "Engineering",
    "Owner":       "platform-team",
}

// Monitor costs
cloudwatch.PutMetricData(&cloudwatch.PutMetricDataInput{
    Namespace: aws.String("NovaCron/Costs"),
    MetricData: []*cloudwatch.MetricDatum{
        {
            MetricName: aws.String("HourlyCost"),
            Value:      aws.Float64(currentCost),
            Unit:       aws.String("None"),
            Timestamp:  aws.Time(time.Now()),
        },
    },
})
```

### 2. Security

```go
// Enable encryption at rest
dbConfig := rds.CreateDBInstanceInput{
    StorageEncrypted: aws.Bool(true),
    KmsKeyId:         aws.String(kmsKeyArn),
}

// Enable VPC endpoints
vpcEndpoints := []string{
    "com.amazonaws.us-east-1.s3",
    "com.amazonaws.us-east-1.ec2",
    "com.amazonaws.us-east-1.rds",
}

// Use Secrets Manager
secretsManager.CreateSecret(&secretsmanager.CreateSecretInput{
    Name:         aws.String("novacron/db/password"),
    SecretString: aws.String(dbPassword),
})
```

### 3. High Availability

```go
// Multi-AZ deployment
multiAZ := true

// Cross-region replication
replicationConfig := s3.PutBucketReplicationInput{
    ReplicationConfiguration: &s3.ReplicationConfiguration{
        Rules: []*s3.ReplicationRule{
            {
                Status: aws.String("Enabled"),
                Destination: &s3.Destination{
                    Bucket: aws.String("arn:aws:s3:::novacron-backup-eu-west-1"),
                },
            },
        },
    },
}
```

### 4. Monitoring & Alerting

```go
// CloudWatch alarms
alarm := cloudwatch.PutMetricAlarmInput{
    AlarmName:          aws.String("NovaCron-HighCPU"),
    ComparisonOperator: aws.String("GreaterThanThreshold"),
    EvaluationPeriods:  aws.Int64(2),
    MetricName:         aws.String("CPUUtilization"),
    Namespace:          aws.String("AWS/EC2"),
    Period:             aws.Int64(300),
    Statistic:          aws.String("Average"),
    Threshold:          aws.Float64(80.0),
    AlarmActions: []*string{
        aws.String(snsTopicArn),
    },
}
```

## Support & Resources

- AWS Marketplace: https://aws.amazon.com/marketplace/seller-profile?id=novacron
- Documentation: https://docs.novacron.io/aws
- Support: aws-support@novacron.io
- Partner Portal: https://partner.novacron.io

## License

Copyright © 2024 NovaCron Technologies. All rights reserved.
