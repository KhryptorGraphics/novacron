# NovaCron Multi-Cloud Federation API Reference

## Overview

The NovaCron Multi-Cloud Federation API provides unified access to VM management, cost optimization, compliance monitoring, and cross-cloud migration across multiple cloud providers including AWS, Azure, GCP, and on-premise infrastructure.

## Base URL

```
https://api.novacron.com/api/multicloud
```

## Authentication

All API endpoints require authentication using JWT tokens. Include the token in the `Authorization` header:

```
Authorization: Bearer <jwt_token>
```

## Provider Management

### List Cloud Providers

Get a list of all registered cloud providers.

```http
GET /providers
```

**Response:**
```json
{
  "aws-prod": {
    "type": "aws",
    "name": "AWS Production",
    "regions": ["us-east-1", "us-west-2", "eu-west-1"],
    "capabilities": ["vm_live_migration", "auto_scaling", "spot_instances"]
  },
  "azure-prod": {
    "type": "azure",
    "name": "Azure Production",
    "regions": ["eastus", "westus2", "westeurope"],
    "capabilities": ["vm_live_migration", "auto_scaling"]
  }
}
```

### Register Cloud Provider

Register a new cloud provider with the system.

```http
POST /providers
```

**Request Body:**
```json
{
  "provider_id": "aws-dev",
  "type": "aws",
  "config": {
    "type": "aws",
    "name": "AWS Development",
    "credentials": {
      "access_key_id": "AKIA...",
      "secret_access_key": "..."
    },
    "default_region": "us-east-1",
    "regions": ["us-east-1", "us-west-2"],
    "endpoints": {
      "api": "https://ec2.us-east-1.amazonaws.com"
    },
    "options": {
      "retry_attempts": 3,
      "timeout": 30
    }
  }
}
```

**Response:**
```json
{
  "message": "Provider registered successfully",
  "provider_id": "aws-dev"
}
```

### Get Provider Details

Get detailed information about a specific provider.

```http
GET /providers/{providerId}
```

**Response:**
```json
{
  "provider_id": "aws-prod",
  "type": "aws",
  "name": "AWS Production",
  "regions": ["us-east-1", "us-west-2", "eu-west-1"],
  "capabilities": ["vm_live_migration", "auto_scaling", "spot_instances"],
  "config": {
    "type": "aws",
    "name": "AWS Production",
    "default_region": "us-east-1",
    "regions": ["us-east-1", "us-west-2", "eu-west-1"]
  }
}
```

### Provider Health Status

Get the current health status of a provider.

```http
GET /providers/{providerId}/health
```

**Response:**
```json
{
  "provider": "aws",
  "overall": "healthy",
  "services": {
    "ec2": "healthy",
    "ebs": "healthy",
    "vpc": "healthy"
  },
  "regions": {
    "us-east-1": "healthy",
    "us-west-2": "healthy",
    "eu-west-1": "degraded"
  },
  "last_checked": "2024-01-15T10:30:00Z",
  "issues": []
}
```

### Provider Metrics

Get performance metrics for a provider.

```http
GET /providers/{providerId}/metrics
```

**Response:**
```json
{
  "provider_id": "aws-prod",
  "request_count": 1250,
  "error_count": 15,
  "success_rate": 98.8,
  "avg_response_time": "250ms",
  "last_updated": "2024-01-15T10:30:00Z"
}
```

## Virtual Machine Management

### List VMs

List VMs across all providers or filter by specific criteria.

```http
GET /vms?provider_id=aws-prod&region=us-east-1&state=running
```

**Query Parameters:**
- `provider_id` (optional): Filter by provider ID
- `provider_type` (optional): Filter by provider type (aws, azure, gcp)
- `region` (optional): Filter by region
- `state` (optional): Filter by VM state
- `name_pattern` (optional): Filter by name pattern

**Response:**
```json
{
  "vms": [
    {
      "id": "i-1234567890abcdef0",
      "name": "web-server-01",
      "provider": "aws",
      "region": "us-east-1",
      "availability_zone": "us-east-1a",
      "instance_type": "t3.medium",
      "state": "running",
      "public_ip": "54.123.45.67",
      "private_ip": "10.0.1.100",
      "cpu": 2,
      "memory": 4096,
      "storage": 20,
      "hourly_rate": 0.0416,
      "monthly_estimate": 30.00,
      "tags": {
        "environment": "production",
        "team": "web"
      },
      "created_at": "2024-01-15T08:00:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "count": 1
}
```

### Create VM

Create a new VM using intelligent provider selection.

```http
POST /vms
```

**Request Body:**
```json
{
  "name": "production-web-server",
  "instance_type": "medium",
  "image_id": "ami-12345678",
  "region": "us-east-1",
  "availability_zone": "us-east-1a",
  "security_groups": ["sg-web", "sg-ssh"],
  "tags": {
    "environment": "production",
    "team": "web-services"
  },
  "cpu": 2,
  "memory": 4096,
  "storage": 20,
  "network_bandwidth": 1000,
  
  "preferred_provider": "aws-prod",
  "required_capabilities": ["auto_scaling", "load_balancing"],
  "cost_optimized": true,
  "low_latency": false,
  "high_availability": true,
  
  "spot_instance": false,
  "compliance_requirements": ["SOC2", "GDPR"],
  "data_residency_regions": ["us-east-1", "us-west-2"],
  
  "custom_options": {
    "monitoring_enabled": true,
    "ebs_optimized": true
  }
}
```

**Response:**
```json
{
  "id": "i-newvm1234567890",
  "name": "production-web-server",
  "provider": "aws",
  "region": "us-east-1",
  "state": "pending",
  "created_at": "2024-01-15T11:00:00Z",
  "updated_at": "2024-01-15T11:00:00Z"
}
```

### Get VM Details

Get detailed information about a specific VM.

```http
GET /vms/{vmId}
```

**Response:**
```json
{
  "id": "i-1234567890abcdef0",
  "name": "web-server-01",
  "provider": "aws",
  "region": "us-east-1",
  "availability_zone": "us-east-1a",
  "instance_type": "t3.medium",
  "state": "running",
  "public_ip": "54.123.45.67",
  "private_ip": "10.0.1.100",
  "image_id": "ami-12345678",
  "key_pair": "my-keypair",
  "security_groups": ["sg-web", "sg-ssh"],
  "tags": {
    "environment": "production",
    "team": "web"
  },
  "cpu": 2,
  "memory": 4096,
  "storage": 20,
  "network_bandwidth": 1000,
  "hourly_rate": 0.0416,
  "monthly_estimate": 30.00,
  "created_at": "2024-01-15T08:00:00Z",
  "updated_at": "2024-01-15T10:30:00Z",
  "metadata": {
    "launch_template": "lt-12345",
    "placement_group": null
  }
}
```

## Cross-Cloud Migration

### List Migrations

List all cross-cloud migrations with optional filtering.

```http
GET /migrations?status=in_progress&source_provider=aws-prod
```

**Query Parameters:**
- `status` (optional): Filter by migration status
- `source_provider` (optional): Filter by source provider
- `destination_provider` (optional): Filter by destination provider
- `vm_id` (optional): Filter by VM ID

**Response:**
```json
{
  "migrations": [
    {
      "migration_id": "migration-1234567890",
      "vm_id": "i-1234567890abcdef0",
      "source_provider_id": "aws-prod",
      "destination_provider_id": "azure-prod",
      "status": "in_progress",
      "progress": 65,
      "started_at": "2024-01-15T10:00:00Z",
      "estimated_completion": "2024-01-15T11:30:00Z"
    }
  ],
  "count": 1
}
```

### Create Migration

Start a new cross-cloud migration.

```http
POST /migrations
```

**Request Body:**
```json
{
  "vm_id": "i-1234567890abcdef0",
  "source_provider_id": "aws-prod",
  "destination_provider_id": "azure-prod",
  "destination_region": "eastus",
  "destination_config": {
    "instance_type": "Standard_D2s_v3",
    "availability_zone": "eastus-1",
    "security_groups": ["nsg-web"]
  },
  "delete_source": false,
  "max_downtime": "5m",
  "rollback": true,
  "options": {
    "compress_data": true,
    "verify_integrity": true
  }
}
```

**Response:**
```json
{
  "migration_id": "migration-1234567890",
  "vm_id": "i-1234567890abcdef0",
  "source_provider_id": "aws-prod",
  "destination_provider_id": "azure-prod",
  "status": "pending",
  "progress": 0,
  "steps": [
    {
      "name": "Pre-migration Validation",
      "status": "pending",
      "progress": 0
    }
  ],
  "start_time": "2024-01-15T11:00:00Z",
  "estimated_completion": "2024-01-15T12:30:00Z"
}
```

### Get Migration Status

Get detailed status of a migration.

```http
GET /migrations/{migrationId}/status
```

**Response:**
```json
{
  "migration_id": "migration-1234567890",
  "vm_id": "i-1234567890abcdef0",
  "source_provider_id": "aws-prod",
  "destination_provider_id": "azure-prod",
  "status": "in_progress",
  "progress": 65,
  "steps": [
    {
      "name": "Pre-migration Validation",
      "status": "completed",
      "start_time": "2024-01-15T10:00:00Z",
      "end_time": "2024-01-15T10:05:00Z",
      "progress": 100
    },
    {
      "name": "VM Export",
      "status": "completed",
      "start_time": "2024-01-15T10:05:00Z",
      "end_time": "2024-01-15T10:45:00Z",
      "progress": 100,
      "details": {
        "format": "vmdk",
        "size": 10737418240
      }
    },
    {
      "name": "Data Transformation",
      "status": "in_progress",
      "start_time": "2024-01-15T10:45:00Z",
      "progress": 60
    }
  ],
  "start_time": "2024-01-15T10:00:00Z",
  "estimated_completion": "2024-01-15T11:30:00Z",
  "metrics": {
    "data_transferred": 6442450944,
    "transfer_rate": 52428800,
    "network_latency": "15ms"
  }
}
```

## Cost Management

### Cost Analysis

Analyze costs across multiple cloud providers.

```http
POST /cost/analysis
```

**Request Body:**
```json
{
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-31T23:59:59Z",
  "period": "monthly",
  "provider_ids": ["aws-prod", "azure-prod"],
  "regions": ["us-east-1", "eastus"],
  "resource_types": ["vm", "storage"]
}
```

**Response:**
```json
{
  "analysis_id": "cost-analysis-1234567890",
  "requested_at": "2024-01-15T11:00:00Z",
  "completed_at": "2024-01-15T11:05:00Z",
  "period": "monthly",
  "total_cost": 2547.85,
  "currency": "USD",
  "by_provider": {
    "aws-prod": {
      "provider": "aws-prod",
      "total_cost": 1425.30,
      "currency": "USD",
      "by_region": {
        "us-east-1": {
          "region": "us-east-1",
          "total_cost": 1425.30,
          "resource_count": 15
        }
      },
      "by_resource_type": {
        "vm": {
          "resource_type": "vm",
          "total_cost": 1200.00,
          "resource_count": 12
        },
        "storage": {
          "resource_type": "storage",
          "total_cost": 225.30,
          "resource_count": 8
        }
      }
    }
  },
  "insights": [
    {
      "type": "top_spending_provider",
      "title": "Highest Spending Provider",
      "description": "aws-prod accounts for 56.1% of total costs",
      "impact": "high",
      "value": 1425.30
    }
  ],
  "recommendations": [
    {
      "type": "reserved_instances",
      "title": "Reserved Instances for AWS",
      "description": "Consider reserved instances for stable workloads in aws-prod to save up to 30% on compute costs",
      "priority": "high",
      "potential_savings": 360.00,
      "implementation": "Purchase 1-year reserved instances for stable workloads"
    }
  ]
}
```

### Cost Optimization

Generate cost optimization recommendations.

```http
POST /cost/optimization
```

**Request Body:**
```json
{
  "start_time": "2024-01-01T00:00:00Z",
  "end_time": "2024-01-31T23:59:59Z",
  "period": "monthly",
  "provider_ids": ["aws-prod", "azure-prod"],
  "goals": {
    "target_savings_percentage": 25.0,
    "right_sizing": true,
    "provider_optimization": true,
    "reserved_instances": true,
    "spot_instances": true
  },
  "constraints": {
    "max_risk": "medium",
    "exclude_providers": [],
    "required_regions": ["us-east-1", "eastus"]
  }
}
```

**Response:**
```json
{
  "plan_id": "optimization-plan-1234567890",
  "requested_at": "2024-01-15T11:00:00Z",
  "completed_at": "2024-01-15T11:10:00Z",
  "actions": [
    {
      "type": "right_size",
      "provider": "aws-prod",
      "description": "Analyze and right-size underutilized resources in aws-prod",
      "impact": "medium",
      "risk": "low",
      "estimated_savings": 254.79,
      "implementation_steps": [
        "Analyze resource utilization metrics",
        "Identify underutilized resources",
        "Resize or terminate unused resources"
      ]
    },
    {
      "type": "migrate_provider",
      "provider": "aws-prod",
      "destination_provider": "azure-prod",
      "description": "Migrate workloads from aws-prod to azure-prod for cost savings",
      "impact": "high",
      "risk": "medium",
      "estimated_savings": 425.50,
      "implementation_steps": [
        "Assess migration compatibility",
        "Plan migration strategy",
        "Execute migration in phases",
        "Validate cost savings"
      ]
    }
  ],
  "potential_savings": {
    "total_potential_savings": 680.29,
    "high_impact_savings": 425.50,
    "medium_impact_savings": 254.79,
    "low_impact_savings": 0.0,
    "confidence": 0.85
  }
}
```

### Cost Forecast

Generate cost forecasts based on historical data and trends.

```http
POST /cost/forecast
```

**Request Body:**
```json
{
  "forecast_months": 6,
  "period": "monthly",
  "provider_ids": ["aws-prod", "azure-prod"],
  "growth_rate": 0.05
}
```

**Response:**
```json
{
  "forecast_id": "forecast-1234567890",
  "generated_at": "2024-01-15T11:00:00Z",
  "period": "monthly",
  "total_forecast_cost": 16287.45,
  "by_month": {
    "2024-02": {
      "month": "2024-02",
      "total_cost": 2675.25,
      "trend": "increasing",
      "confidence": 0.92
    },
    "2024-03": {
      "month": "2024-03",
      "total_cost": 2809.01,
      "trend": "increasing",
      "confidence": 0.88
    }
  },
  "confidence": 0.87,
  "assumptions": [
    "Current usage patterns continue",
    "No major architectural changes",
    "Provider pricing remains stable"
  ]
}
```

## Compliance Management

### Compliance Report

Generate a compliance report across all providers.

```http
POST /compliance/report
```

**Request Body:**
```json
{
  "frameworks": ["SOC2", "HIPAA", "GDPR"]
}
```

**Response:**
```json
{
  "report_id": "compliance-report-1234567890",
  "generated_at": "2024-01-15T11:00:00Z",
  "frameworks": ["SOC2", "HIPAA", "GDPR"],
  "overall_score": 92.3,
  "by_provider": {
    "aws-prod": {
      "provider_id": "aws-prod",
      "provider_type": "aws",
      "overall_score": 95.5,
      "compliance_status": {
        "provider": "aws",
        "overall_score": 95.5,
        "compliances": [
          {
            "name": "SOC2",
            "version": "2017",
            "score": 98.0,
            "status": "compliant",
            "controls": 150,
            "passed": 147,
            "failed": 3,
            "not_applicable": 0
          },
          {
            "name": "HIPAA",
            "version": "2013",
            "score": 94.0,
            "status": "compliant",
            "controls": 78,
            "passed": 73,
            "failed": 5,
            "not_applicable": 0
          }
        ],
        "data_residency": {
          "primary_region": "us-east-1",
          "allowed_regions": ["us-east-1", "us-west-2", "eu-west-1"],
          "restricted_regions": ["ap-southeast-1"],
          "data_location": "United States",
          "cross_border_transfer": true
        },
        "certifications": ["SOC 1", "SOC 2", "SOC 3", "ISO 27001", "PCI DSS"],
        "last_assessment": "2024-01-14T10:00:00Z"
      },
      "violations": []
    }
  },
  "summary": {
    "total_providers": 2,
    "compliant_providers": 2,
    "partially_compliant_providers": 0,
    "non_compliant_providers": 0,
    "total_violations": 0,
    "overall_score": 92.3
  },
  "recommendations": [
    {
      "type": "improve_compliance",
      "provider": "azure-prod",
      "priority": "medium",
      "description": "Provider azure-prod has room for improvement in GDPR compliance (87.5 score)"
    }
  ]
}
```

### Compliance Dashboard

Get compliance dashboard data for real-time monitoring.

```http
GET /compliance/dashboard
```

**Response:**
```json
{
  "overall_score": 92.3,
  "total_violations": 0,
  "by_provider": {
    "aws-prod": {
      "provider_id": "aws-prod",
      "overall_score": 95.5,
      "violation_count": 0,
      "framework_scores": {
        "SOC2": 98.0,
        "HIPAA": 94.0,
        "GDPR": 92.0
      }
    },
    "azure-prod": {
      "provider_id": "azure-prod",
      "overall_score": 89.1,
      "violation_count": 0,
      "framework_scores": {
        "SOC2": 95.0,
        "HIPAA": 90.5,
        "GDPR": 87.5
      }
    }
  },
  "last_updated": "2024-01-15T11:00:00Z"
}
```

### Set Compliance Policy

Create or update compliance policies.

```http
POST /compliance/policies
```

**Request Body:**
```json
{
  "id": "data-residency-policy-1",
  "name": "Data Residency Policy",
  "description": "Ensures data remains within approved geographic regions",
  "type": "data_residency",
  "enabled": true,
  "rules": [
    {
      "id": "rule-1",
      "type": "data_residency",
      "description": "Data must remain in US regions only",
      "parameters": {
        "allowed_regions": ["us-east-1", "us-west-2"],
        "cross_border_transfer": false
      },
      "severity": "critical"
    }
  ],
  "applicable_providers": ["aws-prod", "azure-prod"],
  "applicable_regions": ["us-east-1", "us-west-2"]
}
```

**Response:**
```json
{
  "message": "Compliance policy set successfully",
  "policy_id": "data-residency-policy-1"
}
```

## Multi-Cloud Policies

### List Policies

List all multi-cloud policies.

```http
GET /policies
```

**Response:**
```json
{
  "policies": [
    {
      "id": "cost-control-policy-1",
      "name": "Cost Control Policy",
      "description": "Prevents creation of expensive resources",
      "type": "vm_creation",
      "enabled": true,
      "priority": 100,
      "rules": [
        {
          "id": "rule-1",
          "name": "Max CPU Limit",
          "type": "resource_limits",
          "description": "Limit maximum CPU cores per VM",
          "parameters": {
            "max_cpu": 8,
            "max_memory": 32768
          },
          "enforcement": "hard"
        }
      ],
      "applicable_providers": ["aws-prod", "azure-prod"],
      "created_at": "2024-01-10T09:00:00Z",
      "updated_at": "2024-01-15T10:00:00Z"
    }
  ],
  "count": 1
}
```

### Set Policy

Create or update a multi-cloud policy.

```http
POST /policies
```

**Request Body:**
```json
{
  "id": "security-policy-1",
  "name": "Security Policy",
  "description": "Enforces security requirements for VM creation",
  "type": "vm_creation",
  "enabled": true,
  "priority": 200,
  "rules": [
    {
      "id": "rule-1",
      "name": "Encryption Required",
      "type": "security_requirements",
      "description": "All VMs must have encryption enabled",
      "parameters": {
        "encryption_required": true,
        "min_security_groups": 1
      },
      "enforcement": "hard"
    }
  ],
  "applicable_providers": ["aws-prod", "azure-prod"],
  "applicable_regions": ["us-east-1", "eastus"]
}
```

**Response:**
```json
{
  "message": "Policy set successfully",
  "policy_id": "security-policy-1"
}
```

## Resource Utilization

### Get Resource Utilization

Get resource utilization across all providers.

```http
GET /resources/utilization
```

**Response:**
```json
{
  "total_vms": 45,
  "total_cpu": 180,
  "total_memory": 737280,
  "total_storage": 2250,
  "total_cost": 3547.89,
  "by_provider": {
    "aws-prod": {
      "used_vms": 25,
      "used_cpu": 100,
      "used_memory": 409600,
      "used_storage": 1250,
      "total_cost": 1987.45
    },
    "azure-prod": {
      "used_vms": 20,
      "used_cpu": 80,
      "used_memory": 327680,
      "used_storage": 1000,
      "total_cost": 1560.44
    }
  },
  "by_region": {
    "us-east-1": {
      "region": "us-east-1",
      "total_cost": 2100.00,
      "resource_count": 28
    },
    "eastus": {
      "region": "eastus",
      "total_cost": 1447.89,
      "resource_count": 17
    }
  },
  "trends": {
    "cpu_trend": "stable",
    "memory_trend": "increasing",
    "storage_trend": "increasing",
    "cost_trend": "increasing",
    "trend_period": "30d"
  },
  "recommendations": [
    {
      "type": "consolidation",
      "resource": "aws-prod",
      "provider": "aws-prod",
      "description": "Low VM count (25) with significant cost. Consider consolidating workloads.",
      "potential": "cost savings",
      "confidence": 0.8,
      "impact": "medium"
    }
  ],
  "last_updated": "2024-01-15T11:00:00Z"
}
```

## Dashboard

### Get Multi-Cloud Dashboard

Get comprehensive dashboard data including provider health, metrics, utilization, and compliance.

```http
GET /dashboard
```

**Response:**
```json
{
  "providers": {
    "aws-prod": {
      "provider": "aws",
      "overall": "healthy",
      "services": {
        "ec2": "healthy",
        "ebs": "healthy",
        "vpc": "healthy"
      },
      "regions": {
        "us-east-1": "healthy",
        "us-west-2": "healthy"
      },
      "last_checked": "2024-01-15T11:00:00Z"
    }
  },
  "metrics": {
    "aws-prod": {
      "provider_id": "aws-prod",
      "request_count": 1250,
      "error_count": 15,
      "success_rate": 98.8,
      "avg_response_time": "250ms",
      "last_updated": "2024-01-15T11:00:00Z"
    }
  },
  "utilization": {
    "total_vms": 45,
    "total_cpu": 180,
    "total_memory": 737280,
    "total_storage": 2250,
    "total_cost": 3547.89,
    "last_updated": "2024-01-15T11:00:00Z"
  },
  "compliance": {
    "overall_score": 92.3,
    "total_violations": 0,
    "by_provider": {
      "aws-prod": {
        "provider_id": "aws-prod",
        "overall_score": 95.5,
        "violation_count": 0
      }
    },
    "last_updated": "2024-01-15T11:00:00Z"
  }
}
```

## Error Responses

All API endpoints may return the following error responses:

### 400 Bad Request
```json
{
  "error": "Invalid request body",
  "status": 400,
  "timestamp": "2024-01-15T11:00:00Z",
  "details": "Missing required field: provider_id"
}
```

### 401 Unauthorized
```json
{
  "error": "Authentication required",
  "status": 401,
  "timestamp": "2024-01-15T11:00:00Z",
  "details": "Invalid or missing JWT token"
}
```

### 404 Not Found
```json
{
  "error": "Resource not found",
  "status": 404,
  "timestamp": "2024-01-15T11:00:00Z",
  "details": "Provider with ID 'aws-dev' not found"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "status": 500,
  "timestamp": "2024-01-15T11:00:00Z",
  "details": "Failed to connect to cloud provider API"
}
```

## Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **Standard endpoints**: 1000 requests per hour per user
- **Resource-intensive operations**: 100 requests per hour per user
- **Migration operations**: 10 concurrent migrations per user

Rate limit headers are included in all responses:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642258800
```

## Webhooks

NovaCron supports webhooks for real-time notifications of important events:

### Supported Events
- `vm.created` - VM creation completed
- `vm.deleted` - VM deletion completed
- `migration.started` - Cross-cloud migration started
- `migration.completed` - Cross-cloud migration completed
- `migration.failed` - Cross-cloud migration failed
- `cost.threshold_exceeded` - Cost threshold exceeded
- `compliance.violation_detected` - Compliance violation detected
- `provider.health_degraded` - Provider health degraded

### Webhook Configuration
Webhooks can be configured through the admin interface or API. Each webhook includes:
- Event type
- Target URL
- Authentication (optional)
- Retry policy
- Filtering criteria

## SDK and Client Libraries

Official SDKs are available for popular programming languages:

- **Go**: `go get github.com/novacron/go-sdk`
- **Python**: `pip install novacron-sdk`
- **JavaScript/Node.js**: `npm install @novacron/sdk`
- **Java**: Maven/Gradle dependency available
- **C#/.NET**: NuGet package available

## Support and Documentation

- **API Documentation**: https://docs.novacron.com/api
- **SDK Documentation**: https://docs.novacron.com/sdk
- **Support Portal**: https://support.novacron.com
- **Community Forum**: https://community.novacron.com
- **Status Page**: https://status.novacron.com