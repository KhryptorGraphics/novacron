# NovaCron Infrastructure as Code - Variables
# Variable definitions for multi-environment deployments

# Environment Configuration
variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-east-1"
}

variable "secondary_region" {
  description = "Secondary AWS region for disaster recovery"
  type        = string
  default     = "us-west-2"
}

# Network Configuration
variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones_count" {
  description = "Number of availability zones to use"
  type        = number
  default     = 3
  
  validation {
    condition     = var.availability_zones_count >= 2 && var.availability_zones_count <= 6
    error_message = "Availability zones count must be between 2 and 6."
  }
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the infrastructure"
  type        = list(string)
  default     = ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"]
}

# Application Configuration
variable "application_name" {
  description = "Name of the application"
  type        = string
  default     = "novacron"
}

variable "application_version" {
  description = "Version of the application to deploy"
  type        = string
  default     = "latest"
}

variable "application_port" {
  description = "Port on which the application runs"
  type        = number
  default     = 8080
}

variable "health_check_path" {
  description = "Health check endpoint path"
  type        = string
  default     = "/health"
}

# Compute Configuration
variable "eks_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "node_group_instance_types" {
  description = "EC2 instance types for EKS node groups"
  type        = list(string)
  default     = ["t3.medium", "t3.large"]
}

variable "node_group_desired_size" {
  description = "Desired number of nodes in the node group"
  type        = number
  default     = 3
}

variable "node_group_min_size" {
  description = "Minimum number of nodes in the node group"
  type        = number
  default     = 1
}

variable "node_group_max_size" {
  description = "Maximum number of nodes in the node group"
  type        = number
  default     = 10
}

variable "enable_cluster_autoscaler" {
  description = "Enable cluster autoscaler for EKS"
  type        = bool
  default     = true
}

variable "enable_node_group_spot_instances" {
  description = "Enable spot instances for cost optimization"
  type        = bool
  default     = false
}

# Database Configuration
variable "db_engine" {
  description = "Database engine"
  type        = string
  default     = "postgres"
  
  validation {
    condition     = contains(["postgres", "mysql"], var.db_engine)
    error_message = "Database engine must be either postgres or mysql."
  }
}

variable "db_engine_version" {
  description = "Database engine version"
  type        = string
  default     = "15.4"
}

variable "db_instance_class" {
  description = "RDS instance class"
  type        = string
  default     = "db.t3.micro"
}

variable "db_allocated_storage" {
  description = "Initial storage allocation for RDS instance (GB)"
  type        = number
  default     = 20
}

variable "db_max_allocated_storage" {
  description = "Maximum storage allocation for RDS instance (GB)"
  type        = number
  default     = 100
}

variable "db_backup_retention_period" {
  description = "Backup retention period in days"
  type        = number
  default     = 7
}

variable "enable_db_multi_az" {
  description = "Enable Multi-AZ deployment for RDS"
  type        = bool
  default     = true
}

variable "enable_db_encryption" {
  description = "Enable encryption at rest for RDS"
  type        = bool
  default     = true
}

variable "db_snapshot_identifier" {
  description = "Snapshot ID to restore database from"
  type        = string
  default     = null
}

# Storage Configuration
variable "enable_s3_versioning" {
  description = "Enable S3 bucket versioning"
  type        = bool
  default     = true
}

variable "s3_lifecycle_rules" {
  description = "S3 bucket lifecycle rules"
  type = list(object({
    id     = string
    status = string
    transition = list(object({
      days          = number
      storage_class = string
    }))
    expiration = object({
      days = number
    })
  }))
  default = [
    {
      id     = "logs"
      status = "Enabled"
      transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        }
      ]
      expiration = {
        days = 365
      }
    }
  ]
}

# Monitoring Configuration
variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "enable_container_insights" {
  description = "Enable CloudWatch Container Insights for EKS"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention period in days"
  type        = number
  default     = 30
  
  validation {
    condition = contains([1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653], var.log_retention_days)
    error_message = "Log retention days must be a valid CloudWatch retention period."
  }
}

variable "prometheus_storage_size" {
  description = "Storage size for Prometheus (GB)"
  type        = string
  default     = "20Gi"
}

variable "grafana_admin_password" {
  description = "Admin password for Grafana (will be auto-generated if not provided)"
  type        = string
  default     = null
  sensitive   = true
}

# Security Configuration
variable "enable_waf" {
  description = "Enable AWS WAF for application protection"
  type        = bool
  default     = true
}

variable "enable_shield_advanced" {
  description = "Enable AWS Shield Advanced for DDoS protection"
  type        = bool
  default     = false
}

variable "certificate_domain_name" {
  description = "Domain name for SSL certificate"
  type        = string
  default     = "*.novacron.com"
}

variable "certificate_validation_method" {
  description = "Method for certificate validation"
  type        = string
  default     = "DNS"
  
  validation {
    condition     = contains(["DNS", "EMAIL"], var.certificate_validation_method)
    error_message = "Certificate validation method must be either DNS or EMAIL."
  }
}

# Backup and Disaster Recovery
variable "enable_automated_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_schedule" {
  description = "Cron expression for backup schedule"
  type        = string
  default     = "0 2 * * *"  # Daily at 2 AM
}

variable "cross_region_backup" {
  description = "Enable cross-region backup replication"
  type        = bool
  default     = true
}

# Cost Optimization
variable "enable_spot_instances" {
  description = "Enable spot instances for non-critical workloads"
  type        = bool
  default     = false
}

variable "enable_scheduled_scaling" {
  description = "Enable scheduled scaling for predictable workloads"
  type        = bool
  default     = false
}

variable "off_hours_schedule" {
  description = "Schedule for scaling down during off hours (cron format)"
  type        = string
  default     = "0 20 * * 1-5"  # 8 PM on weekdays
}

variable "on_hours_schedule" {
  description = "Schedule for scaling up during business hours (cron format)"
  type        = string
  default     = "0 8 * * 1-5"   # 8 AM on weekdays
}

# Multi-Cloud Configuration
variable "enable_azure_integration" {
  description = "Enable Azure cloud integration"
  type        = bool
  default     = false
}

variable "enable_gcp_integration" {
  description = "Enable Google Cloud Platform integration"
  type        = bool
  default     = false
}

variable "azure_resource_group_name" {
  description = "Azure resource group name"
  type        = string
  default     = "novacron-rg"
}

variable "azure_location" {
  description = "Azure region location"
  type        = string
  default     = "East US"
}

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
  default     = "novacron-project"
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

# Feature Flags
variable "enable_blue_green_deployment" {
  description = "Enable blue-green deployment strategy"
  type        = bool
  default     = false
}

variable "enable_canary_deployment" {
  description = "Enable canary deployment strategy"
  type        = bool
  default     = true
}

variable "enable_chaos_engineering" {
  description = "Enable chaos engineering tools"
  type        = bool
  default     = false
}

variable "enable_service_mesh" {
  description = "Enable service mesh (Istio)"
  type        = bool
  default     = false
}

# Tags
variable "additional_tags" {
  description = "Additional tags to apply to resources"
  type        = map(string)
  default     = {}
}

# Environment-specific configurations
variable "environment_configs" {
  description = "Environment-specific configuration overrides"
  type = map(object({
    instance_types      = list(string)
    min_size           = number
    max_size           = number
    desired_size       = number
    db_instance_class  = string
    enable_multi_az    = bool
    backup_retention   = number
  }))
  default = {
    dev = {
      instance_types     = ["t3.small"]
      min_size          = 1
      max_size          = 3
      desired_size      = 1
      db_instance_class = "db.t3.micro"
      enable_multi_az   = false
      backup_retention  = 1
    }
    staging = {
      instance_types     = ["t3.medium"]
      min_size          = 2
      max_size          = 6
      desired_size      = 2
      db_instance_class = "db.t3.small"
      enable_multi_az   = true
      backup_retention  = 3
    }
    prod = {
      instance_types     = ["t3.large", "t3.xlarge"]
      min_size          = 3
      max_size          = 20
      desired_size      = 5
      db_instance_class = "db.r5.large"
      enable_multi_az   = true
      backup_retention  = 30
    }
  }
}