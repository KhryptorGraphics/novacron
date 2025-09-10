# NovaCron Terraform Variables
# Variable definitions for infrastructure deployment

# Project Configuration
variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "novacron"
}

variable "environment" {
  description = "Environment name (dev, qa, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "qa", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, qa, staging, production."
  }
}

variable "region" {
  description = "Google Cloud region"
  type        = string
  default     = "us-west2"
}

# Network Configuration
variable "subnet_cidr" {
  description = "CIDR range for the subnet"
  type        = string
  default     = "10.0.0.0/16"
}

variable "pod_cidr" {
  description = "CIDR range for Kubernetes pods"
  type        = string
  default     = "10.244.0.0/16"
}

variable "service_cidr" {
  description = "CIDR range for Kubernetes services"
  type        = string
  default     = "10.96.0.0/16"
}

variable "master_cidr" {
  description = "CIDR range for GKE master nodes"
  type        = string
  default     = "10.100.0.0/28"
}

# Kubernetes Configuration
variable "kubernetes_version" {
  description = "Kubernetes version for GKE cluster"
  type        = string
  default     = "1.28.3-gke.1286000"
}

# Node Pool Configuration - System
variable "system_pool_min_nodes" {
  description = "Minimum number of nodes in system pool"
  type        = number
  default     = 1
}

variable "system_pool_max_nodes" {
  description = "Maximum number of nodes in system pool"
  type        = number
  default     = 3
}

variable "system_pool_machine_type" {
  description = "Machine type for system node pool"
  type        = string
  default     = "e2-standard-4"
}

# Node Pool Configuration - Application
variable "app_pool_min_nodes" {
  description = "Minimum number of nodes in application pool"
  type        = number
  default     = 2
}

variable "app_pool_max_nodes" {
  description = "Maximum number of nodes in application pool"
  type        = number
  default     = 10
}

variable "app_pool_machine_type" {
  description = "Machine type for application node pool"
  type        = string
  default     = "e2-standard-8"
}

# Node Pool Configuration - Monitoring
variable "monitoring_pool_min_nodes" {
  description = "Minimum number of nodes in monitoring pool"
  type        = number
  default     = 1
}

variable "monitoring_pool_max_nodes" {
  description = "Maximum number of nodes in monitoring pool"
  type        = number
  default     = 3
}

variable "monitoring_pool_machine_type" {
  description = "Machine type for monitoring node pool"
  type        = string
  default     = "e2-standard-4"
}

# Database Configuration
variable "db_tier" {
  description = "Database instance tier"
  type        = string
  default     = "db-custom-2-8192"
}

variable "db_disk_size" {
  description = "Database disk size in GB"
  type        = number
  default     = 100
}

variable "db_max_disk_size" {
  description = "Maximum database disk size in GB for auto-resize"
  type        = number
  default     = 1000
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

# Redis Configuration
variable "redis_memory_size" {
  description = "Redis memory size in GB"
  type        = number
  default     = 4
}

# DNS Configuration
variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "novacron.com"
}

variable "manage_dns" {
  description = "Whether to manage DNS records"
  type        = bool
  default     = false
}

# Environment-specific configurations
locals {
  env_configs = {
    dev = {
      system_pool_min_nodes      = 1
      system_pool_max_nodes      = 2
      app_pool_min_nodes         = 1
      app_pool_max_nodes         = 3
      monitoring_pool_min_nodes  = 1
      monitoring_pool_max_nodes  = 2
      db_tier                    = "db-custom-1-4096"
      redis_memory_size          = 2
      system_pool_machine_type   = "e2-standard-2"
      app_pool_machine_type      = "e2-standard-4"
      monitoring_pool_machine_type = "e2-standard-2"
    }
    
    qa = {
      system_pool_min_nodes      = 2
      system_pool_max_nodes      = 3
      app_pool_min_nodes         = 2
      app_pool_max_nodes         = 5
      monitoring_pool_min_nodes  = 1
      monitoring_pool_max_nodes  = 2
      db_tier                    = "db-custom-2-8192"
      redis_memory_size          = 4
      system_pool_machine_type   = "e2-standard-2"
      app_pool_machine_type      = "e2-standard-4"
      monitoring_pool_machine_type = "e2-standard-2"
    }
    
    staging = {
      system_pool_min_nodes      = 2
      system_pool_max_nodes      = 3
      app_pool_min_nodes         = 3
      app_pool_max_nodes         = 8
      monitoring_pool_min_nodes  = 2
      monitoring_pool_max_nodes  = 3
      db_tier                    = "db-custom-4-16384"
      redis_memory_size          = 8
      system_pool_machine_type   = "e2-standard-4"
      app_pool_machine_type      = "e2-standard-8"
      monitoring_pool_machine_type = "e2-standard-4"
    }
    
    production = {
      system_pool_min_nodes      = 3
      system_pool_max_nodes      = 5
      app_pool_min_nodes         = 5
      app_pool_max_nodes         = 20
      monitoring_pool_min_nodes  = 2
      monitoring_pool_max_nodes  = 5
      db_tier                    = "db-custom-8-32768"
      redis_memory_size          = 16
      system_pool_machine_type   = "e2-standard-4"
      app_pool_machine_type      = "e2-standard-16"
      monitoring_pool_machine_type = "e2-standard-8"
    }
  }
  
  # Current environment config
  current_config = local.env_configs[var.environment]
}

# Override defaults with environment-specific values
variable "env_system_pool_min_nodes" {
  description = "Environment-specific minimum nodes for system pool"
  type        = number
  default     = null
}

variable "env_system_pool_max_nodes" {
  description = "Environment-specific maximum nodes for system pool"
  type        = number
  default     = null
}

variable "env_app_pool_min_nodes" {
  description = "Environment-specific minimum nodes for application pool"
  type        = number
  default     = null
}

variable "env_app_pool_max_nodes" {
  description = "Environment-specific maximum nodes for application pool"
  type        = number
  default     = null
}

variable "env_monitoring_pool_min_nodes" {
  description = "Environment-specific minimum nodes for monitoring pool"
  type        = number
  default     = null
}

variable "env_monitoring_pool_max_nodes" {
  description = "Environment-specific maximum nodes for monitoring pool"
  type        = number
  default     = null
}

variable "env_db_tier" {
  description = "Environment-specific database tier"
  type        = string
  default     = null
}

variable "env_redis_memory_size" {
  description = "Environment-specific Redis memory size"
  type        = number
  default     = null
}

# Feature flags
variable "enable_workload_identity" {
  description = "Enable workload identity for the cluster"
  type        = bool
  default     = true
}

variable "enable_network_policy" {
  description = "Enable network policy for the cluster"
  type        = bool
  default     = true
}

variable "enable_private_nodes" {
  description = "Enable private nodes for the cluster"
  type        = bool
  default     = true
}

variable "enable_istio" {
  description = "Enable Istio service mesh"
  type        = bool
  default     = true
}

variable "enable_monitoring" {
  description = "Enable Google Cloud Monitoring"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable Google Cloud Logging"
  type        = bool
  default     = true
}

# Backup configuration
variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}

variable "enable_automated_backup" {
  description = "Enable automated backup"
  type        = bool
  default     = true
}

# Security configuration
variable "enable_shielded_nodes" {
  description = "Enable shielded GKE nodes"
  type        = bool
  default     = true
}

variable "enable_binary_authorization" {
  description = "Enable binary authorization"
  type        = bool
  default     = false
}

variable "enable_pod_security_policy" {
  description = "Enable pod security policy"
  type        = bool
  default     = true
}

# Labels
variable "labels" {
  description = "Additional labels to apply to resources"
  type        = map(string)
  default     = {}
}

# Tags
variable "tags" {
  description = "Additional tags to apply to resources"
  type        = list(string)
  default     = []
}