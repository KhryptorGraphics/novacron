# Project Configuration
variable "project_name" {
  description = "Project name"
  type        = string
  default     = "dwcp-v3"
}

variable "environment" {
  description = "Environment name (staging, production)"
  type        = string
  validation {
    condition     = contains(["staging", "production"], var.environment)
    error_message = "Environment must be staging or production."
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

# Networking
variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "private_subnet_cidrs" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.11.0/24", "10.0.12.0/24", "10.0.13.0/24"]
}

variable "allowed_cidr_blocks" {
  description = "Allowed CIDR blocks for ingress"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "monitoring_cidr_blocks" {
  description = "CIDR blocks for monitoring access"
  type        = list(string)
  default     = ["10.0.0.0/16"]
}

variable "enable_nat_gateway" {
  description = "Enable NAT Gateway for private subnets"
  type        = bool
  default     = true
}

# Load Balancer
variable "enable_https" {
  description = "Enable HTTPS listener"
  type        = bool
  default     = true
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS"
  type        = string
  default     = ""
}

# Kubernetes
variable "kubeconfig_path" {
  description = "Path to kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "kubernetes_namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "dwcp-v3"
}

# Application Configuration
variable "app_replicas" {
  description = "Number of application replicas"
  type        = number
  default     = 3
}

variable "app_cpu_request" {
  description = "CPU request for application pods"
  type        = string
  default     = "500m"
}

variable "app_memory_request" {
  description = "Memory request for application pods"
  type        = string
  default     = "512Mi"
}

variable "app_cpu_limit" {
  description = "CPU limit for application pods"
  type        = string
  default     = "2000m"
}

variable "app_memory_limit" {
  description = "Memory limit for application pods"
  type        = string
  default     = "2Gi"
}

# Auto Scaling
variable "enable_autoscaling" {
  description = "Enable horizontal pod autoscaling"
  type        = bool
  default     = true
}

variable "min_replicas" {
  description = "Minimum number of replicas for autoscaling"
  type        = number
  default     = 3
}

variable "max_replicas" {
  description = "Maximum number of replicas for autoscaling"
  type        = number
  default     = 10
}

variable "cpu_utilization_target" {
  description = "Target CPU utilization percentage for autoscaling"
  type        = number
  default     = 70
}

variable "memory_utilization_target" {
  description = "Target memory utilization percentage for autoscaling"
  type        = number
  default     = 80
}

# Storage
variable "storage_class" {
  description = "Kubernetes storage class"
  type        = string
  default     = "standard"
}

variable "data_volume_size" {
  description = "Size of data persistent volume"
  type        = string
  default     = "10Gi"
}

variable "redis_volume_size" {
  description = "Size of Redis persistent volume"
  type        = string
  default     = "5Gi"
}

# Monitoring
variable "enable_monitoring" {
  description = "Enable Prometheus and Grafana monitoring"
  type        = bool
  default     = true
}

variable "prometheus_retention" {
  description = "Prometheus data retention period"
  type        = string
  default     = "15d"
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

# Feature Flags
variable "enable_feature_flags" {
  description = "Enable feature flag system"
  type        = bool
  default     = true
}

variable "default_rollout_percentage" {
  description = "Default feature rollout percentage"
  type        = number
  default     = 100
  validation {
    condition     = var.default_rollout_percentage >= 0 && var.default_rollout_percentage <= 100
    error_message = "Rollout percentage must be between 0 and 100."
  }
}

# Security
variable "enable_pod_security_policy" {
  description = "Enable pod security policies"
  type        = bool
  default     = true
}

variable "enable_network_policies" {
  description = "Enable Kubernetes network policies"
  type        = bool
  default     = true
}

# Backup
variable "enable_backups" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Backup retention period in days"
  type        = number
  default     = 30
}

# Tags
variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project   = "DWCP v3"
    ManagedBy = "Terraform"
    Component = "Infrastructure"
  }
}

# Redis Configuration
variable "redis_memory_limit" {
  description = "Redis memory limit"
  type        = string
  default     = "512Mi"
}

variable "redis_cpu_limit" {
  description = "Redis CPU limit"
  type        = string
  default     = "500m"
}

# Health Check
variable "health_check_path" {
  description = "Health check endpoint path"
  type        = string
  default     = "/health"
}

variable "health_check_interval" {
  description = "Health check interval in seconds"
  type        = number
  default     = 30
}

variable "health_check_timeout" {
  description = "Health check timeout in seconds"
  type        = number
  default     = 5
}

# DWCP v3 Specific
variable "dwcp_version" {
  description = "DWCP version"
  type        = string
  default     = "3.0.0"
}

variable "enable_codec_v3" {
  description = "Enable DWCP v3 codec"
  type        = bool
  default     = true
}

variable "enable_multiplexing" {
  description = "Enable DWCP v3 multiplexing"
  type        = bool
  default     = true
}

variable "enable_flow_control" {
  description = "Enable DWCP v3 flow control"
  type        = bool
  default     = true
}

variable "max_connections" {
  description = "Maximum concurrent connections"
  type        = number
  default     = 10000
}

variable "buffer_size" {
  description = "Buffer size in bytes"
  type        = number
  default     = 65536
}
