# NovaCron Terraform Outputs
# Output values for use by other Terraform configurations and deployment scripts

# Cluster Information
output "cluster_name" {
  description = "GKE cluster name"
  value       = google_container_cluster.primary.name
}

output "cluster_endpoint" {
  description = "GKE cluster endpoint"
  value       = google_container_cluster.primary.endpoint
  sensitive   = true
}

output "cluster_ca_certificate" {
  description = "GKE cluster CA certificate"
  value       = google_container_cluster.primary.master_auth.0.cluster_ca_certificate
  sensitive   = true
}

output "cluster_location" {
  description = "GKE cluster location"
  value       = google_container_cluster.primary.location
}

output "cluster_zone" {
  description = "GKE cluster zone"
  value       = google_container_cluster.primary.zone
}

# Network Information
output "vpc_name" {
  description = "VPC network name"
  value       = google_compute_network.vpc.name
}

output "vpc_id" {
  description = "VPC network ID"
  value       = google_compute_network.vpc.id
}

output "subnet_name" {
  description = "Subnet name"
  value       = google_compute_subnetwork.subnet.name
}

output "subnet_id" {
  description = "Subnet ID"
  value       = google_compute_subnetwork.subnet.id
}

output "subnet_cidr" {
  description = "Subnet CIDR range"
  value       = google_compute_subnetwork.subnet.ip_cidr_range
}

output "pod_cidr" {
  description = "Pod CIDR range"
  value       = google_compute_subnetwork.subnet.secondary_ip_range[0].ip_cidr_range
}

output "service_cidr" {
  description = "Service CIDR range"
  value       = google_compute_subnetwork.subnet.secondary_ip_range[1].ip_cidr_range
}

# Node Pool Information
output "node_pools" {
  description = "Node pool information"
  value = {
    system = {
      name         = google_container_node_pool.system.name
      machine_type = google_container_node_pool.system.node_config[0].machine_type
      min_nodes    = google_container_node_pool.system.autoscaling[0].min_node_count
      max_nodes    = google_container_node_pool.system.autoscaling[0].max_node_count
    }
    application = {
      name         = google_container_node_pool.application.name
      machine_type = google_container_node_pool.application.node_config[0].machine_type
      min_nodes    = google_container_node_pool.application.autoscaling[0].min_node_count
      max_nodes    = google_container_node_pool.application.autoscaling[0].max_node_count
    }
    monitoring = {
      name         = google_container_node_pool.monitoring.name
      machine_type = google_container_node_pool.monitoring.node_config[0].machine_type
      min_nodes    = google_container_node_pool.monitoring.autoscaling[0].min_node_count
      max_nodes    = google_container_node_pool.monitoring.autoscaling[0].max_node_count
    }
  }
}

# Service Account Information
output "node_service_account_email" {
  description = "Node service account email"
  value       = google_service_account.gke_nodes.email
}

output "node_service_account_name" {
  description = "Node service account name"
  value       = google_service_account.gke_nodes.name
}

# Database Information
output "database_instance_name" {
  description = "Cloud SQL instance name"
  value       = google_sql_database_instance.main.name
}

output "database_connection_name" {
  description = "Cloud SQL connection name"
  value       = google_sql_database_instance.main.connection_name
}

output "database_private_ip_address" {
  description = "Database private IP address"
  value       = google_sql_database_instance.main.private_ip_address
}

output "database_name" {
  description = "Database name"
  value       = google_sql_database.novacron.name
}

output "database_user" {
  description = "Database user"
  value       = google_sql_user.novacron.name
}

# Redis Information
output "redis_instance_name" {
  description = "Redis instance name"
  value       = google_redis_instance.cache.name
}

output "redis_host" {
  description = "Redis host"
  value       = google_redis_instance.cache.host
}

output "redis_port" {
  description = "Redis port"
  value       = google_redis_instance.cache.port
}

output "redis_memory_size" {
  description = "Redis memory size in GB"
  value       = google_redis_instance.cache.memory_size_gb
}

# Storage Information
output "backup_bucket_name" {
  description = "Backup storage bucket name"
  value       = google_storage_bucket.backup.name
}

output "backup_bucket_url" {
  description = "Backup storage bucket URL"
  value       = google_storage_bucket.backup.url
}

# Network Resources
output "ingress_ip_address" {
  description = "Global IP address for ingress"
  value       = google_compute_global_address.ingress_ip.address
}

output "ingress_ip_name" {
  description = "Global IP address name for ingress"
  value       = google_compute_global_address.ingress_ip.name
}

output "nat_ip_addresses" {
  description = "NAT IP addresses"
  value       = google_compute_router_nat.nat.nat_ips
}

# DNS Information (if managed)
output "dns_zone_name" {
  description = "DNS zone name"
  value       = var.manage_dns ? google_dns_managed_zone.main[0].name : null
}

output "dns_zone_name_servers" {
  description = "DNS zone name servers"
  value       = var.manage_dns ? google_dns_managed_zone.main[0].name_servers : null
}

output "dns_zone_dns_name" {
  description = "DNS zone DNS name"
  value       = var.manage_dns ? google_dns_managed_zone.main[0].dns_name : null
}

# Kubectl Connection Command
output "kubectl_connection_command" {
  description = "Command to connect kubectl to the cluster"
  value       = "gcloud container clusters get-credentials ${google_container_cluster.primary.name} --region ${google_container_cluster.primary.location} --project ${var.project_id}"
}

# Environment Information
output "environment" {
  description = "Environment name"
  value       = var.environment
}

output "project_id" {
  description = "Google Cloud Project ID"
  value       = var.project_id
}

output "region" {
  description = "Google Cloud region"
  value       = var.region
}

# Resource Labels
output "common_labels" {
  description = "Common labels applied to resources"
  value = {
    project     = var.project_name
    environment = var.environment
    managed_by  = "terraform"
  }
}

# Kubernetes Configuration
output "kubernetes_config" {
  description = "Kubernetes configuration for applications"
  value = {
    cluster_name               = google_container_cluster.primary.name
    cluster_endpoint          = google_container_cluster.primary.endpoint
    cluster_ca_certificate    = google_container_cluster.primary.master_auth.0.cluster_ca_certificate
    namespace                 = "novacron-${var.environment}"
    service_account_email     = google_service_account.gke_nodes.email
  }
  sensitive = true
}

# Database Configuration
output "database_config" {
  description = "Database configuration for applications"
  value = {
    host             = google_sql_database_instance.main.private_ip_address
    port             = 5432
    database         = google_sql_database.novacron.name
    user             = google_sql_user.novacron.name
    connection_name  = google_sql_database_instance.main.connection_name
    ssl_required     = true
  }
  sensitive = true
}

# Redis Configuration
output "redis_config" {
  description = "Redis configuration for applications"
  value = {
    host = google_redis_instance.cache.host
    port = google_redis_instance.cache.port
    auth_enabled = google_redis_instance.cache.auth_enabled
  }
}

# Monitoring URLs
output "monitoring_urls" {
  description = "Monitoring service URLs"
  value = {
    gke_dashboard     = "https://console.cloud.google.com/kubernetes/clusters/details/${var.region}/${google_container_cluster.primary.name}/details?project=${var.project_id}"
    cloud_sql         = "https://console.cloud.google.com/sql/instances/${google_sql_database_instance.main.name}/overview?project=${var.project_id}"
    redis_instance    = "https://console.cloud.google.com/memorystore/redis/locations/${var.region}/instances/${google_redis_instance.cache.name}/details/overview?project=${var.project_id}"
    vpc_network       = "https://console.cloud.google.com/networking/networks/details/${google_compute_network.vpc.name}?project=${var.project_id}"
    load_balancer     = "https://console.cloud.google.com/net-services/loadbalancing/list/loadBalancers?project=${var.project_id}"
  }
}

# Cost Information
output "estimated_monthly_cost" {
  description = "Estimated monthly cost breakdown"
  value = {
    gke_cluster = {
      management_fee = "$73.00" # GKE management fee
      nodes_estimated = "Varies by node type and count"
    }
    cloud_sql = "Varies by tier and usage"
    redis = "Varies by memory size and tier"
    networking = "Varies by traffic"
    storage = "Varies by usage"
    note = "Actual costs depend on usage patterns and may vary significantly"
  }
}

# Security Information
output "security_config" {
  description = "Security configuration summary"
  value = {
    private_cluster        = google_container_cluster.primary.private_cluster_config[0].enable_private_nodes
    network_policy_enabled = google_container_cluster.primary.network_policy[0].enabled
    workload_identity     = google_container_cluster.primary.workload_identity_config[0].workload_pool != null
    shielded_nodes        = google_container_node_pool.system.node_config[0].shielded_instance_config[0].enable_secure_boot
    master_authorized_networks = length(google_container_cluster.primary.master_authorized_networks_config) > 0
  }
}

# Backup Information
output "backup_config" {
  description = "Backup configuration summary"
  value = {
    database_backup_enabled = google_sql_database_instance.main.settings[0].backup_configuration[0].enabled
    backup_retention_days  = google_sql_database_instance.main.settings[0].backup_configuration[0].backup_retention_settings[0].retained_backups
    point_in_time_recovery = google_sql_database_instance.main.settings[0].backup_configuration[0].point_in_time_recovery_enabled
    backup_bucket         = google_storage_bucket.backup.name
  }
}