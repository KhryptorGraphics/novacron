# NovaCron Infrastructure as Code - Main Terraform Configuration
# Comprehensive infrastructure setup for multi-environment Kubernetes clusters

terraform {
  required_version = ">= 1.6.0"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
  
  backend "gcs" {
    bucket = "novacron-terraform-state"
    prefix = "infrastructure/state"
  }
}

# Provider configurations
provider "google" {
  project = var.project_id
  region  = var.region
}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.primary.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
  }
}

# Data sources
data "google_client_config" "default" {}

data "google_project" "project" {
  project_id = var.project_id
}

# Local values
locals {
  common_labels = {
    project     = "novacron"
    managed_by  = "terraform"
    environment = var.environment
  }
  
  cluster_name = "${var.project_name}-${var.environment}-cluster"
  
  # Network configuration
  network_name    = "${var.project_name}-${var.environment}-vpc"
  subnet_name     = "${var.project_name}-${var.environment}-subnet"
  pod_range_name  = "${var.project_name}-${var.environment}-pods"
  svc_range_name  = "${var.project_name}-${var.environment}-services"
}

# Random suffix for globally unique resources
resource "random_id" "suffix" {
  byte_length = 4
}

# VPC Network
resource "google_compute_network" "vpc" {
  name                    = local.network_name
  auto_create_subnetworks = false
  description             = "VPC for NovaCron ${var.environment} environment"
  
  depends_on = [
    google_project_service.compute,
    google_project_service.container
  ]
}

# Subnet
resource "google_compute_subnetwork" "subnet" {
  name          = local.subnet_name
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc.name
  description   = "Subnet for NovaCron ${var.environment} environment"
  
  secondary_ip_range {
    range_name    = local.pod_range_name
    ip_cidr_range = var.pod_cidr
  }
  
  secondary_ip_range {
    range_name    = local.svc_range_name
    ip_cidr_range = var.service_cidr
  }
  
  private_ip_google_access = true
}

# Cloud Router for NAT
resource "google_compute_router" "router" {
  name    = "${local.network_name}-router"
  region  = var.region
  network = google_compute_network.vpc.id
  
  bgp {
    asn = 64514
  }
}

# Cloud NAT for outbound traffic
resource "google_compute_router_nat" "nat" {
  name                               = "${local.network_name}-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
  
  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

# Firewall Rules
resource "google_compute_firewall" "allow_internal" {
  name    = "${local.network_name}-allow-internal"
  network = google_compute_network.vpc.name
  
  allow {
    protocol = "tcp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "udp"
    ports    = ["0-65535"]
  }
  
  allow {
    protocol = "icmp"
  }
  
  source_ranges = [var.subnet_cidr, var.pod_cidr, var.service_cidr]
}

resource "google_compute_firewall" "allow_health_check" {
  name    = "${local.network_name}-allow-health-check"
  network = google_compute_network.vpc.name
  
  allow {
    protocol = "tcp"
  }
  
  source_ranges = ["130.211.0.0/22", "35.191.0.0/16"]
  target_tags   = ["gke-node"]
}

# Enable required APIs
resource "google_project_service" "compute" {
  service = "compute.googleapis.com"
  
  disable_dependent_services = true
}

resource "google_project_service" "container" {
  service = "container.googleapis.com"
  
  disable_dependent_services = true
  depends_on = [google_project_service.compute]
}

resource "google_project_service" "monitoring" {
  service = "monitoring.googleapis.com"
}

resource "google_project_service" "logging" {
  service = "logging.googleapis.com"
}

resource "google_project_service" "dns" {
  service = "dns.googleapis.com"
}

# GKE Cluster
resource "google_container_cluster" "primary" {
  name     = local.cluster_name
  location = var.region
  
  # Network configuration
  network    = google_compute_network.vpc.name
  subnetwork = google_compute_subnetwork.subnet.name
  
  # Cluster configuration
  min_master_version = var.kubernetes_version
  
  # Remove default node pool
  remove_default_node_pool = true
  initial_node_count       = 1
  
  # Enable network policy
  network_policy {
    enabled = true
  }
  
  # IP allocation policy
  ip_allocation_policy {
    cluster_secondary_range_name  = local.pod_range_name
    services_secondary_range_name = local.svc_range_name
  }
  
  # Private cluster configuration
  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_cidr
    
    master_global_access_config {
      enabled = true
    }
  }
  
  # Master auth configuration
  master_auth {
    client_certificate_config {
      issue_client_certificate = false
    }
  }
  
  # Workload identity
  workload_identity_config {
    workload_pool = "${data.google_project.project.project_id}.svc.id.goog"
  }
  
  # Addons configuration
  addons_config {
    http_load_balancing {
      disabled = false
    }
    
    horizontal_pod_autoscaling {
      disabled = false
    }
    
    network_policy_config {
      disabled = false
    }
    
    istio_config {
      disabled = false
      auth     = "AUTH_MUTUAL_TLS"
    }
    
    cloudrun_config {
      disabled = false
    }
  }
  
  # Monitoring and logging
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }
  
  # Maintenance policy
  maintenance_policy {
    daily_maintenance_window {
      start_time = "03:00"
    }
  }
  
  # Resource labels
  resource_labels = local.common_labels
  
  depends_on = [
    google_project_service.compute,
    google_project_service.container,
    google_compute_subnetwork.subnet,
  ]
}

# Node Pool - System
resource "google_container_node_pool" "system" {
  name       = "system-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  
  # Autoscaling
  autoscaling {
    min_node_count = var.system_pool_min_nodes
    max_node_count = var.system_pool_max_nodes
  }
  
  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  # Node configuration
  node_config {
    preemptible  = false
    machine_type = var.system_pool_machine_type
    disk_size_gb = 50
    disk_type    = "pd-ssd"
    
    # Service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
    
    # Labels and taints
    labels = merge(local.common_labels, {
      node_pool = "system"
    })
    
    taint {
      key    = "node-pool"
      value  = "system"
      effect = "NO_SCHEDULE"
    }
    
    tags = ["gke-node", "${local.cluster_name}-node", "system"]
    
    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded instance
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

# Node Pool - Application
resource "google_container_node_pool" "application" {
  name       = "application-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  
  # Autoscaling
  autoscaling {
    min_node_count = var.app_pool_min_nodes
    max_node_count = var.app_pool_max_nodes
  }
  
  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  # Node configuration
  node_config {
    preemptible  = var.environment != "production"
    machine_type = var.app_pool_machine_type
    disk_size_gb = 100
    disk_type    = "pd-ssd"
    
    # Service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
    
    # Labels
    labels = merge(local.common_labels, {
      node_pool = "application"
    })
    
    tags = ["gke-node", "${local.cluster_name}-node", "application"]
    
    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded instance
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

# Node Pool - Monitoring
resource "google_container_node_pool" "monitoring" {
  name       = "monitoring-pool"
  location   = var.region
  cluster    = google_container_cluster.primary.name
  
  # Autoscaling
  autoscaling {
    min_node_count = var.monitoring_pool_min_nodes
    max_node_count = var.monitoring_pool_max_nodes
  }
  
  # Management
  management {
    auto_repair  = true
    auto_upgrade = true
  }
  
  # Node configuration
  node_config {
    preemptible  = false
    machine_type = var.monitoring_pool_machine_type
    disk_size_gb = 200
    disk_type    = "pd-ssd"
    
    # Service account
    service_account = google_service_account.gke_nodes.email
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform",
    ]
    
    # Labels and taints
    labels = merge(local.common_labels, {
      node_pool = "monitoring"
    })
    
    taint {
      key    = "node-pool"
      value  = "monitoring"
      effect = "NO_SCHEDULE"
    }
    
    tags = ["gke-node", "${local.cluster_name}-node", "monitoring"]
    
    # Workload identity
    workload_metadata_config {
      mode = "GKE_METADATA"
    }
    
    # Shielded instance
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

# Service Account for GKE nodes
resource "google_service_account" "gke_nodes" {
  account_id   = "${var.project_name}-${var.environment}-nodes"
  display_name = "GKE Nodes Service Account - ${var.environment}"
  description  = "Service account for GKE nodes in ${var.environment} environment"
}

# IAM bindings for node service account
resource "google_project_iam_member" "gke_nodes" {
  for_each = toset([
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/monitoring.viewer",
    "roles/stackdriver.resourceMetadata.writer",
    "roles/container.nodeServiceAgent",
  ])
  
  role   = each.value
  member = "serviceAccount:${google_service_account.gke_nodes.email}"
}

# Cloud Storage bucket for backup
resource "google_storage_bucket" "backup" {
  name          = "${var.project_name}-${var.environment}-backup-${random_id.suffix.hex}"
  location      = var.region
  force_destroy = var.environment != "production"
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
  
  labels = local.common_labels
}

# Cloud SQL instance for database
resource "google_sql_database_instance" "main" {
  name             = "${var.project_name}-${var.environment}-db-${random_id.suffix.hex}"
  database_version = "POSTGRES_15"
  region           = var.region
  
  settings {
    tier                        = var.db_tier
    availability_type           = var.environment == "production" ? "REGIONAL" : "ZONAL"
    disk_size                   = var.db_disk_size
    disk_type                   = "PD_SSD"
    disk_autoresize             = true
    disk_autoresize_limit       = var.db_max_disk_size
    deletion_protection_enabled = var.environment == "production"
    
    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      location                       = var.region
      point_in_time_recovery_enabled = true
      backup_retention_settings {
        retained_backups = 30
      }
    }
    
    ip_configuration {
      ipv4_enabled                                  = false
      private_network                               = google_compute_network.vpc.id
      enable_private_path_for_google_cloud_services = true
    }
    
    database_flags {
      name  = "log_checkpoints"
      value = "on"
    }
    
    database_flags {
      name  = "log_connections"
      value = "on"
    }
    
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
    
    database_flags {
      name  = "log_lock_waits"
      value = "on"
    }
    
    database_flags {
      name  = "log_min_duration_statement"
      value = "1000" # Log queries taking longer than 1 second
    }
    
    user_labels = local.common_labels
  }
  
  depends_on = [google_service_networking_connection.private_vpc_connection]
}

# Private service connection for Cloud SQL
resource "google_compute_global_address" "private_ip_address" {
  name          = "${var.project_name}-${var.environment}-private-ip"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
}

# Database
resource "google_sql_database" "novacron" {
  name     = "novacron"
  instance = google_sql_database_instance.main.name
}

# Database user
resource "google_sql_user" "novacron" {
  name     = "novacron"
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

# Redis instance
resource "google_redis_instance" "cache" {
  name           = "${var.project_name}-${var.environment}-cache"
  tier           = var.environment == "production" ? "STANDARD_HA" : "BASIC"
  memory_size_gb = var.redis_memory_size
  region         = var.region
  
  location_id             = "${var.region}-a"
  alternative_location_id = var.environment == "production" ? "${var.region}-b" : null
  
  authorized_network = google_compute_network.vpc.id
  
  redis_version     = "REDIS_7_0"
  display_name      = "NovaCron Cache - ${var.environment}"
  
  labels = local.common_labels
}

# Global IP address for ingress
resource "google_compute_global_address" "ingress_ip" {
  name         = "${var.project_name}-${var.environment}-ingress-ip"
  description  = "Global IP address for NovaCron ${var.environment} ingress"
  address_type = "EXTERNAL"
}

# DNS zone (if managing DNS)
resource "google_dns_managed_zone" "main" {
  count       = var.manage_dns ? 1 : 0
  name        = "${var.project_name}-${var.environment}-zone"
  dns_name    = "${var.domain_name}."
  description = "DNS zone for NovaCron ${var.environment}"
  
  dnssec_config {
    state = "on"
  }
  
  labels = local.common_labels
}

# A record for main domain
resource "google_dns_record_set" "main" {
  count = var.manage_dns ? 1 : 0
  name  = google_dns_managed_zone.main[0].dns_name
  type  = "A"
  ttl   = 300
  
  managed_zone = google_dns_managed_zone.main[0].name
  
  rrdatas = [google_compute_global_address.ingress_ip.address]
}

# A record for API subdomain
resource "google_dns_record_set" "api" {
  count = var.manage_dns ? 1 : 0
  name  = "api.${google_dns_managed_zone.main[0].dns_name}"
  type  = "A"
  ttl   = 300
  
  managed_zone = google_dns_managed_zone.main[0].name
  
  rrdatas = [google_compute_global_address.ingress_ip.address]
}