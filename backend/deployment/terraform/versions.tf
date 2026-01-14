# NovaCron Infrastructure as Code - Provider Versions
# Provider configurations and version constraints

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.31"
    }
    
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.85"
      configuration_aliases = [azurerm.secondary]
    }
    
    google = {
      source  = "hashicorp/google"
      version = "~> 5.10"
      configuration_aliases = [google.secondary]
    }
    
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24"
    }
    
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    
    kubectl = {
      source  = "gavinbunney/kubectl"
      version = "~> 1.14"
    }
    
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
    
    time = {
      source  = "hashicorp/time"
      version = "~> 0.10"
    }
    
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
    
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
    
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
}

# AWS Provider Configuration
provider "aws" {
  region = var.region
  
  # Default tags applied to all resources
  default_tags {
    tags = {
      Project     = "novacron"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "novacron-team"
    }
  }
}

# Secondary AWS region provider for disaster recovery
provider "aws" {
  alias  = "secondary"
  region = var.secondary_region
  
  default_tags {
    tags = {
      Project     = "novacron"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "novacron-team"
      Region      = "secondary"
    }
  }
}

# Azure Provider Configuration (conditional)
provider "azurerm" {
  count = var.enable_azure_integration ? 1 : 0
  
  features {
    resource_group {
      prevent_deletion_if_contains_resources = true
    }
    
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
    
    virtual_machine {
      delete_os_disk_on_deletion     = true
      graceful_shutdown              = false
      skip_shutdown_and_force_delete = false
    }
  }
}

# Google Cloud Provider Configuration (conditional)
provider "google" {
  count   = var.enable_gcp_integration ? 1 : 0
  project = var.gcp_project_id
  region  = var.gcp_region
  
  user_project_override = true
}

# Kubernetes Provider Configuration
# Configured after EKS cluster is created
provider "kubernetes" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
}

# Helm Provider Configuration
provider "helm" {
  kubernetes {
    host                   = data.aws_eks_cluster.cluster.endpoint
    cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
    token                  = data.aws_eks_cluster_auth.cluster.token
  }
}

# Kubectl Provider Configuration
provider "kubectl" {
  host                   = data.aws_eks_cluster.cluster.endpoint
  cluster_ca_certificate = base64decode(data.aws_eks_cluster.cluster.certificate_authority[0].data)
  token                  = data.aws_eks_cluster_auth.cluster.token
  load_config_file       = false
}

# Data sources for EKS cluster authentication
data "aws_eks_cluster" "cluster" {
  name = module.compute.eks_cluster_name
}

data "aws_eks_cluster_auth" "cluster" {
  name = module.compute.eks_cluster_name
}