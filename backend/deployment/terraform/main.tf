# NovaCron Infrastructure as Code - Main Configuration
# Multi-cloud Terraform deployment for enterprise-grade infrastructure

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }

  # Remote state configuration
  backend "s3" {
    bucket         = "novacron-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "novacron-terraform-locks"
    
    # Enable state versioning
    versioning = true
  }
}

# Local variables for common configurations
locals {
  project_name = "novacron"
  environment  = var.environment
  region       = var.region
  
  # Common tags applied to all resources
  common_tags = {
    Project     = local.project_name
    Environment = local.environment
    ManagedBy   = "terraform"
    Owner       = "novacron-team"
    CostCenter  = "engineering"
    Backup      = "required"
  }

  # Network configuration
  vpc_cidr = "10.0.0.0/16"
  availability_zones = data.aws_availability_zones.available.names

  # Application configuration
  app_port = 8080
  health_check_path = "/health"
  
  # Monitoring configuration
  monitoring_namespace = "monitoring"
  logging_namespace    = "logging"
  
  # Security configuration
  allowed_cidr_blocks = var.allowed_cidr_blocks
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# Random password for database
resource "random_password" "db_password" {
  length  = 32
  special = true
}

# Random string for unique resource naming
resource "random_string" "suffix" {
  length  = 8
  special = false
  upper   = false
}

# KMS Key for encryption
resource "aws_kms_key" "novacron" {
  description = "NovaCron encryption key"
  
  key_usage = "ENCRYPT_DECRYPT"
  customer_master_key_spec = "SYMMETRIC_DEFAULT"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow CloudWatch Logs"
        Effect = "Allow"
        Principal = {
          Service = "logs.${data.aws_region.current.name}.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${local.project_name}-kms-key"
  })
}

resource "aws_kms_alias" "novacron" {
  name          = "alias/${local.project_name}-${local.environment}"
  target_key_id = aws_kms_key.novacron.key_id
}

# Import networking module
module "networking" {
  source = "./modules/networking"
  
  project_name       = local.project_name
  environment        = local.environment
  vpc_cidr          = local.vpc_cidr
  availability_zones = local.availability_zones
  common_tags       = local.common_tags
}

# Import security module
module "security" {
  source = "./modules/security"
  
  project_name        = local.project_name
  environment         = local.environment
  vpc_id             = module.networking.vpc_id
  private_subnet_ids = module.networking.private_subnet_ids
  public_subnet_ids  = module.networking.public_subnet_ids
  allowed_cidr_blocks = local.allowed_cidr_blocks
  kms_key_id         = aws_kms_key.novacron.arn
  common_tags        = local.common_tags
}

# Import database module
module "database" {
  source = "./modules/database"
  
  project_name         = local.project_name
  environment          = local.environment
  vpc_id              = module.networking.vpc_id
  private_subnet_ids  = module.networking.private_subnet_ids
  database_password   = random_password.db_password.result
  kms_key_id         = aws_kms_key.novacron.arn
  security_group_ids = [module.security.database_security_group_id]
  common_tags        = local.common_tags
}

# Import compute module for EKS cluster
module "compute" {
  source = "./modules/compute"
  
  project_name         = local.project_name
  environment          = local.environment
  vpc_id              = module.networking.vpc_id
  private_subnet_ids  = module.networking.private_subnet_ids
  public_subnet_ids   = module.networking.public_subnet_ids
  security_group_ids  = [module.security.application_security_group_id]
  kms_key_id         = aws_kms_key.novacron.arn
  common_tags        = local.common_tags
}

# Import monitoring module
module "monitoring" {
  source = "./modules/monitoring"
  
  project_name           = local.project_name
  environment            = local.environment
  vpc_id                = module.networking.vpc_id
  private_subnet_ids    = module.networking.private_subnet_ids
  kms_key_id           = aws_kms_key.novacron.arn
  eks_cluster_name     = module.compute.eks_cluster_name
  eks_cluster_endpoint = module.compute.eks_cluster_endpoint
  common_tags          = local.common_tags
}

# Import application deployment module
module "application" {
  source = "./modules/application"
  
  project_name           = local.project_name
  environment            = local.environment
  vpc_id                = module.networking.vpc_id
  database_endpoint     = module.database.database_endpoint
  database_username     = module.database.database_username
  database_password     = random_password.db_password.result
  eks_cluster_name      = module.compute.eks_cluster_name
  eks_cluster_endpoint  = module.compute.eks_cluster_endpoint
  application_port      = local.app_port
  health_check_path     = local.health_check_path
  common_tags           = local.common_tags

  depends_on = [
    module.compute,
    module.database,
    module.monitoring
  ]
}

# Import storage module
module "storage" {
  source = "./modules/storage"
  
  project_name    = local.project_name
  environment     = local.environment
  kms_key_id     = aws_kms_key.novacron.arn
  common_tags    = local.common_tags
}

# Import observability module
module "observability" {
  source = "./modules/observability"
  
  project_name           = local.project_name
  environment            = local.environment
  vpc_id                = module.networking.vpc_id
  private_subnet_ids    = module.networking.private_subnet_ids
  kms_key_id           = aws_kms_key.novacron.arn
  eks_cluster_name     = module.compute.eks_cluster_name
  log_bucket_name      = module.storage.log_bucket_name
  common_tags          = local.common_tags
}