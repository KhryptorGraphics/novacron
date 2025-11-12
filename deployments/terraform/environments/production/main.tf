# DWCP v3 Production Environment
# Multi-region deployment with disaster recovery

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket         = "novacron-terraform-state-production"
    key            = "dwcp-v3/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-state-lock"
  }
}

# Provider configurations
provider "aws" {
  region = var.primary_region
  alias  = "primary"

  default_tags {
    tags = {
      Project     = "NovaCron"
      Component   = "DWCP-v3"
      Environment = "production"
      ManagedBy   = "Terraform"
    }
  }
}

provider "aws" {
  region = var.secondary_region
  alias  = "secondary"

  default_tags {
    tags = {
      Project     = "NovaCron"
      Component   = "DWCP-v3"
      Environment = "production"
      ManagedBy   = "Terraform"
    }
  }
}

provider "aws" {
  region = var.tertiary_region
  alias  = "tertiary"

  default_tags {
    tags = {
      Project     = "NovaCron"
      Component   = "DWCP-v3"
      Environment = "production"
      ManagedBy   = "Terraform"
    }
  }
}

# Variables
variable "primary_region" {
  description = "Primary AWS region"
  type        = string
  default     = "us-east-1"
}

variable "secondary_region" {
  description = "Secondary AWS region (disaster recovery)"
  type        = string
  default     = "us-west-2"
}

variable "tertiary_region" {
  description = "Tertiary AWS region (global expansion)"
  type        = string
  default     = "eu-west-1"
}

variable "ssh_key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "grafana_admin_password" {
  description = "Grafana admin password"
  type        = string
  sensitive   = true
}

variable "alert_email" {
  description = "Email for alerts"
  type        = string
}

# Primary Region Deployment
module "network_primary" {
  source = "../../modules/dwcp-v3-network"

  providers = {
    aws = aws.primary
  }

  environment        = "production"
  vpc_cidr           = "10.0.0.0/16"
  availability_zones = ["us-east-1a", "us-east-1b", "us-east-1c"]
  enable_rdma        = true
  enable_ipv6        = true
  enable_flow_logs   = true

  tags = {
    Region = var.primary_region
    Role   = "primary"
  }
}

module "compute_primary" {
  source = "../../modules/dwcp-v3-compute"

  providers = {
    aws = aws.primary
  }

  environment                   = "production"
  vpc_id                        = module.network_primary.vpc_id
  datacenter_subnet_ids         = module.network_primary.datacenter_subnet_ids
  internet_subnet_ids           = module.network_primary.internet_subnet_ids
  datacenter_security_group_id  = module.network_primary.datacenter_security_group_id
  internet_security_group_id    = module.network_primary.internet_security_group_id
  datacenter_instance_type      = "p4d.24xlarge"
  internet_instance_type        = "c6in.32xlarge"
  datacenter_min_size           = 5
  datacenter_max_size           = 20
  internet_min_size             = 10
  internet_max_size             = 50
  ssh_key_name                  = var.ssh_key_name

  tags = {
    Region = var.primary_region
    Role   = "primary"
  }
}

module "monitoring_primary" {
  source = "../../modules/dwcp-v3-monitoring"

  providers = {
    aws = aws.primary
  }

  environment              = "production"
  vpc_id                   = module.network_primary.vpc_id
  subnet_ids               = module.network_primary.datacenter_subnet_ids
  monitoring_instance_type = "m6i.4xlarge"
  prometheus_retention_days = 90
  grafana_admin_password   = var.grafana_admin_password
  alert_email              = var.alert_email

  tags = {
    Region = var.primary_region
    Role   = "primary"
  }
}

# Secondary Region Deployment (Disaster Recovery)
module "network_secondary" {
  source = "../../modules/dwcp-v3-network"

  providers = {
    aws = aws.secondary
  }

  environment        = "production"
  vpc_cidr           = "10.1.0.0/16"
  availability_zones = ["us-west-2a", "us-west-2b", "us-west-2c"]
  enable_rdma        = true
  enable_ipv6        = true
  enable_flow_logs   = true

  tags = {
    Region = var.secondary_region
    Role   = "secondary"
  }
}

module "compute_secondary" {
  source = "../../modules/dwcp-v3-compute"

  providers = {
    aws = aws.secondary
  }

  environment                   = "production"
  vpc_id                        = module.network_secondary.vpc_id
  datacenter_subnet_ids         = module.network_secondary.datacenter_subnet_ids
  internet_subnet_ids           = module.network_secondary.internet_subnet_ids
  datacenter_security_group_id  = module.network_secondary.datacenter_security_group_id
  internet_security_group_id    = module.network_secondary.internet_security_group_id
  datacenter_instance_type      = "p4d.24xlarge"
  internet_instance_type        = "c6in.32xlarge"
  datacenter_min_size           = 3
  datacenter_max_size           = 15
  internet_min_size             = 5
  internet_max_size             = 30
  ssh_key_name                  = var.ssh_key_name

  tags = {
    Region = var.secondary_region
    Role   = "secondary"
  }
}

module "monitoring_secondary" {
  source = "../../modules/dwcp-v3-monitoring"

  providers = {
    aws = aws.secondary
  }

  environment              = "production"
  vpc_id                   = module.network_secondary.vpc_id
  subnet_ids               = module.network_secondary.datacenter_subnet_ids
  monitoring_instance_type = "m6i.2xlarge"
  prometheus_retention_days = 30
  grafana_admin_password   = var.grafana_admin_password
  alert_email              = var.alert_email

  tags = {
    Region = var.secondary_region
    Role   = "secondary"
  }
}

# Tertiary Region Deployment (Global Expansion)
module "network_tertiary" {
  source = "../../modules/dwcp-v3-network"

  providers = {
    aws = aws.tertiary
  }

  environment        = "production"
  vpc_cidr           = "10.2.0.0/16"
  availability_zones = ["eu-west-1a", "eu-west-1b", "eu-west-1c"]
  enable_rdma        = false
  enable_ipv6        = true
  enable_flow_logs   = true

  tags = {
    Region = var.tertiary_region
    Role   = "tertiary"
  }
}

module "compute_tertiary" {
  source = "../../modules/dwcp-v3-compute"

  providers = {
    aws = aws.tertiary
  }

  environment                   = "production"
  vpc_id                        = module.network_tertiary.vpc_id
  datacenter_subnet_ids         = module.network_tertiary.datacenter_subnet_ids
  internet_subnet_ids           = module.network_tertiary.internet_subnet_ids
  datacenter_security_group_id  = module.network_tertiary.datacenter_security_group_id
  internet_security_group_id    = module.network_tertiary.internet_security_group_id
  datacenter_instance_type      = "c6in.16xlarge"
  internet_instance_type        = "c6in.16xlarge"
  datacenter_min_size           = 2
  datacenter_max_size           = 10
  internet_min_size             = 5
  internet_max_size             = 20
  ssh_key_name                  = var.ssh_key_name

  tags = {
    Region = var.tertiary_region
    Role   = "tertiary"
  }
}

# VPC Peering between regions
resource "aws_vpc_peering_connection" "primary_secondary" {
  provider = aws.primary

  vpc_id      = module.network_primary.vpc_id
  peer_vpc_id = module.network_secondary.vpc_id
  peer_region = var.secondary_region
  auto_accept = false

  tags = {
    Name        = "dwcp-v3-peering-primary-secondary"
    Environment = "production"
  }
}

resource "aws_vpc_peering_connection_accepter" "primary_secondary" {
  provider = aws.secondary

  vpc_peering_connection_id = aws_vpc_peering_connection.primary_secondary.id
  auto_accept               = true

  tags = {
    Name        = "dwcp-v3-peering-primary-secondary"
    Environment = "production"
  }
}

resource "aws_vpc_peering_connection" "primary_tertiary" {
  provider = aws.primary

  vpc_id      = module.network_primary.vpc_id
  peer_vpc_id = module.network_tertiary.vpc_id
  peer_region = var.tertiary_region
  auto_accept = false

  tags = {
    Name        = "dwcp-v3-peering-primary-tertiary"
    Environment = "production"
  }
}

resource "aws_vpc_peering_connection_accepter" "primary_tertiary" {
  provider = aws.tertiary

  vpc_peering_connection_id = aws_vpc_peering_connection.primary_tertiary.id
  auto_accept               = true

  tags = {
    Name        = "dwcp-v3-peering-primary-tertiary"
    Environment = "production"
  }
}

# S3 bucket for configuration and backups
resource "aws_s3_bucket" "dwcp_config" {
  provider = aws.primary
  bucket   = "novacron-dwcp-v3-config-production"

  tags = {
    Name        = "dwcp-v3-config-production"
    Environment = "production"
  }
}

resource "aws_s3_bucket_versioning" "config_versioning" {
  provider = aws.primary
  bucket   = aws_s3_bucket.dwcp_config.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "config_encryption" {
  provider = aws.primary
  bucket   = aws_s3_bucket.dwcp_config.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Backup replication to secondary region
resource "aws_s3_bucket_replication_configuration" "config_replication" {
  provider = aws.primary
  bucket   = aws_s3_bucket.dwcp_config.id
  role     = aws_iam_role.replication.arn

  rule {
    id     = "replicate-to-secondary"
    status = "Enabled"

    destination {
      bucket        = aws_s3_bucket.dwcp_config_replica.arn
      storage_class = "STANDARD_IA"
    }
  }

  depends_on = [aws_s3_bucket_versioning.config_versioning]
}

resource "aws_s3_bucket" "dwcp_config_replica" {
  provider = aws.secondary
  bucket   = "novacron-dwcp-v3-config-production-replica"

  tags = {
    Name        = "dwcp-v3-config-production-replica"
    Environment = "production"
  }
}

resource "aws_iam_role" "replication" {
  provider = aws.primary
  name     = "dwcp-v3-s3-replication-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "s3.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "replication_policy" {
  provider = aws.primary
  name     = "dwcp-v3-s3-replication-policy"
  role     = aws_iam_role.replication.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetReplicationConfiguration",
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.dwcp_config.arn
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObjectVersionForReplication",
          "s3:GetObjectVersionAcl"
        ]
        Resource = "${aws_s3_bucket.dwcp_config.arn}/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ReplicateObject",
          "s3:ReplicateDelete"
        ]
        Resource = "${aws_s3_bucket.dwcp_config_replica.arn}/*"
      }
    ]
  })
}

# Route53 for DNS (global routing)
resource "aws_route53_zone" "dwcp" {
  provider = aws.primary
  name     = "dwcp.novacron.io"

  tags = {
    Name        = "dwcp-v3-zone"
    Environment = "production"
  }
}

resource "aws_route53_record" "dwcp_primary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.dwcp.zone_id
  name     = "api.dwcp.novacron.io"
  type     = "A"

  weighted_routing_policy {
    weight = 70
  }

  set_identifier = "primary"
  alias {
    name                   = module.compute_primary.internet_nlb_dns
    zone_id                = module.compute_primary.internet_nlb_dns
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "dwcp_secondary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.dwcp.zone_id
  name     = "api.dwcp.novacron.io"
  type     = "A"

  weighted_routing_policy {
    weight = 20
  }

  set_identifier = "secondary"
  alias {
    name                   = module.compute_secondary.internet_nlb_dns
    zone_id                = module.compute_secondary.internet_nlb_dns
    evaluate_target_health = true
  }
}

resource "aws_route53_record" "dwcp_tertiary" {
  provider = aws.primary
  zone_id  = aws_route53_zone.dwcp.zone_id
  name     = "api.dwcp.novacron.io"
  type     = "A"

  weighted_routing_policy {
    weight = 10
  }

  set_identifier = "tertiary"
  alias {
    name                   = module.compute_tertiary.internet_nlb_dns
    zone_id                = module.compute_tertiary.internet_nlb_dns
    evaluate_target_health = true
  }
}

# Outputs
output "primary_region_endpoints" {
  description = "Primary region endpoints"
  value = {
    prometheus      = module.monitoring_primary.prometheus_url
    grafana         = module.monitoring_primary.grafana_url
    datacenter_nlb  = module.compute_primary.datacenter_nlb_dns
    internet_nlb    = module.compute_primary.internet_nlb_dns
  }
}

output "secondary_region_endpoints" {
  description = "Secondary region endpoints"
  value = {
    prometheus      = module.monitoring_secondary.prometheus_url
    grafana         = module.monitoring_secondary.grafana_url
    datacenter_nlb  = module.compute_secondary.datacenter_nlb_dns
    internet_nlb    = module.compute_secondary.internet_nlb_dns
  }
}

output "global_dns" {
  description = "Global DNS endpoint"
  value       = "api.dwcp.novacron.io"
}

output "config_bucket" {
  description = "S3 configuration bucket"
  value       = aws_s3_bucket.dwcp_config.id
}
