# NovaCron Infrastructure as Code - Outputs
# Output values for infrastructure components

# General Information
output "project_name" {
  description = "Project name"
  value       = local.project_name
}

output "environment" {
  description = "Deployment environment"
  value       = local.environment
}

output "region" {
  description = "AWS region"
  value       = local.region
}

output "account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}

# Networking Outputs
output "vpc_id" {
  description = "ID of the VPC"
  value       = module.networking.vpc_id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = module.networking.vpc_cidr_block
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = module.networking.public_subnet_ids
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = module.networking.private_subnet_ids
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = module.networking.database_subnet_ids
}

output "nat_gateway_ips" {
  description = "Elastic IP addresses of the NAT Gateways"
  value       = module.networking.nat_gateway_ips
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = module.networking.internet_gateway_id
}

# Security Outputs
output "application_security_group_id" {
  description = "ID of the application security group"
  value       = module.security.application_security_group_id
}

output "database_security_group_id" {
  description = "ID of the database security group"
  value       = module.security.database_security_group_id
}

output "load_balancer_security_group_id" {
  description = "ID of the load balancer security group"
  value       = module.security.load_balancer_security_group_id
}

output "kms_key_id" {
  description = "ID of the KMS key for encryption"
  value       = aws_kms_key.novacron.key_id
}

output "kms_key_arn" {
  description = "ARN of the KMS key for encryption"
  value       = aws_kms_key.novacron.arn
}

# Database Outputs
output "database_endpoint" {
  description = "RDS instance endpoint"
  value       = module.database.database_endpoint
  sensitive   = true
}

output "database_port" {
  description = "RDS instance port"
  value       = module.database.database_port
}

output "database_name" {
  description = "Database name"
  value       = module.database.database_name
}

output "database_username" {
  description = "Database master username"
  value       = module.database.database_username
  sensitive   = true
}

output "database_backup_window" {
  description = "Database backup window"
  value       = module.database.backup_window
}

output "database_maintenance_window" {
  description = "Database maintenance window"
  value       = module.database.maintenance_window
}

# Compute Outputs
output "eks_cluster_name" {
  description = "Name of the EKS cluster"
  value       = module.compute.eks_cluster_name
}

output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.compute.eks_cluster_endpoint
}

output "eks_cluster_version" {
  description = "Version of the EKS cluster"
  value       = module.compute.eks_cluster_version
}

output "eks_cluster_arn" {
  description = "ARN of the EKS cluster"
  value       = module.compute.eks_cluster_arn
}

output "eks_node_group_arn" {
  description = "ARN of the EKS node group"
  value       = module.compute.eks_node_group_arn
}

output "eks_oidc_issuer_url" {
  description = "OIDC issuer URL for the EKS cluster"
  value       = module.compute.eks_oidc_issuer_url
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.compute.eks_cluster_security_group_id
}

# Application Outputs
output "application_load_balancer_arn" {
  description = "ARN of the application load balancer"
  value       = module.application.load_balancer_arn
}

output "application_load_balancer_dns_name" {
  description = "DNS name of the application load balancer"
  value       = module.application.load_balancer_dns_name
}

output "application_load_balancer_zone_id" {
  description = "Zone ID of the application load balancer"
  value       = module.application.load_balancer_zone_id
}

output "application_target_group_arn" {
  description = "ARN of the application target group"
  value       = module.application.target_group_arn
}

output "application_url" {
  description = "URL to access the application"
  value       = "https://${module.application.load_balancer_dns_name}"
}

# Storage Outputs
output "s3_bucket_name" {
  description = "Name of the S3 bucket for application data"
  value       = module.storage.bucket_name
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = module.storage.bucket_arn
}

output "log_bucket_name" {
  description = "Name of the S3 bucket for logs"
  value       = module.storage.log_bucket_name
}

output "backup_bucket_name" {
  description = "Name of the S3 bucket for backups"
  value       = module.storage.backup_bucket_name
}

# Monitoring Outputs
output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group"
  value       = module.monitoring.log_group_name
}

output "cloudwatch_log_group_arn" {
  description = "ARN of the CloudWatch log group"
  value       = module.monitoring.log_group_arn
}

output "prometheus_endpoint" {
  description = "Prometheus endpoint URL"
  value       = module.monitoring.prometheus_endpoint
  sensitive   = true
}

output "grafana_endpoint" {
  description = "Grafana endpoint URL"
  value       = module.monitoring.grafana_endpoint
  sensitive   = true
}

output "grafana_admin_password" {
  description = "Grafana admin password"
  value       = module.monitoring.grafana_admin_password
  sensitive   = true
}

# Observability Outputs
output "elasticsearch_endpoint" {
  description = "Elasticsearch cluster endpoint"
  value       = module.observability.elasticsearch_endpoint
  sensitive   = true
}

output "kibana_endpoint" {
  description = "Kibana endpoint URL"
  value       = module.observability.kibana_endpoint
  sensitive   = true
}

output "jaeger_endpoint" {
  description = "Jaeger endpoint URL"
  value       = module.observability.jaeger_endpoint
  sensitive   = true
}

# Certificate Outputs
output "certificate_arn" {
  description = "ARN of the SSL certificate"
  value       = module.security.certificate_arn
}

output "certificate_validation_records" {
  description = "Certificate validation DNS records"
  value       = module.security.certificate_validation_records
}

# IAM Outputs
output "eks_cluster_role_arn" {
  description = "ARN of the EKS cluster IAM role"
  value       = module.compute.eks_cluster_role_arn
}

output "eks_node_group_role_arn" {
  description = "ARN of the EKS node group IAM role"
  value       = module.compute.eks_node_group_role_arn
}

output "application_role_arn" {
  description = "ARN of the application IAM role"
  value       = module.application.application_role_arn
}

# Auto Scaling Outputs
output "auto_scaling_group_arn" {
  description = "ARN of the Auto Scaling Group"
  value       = module.compute.auto_scaling_group_arn
}

output "auto_scaling_group_name" {
  description = "Name of the Auto Scaling Group"
  value       = module.compute.auto_scaling_group_name
}

# DNS and Route53 Outputs
output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = module.networking.route53_zone_id
}

output "route53_zone_name" {
  description = "Route53 hosted zone name"
  value       = module.networking.route53_zone_name
}

output "dns_name" {
  description = "Primary DNS name for the application"
  value       = module.networking.dns_name
}

# Backup Outputs
output "backup_vault_name" {
  description = "AWS Backup vault name"
  value       = module.storage.backup_vault_name
}

output "backup_plan_arn" {
  description = "AWS Backup plan ARN"
  value       = module.storage.backup_plan_arn
}

# Cost Optimization Outputs
output "estimated_monthly_cost" {
  description = "Estimated monthly cost (USD)"
  value       = module.compute.estimated_monthly_cost
}

output "cost_optimization_recommendations" {
  description = "Cost optimization recommendations"
  value = [
    "Enable spot instances for non-critical workloads",
    "Use scheduled scaling for predictable workloads",
    "Implement S3 lifecycle policies for log retention",
    "Right-size instances based on actual usage",
    "Enable detailed monitoring for better optimization"
  ]
}

# Configuration Information
output "kubectl_config" {
  description = "Kubectl configuration command"
  value       = "aws eks update-kubeconfig --region ${local.region} --name ${module.compute.eks_cluster_name}"
}

output "database_connection_string" {
  description = "Database connection string template"
  value       = "postgresql://${module.database.database_username}:PASSWORD@${module.database.database_endpoint}:${module.database.database_port}/${module.database.database_name}"
  sensitive   = true
}

# Health Check URLs
output "health_check_urls" {
  description = "Application health check URLs"
  value = {
    application = "https://${module.application.load_balancer_dns_name}/health"
    prometheus  = module.monitoring.prometheus_health_url
    grafana     = "${module.monitoring.grafana_endpoint}/api/health"
    elasticsearch = "${module.observability.elasticsearch_endpoint}/_cluster/health"
  }
  sensitive = true
}

# Resource Tags
output "common_tags" {
  description = "Common tags applied to all resources"
  value       = local.common_tags
}

# Multi-cloud Integration Status
output "multi_cloud_status" {
  description = "Multi-cloud integration status"
  value = {
    aws_primary = "enabled"
    azure       = var.enable_azure_integration ? "enabled" : "disabled"
    gcp         = var.enable_gcp_integration ? "enabled" : "disabled"
  }
}

# Security Compliance
output "security_compliance_status" {
  description = "Security compliance status"
  value = {
    encryption_at_rest  = "enabled"
    encryption_in_transit = "enabled"
    vpc_flow_logs      = "enabled"
    waf_protection     = var.enable_waf ? "enabled" : "disabled"
    certificate_manager = "enabled"
    secrets_manager    = "enabled"
  }
}

# Deployment Information
output "deployment_timestamp" {
  description = "Deployment timestamp"
  value       = timestamp()
}

output "terraform_version" {
  description = "Terraform version used for deployment"
  value       = "~> 1.5"
}

# Connection Commands
output "connection_commands" {
  description = "Commands to connect to various services"
  value = {
    kubectl         = "aws eks update-kubeconfig --region ${local.region} --name ${module.compute.eks_cluster_name}"
    database        = "psql -h ${module.database.database_endpoint} -p ${module.database.database_port} -U ${module.database.database_username} -d ${module.database.database_name}"
    ssh_bastion     = module.networking.bastion_connection_command
    port_forward    = "kubectl port-forward -n monitoring svc/prometheus-server 9090:80"
  }
  sensitive = true
}