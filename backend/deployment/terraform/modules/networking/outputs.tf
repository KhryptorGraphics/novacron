# NovaCron Networking Module - Outputs

output "vpc_id" {
  description = "ID of the VPC"
  value       = aws_vpc.main.id
}

output "vpc_cidr_block" {
  description = "CIDR block of the VPC"
  value       = aws_vpc.main.cidr_block
}

output "internet_gateway_id" {
  description = "ID of the Internet Gateway"
  value       = aws_internet_gateway.main.id
}

output "public_subnet_ids" {
  description = "IDs of the public subnets"
  value       = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  description = "IDs of the private subnets"
  value       = aws_subnet.private[*].id
}

output "database_subnet_ids" {
  description = "IDs of the database subnets"
  value       = aws_subnet.database[*].id
}

output "database_subnet_group_name" {
  description = "Name of the database subnet group"
  value       = aws_db_subnet_group.main.name
}

output "nat_gateway_ids" {
  description = "IDs of the NAT Gateways"
  value       = aws_nat_gateway.main[*].id
}

output "nat_gateway_ips" {
  description = "Elastic IP addresses of the NAT Gateways"
  value       = aws_eip.nat[*].public_ip
}

output "route53_zone_id" {
  description = "Route53 hosted zone ID"
  value       = ""  # To be implemented if needed
}

output "route53_zone_name" {
  description = "Route53 hosted zone name"
  value       = ""  # To be implemented if needed
}

output "dns_name" {
  description = "Primary DNS name for the application"
  value       = ""  # To be implemented if needed
}

output "bastion_connection_command" {
  description = "Command to connect to bastion host"
  value       = ""  # To be implemented if needed
}

output "vpc_flow_log_group_name" {
  description = "Name of VPC flow log CloudWatch group"
  value       = aws_cloudwatch_log_group.vpc_flow_log.name
}

output "vpc_endpoints" {
  description = "VPC endpoint information"
  value = {
    s3      = aws_vpc_endpoint.s3.id
    ec2     = aws_vpc_endpoint.ec2.id
    ecr_dkr = aws_vpc_endpoint.ecr_dkr.id
    ecr_api = aws_vpc_endpoint.ecr_api.id
  }
}