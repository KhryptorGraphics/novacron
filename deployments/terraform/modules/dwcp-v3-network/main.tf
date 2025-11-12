# DWCP v3 Network Infrastructure Module
# Creates VPC, subnets, security groups, and RDMA-enabled networking

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Variables
variable "environment" {
  description = "Environment name (production, staging, development)"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "List of availability zones"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b", "us-east-1c"]
}

variable "enable_rdma" {
  description = "Enable RDMA-capable networking (EFA)"
  type        = bool
  default     = true
}

variable "enable_ipv6" {
  description = "Enable IPv6 support"
  type        = bool
  default     = true
}

variable "enable_flow_logs" {
  description = "Enable VPC flow logs"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project   = "NovaCron"
    Component = "DWCP-v3"
  }
}

# VPC
resource "aws_vpc" "dwcp_vpc" {
  cidr_block                       = var.vpc_cidr
  enable_dns_hostnames             = true
  enable_dns_support               = true
  assign_generated_ipv6_cidr_block = var.enable_ipv6

  tags = merge(var.tags, {
    Name        = "dwcp-v3-vpc-${var.environment}"
    Environment = var.environment
  })
}

# Internet Gateway
resource "aws_internet_gateway" "dwcp_igw" {
  vpc_id = aws_vpc.dwcp_vpc.id

  tags = merge(var.tags, {
    Name        = "dwcp-v3-igw-${var.environment}"
    Environment = var.environment
  })
}

# Datacenter Subnets (Private, RDMA-enabled)
resource "aws_subnet" "datacenter_subnets" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.dwcp_vpc.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = var.availability_zones[count.index]

  # Enable IPv6 if configured
  ipv6_cidr_block                 = var.enable_ipv6 ? cidrsubnet(aws_vpc.dwcp_vpc.ipv6_cidr_block, 8, count.index) : null
  assign_ipv6_address_on_creation = var.enable_ipv6

  tags = merge(var.tags, {
    Name        = "dwcp-v3-datacenter-subnet-${count.index + 1}-${var.environment}"
    Type        = "datacenter"
    RDMA        = "enabled"
    Environment = var.environment
  })
}

# Internet Subnets (Public)
resource "aws_subnet" "internet_subnets" {
  count                   = length(var.availability_zones)
  vpc_id                  = aws_vpc.dwcp_vpc.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 10)
  availability_zone       = var.availability_zones[count.index]
  map_public_ip_on_launch = true

  ipv6_cidr_block                 = var.enable_ipv6 ? cidrsubnet(aws_vpc.dwcp_vpc.ipv6_cidr_block, 8, count.index + 10) : null
  assign_ipv6_address_on_creation = var.enable_ipv6

  tags = merge(var.tags, {
    Name        = "dwcp-v3-internet-subnet-${count.index + 1}-${var.environment}"
    Type        = "internet"
    Public      = "true"
    Environment = var.environment
  })
}

# NAT Gateways for private subnets
resource "aws_eip" "nat_eips" {
  count  = length(var.availability_zones)
  domain = "vpc"

  tags = merge(var.tags, {
    Name        = "dwcp-v3-nat-eip-${count.index + 1}-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_nat_gateway" "nat_gateways" {
  count         = length(var.availability_zones)
  allocation_id = aws_eip.nat_eips[count.index].id
  subnet_id     = aws_subnet.internet_subnets[count.index].id

  tags = merge(var.tags, {
    Name        = "dwcp-v3-nat-${count.index + 1}-${var.environment}"
    Environment = var.environment
  })

  depends_on = [aws_internet_gateway.dwcp_igw]
}

# Route Tables
resource "aws_route_table" "datacenter_rt" {
  count  = length(var.availability_zones)
  vpc_id = aws_vpc.dwcp_vpc.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateways[count.index].id
  }

  dynamic "route" {
    for_each = var.enable_ipv6 ? [1] : []
    content {
      ipv6_cidr_block = "::/0"
      gateway_id      = aws_internet_gateway.dwcp_igw.id
    }
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-datacenter-rt-${count.index + 1}-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_route_table" "internet_rt" {
  vpc_id = aws_vpc.dwcp_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.dwcp_igw.id
  }

  dynamic "route" {
    for_each = var.enable_ipv6 ? [1] : []
    content {
      ipv6_cidr_block = "::/0"
      gateway_id      = aws_internet_gateway.dwcp_igw.id
    }
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-internet-rt-${var.environment}"
    Environment = var.environment
  })
}

# Route Table Associations
resource "aws_route_table_association" "datacenter_rta" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.datacenter_subnets[count.index].id
  route_table_id = aws_route_table.datacenter_rt[count.index].id
}

resource "aws_route_table_association" "internet_rta" {
  count          = length(var.availability_zones)
  subnet_id      = aws_subnet.internet_subnets[count.index].id
  route_table_id = aws_route_table.internet_rt.id
}

# Security Groups
resource "aws_security_group" "dwcp_datacenter_sg" {
  name        = "dwcp-v3-datacenter-sg-${var.environment}"
  description = "Security group for DWCP v3 datacenter nodes (RDMA-enabled)"
  vpc_id      = aws_vpc.dwcp_vpc.id

  # RDMA/InfiniBand traffic
  ingress {
    description = "RDMA traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
  }

  # DWCP control plane
  ingress {
    description = "DWCP control plane"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  # Metrics endpoint
  ingress {
    description = "Prometheus metrics"
    from_port   = 9100
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  # Health checks
  ingress {
    description = "Health checks"
    from_port   = 9100
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  # SSH (restricted)
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"] # Adjust to your bastion CIDR
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-datacenter-sg-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_security_group" "dwcp_internet_sg" {
  name        = "dwcp-v3-internet-sg-${var.environment}"
  description = "Security group for DWCP v3 internet nodes"
  vpc_id      = aws_vpc.dwcp_vpc.id

  # DWCP protocol (TCP/BBR)
  ingress {
    description = "DWCP internet traffic"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # QUIC support
  ingress {
    description = "QUIC traffic"
    from_port   = 443
    to_port     = 443
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Internal VPC traffic
  ingress {
    description = "VPC internal traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = [var.vpc_cidr]
  }

  # Metrics
  ingress {
    description = "Prometheus metrics"
    from_port   = 9100
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = [var.vpc_cidr]
  }

  # SSH (restricted)
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  egress {
    description = "All outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-internet-sg-${var.environment}"
    Environment = var.environment
  })
}

# Network ACLs (additional layer)
resource "aws_network_acl" "dwcp_datacenter_acl" {
  vpc_id     = aws_vpc.dwcp_vpc.id
  subnet_ids = aws_subnet.datacenter_subnets[*].id

  # Ingress rules
  ingress {
    protocol   = -1
    rule_no    = 100
    action     = "allow"
    cidr_block = var.vpc_cidr
    from_port  = 0
    to_port    = 0
  }

  # Egress rules
  egress {
    protocol   = -1
    rule_no    = 100
    action     = "allow"
    cidr_block = "0.0.0.0/0"
    from_port  = 0
    to_port    = 0
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-datacenter-acl-${var.environment}"
    Environment = var.environment
  })
}

# VPC Flow Logs
resource "aws_cloudwatch_log_group" "flow_logs" {
  count             = var.enable_flow_logs ? 1 : 0
  name              = "/aws/vpc/dwcp-v3-${var.environment}"
  retention_in_days = 30

  tags = merge(var.tags, {
    Name        = "dwcp-v3-flow-logs-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_iam_role" "flow_logs_role" {
  count = var.enable_flow_logs ? 1 : 0
  name  = "dwcp-v3-flow-logs-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "vpc-flow-logs.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "dwcp-v3-flow-logs-role-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "flow_logs_policy" {
  count = var.enable_flow_logs ? 1 : 0
  name  = "dwcp-v3-flow-logs-policy-${var.environment}"
  role  = aws_iam_role.flow_logs_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogGroups",
          "logs:DescribeLogStreams"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

resource "aws_flow_log" "dwcp_flow_log" {
  count                = var.enable_flow_logs ? 1 : 0
  iam_role_arn         = aws_iam_role.flow_logs_role[0].arn
  log_destination      = aws_cloudwatch_log_group.flow_logs[0].arn
  traffic_type         = "ALL"
  vpc_id               = aws_vpc.dwcp_vpc.id
  max_aggregation_interval = 60

  tags = merge(var.tags, {
    Name        = "dwcp-v3-flow-log-${var.environment}"
    Environment = var.environment
  })
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.dwcp_vpc.id
}

output "vpc_cidr" {
  description = "VPC CIDR block"
  value       = aws_vpc.dwcp_vpc.cidr_block
}

output "datacenter_subnet_ids" {
  description = "List of datacenter subnet IDs"
  value       = aws_subnet.datacenter_subnets[*].id
}

output "internet_subnet_ids" {
  description = "List of internet subnet IDs"
  value       = aws_subnet.internet_subnets[*].id
}

output "datacenter_security_group_id" {
  description = "Security group ID for datacenter nodes"
  value       = aws_security_group.dwcp_datacenter_sg.id
}

output "internet_security_group_id" {
  description = "Security group ID for internet nodes"
  value       = aws_security_group.dwcp_internet_sg.id
}

output "nat_gateway_ips" {
  description = "Elastic IPs of NAT gateways"
  value       = aws_eip.nat_eips[*].public_ip
}
