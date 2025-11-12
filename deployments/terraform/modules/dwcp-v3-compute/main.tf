# DWCP v3 Compute Infrastructure Module
# Creates EC2 instances, auto-scaling groups, and load balancers for DWCP v3

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
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_id" {
  description = "VPC ID"
  type        = string
}

variable "datacenter_subnet_ids" {
  description = "List of datacenter subnet IDs"
  type        = list(string)
}

variable "internet_subnet_ids" {
  description = "List of internet subnet IDs"
  type        = list(string)
}

variable "datacenter_security_group_id" {
  description = "Security group ID for datacenter nodes"
  type        = string
}

variable "internet_security_group_id" {
  description = "Security group ID for internet nodes"
  type        = string
}

variable "datacenter_instance_type" {
  description = "Instance type for datacenter nodes (RDMA-capable)"
  type        = string
  default     = "p4d.24xlarge" # EFA-enabled, 400 Gbps network
}

variable "internet_instance_type" {
  description = "Instance type for internet nodes"
  type        = string
  default     = "c6in.32xlarge" # Network-optimized, 200 Gbps
}

variable "datacenter_min_size" {
  description = "Minimum number of datacenter nodes"
  type        = number
  default     = 3
}

variable "datacenter_max_size" {
  description = "Maximum number of datacenter nodes"
  type        = number
  default     = 10
}

variable "internet_min_size" {
  description = "Minimum number of internet nodes"
  type        = number
  default     = 5
}

variable "internet_max_size" {
  description = "Maximum number of internet nodes"
  type        = number
  default     = 20
}

variable "ssh_key_name" {
  description = "SSH key pair name"
  type        = string
}

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default = {
    Project   = "NovaCron"
    Component = "DWCP-v3"
  }
}

# Data sources
data "aws_ami" "ubuntu_22_04" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# IAM Role for EC2 instances
resource "aws_iam_role" "dwcp_instance_role" {
  name = "dwcp-v3-instance-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = merge(var.tags, {
    Name        = "dwcp-v3-instance-role-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "dwcp_instance_policy" {
  name = "dwcp-v3-instance-policy-${var.environment}"
  role = aws_iam_role.dwcp_instance_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeTags",
          "ec2:DescribeNetworkInterfaces",
          "autoscaling:DescribeAutoScalingGroups",
          "autoscaling:DescribeAutoScalingInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/dwcp/*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "arn:aws:s3:::dwcp-${var.environment}-config/*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "dwcp_instance_profile" {
  name = "dwcp-v3-instance-profile-${var.environment}"
  role = aws_iam_role.dwcp_instance_role.name

  tags = merge(var.tags, {
    Name        = "dwcp-v3-instance-profile-${var.environment}"
    Environment = var.environment
  })
}

# User data script
locals {
  user_data_datacenter = <<-EOF
    #!/bin/bash
    set -e

    # Update system
    apt-get update
    apt-get upgrade -y

    # Install CloudWatch agent
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    dpkg -i amazon-cloudwatch-agent.deb

    # Configure kernel for RDMA
    cat >> /etc/sysctl.conf <<EOL
    net.core.rmem_max=134217728
    net.core.wmem_max=134217728
    vm.zone_reclaim_mode=0
    kernel.numa_balancing=0
    EOL
    sysctl -p

    # Install EFA driver
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
    tar -xf aws-efa-installer-latest.tar.gz
    cd aws-efa-installer
    ./efa_installer.sh -y

    # Configure DWCP
    mkdir -p /etc/dwcp /var/lib/dwcp /var/log/dwcp

    # Signal completion
    /opt/aws/bin/cfn-signal -e $? --stack ${var.environment} --resource DatacenterASG
  EOF

  user_data_internet = <<-EOF
    #!/bin/bash
    set -e

    # Update system
    apt-get update
    apt-get upgrade -y

    # Install CloudWatch agent
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    dpkg -i amazon-cloudwatch-agent.deb

    # Configure kernel for internet mode
    cat >> /etc/sysctl.conf <<EOL
    net.ipv4.tcp_congestion_control=bbr
    net.core.default_qdisc=fq
    net.ipv4.tcp_rmem=4096 87380 134217728
    net.ipv4.tcp_wmem=4096 65536 134217728
    EOL
    sysctl -p

    # Load BBR module
    modprobe tcp_bbr
    echo 'tcp_bbr' >> /etc/modules-load.d/dwcp.conf

    # Configure DWCP
    mkdir -p /etc/dwcp /var/lib/dwcp /var/log/dwcp

    # Signal completion
    /opt/aws/bin/cfn-signal -e $? --stack ${var.environment} --resource InternetASG
  EOF
}

# Launch Templates
resource "aws_launch_template" "datacenter_lt" {
  name_prefix   = "dwcp-v3-datacenter-${var.environment}-"
  image_id      = data.aws_ami.ubuntu_22_04.id
  instance_type = var.datacenter_instance_type
  key_name      = var.ssh_key_name
  user_data     = base64encode(local.user_data_datacenter)

  iam_instance_profile {
    name = aws_iam_instance_profile.dwcp_instance_profile.name
  }

  network_interfaces {
    associate_public_ip_address = false
    delete_on_termination       = true
    security_groups             = [var.datacenter_security_group_id]

    # Enable EFA for RDMA
    interface_type = "efa"
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 100
      volume_type           = "gp3"
      iops                  = 16000
      throughput            = 1000
      delete_on_termination = true
      encrypted             = true
    }
  }

  # Instance metadata options
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  # Placement for optimal NUMA/network performance
  placement {
    group_name = aws_placement_group.datacenter_pg.name
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.tags, {
      Name        = "dwcp-v3-datacenter-${var.environment}"
      Environment = var.environment
      Mode        = "datacenter"
    })
  }

  tag_specifications {
    resource_type = "volume"
    tags = merge(var.tags, {
      Name        = "dwcp-v3-datacenter-volume-${var.environment}"
      Environment = var.environment
    })
  }
}

resource "aws_launch_template" "internet_lt" {
  name_prefix   = "dwcp-v3-internet-${var.environment}-"
  image_id      = data.aws_ami.ubuntu_22_04.id
  instance_type = var.internet_instance_type
  key_name      = var.ssh_key_name
  user_data     = base64encode(local.user_data_internet)

  iam_instance_profile {
    name = aws_iam_instance_profile.dwcp_instance_profile.name
  }

  network_interfaces {
    associate_public_ip_address = true
    delete_on_termination       = true
    security_groups             = [var.internet_security_group_id]
  }

  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size           = 50
      volume_type           = "gp3"
      iops                  = 3000
      throughput            = 125
      delete_on_termination = true
      encrypted             = true
    }
  }

  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  tag_specifications {
    resource_type = "instance"
    tags = merge(var.tags, {
      Name        = "dwcp-v3-internet-${var.environment}"
      Environment = var.environment
      Mode        = "internet"
    })
  }
}

# Placement Group for datacenter nodes (cluster placement for low latency)
resource "aws_placement_group" "datacenter_pg" {
  name     = "dwcp-v3-datacenter-pg-${var.environment}"
  strategy = "cluster"

  tags = merge(var.tags, {
    Name        = "dwcp-v3-datacenter-pg-${var.environment}"
    Environment = var.environment
  })
}

# Auto Scaling Groups
resource "aws_autoscaling_group" "datacenter_asg" {
  name                = "dwcp-v3-datacenter-asg-${var.environment}"
  vpc_zone_identifier = var.datacenter_subnet_ids
  min_size            = var.datacenter_min_size
  max_size            = var.datacenter_max_size
  desired_capacity    = var.datacenter_min_size
  health_check_type   = "ELB"
  health_check_grace_period = 300

  launch_template {
    id      = aws_launch_template.datacenter_lt.id
    version = "$Latest"
  }

  target_group_arns = [aws_lb_target_group.datacenter_tg.arn]

  tag {
    key                 = "Name"
    value               = "dwcp-v3-datacenter-${var.environment}"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }

  tag {
    key                 = "Mode"
    value               = "datacenter"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_autoscaling_group" "internet_asg" {
  name                = "dwcp-v3-internet-asg-${var.environment}"
  vpc_zone_identifier = var.internet_subnet_ids
  min_size            = var.internet_min_size
  max_size            = var.internet_max_size
  desired_capacity    = var.internet_min_size
  health_check_type   = "ELB"
  health_check_grace_period = 300

  launch_template {
    id      = aws_launch_template.internet_lt.id
    version = "$Latest"
  }

  target_group_arns = [aws_lb_target_group.internet_tg.arn]

  tag {
    key                 = "Name"
    value               = "dwcp-v3-internet-${var.environment}"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }

  tag {
    key                 = "Mode"
    value               = "internet"
    propagate_at_launch = true
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Policies
resource "aws_autoscaling_policy" "datacenter_scale_up" {
  name                   = "dwcp-v3-datacenter-scale-up-${var.environment}"
  scaling_adjustment     = 2
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.datacenter_asg.name
}

resource "aws_autoscaling_policy" "datacenter_scale_down" {
  name                   = "dwcp-v3-datacenter-scale-down-${var.environment}"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.datacenter_asg.name
}

resource "aws_autoscaling_policy" "internet_scale_up" {
  name                   = "dwcp-v3-internet-scale-up-${var.environment}"
  scaling_adjustment     = 3
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.internet_asg.name
}

resource "aws_autoscaling_policy" "internet_scale_down" {
  name                   = "dwcp-v3-internet-scale-down-${var.environment}"
  scaling_adjustment     = -1
  adjustment_type        = "ChangeInCapacity"
  cooldown               = 300
  autoscaling_group_name = aws_autoscaling_group.internet_asg.name
}

# Load Balancers
resource "aws_lb" "datacenter_nlb" {
  name               = "dwcp-v3-datacenter-nlb-${var.environment}"
  internal           = true
  load_balancer_type = "network"
  subnets            = var.datacenter_subnet_ids

  enable_cross_zone_load_balancing = true

  tags = merge(var.tags, {
    Name        = "dwcp-v3-datacenter-nlb-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_lb" "internet_nlb" {
  name               = "dwcp-v3-internet-nlb-${var.environment}"
  internal           = false
  load_balancer_type = "network"
  subnets            = var.internet_subnet_ids

  enable_cross_zone_load_balancing = true

  tags = merge(var.tags, {
    Name        = "dwcp-v3-internet-nlb-${var.environment}"
    Environment = var.environment
  })
}

# Target Groups
resource "aws_lb_target_group" "datacenter_tg" {
  name     = "dwcp-v3-datacenter-tg-${var.environment}"
  port     = 8080
  protocol = "TCP"
  vpc_id   = var.vpc_id

  health_check {
    protocol            = "HTTP"
    path                = "/health"
    port                = "9100"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 10
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-datacenter-tg-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_lb_target_group" "internet_tg" {
  name     = "dwcp-v3-internet-tg-${var.environment}"
  port     = 443
  protocol = "TCP"
  vpc_id   = var.vpc_id

  health_check {
    protocol            = "HTTP"
    path                = "/health"
    port                = "9100"
    healthy_threshold   = 2
    unhealthy_threshold = 2
    timeout             = 5
    interval            = 10
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-internet-tg-${var.environment}"
    Environment = var.environment
  })
}

# Listeners
resource "aws_lb_listener" "datacenter_listener" {
  load_balancer_arn = aws_lb.datacenter_nlb.arn
  port              = 8080
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.datacenter_tg.arn
  }
}

resource "aws_lb_listener" "internet_listener" {
  load_balancer_arn = aws_lb.internet_nlb.arn
  port              = 443
  protocol          = "TCP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.internet_tg.arn
  }
}

# Outputs
output "datacenter_asg_name" {
  description = "Datacenter auto-scaling group name"
  value       = aws_autoscaling_group.datacenter_asg.name
}

output "internet_asg_name" {
  description = "Internet auto-scaling group name"
  value       = aws_autoscaling_group.internet_asg.name
}

output "datacenter_nlb_dns" {
  description = "Datacenter NLB DNS name"
  value       = aws_lb.datacenter_nlb.dns_name
}

output "internet_nlb_dns" {
  description = "Internet NLB DNS name"
  value       = aws_lb.internet_nlb.dns_name
}
