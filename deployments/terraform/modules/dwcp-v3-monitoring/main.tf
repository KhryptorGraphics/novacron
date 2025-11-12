# DWCP v3 Monitoring Infrastructure Module
# Deploys Prometheus, Grafana, and CloudWatch integration

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
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

variable "subnet_ids" {
  description = "List of subnet IDs for monitoring infrastructure"
  type        = list(string)
}

variable "monitoring_instance_type" {
  description = "Instance type for monitoring nodes"
  type        = string
  default     = "m6i.2xlarge"
}

variable "prometheus_retention_days" {
  description = "Prometheus data retention in days"
  type        = number
  default     = 30
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

variable "tags" {
  description = "Common tags"
  type        = map(string)
  default = {
    Project   = "NovaCron"
    Component = "DWCP-v3-Monitoring"
  }
}

# Security Group for monitoring
resource "aws_security_group" "monitoring_sg" {
  name        = "dwcp-v3-monitoring-sg-${var.environment}"
  description = "Security group for DWCP v3 monitoring infrastructure"
  vpc_id      = var.vpc_id

  # Prometheus
  ingress {
    description = "Prometheus UI"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  # Grafana
  ingress {
    description = "Grafana UI"
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  # Alertmanager
  ingress {
    description = "Alertmanager UI"
    from_port   = 9093
    to_port     = 9093
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  # Node Exporter
  ingress {
    description = "Node Exporter"
    from_port   = 9100
    to_port     = 9100
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }

  # SSH
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
    Name        = "dwcp-v3-monitoring-sg-${var.environment}"
    Environment = var.environment
  })
}

# IAM Role for monitoring instances
resource "aws_iam_role" "monitoring_role" {
  name = "dwcp-v3-monitoring-role-${var.environment}"

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
    Name        = "dwcp-v3-monitoring-role-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_iam_role_policy" "monitoring_policy" {
  name = "dwcp-v3-monitoring-policy-${var.environment}"
  role = aws_iam_role.monitoring_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ec2:DescribeInstances",
          "ec2:DescribeTags",
          "autoscaling:DescribeAutoScalingGroups"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics",
          "cloudwatch:PutMetricData"
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
        Resource = "arn:aws:logs:*:*:log-group:/aws/dwcp/monitoring/*"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "monitoring_profile" {
  name = "dwcp-v3-monitoring-profile-${var.environment}"
  role = aws_iam_role.monitoring_role.name

  tags = merge(var.tags, {
    Name        = "dwcp-v3-monitoring-profile-${var.environment}"
    Environment = var.environment
  })
}

# Data source for AMI
data "aws_ami" "ubuntu_22_04" {
  most_recent = true
  owners      = ["099720109477"]

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# EBS volumes for Prometheus and Grafana data
resource "aws_ebs_volume" "prometheus_data" {
  availability_zone = data.aws_subnet.monitoring_subnet.availability_zone
  size              = 500
  type              = "gp3"
  iops              = 16000
  throughput        = 1000
  encrypted         = true

  tags = merge(var.tags, {
    Name        = "dwcp-v3-prometheus-data-${var.environment}"
    Environment = var.environment
    Component   = "prometheus"
  })
}

resource "aws_ebs_volume" "grafana_data" {
  availability_zone = data.aws_subnet.monitoring_subnet.availability_zone
  size              = 100
  type              = "gp3"
  iops              = 3000
  encrypted         = true

  tags = merge(var.tags, {
    Name        = "dwcp-v3-grafana-data-${var.environment}"
    Environment = var.environment
    Component   = "grafana"
  })
}

data "aws_subnet" "monitoring_subnet" {
  id = var.subnet_ids[0]
}

# User data for monitoring instance
locals {
  monitoring_user_data = <<-EOF
    #!/bin/bash
    set -e

    # Update system
    apt-get update
    apt-get upgrade -y

    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu

    # Install Docker Compose
    curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose

    # Wait for EBS volumes to attach
    while [ ! -e /dev/xvdf ]; do sleep 1; done
    while [ ! -e /dev/xvdg ]; do sleep 1; done

    # Format and mount Prometheus volume
    if ! blkid /dev/xvdf; then
      mkfs.ext4 /dev/xvdf
    fi
    mkdir -p /data/prometheus
    mount /dev/xvdf /data/prometheus
    echo '/dev/xvdf /data/prometheus ext4 defaults,nofail 0 2' >> /etc/fstab

    # Format and mount Grafana volume
    if ! blkid /dev/xvdg; then
      mkfs.ext4 /dev/xvdg
    fi
    mkdir -p /data/grafana
    mount /dev/xvdg /data/grafana
    echo '/dev/xvdg /data/grafana ext4 defaults,nofail 0 2' >> /etc/fstab

    # Set permissions
    chown -R 65534:65534 /data/prometheus
    chown -R 472:472 /data/grafana

    # Create monitoring stack
    mkdir -p /opt/monitoring
    cd /opt/monitoring

    # Create Prometheus configuration
    cat > prometheus.yml <<'EOL'
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    environment: ${var.environment}
    cluster: dwcp-v3

alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

rule_files:
  - /etc/prometheus/alerts/*.yml

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # DWCP v3 nodes discovery
  - job_name: 'dwcp-datacenter'
    ec2_sd_configs:
      - region: us-east-1
        port: 9100
        filters:
          - name: tag:Mode
            values:
              - datacenter
          - name: tag:Environment
            values:
              - ${var.environment}
    relabel_configs:
      - source_labels: [__meta_ec2_tag_Name]
        target_label: instance
      - source_labels: [__meta_ec2_private_ip]
        target_label: __address__
        replacement: $1:9100

  - job_name: 'dwcp-internet'
    ec2_sd_configs:
      - region: us-east-1
        port: 9100
        filters:
          - name: tag:Mode
            values:
              - internet
          - name: tag:Environment
            values:
              - ${var.environment}
    relabel_configs:
      - source_labels: [__meta_ec2_tag_Name]
        target_label: instance
      - source_labels: [__meta_ec2_private_ip]
        target_label: __address__
        replacement: $1:9100

  - job_name: 'node-exporter'
    ec2_sd_configs:
      - region: us-east-1
        port: 9100
        filters:
          - name: tag:Environment
            values:
              - ${var.environment}
    relabel_configs:
      - source_labels: [__meta_ec2_tag_Name]
        target_label: instance

  # CloudWatch metrics
  - job_name: 'cloudwatch'
    static_configs:
      - targets: ['cloudwatch-exporter:9106']
EOL

    # Create alert rules
    mkdir -p alerts
    cat > alerts/dwcp.yml <<'EOL'
groups:
  - name: dwcp_v3_alerts
    interval: 30s
    rules:
      - alert: DWCPHighLatency
        expr: dwcp_latency_p99_us > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p99 latency on {{ $labels.instance }}"
          description: "p99 latency is {{ $value }}us on {{ $labels.instance }}"

      - alert: DWCPLowThroughput
        expr: dwcp_throughput_gbps < 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Low throughput on {{ $labels.instance }}"
          description: "Throughput is {{ $value }} Gbps on {{ $labels.instance }}"

      - alert: DWCPNodeDown
        expr: up{job=~"dwcp-.*"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "DWCP node {{ $labels.instance }} is down"
          description: "DWCP node {{ $labels.instance }} has been down for more than 2 minutes"

      - alert: DWCPHighErrorRate
        expr: rate(dwcp_errors_total[5m]) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.instance }}"
          description: "Error rate is {{ $value }} errors/sec on {{ $labels.instance }}"

      - alert: DWCPConnectionSaturation
        expr: dwcp_active_connections / dwcp_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Connection saturation on {{ $labels.instance }}"
          description: "Connection usage is {{ $value | humanizePercentage }} on {{ $labels.instance }}"
EOL

    # Create Alertmanager configuration
    cat > alertmanager.yml <<'EOL'
global:
  resolve_timeout: 5m
  smtp_smarthost: 'localhost:25'
  smtp_from: 'alertmanager@novacron.io'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true
    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
    email_configs:
      - to: '${var.alert_email}'

  - name: 'critical'
    email_configs:
      - to: '${var.alert_email}'
        send_resolved: true

  - name: 'warning'
    email_configs:
      - to: '${var.alert_email}'
        send_resolved: false

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'instance']
EOL

    # Create Docker Compose file
    cat > docker-compose.yml <<'EOL'
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:v2.47.0
    container_name: prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=${var.prometheus_retention_days}d'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--web.enable-lifecycle'
    ports:
      - "9090:9090"
    volumes:
      - /data/prometheus:/prometheus
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - ./alerts:/etc/prometheus/alerts
    restart: unless-stopped
    networks:
      - monitoring

  alertmanager:
    image: prom/alertmanager:v0.26.0
    container_name: alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager-data:/alertmanager
    restart: unless-stopped
    networks:
      - monitoring

  grafana:
    image: grafana/grafana:10.1.5
    container_name: grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${var.grafana_admin_password}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_SERVER_ROOT_URL=http://grafana.novacron.io
    ports:
      - "3000:3000"
    volumes:
      - /data/grafana:/var/lib/grafana
      - ./grafana-provisioning:/etc/grafana/provisioning
    restart: unless-stopped
    networks:
      - monitoring

  node-exporter:
    image: prom/node-exporter:v1.6.1
    container_name: node-exporter
    command:
      - '--path.procfs=/host/proc'
      - '--path.sysfs=/host/sys'
      - '--path.rootfs=/rootfs'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    restart: unless-stopped
    networks:
      - monitoring

  cloudwatch-exporter:
    image: prom/cloudwatch-exporter:v0.15.5
    container_name: cloudwatch-exporter
    ports:
      - "9106:9106"
    volumes:
      - ./cloudwatch-exporter.yml:/config/config.yml
    command:
      - '/config/config.yml'
    restart: unless-stopped
    networks:
      - monitoring

networks:
  monitoring:
    driver: bridge

volumes:
  alertmanager-data:
EOL

    # Create Grafana provisioning
    mkdir -p grafana-provisioning/datasources grafana-provisioning/dashboards

    cat > grafana-provisioning/datasources/prometheus.yml <<'EOL'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
EOL

    cat > grafana-provisioning/dashboards/dashboard.yml <<'EOL'
apiVersion: 1

providers:
  - name: 'DWCP v3'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: false
    options:
      path: /etc/grafana/provisioning/dashboards
EOL

    # Start monitoring stack
    docker-compose up -d

    # Wait for services
    sleep 30

    # Install CloudWatch agent
    wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
    dpkg -i amazon-cloudwatch-agent.deb

    echo "Monitoring stack deployed successfully"
  EOF
}

# Launch monitoring instance
resource "aws_instance" "monitoring" {
  ami                    = data.aws_ami.ubuntu_22_04.id
  instance_type          = var.monitoring_instance_type
  subnet_id              = var.subnet_ids[0]
  vpc_security_group_ids = [aws_security_group.monitoring_sg.id]
  iam_instance_profile   = aws_iam_instance_profile.monitoring_profile.name
  user_data              = local.monitoring_user_data

  root_block_device {
    volume_size           = 50
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-monitoring-${var.environment}"
    Environment = var.environment
    Component   = "monitoring"
  })

  lifecycle {
    ignore_changes = [user_data]
  }
}

# Attach EBS volumes
resource "aws_volume_attachment" "prometheus_attachment" {
  device_name = "/dev/xvdf"
  volume_id   = aws_ebs_volume.prometheus_data.id
  instance_id = aws_instance.monitoring.id
}

resource "aws_volume_attachment" "grafana_attachment" {
  device_name = "/dev/xvdg"
  volume_id   = aws_ebs_volume.grafana_data.id
  instance_id = aws_instance.monitoring.id
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "monitoring_logs" {
  name              = "/aws/dwcp/monitoring/${var.environment}"
  retention_in_days = 30

  tags = merge(var.tags, {
    Name        = "dwcp-v3-monitoring-logs-${var.environment}"
    Environment = var.environment
  })
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "monitoring_cpu" {
  alarm_name          = "dwcp-v3-monitoring-high-cpu-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "Monitoring instance CPU utilization is too high"
  alarm_actions       = [aws_sns_topic.alerts.arn]

  dimensions = {
    InstanceId = aws_instance.monitoring.id
  }

  tags = merge(var.tags, {
    Name        = "dwcp-v3-monitoring-cpu-alarm-${var.environment}"
    Environment = var.environment
  })
}

# SNS Topic for alerts
resource "aws_sns_topic" "alerts" {
  name = "dwcp-v3-monitoring-alerts-${var.environment}"

  tags = merge(var.tags, {
    Name        = "dwcp-v3-monitoring-alerts-${var.environment}"
    Environment = var.environment
  })
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# Outputs
output "prometheus_url" {
  description = "Prometheus URL"
  value       = "http://${aws_instance.monitoring.private_ip}:9090"
}

output "grafana_url" {
  description = "Grafana URL"
  value       = "http://${aws_instance.monitoring.private_ip}:3000"
}

output "alertmanager_url" {
  description = "Alertmanager URL"
  value       = "http://${aws_instance.monitoring.private_ip}:9093"
}

output "monitoring_instance_id" {
  description = "Monitoring instance ID"
  value       = aws_instance.monitoring.id
}
