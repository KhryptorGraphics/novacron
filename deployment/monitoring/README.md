# NovaCron Monitoring Stack Deployment

## Overview

This directory contains the complete monitoring and observability stack for NovaCron, including metrics collection, alerting, logging, distributed tracing, and automated incident response.

## Quick Start

### Prerequisites

1. **System Requirements**:
   - Docker Engine 20.10+
   - Docker Compose 2.0+
   - Minimum 16GB RAM available for monitoring stack
   - 100GB+ free disk space for data retention

2. **Environment Setup**:
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit environment variables
   vim .env
   ```

3. **Required Environment Variables**:
   ```bash
   # Notification Configuration
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
   PAGERDUTY_INTEGRATION_KEY=your-pagerduty-key
   SMTP_PASSWORD=your-smtp-password
   
   # Database Passwords
   POSTGRES_PASSWORD=your-postgres-password
   REDIS_PASSWORD=your-redis-password
   
   # Security
   GRAFANA_ADMIN_PASSWORD=secure-password
   
   # External Integrations
   JIRA_URL=https://your-company.atlassian.net
   JIRA_USERNAME=service-account@company.com
   JIRA_TOKEN=your-jira-api-token
   ```

### Deployment

1. **Start the monitoring stack**:
   ```bash
   # Start all monitoring services
   docker-compose -f docker-compose.monitoring.yml up -d
   
   # Verify all services are running
   docker-compose -f docker-compose.monitoring.yml ps
   ```

2. **Access monitoring services**:
   - **Grafana**: http://localhost:3000 (admin/password from env)
   - **Prometheus**: http://localhost:9090
   - **Alertmanager**: http://localhost:9093
   - **Kibana**: http://localhost:5601
   - **Jaeger**: http://localhost:16686
   - **Monitoring Dashboard**: http://localhost:8090

3. **Import Grafana dashboards**:
   ```bash
   # Dashboards are automatically provisioned from ./grafana/dashboards/
   # Manual import available via Grafana UI if needed
   ```

### Verification

1. **Check service health**:
   ```bash
   # View monitoring stack health
   docker logs novacron-monitoring-healthcheck
   
   # Check individual services
   curl http://localhost:9090/-/healthy    # Prometheus
   curl http://localhost:9093/-/healthy    # Alertmanager
   curl http://localhost:3000/api/health   # Grafana
   curl http://localhost:9200/_cluster/health  # Elasticsearch
   ```

2. **Verify metrics collection**:
   ```bash
   # Check if metrics are being collected
   curl "http://localhost:9090/api/v1/query?query=up"
   
   # Verify custom NovaCron metrics
   curl "http://localhost:9090/api/v1/query?query=novacron_vm_count"
   ```

3. **Test alerting**:
   ```bash
   # Trigger a test alert
   curl -XPOST http://localhost:9093/api/v1/alerts \
     -H "Content-Type: application/json" \
     -d @test-alert.json
   ```

## Architecture Components

### Core Monitoring Stack

| Service | Port | Purpose | Resource Requirements |
|---------|------|---------|----------------------|
| Prometheus | 9090 | Metrics collection & storage | 4 CPU, 16GB RAM, 500GB storage |
| Alertmanager | 9093 | Alert routing & notifications | 2 CPU, 4GB RAM, 10GB storage |
| Grafana | 3000 | Visualization & dashboards | 2 CPU, 4GB RAM, 20GB storage |
| Elasticsearch | 9200 | Log storage & search | 8 CPU, 32GB RAM, 2TB storage |
| Kibana | 5601 | Log analysis & visualization | 2 CPU, 4GB RAM, 10GB storage |
| Jaeger | 16686 | Distributed tracing | 4 CPU, 8GB RAM, variable storage |

### Data Collection

| Service | Port | Purpose |
|---------|------|---------|
| Fluentd | 24224 | Log aggregation |
| Node Exporter | 9100 | System metrics |
| cAdvisor | 8080 | Container metrics |
| Blackbox Exporter | 9115 | External service monitoring |
| Redis Exporter | 9121 | Redis metrics |
| Postgres Exporter | 9187 | PostgreSQL metrics |
| Libvirt Exporter | 9177 | VM hypervisor metrics |

### Long-term Storage

| Service | Port | Purpose |
|---------|------|---------|
| Thanos Sidecar | 10901/10902 | Prometheus long-term storage |
| Thanos Query | 19192 | Unified query interface |

## Configuration

### Prometheus Configuration

**Key Features**:
- 15-second scrape interval for most metrics
- 5-second interval for critical metrics
- 7-day local retention
- Remote write to Thanos for long-term storage
- Federation support for multi-cluster deployments

**Custom Metrics**:
- `novacron_vm_*`: Virtual machine metrics
- `novacron_storage_*`: Storage performance and utilization
- `novacron_network_*`: Network performance metrics
- `novacron_migration_*`: VM migration statistics
- `novacron_security_*`: Security event metrics

### Alerting Configuration

**Alert Categories**:
- **Critical**: Service down, SLA breaches, security incidents
- **Warning**: Performance degradation, resource exhaustion
- **Capacity**: Proactive scaling alerts

**Notification Routing**:
- Critical alerts → PagerDuty + Slack + SMS
- Security alerts → Dedicated security team channels
- SLA breaches → Executive notifications
- Capacity warnings → Operations and capacity planning teams

### Dashboard Organization

**Executive Dashboards**:
- System health overview
- SLA compliance tracking
- Business impact metrics

**Operational Dashboards**:
- VM management and performance
- Storage performance and capacity
- Network topology and health
- Migration tracking and analytics

**Troubleshooting Dashboards**:
- Error analysis and patterns
- Resource utilization deep-dive
- Performance bottleneck identification

## Maintenance

### Daily Tasks

1. **Health Check Review**:
   ```bash
   # Check monitoring stack health
   ./scripts/daily-health-check.sh
   ```

2. **Alert Review**:
   ```bash
   # Review overnight alerts
   ./scripts/alert-summary.sh --since="24h"
   ```

3. **Capacity Monitoring**:
   ```bash
   # Check resource usage trends
   ./scripts/capacity-report.sh
   ```

### Weekly Tasks

1. **Performance Review**:
   ```bash
   # Generate weekly performance report
   ./scripts/weekly-performance-report.sh
   ```

2. **Alert Tuning**:
   ```bash
   # Review alert thresholds and noise
   ./scripts/alert-analysis.sh --period="7d"
   ```

### Monthly Tasks

1. **Retention Management**:
   ```bash
   # Clean up old data based on retention policies
   ./scripts/cleanup-old-data.sh
   ```

2. **Cost Optimization**:
   ```bash
   # Review monitoring infrastructure costs
   ./scripts/cost-analysis.sh
   ```

## Troubleshooting

### Common Issues

1. **High Memory Usage (Prometheus)**:
   ```bash
   # Check series count and cardinality
   curl http://localhost:9090/api/v1/label/__name__/values | jq length
   
   # Identify high cardinality metrics
   curl http://localhost:9090/api/v1/query?query=prometheus_tsdb_symbol_table_size_bytes
   ```

2. **Disk Space Issues (Elasticsearch)**:
   ```bash
   # Check index sizes
   curl "http://localhost:9200/_cat/indices?v&s=store.size:desc"
   
   # Clean up old indices
   curl -X DELETE "http://localhost:9200/novacron-*-$(date -d '30 days ago' '+%Y.%m.*')"
   ```

3. **Missing Metrics**:
   ```bash
   # Check scrape targets
   curl http://localhost:9090/api/v1/targets
   
   # Verify service discovery
   curl http://localhost:9090/api/v1/targets?state=down
   ```

### Performance Tuning

1. **Prometheus Query Optimization**:
   ```bash
   # Enable query logging
   # Add to prometheus.yml:
   # global:
   #   query_log_file: /prometheus/query.log
   ```

2. **Elasticsearch Performance**:
   ```bash
   # Monitor cluster health
   curl "http://localhost:9200/_cluster/health?pretty"
   
   # Check node statistics
   curl "http://localhost:9200/_nodes/stats?pretty"
   ```

3. **Grafana Dashboard Performance**:
   - Use recording rules for expensive queries
   - Limit time ranges on heavy dashboards
   - Optimize panel queries with appropriate intervals

## Security

### Access Control

1. **Grafana Security**:
   - Change default admin password
   - Configure LDAP/OAuth integration
   - Set up proper role-based access control

2. **Prometheus Security**:
   - Enable basic authentication
   - Configure TLS for scrape targets
   - Restrict query API access

3. **Network Security**:
   - Use dedicated monitoring network
   - Configure firewalls for monitoring ports
   - Enable TLS for inter-service communication

### Data Protection

1. **Sensitive Data**:
   - Scrub sensitive information from logs
   - Use secret management for credentials
   - Implement data retention policies

2. **Encryption**:
   - Enable TLS for all web interfaces
   - Encrypt data at rest (Elasticsearch)
   - Use encrypted communications between services

## Scaling

### Horizontal Scaling

1. **Prometheus**:
   ```bash
   # Use federation for multiple Prometheus instances
   # Configure in prometheus.yml:
   # - job_name: 'federate'
   #   scrape_interval: 15s
   #   honor_labels: true
   #   metrics_path: '/federate'
   ```

2. **Elasticsearch**:
   ```bash
   # Add more nodes to the cluster
   # Update docker-compose to include additional ES nodes
   ```

### Vertical Scaling

1. **Resource Limits**:
   ```yaml
   # Update docker-compose resource limits
   deploy:
     resources:
       limits:
         memory: 32G
         cpus: '8'
   ```

## Integration

### CI/CD Integration

1. **Build Metrics**:
   ```bash
   # Add build metrics to Prometheus
   # Use pushgateway for batch job metrics
   ```

2. **Deployment Tracking**:
   ```bash
   # Add deployment annotations to Grafana
   # Use Grafana annotations API
   ```

### External Systems

1. **Cloud Provider Integration**:
   - AWS CloudWatch metrics
   - Azure Monitor integration
   - GCP Monitoring

2. **Service Mesh Integration**:
   - Istio metrics collection
   - Linkerd observability
   - Consul Connect monitoring

## Support

### Documentation

- **Architecture**: [MONITORING_ARCHITECTURE.md](../docs/monitoring/MONITORING_ARCHITECTURE.md)
- **Runbooks**: [./runbooks/](./runbooks/)
- **API Documentation**: [./api/](./api/)

### Getting Help

- **Slack**: #ops-monitoring
- **Email**: ops-team@company.com
- **On-call**: PagerDuty escalation

### Contributing

1. **Making Changes**:
   - Test changes in development environment
   - Update documentation
   - Submit pull request with monitoring team review

2. **Adding New Dashboards**:
   - Follow dashboard design guidelines
   - Include appropriate documentation
   - Test with various data scenarios