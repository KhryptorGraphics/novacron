#!/usr/bin/env node

/**
 * Metrics Collection and Analysis Command for NovaCron
 */

const { execSync } = require('child_process');

module.exports = {
  name: 'monitoring:metrics',
  description: 'Collect and analyze system metrics',
  usage: '/monitoring:metrics [--vm <vm-id>] [--period <1h|24h|7d>] [--export]',
  
  async execute(args) {
    const vmIndex = args.indexOf('--vm');
    const vmId = vmIndex !== -1 ? args[vmIndex + 1] : null;
    const periodIndex = args.indexOf('--period');
    const period = periodIndex !== -1 ? args[periodIndex + 1] : '1h';
    const shouldExport = args.includes('--export');
    
    console.log('📈 Collecting metrics...\n');
    
    try {
      // Determine endpoint based on scope
      let endpoint = vmId 
        ? `/api/metrics/vms/${vmId}?period=${period}`
        : `/api/metrics/cluster?period=${period}`;
      
      const metricsResponse = execSync(
        `curl -s http://localhost:8090${endpoint}`,
        { encoding: 'utf8' }
      );
      const metrics = JSON.parse(metricsResponse);
      
      // Display metrics summary
      console.log(`📊 Metrics Summary (${period}):`);
      console.log(`  Time Range: ${metrics.start_time} to ${metrics.end_time}`);
      
      if (metrics.cpu) {
        console.log('\n⚡ CPU Metrics:');
        console.log(`  Average Usage: ${metrics.cpu.average}%`);
        console.log(`  Peak Usage: ${metrics.cpu.peak}%`);
        console.log(`  95th Percentile: ${metrics.cpu.p95}%`);
      }
      
      if (metrics.memory) {
        console.log('\n💾 Memory Metrics:');
        console.log(`  Average Usage: ${metrics.memory.average}%`);
        console.log(`  Peak Usage: ${metrics.memory.peak}%`);
        console.log(`  Available: ${metrics.memory.available_gb} GB`);
      }
      
      if (metrics.storage) {
        console.log('\n💿 Storage Metrics:');
        console.log(`  Read IOPS: ${metrics.storage.read_iops}`);
        console.log(`  Write IOPS: ${metrics.storage.write_iops}`);
        console.log(`  Throughput: ${metrics.storage.throughput_mbps} MB/s`);
      }
      
      if (metrics.network) {
        console.log('\n🌐 Network Metrics:');
        console.log(`  Ingress: ${metrics.network.ingress_mbps} MB/s`);
        console.log(`  Egress: ${metrics.network.egress_mbps} MB/s`);
        console.log(`  Packet Loss: ${metrics.network.packet_loss}%`);
        console.log(`  Latency: ${metrics.network.latency_ms} ms`);
      }
      
      if (metrics.migrations) {
        console.log('\n🚀 Migration Metrics:');
        console.log(`  Total Migrations: ${metrics.migrations.total}`);
        console.log(`  Success Rate: ${metrics.migrations.success_rate}%`);
        console.log(`  Average Duration: ${metrics.migrations.avg_duration_seconds}s`);
      }
      
      // Export metrics if requested
      if (shouldExport) {
        const filename = `metrics-${vmId || 'cluster'}-${period}-${Date.now()}.json`;
        require('fs').writeFileSync(filename, JSON.stringify(metrics, null, 2));
        console.log(`\n💾 Metrics exported to: ${filename}`);
      }
      
      // Check for anomalies
      if (metrics.anomalies && metrics.anomalies.length > 0) {
        console.log('\n⚠️ Detected Anomalies:');
        metrics.anomalies.forEach(anomaly => {
          console.log(`  - ${anomaly.timestamp}: ${anomaly.type} - ${anomaly.description}`);
        });
      }
      
      return {
        success: true,
        period: period,
        scope: vmId ? `VM ${vmId}` : 'Cluster',
        metrics: metrics
      };
      
    } catch (error) {
      console.error('❌ Failed to collect metrics:', error.message);
      return {
        error: error.message,
        suggestion: 'Check if Prometheus is running on port 9090'
      };
    }
  }
};