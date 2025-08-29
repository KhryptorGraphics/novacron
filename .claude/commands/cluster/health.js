#!/usr/bin/env node

/**
 * Cluster Health Check Command for NovaCron
 */

const { execSync } = require('child_process');

module.exports = {
  name: 'cluster:health',
  description: 'Check cluster health and node status',
  usage: '/cluster:health [--detailed] [--node <node-id>]',
  
  async execute(args) {
    const detailed = args.includes('--detailed');
    const nodeIndex = args.indexOf('--node');
    const specificNode = nodeIndex !== -1 ? args[nodeIndex + 1] : null;
    
    console.log('🏥 Checking cluster health...\n');
    
    try {
      // Get cluster status
      const clusterStatus = execSync(
        'curl -s http://localhost:8090/api/cluster/status',
        { encoding: 'utf8' }
      );
      const cluster = JSON.parse(clusterStatus);
      
      console.log('📊 Cluster Overview:');
      console.log(`  Status: ${cluster.healthy ? '✅ Healthy' : '⚠️ Degraded'}`);
      console.log(`  Nodes: ${cluster.total_nodes} (${cluster.active_nodes} active)`);
      console.log(`  VMs: ${cluster.total_vms} running`);
      console.log(`  CPU Usage: ${cluster.cpu_usage}%`);
      console.log(`  Memory Usage: ${cluster.memory_usage}%`);
      console.log(`  Storage Usage: ${cluster.storage_usage}%`);
      
      if (detailed || specificNode) {
        console.log('\n📋 Node Details:');
        
        const nodesEndpoint = specificNode 
          ? `/api/nodes/${specificNode}`
          : '/api/nodes';
        
        const nodesStatus = execSync(
          `curl -s http://localhost:8090${nodesEndpoint}`,
          { encoding: 'utf8' }
        );
        const nodes = JSON.parse(nodesStatus);
        const nodeList = Array.isArray(nodes) ? nodes : [nodes];
        
        nodeList.forEach(node => {
          console.log(`\n  Node: ${node.id}`);
          console.log(`    Status: ${node.status}`);
          console.log(`    VMs: ${node.vm_count}`);
          console.log(`    CPU: ${node.cpu_cores} cores (${node.cpu_usage}% used)`);
          console.log(`    Memory: ${node.memory_gb}GB (${node.memory_usage}% used)`);
          console.log(`    Storage: ${node.storage_gb}GB (${node.storage_usage}% used)`);
          console.log(`    Uptime: ${node.uptime}`);
          
          if (node.alerts && node.alerts.length > 0) {
            console.log('    ⚠️ Alerts:');
            node.alerts.forEach(alert => {
              console.log(`      - ${alert.severity}: ${alert.message}`);
            });
          }
        });
      }
      
      // Check for any critical issues
      if (cluster.critical_alerts && cluster.critical_alerts.length > 0) {
        console.log('\n🚨 Critical Alerts:');
        cluster.critical_alerts.forEach(alert => {
          console.log(`  - ${alert.timestamp}: ${alert.message}`);
        });
      }
      
      return {
        success: true,
        healthy: cluster.healthy,
        nodes: cluster.total_nodes,
        active_nodes: cluster.active_nodes,
        total_vms: cluster.total_vms
      };
      
    } catch (error) {
      console.error('❌ Failed to check cluster health:', error.message);
      return {
        error: error.message,
        suggestion: 'Ensure the API server is running on port 8090'
      };
    }
  }
};