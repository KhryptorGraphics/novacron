#!/usr/bin/env node

/**
 * VM Lifecycle Management Command for NovaCron
 */

const { execSync } = require('child_process');

module.exports = {
  name: 'vm:manage',
  description: 'Manage VM lifecycle operations',
  usage: '/vm:manage <action> <vm-id> [options]',
  
  async execute(args) {
    const [action, vmId, ...options] = args;
    
    const validActions = ['start', 'stop', 'restart', 'pause', 'resume', 'snapshot', 'status'];
    
    if (!action || !validActions.includes(action)) {
      return {
        error: `Invalid action. Valid actions: ${validActions.join(', ')}`
      };
    }
    
    if (!vmId && action !== 'list') {
      return {
        error: 'VM ID is required for this action'
      };
    }
    
    console.log(`🖥️ Executing ${action} on VM ${vmId}`);
    
    try {
      let endpoint = '';
      let method = 'GET';
      let data = null;
      
      switch(action) {
        case 'start':
          endpoint = `/api/vms/${vmId}/start`;
          method = 'POST';
          break;
        case 'stop':
          endpoint = `/api/vms/${vmId}/stop`;
          method = 'POST';
          break;
        case 'restart':
          endpoint = `/api/vms/${vmId}/restart`;
          method = 'POST';
          break;
        case 'pause':
          endpoint = `/api/vms/${vmId}/pause`;
          method = 'POST';
          break;
        case 'resume':
          endpoint = `/api/vms/${vmId}/resume`;
          method = 'POST';
          break;
        case 'snapshot':
          endpoint = `/api/vms/${vmId}/snapshots`;
          method = 'POST';
          data = { name: options[0] || `snapshot-${Date.now()}` };
          break;
        case 'status':
          endpoint = `/api/vms/${vmId}`;
          break;
      }
      
      let command = `curl -s -X ${method}`;
      if (data) {
        command += ` -H "Content-Type: application/json" -d '${JSON.stringify(data)}'`;
      }
      command += ` http://localhost:8090${endpoint}`;
      
      const response = execSync(command, { encoding: 'utf8' });
      const result = JSON.parse(response);
      
      console.log(`✅ Action '${action}' completed successfully`);
      
      if (action === 'status') {
        console.log('\n📊 VM Status:');
        console.log(`  ID: ${result.id}`);
        console.log(`  Name: ${result.name}`);
        console.log(`  State: ${result.state}`);
        console.log(`  CPU: ${result.cpu_cores} cores`);
        console.log(`  Memory: ${result.memory_mb} MB`);
        console.log(`  Storage: ${result.storage_gb} GB`);
        console.log(`  Node: ${result.node}`);
      }
      
      return {
        success: true,
        action: action,
        vm_id: vmId,
        result: result
      };
      
    } catch (error) {
      console.error(`❌ Failed to ${action} VM:`, error.message);
      return {
        error: error.message
      };
    }
  }
};