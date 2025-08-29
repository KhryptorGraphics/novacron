#!/usr/bin/env node

/**
 * VM Migration Command for NovaCron
 * Handles cold, warm, and live migrations with WAN optimization
 */

const { execSync } = require('child_process');
const path = require('path');

module.exports = {
  name: 'vm:migrate',
  description: 'Migrate a VM between nodes with optimization',
  usage: '/vm:migrate <vm-id> <destination> [--type cold|warm|live] [--optimize]',
  
  async execute(args) {
    const [vmId, destination, ...options] = args;
    
    if (!vmId || !destination) {
      return {
        error: 'Usage: /vm:migrate <vm-id> <destination> [--type cold|warm|live]'
      };
    }
    
    const migrationType = options.includes('--live') ? 'live' : 
                         options.includes('--warm') ? 'warm' : 'cold';
    const optimize = options.includes('--optimize');
    
    console.log(`🚀 Initiating ${migrationType} migration of VM ${vmId} to ${destination}`);
    
    try {
      // Validate VM exists
      console.log('📋 Validating VM state...');
      execSync(`curl -s http://localhost:8090/api/vms/${vmId}`, { stdio: 'pipe' });
      
      // Prepare migration request
      const migrationData = {
        vm_id: vmId,
        destination: destination,
        type: migrationType,
        optimize_wan: optimize,
        compression: optimize ? 'zstd' : 'none',
        bandwidth_limit: optimize ? 100 : 0 // MB/s
      };
      
      console.log('🔄 Starting migration process...');
      const response = execSync(
        `curl -X POST -H "Content-Type: application/json" ` +
        `-d '${JSON.stringify(migrationData)}' ` +
        `http://localhost:8090/api/migrations`,
        { encoding: 'utf8' }
      );
      
      const result = JSON.parse(response);
      
      if (result.migration_id) {
        console.log(`✅ Migration initiated successfully`);
        console.log(`📊 Migration ID: ${result.migration_id}`);
        console.log(`📈 Track progress at: http://localhost:8092/migrations/${result.migration_id}`);
        
        // Monitor migration progress
        let status = 'in_progress';
        while (status === 'in_progress') {
          await new Promise(resolve => setTimeout(resolve, 2000));
          const statusResponse = execSync(
            `curl -s http://localhost:8090/api/migrations/${result.migration_id}/status`,
            { encoding: 'utf8' }
          );
          const statusData = JSON.parse(statusResponse);
          status = statusData.status;
          console.log(`⏳ Progress: ${statusData.progress}% - ${statusData.phase}`);
        }
        
        if (status === 'completed') {
          console.log('🎉 Migration completed successfully!');
        } else {
          console.log(`⚠️ Migration ended with status: ${status}`);
        }
      }
      
      return {
        success: true,
        migration_id: result.migration_id,
        type: migrationType,
        destination: destination
      };
      
    } catch (error) {
      console.error('❌ Migration failed:', error.message);
      return {
        error: error.message,
        suggestion: 'Check VM state and destination node availability'
      };
    }
  }
};