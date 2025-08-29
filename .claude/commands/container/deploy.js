#!/usr/bin/env node

/**
 * Container Deployment Command for NovaCron
 */

const { execSync } = require('child_process');

module.exports = {
  name: 'container:deploy',
  description: 'Deploy and manage containers',
  usage: '/container:deploy <image> [--name <name>] [--replicas <n>] [--cpu <cores>] [--memory <mb>]',
  
  async execute(args) {
    const [image, ...options] = args;
    
    if (!image) {
      return {
        error: 'Container image is required. Usage: /container:deploy <image> [options]'
      };
    }
    
    // Parse options
    const opts = {};
    for (let i = 0; i < options.length; i += 2) {
      if (options[i].startsWith('--')) {
        const key = options[i].substring(2);
        opts[key] = options[i + 1];
      }
    }
    
    const containerSpec = {
      image: image,
      name: opts.name || `container-${Date.now()}`,
      replicas: parseInt(opts.replicas) || 1,
      resources: {
        cpu: parseInt(opts.cpu) || 1,
        memory_mb: parseInt(opts.memory) || 512
      },
      driver: 'containerd'
    };
    
    console.log('🐳 Deploying container...');
    console.log(`  Image: ${containerSpec.image}`);
    console.log(`  Name: ${containerSpec.name}`);
    console.log(`  Replicas: ${containerSpec.replicas}`);
    console.log(`  Resources: ${containerSpec.resources.cpu} CPU, ${containerSpec.resources.memory_mb}MB RAM`);
    
    try {
      // Deploy container via API
      const response = execSync(
        `curl -X POST -H "Content-Type: application/json" ` +
        `-d '${JSON.stringify(containerSpec)}' ` +
        `http://localhost:8090/api/containers`,
        { encoding: 'utf8' }
      );
      
      const result = JSON.parse(response);
      
      if (result.container_id || result.deployment_id) {
        console.log('✅ Container deployment initiated successfully');
        console.log(`📦 Container ID: ${result.container_id || result.deployment_id}`);
        
        // Wait for container to be ready
        console.log('⏳ Waiting for container to be ready...');
        let attempts = 0;
        let ready = false;
        
        while (!ready && attempts < 30) {
          await new Promise(resolve => setTimeout(resolve, 2000));
          try {
            const statusResponse = execSync(
              `curl -s http://localhost:8090/api/containers/${result.container_id || result.deployment_id}/status`,
              { encoding: 'utf8' }
            );
            const status = JSON.parse(statusResponse);
            ready = status.state === 'running';
            attempts++;
          } catch (e) {
            attempts++;
          }
        }
        
        if (ready) {
          console.log('🎉 Container is running!');
        } else {
          console.log('⚠️ Container deployment timeout - check status manually');
        }
      }
      
      return {
        success: true,
        container_id: result.container_id || result.deployment_id,
        name: containerSpec.name,
        image: containerSpec.image
      };
      
    } catch (error) {
      console.error('❌ Container deployment failed:', error.message);
      return {
        error: error.message,
        suggestion: 'Check if containerd driver is enabled and Docker is running'
      };
    }
  }
};