const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

class LoadTestCleanup {
  constructor() {
    this.environment = process.env.ENVIRONMENT || 'local';
    this.baseURL = this.getBaseURL();
    this.cleanupLog = [];
    this.dryRun = process.env.DRY_RUN === 'true';
  }

  getBaseURL() {
    const urls = {
      local: 'http://localhost:8080',
      staging: 'https://staging.novacron.com',
      production: 'https://api.novacron.com'
    };
    return urls[this.environment] || urls.local;
  }

  async performCleanup() {
    console.log(`Starting load test cleanup for environment: ${this.environment}`);
    console.log(`Base URL: ${this.baseURL}`);
    console.log(`Dry run mode: ${this.dryRun}`);

    try {
      // Authenticate as admin
      const token = await this.authenticate();
      if (!token) {
        throw new Error('Failed to authenticate for cleanup');
      }

      // Perform cleanup operations
      await Promise.all([
        this.cleanupTestVMs(token),
        this.cleanupTestVolumes(token),
        this.cleanupTestSnapshots(token),
        this.cleanupTestUsers(token),
        this.cleanupLogFiles(),
        this.cleanupReportFiles(),
        this.cleanupTempData()
      ]);

      // Generate cleanup report
      await this.generateCleanupReport();

      console.log('Cleanup completed successfully');

    } catch (error) {
      console.error('Cleanup failed:', error);
      process.exit(1);
    }
  }

  async authenticate() {
    const loginPayload = {
      username: process.env.ADMIN_USERNAME || 'admin',
      password: process.env.ADMIN_PASSWORD || 'admin123'
    };

    try {
      const response = await this.makeRequest('POST', '/api/auth/login', loginPayload);
      
      if (response.status === 200) {
        const data = JSON.parse(response.data);
        console.log('Authentication successful');
        return data.token;
      } else {
        console.error(`Authentication failed: ${response.status}`);
        return null;
      }
    } catch (error) {
      console.error('Authentication error:', error);
      return null;
    }
  }

  async cleanupTestVMs(token) {
    console.log('Cleaning up test VMs...');
    
    try {
      // Get all VMs
      const vmsResponse = await this.makeRequest('GET', '/api/vms', null, token);
      
      if (vmsResponse.status !== 200) {
        console.error(`Failed to fetch VMs: ${vmsResponse.status}`);
        return;
      }

      const vms = JSON.parse(vmsResponse.data);
      const testVMs = vms.filter(vm => this.isTestResource(vm));

      console.log(`Found ${testVMs.length} test VMs to clean up`);

      for (const vm of testVMs) {
        await this.cleanupSingleVM(token, vm);
      }

      this.cleanupLog.push({
        category: 'VMs',
        action: 'cleanup',
        count: testVMs.length,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('VM cleanup error:', error);
    }
  }

  async cleanupSingleVM(token, vm) {
    console.log(`Cleaning up VM: ${vm.name} (${vm.id})`);

    try {
      // Stop VM if running
      if (vm.state === 'running') {
        if (!this.dryRun) {
          await this.makeRequest('POST', `/api/vms/${vm.id}/stop`, null, token);
          console.log(`  Stopped VM: ${vm.name}`);
          
          // Wait for VM to stop
          await this.sleep(2000);
        } else {
          console.log(`  [DRY RUN] Would stop VM: ${vm.name}`);
        }
      }

      // Delete VM
      if (!this.dryRun) {
        const deleteResponse = await this.makeRequest('DELETE', `/api/vms/${vm.id}`, null, token);
        if (deleteResponse.status === 200) {
          console.log(`  Deleted VM: ${vm.name}`);
        } else {
          console.error(`  Failed to delete VM ${vm.name}: ${deleteResponse.status}`);
        }
      } else {
        console.log(`  [DRY RUN] Would delete VM: ${vm.name}`);
      }

    } catch (error) {
      console.error(`Error cleaning up VM ${vm.name}:`, error);
    }
  }

  async cleanupTestVolumes(token) {
    console.log('Cleaning up test storage volumes...');
    
    try {
      const volumesResponse = await this.makeRequest('GET', '/api/storage/volumes', null, token);
      
      if (volumesResponse.status !== 200) {
        console.error(`Failed to fetch volumes: ${volumesResponse.status}`);
        return;
      }

      const volumes = JSON.parse(volumesResponse.data);
      const testVolumes = volumes.filter(volume => this.isTestResource(volume));

      console.log(`Found ${testVolumes.length} test volumes to clean up`);

      for (const volume of testVolumes) {
        if (!this.dryRun) {
          const deleteResponse = await this.makeRequest('DELETE', `/api/storage/volumes/${volume.id}`, null, token);
          if (deleteResponse.status === 200) {
            console.log(`  Deleted volume: ${volume.name}`);
          } else {
            console.error(`  Failed to delete volume ${volume.name}: ${deleteResponse.status}`);
          }
        } else {
          console.log(`  [DRY RUN] Would delete volume: ${volume.name}`);
        }
      }

      this.cleanupLog.push({
        category: 'Volumes',
        action: 'cleanup',
        count: testVolumes.length,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Volume cleanup error:', error);
    }
  }

  async cleanupTestSnapshots(token) {
    console.log('Cleaning up test snapshots...');
    
    try {
      // Get snapshots through VM API
      const vmsResponse = await this.makeRequest('GET', '/api/vms', null, token);
      
      if (vmsResponse.status === 200) {
        const vms = JSON.parse(vmsResponse.data);
        let snapshotCount = 0;

        for (const vm of vms) {
          if (this.isTestResource(vm)) {
            // Get VM snapshots (placeholder - actual endpoint may vary)
            try {
              const snapshotsResponse = await this.makeRequest('GET', `/api/vms/${vm.id}/snapshots`, null, token);
              if (snapshotsResponse.status === 200) {
                const snapshots = JSON.parse(snapshotsResponse.data);
                
                for (const snapshot of snapshots) {
                  if (!this.dryRun) {
                    await this.makeRequest('DELETE', `/api/vms/${vm.id}/snapshots/${snapshot.id}`, null, token);
                    console.log(`  Deleted snapshot: ${snapshot.name}`);
                  } else {
                    console.log(`  [DRY RUN] Would delete snapshot: ${snapshot.name}`);
                  }
                  snapshotCount++;
                }
              }
            } catch (error) {
              console.warn(`Could not fetch snapshots for VM ${vm.id}: ${error.message}`);
            }
          }
        }

        this.cleanupLog.push({
          category: 'Snapshots',
          action: 'cleanup',
          count: snapshotCount,
          timestamp: new Date().toISOString()
        });
      }

    } catch (error) {
      console.error('Snapshot cleanup error:', error);
    }
  }

  async cleanupTestUsers(token) {
    console.log('Cleaning up test users...');
    
    try {
      // Only clean up test users, not system users
      const usersResponse = await this.makeRequest('GET', '/api/admin/users', null, token);
      
      if (usersResponse.status === 200) {
        const users = JSON.parse(usersResponse.data);
        const testUsers = users.filter(user => 
          user.username && (
            user.username.includes('test-user') ||
            user.username.includes('load-test') ||
            user.username.includes('benchmark')
          )
        );

        console.log(`Found ${testUsers.length} test users to clean up`);

        for (const user of testUsers) {
          if (!this.dryRun) {
            const deleteResponse = await this.makeRequest('DELETE', `/api/admin/users/${user.id}`, null, token);
            if (deleteResponse.status === 200) {
              console.log(`  Deleted test user: ${user.username}`);
            }
          } else {
            console.log(`  [DRY RUN] Would delete test user: ${user.username}`);
          }
        }

        this.cleanupLog.push({
          category: 'Users',
          action: 'cleanup',
          count: testUsers.length,
          timestamp: new Date().toISOString()
        });
      }

    } catch (error) {
      console.error('User cleanup error:', error);
    }
  }

  async cleanupLogFiles() {
    console.log('Cleaning up test log files...');
    
    const logDirs = [
      path.join(__dirname, '../reports'),
      '/tmp/novacron-test-logs',
      './test-logs'
    ];

    let fileCount = 0;

    for (const logDir of logDirs) {
      if (fs.existsSync(logDir)) {
        const files = fs.readdirSync(logDir);
        
        for (const file of files) {
          const filePath = path.join(logDir, file);
          const stats = fs.statSync(filePath);
          
          // Clean up files older than 24 hours or with test prefixes
          const isOldFile = Date.now() - stats.mtime.getTime() > 24 * 60 * 60 * 1000;
          const isTestFile = file.includes('test') || file.includes('load') || file.includes('bench');
          
          if (isOldFile || isTestFile) {
            if (!this.dryRun) {
              fs.unlinkSync(filePath);
              console.log(`  Deleted log file: ${file}`);
            } else {
              console.log(`  [DRY RUN] Would delete log file: ${file}`);
            }
            fileCount++;
          }
        }
      }
    }

    this.cleanupLog.push({
      category: 'LogFiles',
      action: 'cleanup',
      count: fileCount,
      timestamp: new Date().toISOString()
    });
  }

  async cleanupReportFiles() {
    console.log('Cleaning up old report files...');
    
    const reportsDir = path.join(__dirname, '../reports');
    let reportCount = 0;

    if (fs.existsSync(reportsDir)) {
      const files = fs.readdirSync(reportsDir);
      const cutoffTime = Date.now() - (7 * 24 * 60 * 60 * 1000); // 7 days

      for (const file of files) {
        const filePath = path.join(reportsDir, file);
        const stats = fs.statSync(filePath);
        
        // Keep recent reports, clean up old ones
        if (stats.mtime.getTime() < cutoffTime && 
            (file.includes('load-test-report') || file.includes('monitoring-report'))) {
          
          if (!this.dryRun) {
            fs.unlinkSync(filePath);
            console.log(`  Deleted old report: ${file}`);
          } else {
            console.log(`  [DRY RUN] Would delete old report: ${file}`);
          }
          reportCount++;
        }
      }
    }

    this.cleanupLog.push({
      category: 'Reports',
      action: 'cleanup',
      count: reportCount,
      timestamp: new Date().toISOString()
    });
  }

  async cleanupTempData() {
    console.log('Cleaning up temporary test data...');
    
    const tempDirs = [
      '/tmp/k6-*',
      '/tmp/novacron-*',
      './temp-*'
    ];

    let tempFileCount = 0;

    tempDirs.forEach(pattern => {
      try {
        const { execSync } = require('child_process');
        const files = execSync(`ls -d ${pattern} 2>/dev/null || true`, { encoding: 'utf8' });
        
        files.split('\n').forEach(file => {
          if (file.trim()) {
            if (!this.dryRun) {
              execSync(`rm -rf "${file}"`);
              console.log(`  Deleted temp data: ${file}`);
            } else {
              console.log(`  [DRY RUN] Would delete temp data: ${file}`);
            }
            tempFileCount++;
          }
        });
      } catch (error) {
        // Ignore errors for temp file cleanup
      }
    });

    this.cleanupLog.push({
      category: 'TempData',
      action: 'cleanup',
      count: tempFileCount,
      timestamp: new Date().toISOString()
    });
  }

  isTestResource(resource) {
    if (!resource || !resource.name) return false;
    
    const testPatterns = [
      'test-',
      'load-test',
      'benchmark',
      'stress-test',
      'perf-test',
      'db-test',
      'fed-migration',
      'sync-test',
      'txn-test'
    ];

    return testPatterns.some(pattern => 
      resource.name.toLowerCase().includes(pattern) ||
      (resource.tags && Object.values(resource.tags).some(tag => 
        tag.toLowerCase().includes(pattern) || tag.includes('test')
      ))
    );
  }

  async makeRequest(method, endpoint, data = null, token = null) {
    return new Promise((resolve) => {
      const url = new URL(endpoint, this.baseURL);
      const options = {
        method: method,
        headers: {
          'Content-Type': 'application/json',
          'User-Agent': 'NovaCron-LoadTest-Cleanup/1.0'
        }
      };

      if (token) {
        options.headers['Authorization'] = `Bearer ${token}`;
      }

      const client = url.protocol === 'https:' ? https : http;
      
      const req = client.request(url, options, (res) => {
        let responseData = '';
        
        res.on('data', chunk => responseData += chunk);
        res.on('end', () => {
          resolve({
            status: res.statusCode,
            data: responseData,
            headers: res.headers
          });
        });
      });

      req.on('error', (error) => {
        resolve({
          status: 0,
          data: null,
          error: error.message
        });
      });

      req.setTimeout(30000, () => {
        req.destroy();
        resolve({
          status: 0,
          data: null,
          error: 'Request timeout'
        });
      });

      if (data) {
        req.write(JSON.stringify(data));
      }
      
      req.end();
    });
  }

  async generateCleanupReport() {
    const report = {
      cleanup: {
        timestamp: new Date().toISOString(),
        environment: this.environment,
        dryRun: this.dryRun,
        summary: this.generateCleanupSummary(),
        details: this.cleanupLog
      }
    };

    const reportPath = path.join(__dirname, '../reports', 
      `cleanup-report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    
    // Ensure reports directory exists
    const reportsDir = path.dirname(reportPath);
    if (!fs.existsSync(reportsDir)) {
      fs.mkdirSync(reportsDir, { recursive: true });
    }

    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    console.log(`Cleanup report generated: ${reportPath}`);

    // Also generate a summary
    this.generateCleanupSummary();
  }

  generateCleanupSummary() {
    const summary = {
      totalOperations: this.cleanupLog.length,
      resourcesCleaned: {},
      duration: 'completed',
      status: this.dryRun ? 'dry-run' : 'executed'
    };

    this.cleanupLog.forEach(entry => {
      summary.resourcesCleaned[entry.category] = entry.count;
    });

    console.log('\n=== Cleanup Summary ===');
    console.log(`Mode: ${this.dryRun ? 'DRY RUN' : 'EXECUTED'}`);
    console.log(`Environment: ${this.environment}`);
    console.log('Resources cleaned:');
    
    Object.entries(summary.resourcesCleaned).forEach(([category, count]) => {
      console.log(`  ${category}: ${count}`);
    });
    
    console.log('=== End Cleanup Summary ===\n');

    return summary;
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// CLI execution
if (require.main === module) {
  const cleanup = new LoadTestCleanup();
  
  // Parse command line arguments
  const args = process.argv.slice(2);
  if (args.includes('--dry-run')) {
    process.env.DRY_RUN = 'true';
  }

  if (args.includes('--help')) {
    console.log(`
NovaCron Load Test Cleanup Tool

Usage: node cleanup-test-data.js [options]

Options:
  --dry-run     Show what would be cleaned up without actually deleting
  --help        Show this help message

Environment Variables:
  ENVIRONMENT          Target environment (local, staging, production)
  ADMIN_USERNAME       Admin username for authentication
  ADMIN_PASSWORD       Admin password for authentication
  DRY_RUN             Set to 'true' for dry run mode

Examples:
  node cleanup-test-data.js --dry-run
  ENVIRONMENT=staging node cleanup-test-data.js
  DRY_RUN=true node cleanup-test-data.js
`);
    process.exit(0);
  }

  cleanup.performCleanup()
    .then(() => {
      console.log('Cleanup script completed successfully');
      process.exit(0);
    })
    .catch((error) => {
      console.error('Cleanup script failed:', error);
      process.exit(1);
    });
}

module.exports = LoadTestCleanup;