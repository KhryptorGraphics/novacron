/**
 * Test Environment Manager
 * 
 * Manages the setup and teardown of the complete test environment
 * including services, databases, test data, and monitoring.
 */

const { spawn, exec } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');
const axios = require('axios');
const { Pool } = require('pg');
const Redis = require('redis');

const execAsync = promisify(exec);

class TestEnvironment {
  constructor(config = {}) {
    this.config = {
      // Service URLs
      apiUrl: config.apiUrl || process.env.NOVACRON_API_URL || 'http://localhost:8090',
      frontendUrl: config.frontendUrl || process.env.NOVACRON_UI_URL || 'http://localhost:8092',
      
      // Database configuration
      dbUrl: config.dbUrl || process.env.DB_URL || 'postgresql://postgres:postgres@localhost:5432/novacron_test',
      redisUrl: config.redisUrl || process.env.REDIS_URL || 'redis://localhost:6379',
      
      // Test configuration
      setupTimeout: config.setupTimeout || 120000, // 2 minutes
      serviceStartTimeout: config.serviceStartTimeout || 60000, // 1 minute
      cleanupTimeout: config.cleanupTimeout || 30000, // 30 seconds
      
      // Docker configuration
      useDocker: config.useDocker !== undefined ? config.useDocker : process.env.NOVACRON_USE_DOCKER === 'true',
      dockerComposeFile: config.dockerComposeFile || 'docker-compose.test.yml',
      
      ...config
    };
    
    this.services = new Map();
    this.dbPool = null;
    this.redisClient = null;
    this.isSetup = false;
    this.testDataIds = new Set();
  }

  /**
   * Setup the complete test environment
   */
  async setup() {
    console.log('🚀 Setting up test environment...');
    
    try {
      if (this.config.useDocker) {
        await this.setupDockerEnvironment();
      } else {
        await this.setupLocalEnvironment();
      }
      
      await this.initializeDatabaseConnections();
      await this.waitForServicesToBeReady();
      await this.setupTestDatabase();
      
      this.isSetup = true;
      console.log('✅ Test environment setup complete');
      
    } catch (error) {
      console.error('❌ Failed to setup test environment:', error);
      await this.cleanup();
      throw error;
    }
  }

  /**
   * Setup Docker-based test environment
   */
  async setupDockerEnvironment() {
    console.log('🐳 Setting up Docker test environment...');
    
    const dockerComposePath = path.join(__dirname, '../../..', this.config.dockerComposeFile);
    
    try {
      // Check if docker-compose file exists
      await fs.access(dockerComposePath);
    } catch (error) {
      throw new Error(`Docker compose file not found: ${dockerComposePath}`);
    }
    
    // Stop any existing containers
    await this.executeCommand(`docker-compose -f ${dockerComposePath} down -v`);
    
    // Start services
    const composeCommand = `docker-compose -f ${dockerComposePath} up -d`;
    console.log(`Executing: ${composeCommand}`);
    
    await this.executeCommand(composeCommand);
    
    // Store service information
    this.services.set('docker-compose', {
      type: 'docker-compose',
      configFile: dockerComposePath,
      started: true
    });
    
    console.log('✅ Docker environment started');
  }

  /**
   * Setup local test environment
   */
  async setupLocalEnvironment() {
    console.log('🏠 Setting up local test environment...');
    
    const services = [
      {
        name: 'api-server',
        command: 'npm run start:api',
        cwd: path.join(__dirname, '../../..'),
        port: 8090,
        healthCheck: '/api/v1/health'
      },
      {
        name: 'frontend',
        command: 'npm run start:frontend',
        cwd: path.join(__dirname, '../../..'),
        port: 8092,
        healthCheck: '/'
      }
    ];
    
    for (const service of services) {
      await this.startLocalService(service);
    }
  }

  /**
   * Start a local service
   */
  async startLocalService(serviceConfig) {
    console.log(`Starting ${serviceConfig.name}...`);
    
    const child = spawn('bash', ['-c', serviceConfig.command], {
      cwd: serviceConfig.cwd,
      stdio: ['ignore', 'pipe', 'pipe'],
      env: {
        ...process.env,
        NODE_ENV: 'test',
        PORT: serviceConfig.port
      }
    });
    
    this.services.set(serviceConfig.name, {
      type: 'local',
      process: child,
      config: serviceConfig,
      started: true
    });
    
    // Handle process output
    child.stdout.on('data', (data) => {
      if (process.env.NOVACRON_TEST_VERBOSE === 'true') {
        console.log(`[${serviceConfig.name}] ${data.toString()}`);
      }
    });
    
    child.stderr.on('data', (data) => {
      console.error(`[${serviceConfig.name}] ${data.toString()}`);
    });
    
    child.on('error', (error) => {
      console.error(`[${serviceConfig.name}] Process error:`, error);
    });
    
    child.on('exit', (code, signal) => {
      if (code !== 0 && code !== null) {
        console.error(`[${serviceConfig.name}] Exited with code ${code}, signal ${signal}`);
      }
    });
  }

  /**
   * Initialize database connections
   */
  async initializeDatabaseConnections() {
    console.log('🗄️ Initializing database connections...');
    
    // PostgreSQL connection
    this.dbPool = new Pool({
      connectionString: this.config.dbUrl,
      max: 10,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 10000
    });
    
    // Redis connection
    this.redisClient = Redis.createClient({
      url: this.config.redisUrl,
      socket: {
        connectTimeout: 10000
      }
    });
    
    await this.redisClient.connect();
    
    console.log('✅ Database connections initialized');
  }

  /**
   * Wait for all services to be ready
   */
  async waitForServicesToBeReady() {
    console.log('⏳ Waiting for services to be ready...');
    
    const services = [
      { name: 'api-server', url: this.config.apiUrl + '/api/v1/health' },
      { name: 'frontend', url: this.config.frontendUrl },
      { name: 'database', check: () => this.checkDatabase() },
      { name: 'redis', check: () => this.checkRedis() }
    ];
    
    const startTime = Date.now();
    const timeout = this.config.setupTimeout;
    
    for (const service of services) {
      let isReady = false;
      let attempts = 0;
      const maxAttempts = Math.floor(timeout / 5000); // 5-second intervals
      
      while (!isReady && attempts < maxAttempts) {
        try {
          if (service.url) {
            const response = await axios.get(service.url, { timeout: 5000 });
            isReady = response.status === 200;
          } else if (service.check) {
            isReady = await service.check();
          }
          
          if (isReady) {
            console.log(`✅ ${service.name} is ready`);
          } else {
            attempts++;
            await new Promise(resolve => setTimeout(resolve, 5000));
          }
        } catch (error) {
          attempts++;
          if (attempts >= maxAttempts) {
            throw new Error(`Service ${service.name} failed to become ready: ${error.message}`);
          }
          await new Promise(resolve => setTimeout(resolve, 5000));
        }
      }
      
      if (!isReady) {
        throw new Error(`Service ${service.name} failed to become ready within timeout`);
      }
    }
    
    const readyTime = Date.now() - startTime;
    console.log(`✅ All services ready in ${readyTime}ms`);
  }

  /**
   * Wait for specific services to be ready
   */
  async waitForServices(serviceNames, timeout = 60000) {
    const serviceMap = {
      'api-server': { url: this.config.apiUrl + '/api/v1/health' },
      'frontend': { url: this.config.frontendUrl },
      'database': { check: () => this.checkDatabase() },
      'redis': { check: () => this.checkRedis() },
      'jupyter-hub': { url: this.config.apiUrl + '/api/v1/jupyter/health' },
      'mle-star-service': { url: this.config.apiUrl + '/api/v1/mle-star/health' },
      'model-registry': { url: this.config.apiUrl + '/api/v1/models/health' }
    };
    
    for (const serviceName of serviceNames) {
      const service = serviceMap[serviceName];
      if (!service) {
        throw new Error(`Unknown service: ${serviceName}`);
      }
      
      await this.waitForSingleService(serviceName, service, timeout);
    }
  }

  /**
   * Wait for a single service to be ready
   */
  async waitForSingleService(name, service, timeout) {
    console.log(`⏳ Waiting for ${name} to be ready...`);
    
    const startTime = Date.now();
    let isReady = false;
    
    while (!isReady && (Date.now() - startTime) < timeout) {
      try {
        if (service.url) {
          const response = await axios.get(service.url, { timeout: 5000 });
          isReady = response.status === 200;
        } else if (service.check) {
          isReady = await service.check();
        }
        
        if (!isReady) {
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      } catch (error) {
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    if (!isReady) {
      throw new Error(`Service ${name} failed to become ready within ${timeout}ms`);
    }
    
    console.log(`✅ ${name} is ready`);
  }

  /**
   * Check database connectivity
   */
  async checkDatabase() {
    try {
      const result = await this.dbPool.query('SELECT 1');
      return result.rows.length > 0;
    } catch (error) {
      return false;
    }
  }

  /**
   * Check Redis connectivity
   */
  async checkRedis() {
    try {
      await this.redisClient.ping();
      return true;
    } catch (error) {
      return false;
    }
  }

  /**
   * Setup test database schema and data
   */
  async setupTestDatabase() {
    console.log('🗄️ Setting up test database...');
    
    try {
      // Run database migrations
      await this.executeCommand('npm run db:migrate:test');
      
      // Clear existing test data
      await this.cleanupTestData();
      
      console.log('✅ Test database setup complete');
    } catch (error) {
      console.warn('⚠️ Database setup warning:', error.message);
    }
  }

  /**
   * Wait for VM to reach specific state
   */
  async waitForVMState(vmId, expectedState, timeout = 60000) {
    const startTime = Date.now();
    
    while ((Date.now() - startTime) < timeout) {
      try {
        const response = await axios.get(`${this.config.apiUrl}/api/v1/vms/${vmId}`);
        const currentState = response.data.state;
        
        if (currentState === expectedState) {
          return true;
        }
        
        await new Promise(resolve => setTimeout(resolve, 2000));
      } catch (error) {
        if (error.response && error.response.status === 404 && expectedState === 'deleted') {
          return true;
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    throw new Error(`VM ${vmId} failed to reach state ${expectedState} within ${timeout}ms`);
  }

  /**
   * Clean up test data
   */
  async cleanupTestData() {
    if (!this.dbPool) return;
    
    try {
      // Clean up VMs created during tests
      await this.dbPool.query(`
        DELETE FROM vms 
        WHERE name LIKE '%test%' 
        OR metadata->>'test-type' IS NOT NULL
      `);
      
      // Clean up test users
      await this.dbPool.query(`
        DELETE FROM users 
        WHERE username LIKE '%test%'
        OR email LIKE '%test%'
      `);
      
      // Clean up test projects
      await this.dbPool.query(`
        DELETE FROM mle_projects 
        WHERE name LIKE '%test%'
      `);
      
      // Clear Redis test data
      if (this.redisClient) {
        const keys = await this.redisClient.keys('test:*');
        if (keys.length > 0) {
          await this.redisClient.del(keys);
        }
      }
      
    } catch (error) {
      console.warn('⚠️ Error cleaning up test data:', error.message);
    }
  }

  /**
   * Track test data for cleanup
   */
  trackTestData(type, id) {
    this.testDataIds.add(`${type}:${id}`);
  }

  /**
   * Get system metrics
   */
  async getSystemMetrics() {
    try {
      const response = await axios.get(`${this.config.apiUrl}/api/v1/metrics`);
      return response.data;
    } catch (error) {
      console.warn('Failed to get system metrics:', error.message);
      return null;
    }
  }

  /**
   * Execute shell command
   */
  async executeCommand(command) {
    try {
      const { stdout, stderr } = await execAsync(command);
      if (stderr && !stderr.includes('Warning')) {
        console.warn('Command stderr:', stderr);
      }
      return stdout;
    } catch (error) {
      console.error(`Command failed: ${command}`);
      console.error('Error:', error.message);
      throw error;
    }
  }

  /**
   * Cleanup the test environment
   */
  async cleanup() {
    if (!this.isSetup) return;
    
    console.log('🧹 Cleaning up test environment...');
    
    try {
      // Cleanup test data
      await this.cleanupTestData();
      
      // Close database connections
      if (this.dbPool) {
        await this.dbPool.end();
        this.dbPool = null;
      }
      
      if (this.redisClient) {
        await this.redisClient.quit();
        this.redisClient = null;
      }
      
      // Stop services
      for (const [serviceName, service] of this.services) {
        await this.stopService(serviceName, service);
      }
      
      this.services.clear();
      this.isSetup = false;
      
      console.log('✅ Test environment cleanup complete');
    } catch (error) {
      console.error('❌ Error during cleanup:', error);
    }
  }

  /**
   * Stop a service
   */
  async stopService(serviceName, service) {
    try {
      if (service.type === 'docker-compose') {
        await this.executeCommand(`docker-compose -f ${service.configFile} down -v`);
      } else if (service.type === 'local' && service.process) {
        service.process.kill('SIGTERM');
        
        // Wait for graceful shutdown
        await new Promise((resolve) => {
          const timeout = setTimeout(() => {
            service.process.kill('SIGKILL');
            resolve();
          }, 5000);
          
          service.process.on('exit', () => {
            clearTimeout(timeout);
            resolve();
          });
        });
      }
      
      console.log(`✅ Stopped ${serviceName}`);
    } catch (error) {
      console.warn(`⚠️ Error stopping ${serviceName}:`, error.message);
    }
  }

  /**
   * Get database pool for direct queries
   */
  getDbPool() {
    return this.dbPool;
  }

  /**
   * Get Redis client for direct operations
   */
  getRedisClient() {
    return this.redisClient;
  }

  /**
   * Check if environment is ready
   */
  isReady() {
    return this.isSetup;
  }
}

module.exports = TestEnvironment;