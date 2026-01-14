/**
 * NovaCron JavaScript SDK Client
 */

const axios = require('axios');
const WebSocket = require('ws');
const EventEmitter = require('events');

/**
 * NovaCron API Client
 */
class NovaCronClient extends EventEmitter {
  /**
   * Create a new NovaCron client
   * @param {Object} config - Configuration options
   * @param {string} config.baseURL - Base URL for the API
   * @param {string} [config.apiToken] - JWT token for authentication
   * @param {string} [config.username] - Username for basic auth
   * @param {string} [config.password] - Password for basic auth
   * @param {number} [config.timeout=30000] - Request timeout in milliseconds
   * @param {number} [config.maxRetries=3] - Maximum number of retries
   * @param {number} [config.retryDelay=1000] - Delay between retries in milliseconds
   */
  constructor(config) {
    super();
    
    if (!config.baseURL) {
      throw new Error('baseURL is required');
    }

    this.baseURL = config.baseURL.replace(/\/$/, '');
    this.apiToken = config.apiToken;
    this.username = config.username;
    this.password = config.password;
    this.timeout = config.timeout || 30000;
    this.maxRetries = config.maxRetries || 3;
    this.retryDelay = config.retryDelay || 1000;

    // Create axios instance
    this.http = axios.create({
      baseURL: this.baseURL,
      timeout: this.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    // Set up interceptors
    this._setupInterceptors();
  }

  /**
   * Set up axios interceptors
   * @private
   */
  _setupInterceptors() {
    // Request interceptor to add authentication
    this.http.interceptors.request.use((config) => {
      if (this.apiToken) {
        config.headers.Authorization = `Bearer ${this.apiToken}`;
      } else if (this.username && this.password) {
        config.auth = {
          username: this.username,
          password: this.password,
        };
      }
      return config;
    });

    // Response interceptor for error handling
    this.http.interceptors.response.use(
      (response) => response,
      async (error) => {
        const { config, response } = error;

        // Don't retry if we've exceeded max retries
        if (config._retryCount >= this.maxRetries) {
          return Promise.reject(error);
        }

        // Only retry on network errors or 5xx responses
        const shouldRetry = !response || response.status >= 500;
        if (!shouldRetry) {
          return Promise.reject(error);
        }

        // Increment retry count
        config._retryCount = (config._retryCount || 0) + 1;

        // Wait before retrying
        await new Promise(resolve => 
          setTimeout(resolve, this.retryDelay * config._retryCount)
        );

        return this.http.request(config);
      }
    );
  }

  /**
   * Set API token
   * @param {string} token - JWT token
   */
  setApiToken(token) {
    this.apiToken = token;
  }

  // VM Management Methods

  /**
   * Create a new VM
   * @param {Object} request - VM creation request
   * @returns {Promise<Object>} Created VM
   */
  async createVM(request) {
    const response = await this.http.post('/api/vms', request);
    return response.data;
  }

  /**
   * Get VM by ID
   * @param {string} vmId - VM identifier
   * @returns {Promise<Object>} VM object
   */
  async getVM(vmId) {
    const response = await this.http.get(`/api/vms/${vmId}`);
    return response.data;
  }

  /**
   * List VMs with optional filtering
   * @param {Object} [options] - Filter options
   * @param {string} [options.tenantId] - Filter by tenant ID
   * @param {string} [options.state] - Filter by VM state
   * @param {string} [options.nodeId] - Filter by node ID
   * @returns {Promise<Array>} List of VMs
   */
  async listVMs(options = {}) {
    const params = {};
    if (options.tenantId) params.tenant_id = options.tenantId;
    if (options.state) params.state = options.state;
    if (options.nodeId) params.node_id = options.nodeId;

    const response = await this.http.get('/api/vms', { params });
    return response.data;
  }

  /**
   * Update VM configuration
   * @param {string} vmId - VM identifier
   * @param {Object} updates - Update request
   * @returns {Promise<Object>} Updated VM
   */
  async updateVM(vmId, updates) {
    const response = await this.http.put(`/api/vms/${vmId}`, updates);
    return response.data;
  }

  /**
   * Delete VM
   * @param {string} vmId - VM identifier
   * @returns {Promise<void>}
   */
  async deleteVM(vmId) {
    await this.http.delete(`/api/vms/${vmId}`);
  }

  // VM Lifecycle Methods

  /**
   * Start VM
   * @param {string} vmId - VM identifier
   * @returns {Promise<void>}
   */
  async startVM(vmId) {
    await this.http.post(`/api/vms/${vmId}/start`);
  }

  /**
   * Stop VM
   * @param {string} vmId - VM identifier
   * @param {boolean} [force=false] - Force stop
   * @returns {Promise<void>}
   */
  async stopVM(vmId, force = false) {
    await this.http.post(`/api/vms/${vmId}/stop`, { force });
  }

  /**
   * Restart VM
   * @param {string} vmId - VM identifier
   * @returns {Promise<void>}
   */
  async restartVM(vmId) {
    await this.http.post(`/api/vms/${vmId}/restart`);
  }

  /**
   * Pause VM
   * @param {string} vmId - VM identifier
   * @returns {Promise<void>}
   */
  async pauseVM(vmId) {
    await this.http.post(`/api/vms/${vmId}/pause`);
  }

  /**
   * Resume paused VM
   * @param {string} vmId - VM identifier
   * @returns {Promise<void>}
   */
  async resumeVM(vmId) {
    await this.http.post(`/api/vms/${vmId}/resume`);
  }

  // Metrics and Monitoring

  /**
   * Get VM metrics
   * @param {string} vmId - VM identifier
   * @param {Object} [options] - Metrics options
   * @param {Date} [options.startTime] - Start time
   * @param {Date} [options.endTime] - End time
   * @returns {Promise<Object>} VM metrics
   */
  async getVMMetrics(vmId, options = {}) {
    const params = {};
    if (options.startTime) params.start = options.startTime.toISOString();
    if (options.endTime) params.end = options.endTime.toISOString();

    const response = await this.http.get(`/api/vms/${vmId}/metrics`, { params });
    return response.data;
  }

  /**
   * Get system metrics
   * @param {Object} [options] - Metrics options
   * @param {string} [options.nodeId] - Node ID filter
   * @param {Date} [options.startTime] - Start time
   * @param {Date} [options.endTime] - End time
   * @returns {Promise<Object>} System metrics
   */
  async getSystemMetrics(options = {}) {
    const params = {};
    if (options.nodeId) params.node_id = options.nodeId;
    if (options.startTime) params.start = options.startTime.toISOString();
    if (options.endTime) params.end = options.endTime.toISOString();

    const response = await this.http.get('/api/metrics/system', { params });
    return response.data;
  }

  // Migration Methods

  /**
   * Migrate VM to another node
   * @param {string} vmId - VM identifier
   * @param {Object} request - Migration request
   * @returns {Promise<Object>} Migration object
   */
  async migrateVM(vmId, request) {
    const response = await this.http.post(`/api/vms/${vmId}/migrate`, request);
    return response.data;
  }

  /**
   * Get migration status
   * @param {string} migrationId - Migration identifier
   * @returns {Promise<Object>} Migration object
   */
  async getMigration(migrationId) {
    const response = await this.http.get(`/api/migrations/${migrationId}`);
    return response.data;
  }

  /**
   * List migrations
   * @param {Object} [options] - Filter options
   * @param {string} [options.vmId] - Filter by VM ID
   * @param {string} [options.status] - Filter by status
   * @returns {Promise<Array>} List of migrations
   */
  async listMigrations(options = {}) {
    const params = {};
    if (options.vmId) params.vm_id = options.vmId;
    if (options.status) params.status = options.status;

    const response = await this.http.get('/api/migrations', { params });
    return response.data;
  }

  /**
   * Cancel ongoing migration
   * @param {string} migrationId - Migration identifier
   * @returns {Promise<void>}
   */
  async cancelMigration(migrationId) {
    await this.http.post(`/api/migrations/${migrationId}/cancel`);
  }

  // Template Methods

  /**
   * Create VM template
   * @param {Object} template - Template object
   * @returns {Promise<Object>} Created template
   */
  async createVMTemplate(template) {
    const response = await this.http.post('/api/templates', template);
    return response.data;
  }

  /**
   * Get VM template
   * @param {string} templateId - Template identifier
   * @returns {Promise<Object>} Template object
   */
  async getVMTemplate(templateId) {
    const response = await this.http.get(`/api/templates/${templateId}`);
    return response.data;
  }

  /**
   * List VM templates
   * @returns {Promise<Array>} List of templates
   */
  async listVMTemplates() {
    const response = await this.http.get('/api/templates');
    return response.data;
  }

  /**
   * Update VM template
   * @param {string} templateId - Template identifier
   * @param {Object} template - Template object
   * @returns {Promise<Object>} Updated template
   */
  async updateVMTemplate(templateId, template) {
    const response = await this.http.put(`/api/templates/${templateId}`, template);
    return response.data;
  }

  /**
   * Delete VM template
   * @param {string} templateId - Template identifier
   * @returns {Promise<void>}
   */
  async deleteVMTemplate(templateId) {
    await this.http.delete(`/api/templates/${templateId}`);
  }

  // Node Management

  /**
   * List cluster nodes
   * @returns {Promise<Array>} List of nodes
   */
  async listNodes() {
    const response = await this.http.get('/api/nodes');
    return response.data;
  }

  /**
   * Get node information
   * @param {string} nodeId - Node identifier
   * @returns {Promise<Object>} Node object
   */
  async getNode(nodeId) {
    const response = await this.http.get(`/api/nodes/${nodeId}`);
    return response.data;
  }

  /**
   * Get node metrics
   * @param {string} nodeId - Node identifier
   * @param {Object} [options] - Metrics options
   * @returns {Promise<Object>} Node metrics
   */
  async getNodeMetrics(nodeId, options = {}) {
    const params = {};
    if (options.startTime) params.start = options.startTime.toISOString();
    if (options.endTime) params.end = options.endTime.toISOString();

    const response = await this.http.get(`/api/nodes/${nodeId}/metrics`, { params });
    return response.data;
  }

  // WebSocket Methods

  /**
   * Stream VM events via WebSocket
   * @param {Object} [options] - Stream options
   * @param {string} [options.vmId] - Filter by VM ID
   * @returns {EventEmitter} Event emitter for streaming events
   */
  streamVMEvents(options = {}) {
    const wsUrl = this.baseURL.replace('http://', 'ws://').replace('https://', 'wss://');
    const url = new URL('/ws/events', wsUrl);
    
    if (options.vmId) {
      url.searchParams.set('vm_id', options.vmId);
    }

    const headers = {};
    if (this.apiToken) {
      headers.Authorization = `Bearer ${this.apiToken}`;
    }

    const ws = new WebSocket(url.toString(), { headers });
    const emitter = new EventEmitter();

    ws.on('open', () => {
      emitter.emit('connected');
    });

    ws.on('message', (data) => {
      try {
        const event = JSON.parse(data.toString());
        emitter.emit('event', event);
      } catch (error) {
        emitter.emit('error', error);
      }
    });

    ws.on('error', (error) => {
      emitter.emit('error', error);
    });

    ws.on('close', () => {
      emitter.emit('disconnected');
    });

    // Add close method to emitter
    emitter.close = () => ws.close();

    return emitter;
  }

  /**
   * Stream real-time metrics
   * @param {Object} [options] - Stream options
   * @param {string} [options.vmId] - Filter by VM ID
   * @param {number} [options.interval=5] - Update interval in seconds
   * @returns {EventEmitter} Event emitter for streaming metrics
   */
  streamMetrics(options = {}) {
    const wsUrl = this.baseURL.replace('http://', 'ws://').replace('https://', 'wss://');
    const url = new URL('/ws/metrics', wsUrl);
    
    url.searchParams.set('interval', (options.interval || 5).toString());
    if (options.vmId) {
      url.searchParams.set('vm_id', options.vmId);
    }

    const headers = {};
    if (this.apiToken) {
      headers.Authorization = `Bearer ${this.apiToken}`;
    }

    const ws = new WebSocket(url.toString(), { headers });
    const emitter = new EventEmitter();

    ws.on('open', () => {
      emitter.emit('connected');
    });

    ws.on('message', (data) => {
      try {
        const metrics = JSON.parse(data.toString());
        emitter.emit('metrics', metrics);
      } catch (error) {
        emitter.emit('error', error);
      }
    });

    ws.on('error', (error) => {
      emitter.emit('error', error);
    });

    ws.on('close', () => {
      emitter.emit('disconnected');
    });

    // Add close method to emitter
    emitter.close = () => ws.close();

    return emitter;
  }

  // Health and Status

  /**
   * Check API health
   * @returns {Promise<Object>} Health status
   */
  async healthCheck() {
    const response = await this.http.get('/health');
    return response.data;
  }

  /**
   * Get API version information
   * @returns {Promise<Object>} Version information
   */
  async getVersion() {
    const response = await this.http.get('/version');
    return response.data;
  }

  // Batch Operations

  /**
   * Start multiple VMs
   * @param {Array<string>} vmIds - List of VM identifiers
   * @returns {Promise<Object>} Results mapping VM ID to success status
   */
  async batchStartVMs(vmIds) {
    const results = {};
    const promises = vmIds.map(async (vmId) => {
      try {
        await this.startVM(vmId);
        results[vmId] = true;
      } catch (error) {
        results[vmId] = false;
      }
    });

    await Promise.all(promises);
    return results;
  }

  /**
   * Stop multiple VMs
   * @param {Array<string>} vmIds - List of VM identifiers
   * @param {boolean} [force=false] - Force stop
   * @returns {Promise<Object>} Results mapping VM ID to success status
   */
  async batchStopVMs(vmIds, force = false) {
    const results = {};
    const promises = vmIds.map(async (vmId) => {
      try {
        await this.stopVM(vmId, force);
        results[vmId] = true;
      } catch (error) {
        results[vmId] = false;
      }
    });

    await Promise.all(promises);
    return results;
  }

  // Authentication

  /**
   * Authenticate and get JWT token
   * @param {string} username - Username
   * @param {string} password - Password
   * @returns {Promise<string>} JWT token
   */
  async authenticate(username, password) {
    const response = await this.http.post('/api/auth/login', {
      username,
      password,
    });

    const token = response.data.token;
    if (token) {
      this.setApiToken(token);
      return token;
    } else {
      throw new Error('Failed to get authentication token');
    }
  }

  /**
   * Refresh JWT token
   * @returns {Promise<string>} New JWT token
   */
  async refreshToken() {
    const response = await this.http.post('/api/auth/refresh');
    
    const token = response.data.token;
    if (token) {
      this.setApiToken(token);
      return token;
    } else {
      throw new Error('Failed to refresh authentication token');
    }
  }
}

module.exports = NovaCronClient;