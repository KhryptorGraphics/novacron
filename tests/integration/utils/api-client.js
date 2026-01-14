/**
 * API Client for Integration Tests
 * 
 * Enhanced HTTP client for testing NovaCron APIs with features like:
 * - Request/response timing
 * - Automatic retries
 * - Authentication handling
 * - Request/response logging
 * - Error handling and reporting
 */

const axios = require('axios');
const { performance } = require('perf_hooks');

class APIClient {
  constructor(config = {}) {
    this.config = {
      baseURL: config.baseURL || 'http://localhost:8090',
      timeout: config.timeout || 30000,
      retries: config.retries || 3,
      retryDelay: config.retryDelay || 1000,
      logRequests: config.logRequests !== undefined ? config.logRequests : process.env.NOVACRON_TEST_DEBUG === 'true',
      ...config
    };
    
    this.client = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'NovaCron-IntegrationTests/1.0',
        ...config.headers
      }
    });
    
    this.setupInterceptors();
    this.requestCount = 0;
    this.responseMetrics = [];
  }

  /**
   * Setup request and response interceptors
   */
  setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        config.requestStart = performance.now();
        config.requestId = ++this.requestCount;
        
        if (this.config.logRequests) {
          console.log(`🔄 [${config.requestId}] ${config.method.toUpperCase()} ${config.url}`);
          if (config.data) {
            console.log(`📤 [${config.requestId}] Request data:`, JSON.stringify(config.data, null, 2));
          }
        }
        
        return config;
      },
      (error) => {
        console.error('❌ Request interceptor error:', error);
        return Promise.reject(error);
      }
    );
    
    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        const responseTime = performance.now() - response.config.requestStart;
        response.responseTime = responseTime;
        
        this.responseMetrics.push({
          requestId: response.config.requestId,
          method: response.config.method.toUpperCase(),
          url: response.config.url,
          status: response.status,
          responseTime: responseTime,
          timestamp: new Date().toISOString()
        });
        
        if (this.config.logRequests) {
          console.log(`✅ [${response.config.requestId}] ${response.status} (${responseTime.toFixed(2)}ms)`);
          if (response.data && this.config.logResponses) {
            console.log(`📥 [${response.config.requestId}] Response data:`, JSON.stringify(response.data, null, 2));
          }
        }
        
        return response;
      },
      (error) => {
        if (error.config) {
          const responseTime = performance.now() - error.config.requestStart;
          
          this.responseMetrics.push({
            requestId: error.config.requestId,
            method: error.config.method.toUpperCase(),
            url: error.config.url,
            status: error.response?.status || 0,
            responseTime: responseTime,
            error: error.message,
            timestamp: new Date().toISOString()
          });
          
          if (this.config.logRequests) {
            console.error(`❌ [${error.config.requestId}] ${error.response?.status || 'Network Error'} (${responseTime.toFixed(2)}ms): ${error.message}`);
          }
        }
        
        return Promise.reject(error);
      }
    );
  }

  /**
   * Make HTTP GET request with retry logic
   */
  async get(url, config = {}) {
    return this.requestWithRetry('get', url, undefined, config);
  }

  /**
   * Make HTTP POST request with retry logic
   */
  async post(url, data, config = {}) {
    return this.requestWithRetry('post', url, data, config);
  }

  /**
   * Make HTTP PUT request with retry logic
   */
  async put(url, data, config = {}) {
    return this.requestWithRetry('put', url, data, config);
  }

  /**
   * Make HTTP DELETE request with retry logic
   */
  async delete(url, config = {}) {
    return this.requestWithRetry('delete', url, undefined, config);
  }

  /**
   * Make HTTP PATCH request with retry logic
   */
  async patch(url, data, config = {}) {
    return this.requestWithRetry('patch', url, data, config);
  }

  /**
   * Make request with automatic retry logic
   */
  async requestWithRetry(method, url, data, config = {}) {
    const maxRetries = config.retries !== undefined ? config.retries : this.config.retries;
    const retryDelay = config.retryDelay !== undefined ? config.retryDelay : this.config.retryDelay;
    
    let lastError;
    
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      try {
        const args = [url];
        if (data !== undefined) {
          args.push(data);
        }
        if (Object.keys(config).length > 0) {
          args.push(config);
        }
        
        return await this.client[method](...args);
      } catch (error) {
        lastError = error;
        
        // Don't retry on client errors (4xx) unless specifically configured
        if (error.response && error.response.status >= 400 && error.response.status < 500) {
          if (!config.retryOn4xx) {
            throw error;
          }
        }
        
        // Don't retry on the last attempt
        if (attempt === maxRetries) {
          break;
        }
        
        const delay = retryDelay * Math.pow(2, attempt); // Exponential backoff
        
        if (this.config.logRequests) {
          console.warn(`⚠️ Request failed (attempt ${attempt + 1}/${maxRetries + 1}), retrying in ${delay}ms...`);
        }
        
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    
    throw lastError;
  }

  /**
   * Set authentication token
   */
  setAuthToken(token) {
    this.client.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  }

  /**
   * Remove authentication token
   */
  clearAuthToken() {
    delete this.client.defaults.headers.common['Authorization'];
  }

  /**
   * Set custom headers
   */
  setHeaders(headers) {
    Object.assign(this.client.defaults.headers.common, headers);
  }

  /**
   * Make batch requests in parallel
   */
  async batchRequests(requests, options = {}) {
    const { concurrency = 10, failFast = false } = options;
    
    const results = [];
    const errors = [];
    
    // Process requests in batches
    for (let i = 0; i < requests.length; i += concurrency) {
      const batch = requests.slice(i, i + concurrency);
      
      const batchPromises = batch.map(async (request, index) => {
        try {
          const { method, url, data, config = {} } = request;
          
          let response;
          switch (method.toLowerCase()) {
            case 'get':
              response = await this.get(url, config);
              break;
            case 'post':
              response = await this.post(url, data, config);
              break;
            case 'put':
              response = await this.put(url, data, config);
              break;
            case 'delete':
              response = await this.delete(url, config);
              break;
            case 'patch':
              response = await this.patch(url, data, config);
              break;
            default:
              throw new Error(`Unsupported method: ${method}`);
          }
          
          return {
            index: i + index,
            success: true,
            response: response,
            data: response.data
          };
        } catch (error) {
          const result = {
            index: i + index,
            success: false,
            error: error,
            request: request
          };
          
          if (failFast) {
            throw result;
          }
          
          return result;
        }
      });
      
      const batchResults = await Promise.allSettled(batchPromises);
      
      batchResults.forEach(result => {
        if (result.status === 'fulfilled') {
          if (result.value.success) {
            results.push(result.value);
          } else {
            errors.push(result.value);
          }
        } else {
          errors.push({
            success: false,
            error: result.reason
          });
        }
      });
    }
    
    return {
      results,
      errors,
      successCount: results.length,
      errorCount: errors.length,
      totalCount: requests.length,
      successRate: results.length / requests.length
    };
  }

  /**
   * Upload file
   */
  async uploadFile(url, filePath, fieldName = 'file', additionalFields = {}) {
    const FormData = require('form-data');
    const fs = require('fs');
    
    const form = new FormData();
    form.append(fieldName, fs.createReadStream(filePath));
    
    // Add additional fields
    Object.entries(additionalFields).forEach(([key, value]) => {
      form.append(key, value);
    });
    
    return this.post(url, form, {
      headers: {
        ...form.getHeaders()
      }
    });
  }

  /**
   * Download file
   */
  async downloadFile(url, outputPath) {
    const fs = require('fs').promises;
    const response = await this.client.get(url, {
      responseType: 'stream'
    });
    
    const writer = fs.createWriteStream(outputPath);
    response.data.pipe(writer);
    
    return new Promise((resolve, reject) => {
      writer.on('finish', resolve);
      writer.on('error', reject);
    });
  }

  /**
   * Health check endpoint
   */
  async healthCheck(endpoint = '/api/v1/health') {
    try {
      const response = await this.get(endpoint, { timeout: 5000 });
      return {
        healthy: response.status === 200,
        status: response.data
      };
    } catch (error) {
      return {
        healthy: false,
        error: error.message
      };
    }
  }

  /**
   * Wait for endpoint to be healthy
   */
  async waitForHealthy(endpoint = '/api/v1/health', timeout = 60000) {
    const startTime = Date.now();
    
    while ((Date.now() - startTime) < timeout) {
      const health = await this.healthCheck(endpoint);
      if (health.healthy) {
        return true;
      }
      
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    throw new Error(`Endpoint ${endpoint} failed to become healthy within ${timeout}ms`);
  }

  /**
   * Get performance metrics
   */
  getMetrics() {
    const metrics = {
      totalRequests: this.responseMetrics.length,
      averageResponseTime: this.responseMetrics.reduce((sum, m) => sum + m.responseTime, 0) / this.responseMetrics.length,
      successRate: this.responseMetrics.filter(m => m.status >= 200 && m.status < 400).length / this.responseMetrics.length,
      errorRate: this.responseMetrics.filter(m => m.status >= 400 || m.error).length / this.responseMetrics.length
    };
    
    // Response time percentiles
    const sortedTimes = this.responseMetrics.map(m => m.responseTime).sort((a, b) => a - b);
    if (sortedTimes.length > 0) {
      metrics.p50ResponseTime = sortedTimes[Math.floor(sortedTimes.length * 0.5)];
      metrics.p95ResponseTime = sortedTimes[Math.floor(sortedTimes.length * 0.95)];
      metrics.p99ResponseTime = sortedTimes[Math.floor(sortedTimes.length * 0.99)];
      metrics.minResponseTime = sortedTimes[0];
      metrics.maxResponseTime = sortedTimes[sortedTimes.length - 1];
    }
    
    return metrics;
  }

  /**
   * Reset metrics
   */
  resetMetrics() {
    this.responseMetrics = [];
    this.requestCount = 0;
  }

  /**
   * Get detailed request history
   */
  getRequestHistory() {
    return [...this.responseMetrics];
  }

  /**
   * Get requests by status code
   */
  getRequestsByStatus(statusCode) {
    return this.responseMetrics.filter(m => m.status === statusCode);
  }

  /**
   * Get failed requests
   */
  getFailedRequests() {
    return this.responseMetrics.filter(m => m.status >= 400 || m.error);
  }

  /**
   * Create a copy of the client with different configuration
   */
  clone(config = {}) {
    return new APIClient({
      ...this.config,
      ...config
    });
  }
}

module.exports = APIClient;