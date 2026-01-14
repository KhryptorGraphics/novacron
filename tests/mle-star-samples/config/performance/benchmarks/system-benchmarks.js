/**
 * System Performance Benchmarks
 * Comprehensive benchmarks for NovaCron system operations
 */

class SystemBenchmark {
  constructor(config) {
    this.config = config;
    this.framework = config.framework;
  }

  async measureLatency(operation, iterations = 100) {
    const latencies = [];
    
    for (let i = 0; i < iterations; i++) {
      const start = process.hrtime.bigint();
      await operation();
      const end = process.hrtime.bigint();
      
      latencies.push(Number(end - start) / 1000000); // Convert to milliseconds
    }
    
    return {
      min: Math.min(...latencies),
      max: Math.max(...latencies),
      avg: latencies.reduce((a, b) => a + b) / latencies.length,
      p50: this.framework.calculateMedian(latencies),
      p95: this.framework.calculatePercentile(latencies, 95),
      p99: this.framework.calculatePercentile(latencies, 99),
      samples: latencies.length
    };
  }

  async measureThroughput(operation, duration = 10000) {
    const operations = [];
    const startTime = Date.now();
    let operationCount = 0;
    
    while (Date.now() - startTime < duration) {
      const opStart = Date.now();
      await operation();
      const opEnd = Date.now();
      
      operations.push({
        duration: opEnd - opStart,
        timestamp: opEnd
      });
      operationCount++;
    }
    
    const totalTime = Date.now() - startTime;
    
    return {
      operationsPerSecond: (operationCount / totalTime) * 1000,
      totalOperations: operationCount,
      totalTime,
      avgOperationTime: operations.reduce((sum, op) => sum + op.duration, 0) / operations.length,
      operations
    };
  }
}

// VM Operations Benchmark
class VMOperationsBenchmark extends SystemBenchmark {
  constructor(config) {
    super(config);
    this.vmService = config.vmService || this.createMockVMService();
  }

  createMockVMService() {
    return {
      create: async (config) => {
        // Simulate VM creation time
        await new Promise(resolve => setTimeout(resolve, Math.random() * 2000 + 1000));
        return { id: `vm_${Date.now()}`, status: 'created' };
      },
      start: async (vmId) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 1500 + 500));
        return { id: vmId, status: 'running' };
      },
      stop: async (vmId) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 300));
        return { id: vmId, status: 'stopped' };
      },
      migrate: async (vmId, targetHost) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 5000 + 2000));
        return { id: vmId, status: 'migrated', host: targetHost };
      },
      delete: async (vmId) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 800 + 200));
        return { id: vmId, status: 'deleted' };
      }
    };
  }

  async run() {
    console.log('Running VM Operations Benchmark...');
    
    const results = {
      vmCreation: await this.benchmarkVMCreation(),
      vmStartStop: await this.benchmarkVMStartStop(),
      vmMigration: await this.benchmarkVMMigration(),
      vmDeletion: await this.benchmarkVMDeletion(),
      concurrentOperations: await this.benchmarkConcurrentOperations()
    };

    return {
      success: true,
      timestamp: Date.now(),
      results
    };
  }

  async benchmarkVMCreation() {
    console.log('  Benchmarking VM Creation...');
    
    const createOperation = async () => {
      return await this.vmService.create({
        cpu: 2,
        memory: '4GB',
        disk: '50GB',
        os: 'ubuntu-20.04'
      });
    };

    const latency = await this.measureLatency(createOperation, 20);
    const throughput = await this.measureThroughput(createOperation, 30000);
    
    return {
      latency,
      throughput,
      recommendations: this.generateVMRecommendations('creation', latency, throughput)
    };
  }

  async benchmarkVMStartStop() {
    console.log('  Benchmarking VM Start/Stop...');
    
    // Create VMs for testing
    const testVMs = [];
    for (let i = 0; i < 10; i++) {
      const vm = await this.vmService.create({ name: `benchmark_vm_${i}` });
      testVMs.push(vm);
    }

    const startOperation = async () => {
      const vm = testVMs[Math.floor(Math.random() * testVMs.length)];
      return await this.vmService.start(vm.id);
    };

    const stopOperation = async () => {
      const vm = testVMs[Math.floor(Math.random() * testVMs.length)];
      return await this.vmService.stop(vm.id);
    };

    const startLatency = await this.measureLatency(startOperation, 30);
    const stopLatency = await this.measureLatency(stopOperation, 30);
    
    // Cleanup
    for (const vm of testVMs) {
      await this.vmService.delete(vm.id);
    }

    return {
      start: {
        latency: startLatency,
        recommendations: this.generateVMRecommendations('start', startLatency)
      },
      stop: {
        latency: stopLatency,
        recommendations: this.generateVMRecommendations('stop', stopLatency)
      }
    };
  }

  async benchmarkVMMigration() {
    console.log('  Benchmarking VM Migration...');
    
    const migrateOperation = async () => {
      const vm = await this.vmService.create({ name: 'migration_test' });
      const result = await this.vmService.migrate(vm.id, 'host_2');
      await this.vmService.delete(vm.id);
      return result;
    };

    const latency = await this.measureLatency(migrateOperation, 10);
    
    return {
      latency,
      recommendations: this.generateVMRecommendations('migration', latency)
    };
  }

  async benchmarkVMDeletion() {
    console.log('  Benchmarking VM Deletion...');
    
    const deleteOperation = async () => {
      const vm = await this.vmService.create({ name: 'delete_test' });
      return await this.vmService.delete(vm.id);
    };

    const latency = await this.measureLatency(deleteOperation, 20);
    const throughput = await this.measureThroughput(deleteOperation, 15000);
    
    return {
      latency,
      throughput,
      recommendations: this.generateVMRecommendations('deletion', latency, throughput)
    };
  }

  async benchmarkConcurrentOperations() {
    console.log('  Benchmarking Concurrent VM Operations...');
    
    const concurrencyLevels = [1, 5, 10, 20, 50];
    const results = {};

    for (const concurrency of concurrencyLevels) {
      console.log(`    Testing concurrency level: ${concurrency}`);
      
      const startTime = Date.now();
      const promises = [];
      
      for (let i = 0; i < concurrency; i++) {
        promises.push(this.performMixedOperations());
      }
      
      await Promise.all(promises);
      const totalTime = Date.now() - startTime;
      
      results[`concurrency_${concurrency}`] = {
        totalTime,
        operationsPerSecond: (concurrency / totalTime) * 1000,
        avgTimePerOperation: totalTime / concurrency
      };
    }

    return {
      results,
      recommendations: this.generateConcurrencyRecommendations(results)
    };
  }

  async performMixedOperations() {
    const operations = ['create', 'start', 'stop', 'delete'];
    const operation = operations[Math.floor(Math.random() * operations.length)];
    
    switch (operation) {
      case 'create':
        const vm = await this.vmService.create({ name: `mixed_op_${Date.now()}` });
        await this.vmService.delete(vm.id);
        break;
      case 'start':
        const startVm = await this.vmService.create({ name: `start_op_${Date.now()}` });
        await this.vmService.start(startVm.id);
        await this.vmService.delete(startVm.id);
        break;
      case 'stop':
        const stopVm = await this.vmService.create({ name: `stop_op_${Date.now()}` });
        await this.vmService.start(stopVm.id);
        await this.vmService.stop(stopVm.id);
        await this.vmService.delete(stopVm.id);
        break;
      case 'delete':
        const deleteVm = await this.vmService.create({ name: `delete_op_${Date.now()}` });
        await this.vmService.delete(deleteVm.id);
        break;
    }
  }

  generateVMRecommendations(operation, latency, throughput = null) {
    const recommendations = [];
    
    if (latency.avg > 5000) {
      recommendations.push({
        type: 'performance',
        severity: 'high',
        message: `${operation} operations are slow (${latency.avg.toFixed(2)}ms avg). Consider optimizing storage or network.`
      });
    }
    
    if (latency.p99 > latency.avg * 3) {
      recommendations.push({
        type: 'reliability',
        severity: 'medium',
        message: `High ${operation} latency variance detected. P99 is ${latency.p99.toFixed(2)}ms vs ${latency.avg.toFixed(2)}ms avg.`
      });
    }
    
    if (throughput && throughput.operationsPerSecond < 1) {
      recommendations.push({
        type: 'scalability',
        severity: 'high',
        message: `Low ${operation} throughput (${throughput.operationsPerSecond.toFixed(2)} ops/sec). Consider parallel processing.`
      });
    }
    
    return recommendations;
  }

  generateConcurrencyRecommendations(results) {
    const recommendations = [];
    const concurrencyLevels = Object.keys(results).map(k => parseInt(k.split('_')[1]));
    
    // Find optimal concurrency level
    let bestThroughput = 0;
    let optimalConcurrency = 1;
    
    concurrencyLevels.forEach(level => {
      const result = results[`concurrency_${level}`];
      if (result.operationsPerSecond > bestThroughput) {
        bestThroughput = result.operationsPerSecond;
        optimalConcurrency = level;
      }
    });
    
    recommendations.push({
      type: 'optimization',
      severity: 'info',
      message: `Optimal concurrency level appears to be ${optimalConcurrency} with ${bestThroughput.toFixed(2)} ops/sec.`
    });
    
    // Check for performance degradation at high concurrency
    const highConcurrency = Math.max(...concurrencyLevels);
    const highConcurrencyResult = results[`concurrency_${highConcurrency}`];
    
    if (highConcurrencyResult.operationsPerSecond < bestThroughput * 0.7) {
      recommendations.push({
        type: 'scalability',
        severity: 'medium',
        message: `Performance degrades significantly at high concurrency (${highConcurrency}). Consider resource limits.`
      });
    }
    
    return recommendations;
  }
}

// Database Performance Benchmark
class DatabaseBenchmark extends SystemBenchmark {
  constructor(config) {
    super(config);
    this.dbService = config.dbService || this.createMockDBService();
  }

  createMockDBService() {
    const data = new Map();
    
    return {
      query: async (sql, params = []) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 100 + 10));
        return { rows: [], rowCount: Math.floor(Math.random() * 1000) };
      },
      insert: async (table, data) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 50 + 5));
        return { insertedId: Date.now() };
      },
      update: async (table, id, data) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 30 + 5));
        return { modifiedCount: 1 };
      },
      delete: async (table, id) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 20 + 3));
        return { deletedCount: 1 };
      },
      transaction: async (operations) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 200 + 50));
        return { success: true };
      }
    };
  }

  async run() {
    console.log('Running Database Performance Benchmark...');
    
    const results = {
      queryPerformance: await this.benchmarkQueries(),
      insertPerformance: await this.benchmarkInserts(),
      updatePerformance: await this.benchmarkUpdates(),
      deletePerformance: await this.benchmarkDeletes(),
      transactionPerformance: await this.benchmarkTransactions(),
      concurrentLoad: await this.benchmarkConcurrentLoad()
    };

    return {
      success: true,
      timestamp: Date.now(),
      results
    };
  }

  async benchmarkQueries() {
    console.log('  Benchmarking Database Queries...');
    
    const queryTypes = {
      simple: () => this.dbService.query('SELECT * FROM users LIMIT 10'),
      complex: () => this.dbService.query(`
        SELECT u.*, p.name as profile_name 
        FROM users u 
        JOIN profiles p ON u.id = p.user_id 
        WHERE u.created_at > ? 
        ORDER BY u.created_at DESC 
        LIMIT 100
      `, [new Date('2023-01-01')]),
      aggregate: () => this.dbService.query(`
        SELECT COUNT(*) as total, AVG(age) as avg_age, status
        FROM users 
        GROUP BY status 
        HAVING COUNT(*) > 10
      `)
    };

    const results = {};
    
    for (const [type, query] of Object.entries(queryTypes)) {
      const latency = await this.measureLatency(query, 50);
      const throughput = await this.measureThroughput(query, 10000);
      
      results[type] = {
        latency,
        throughput,
        recommendations: this.generateQueryRecommendations(type, latency, throughput)
      };
    }

    return results;
  }

  async benchmarkInserts() {
    console.log('  Benchmarking Database Inserts...');
    
    const insertOperation = () => this.dbService.insert('users', {
      name: `User_${Date.now()}`,
      email: `user${Date.now()}@example.com`,
      age: Math.floor(Math.random() * 80) + 18,
      created_at: new Date()
    });

    const latency = await this.measureLatency(insertOperation, 100);
    const throughput = await this.measureThroughput(insertOperation, 15000);
    
    return {
      latency,
      throughput,
      batchInsertPerformance: await this.benchmarkBatchInserts(),
      recommendations: this.generateInsertRecommendations(latency, throughput)
    };
  }

  async benchmarkBatchInserts() {
    const batchSizes = [1, 10, 50, 100, 500];
    const results = {};
    
    for (const size of batchSizes) {
      const batchOperation = async () => {
        const batch = [];
        for (let i = 0; i < size; i++) {
          batch.push({
            name: `BatchUser_${i}_${Date.now()}`,
            email: `batch${i}_${Date.now()}@example.com`,
            age: Math.floor(Math.random() * 80) + 18
          });
        }
        
        // Simulate batch insert
        for (const item of batch) {
          await this.dbService.insert('users', item);
        }
      };
      
      const latency = await this.measureLatency(batchOperation, 10);
      
      results[`batch_${size}`] = {
        latency,
        itemsPerSecond: (size / latency.avg) * 1000
      };
    }
    
    return results;
  }

  async benchmarkUpdates() {
    console.log('  Benchmarking Database Updates...');
    
    const updateOperation = () => this.dbService.update('users', 1, {
      last_updated: new Date(),
      status: 'active'
    });

    const latency = await this.measureLatency(updateOperation, 50);
    const throughput = await this.measureThroughput(updateOperation, 10000);
    
    return {
      latency,
      throughput,
      recommendations: this.generateUpdateRecommendations(latency, throughput)
    };
  }

  async benchmarkDeletes() {
    console.log('  Benchmarking Database Deletes...');
    
    const deleteOperation = () => this.dbService.delete('users', Math.floor(Math.random() * 1000) + 1);

    const latency = await this.measureLatency(deleteOperation, 30);
    const throughput = await this.measureThroughput(deleteOperation, 8000);
    
    return {
      latency,
      throughput,
      recommendations: this.generateDeleteRecommendations(latency, throughput)
    };
  }

  async benchmarkTransactions() {
    console.log('  Benchmarking Database Transactions...');
    
    const transactionOperation = () => this.dbService.transaction([
      { type: 'insert', table: 'users', data: { name: 'Transaction User' } },
      { type: 'update', table: 'profiles', id: 1, data: { updated: true } },
      { type: 'delete', table: 'temp_data', id: 1 }
    ]);

    const latency = await this.measureLatency(transactionOperation, 20);
    const throughput = await this.measureThroughput(transactionOperation, 10000);
    
    return {
      latency,
      throughput,
      recommendations: this.generateTransactionRecommendations(latency, throughput)
    };
  }

  async benchmarkConcurrentLoad() {
    console.log('  Benchmarking Concurrent Database Load...');
    
    const concurrencyLevels = [1, 5, 10, 25, 50, 100];
    const results = {};
    
    for (const concurrency of concurrencyLevels) {
      console.log(`    Testing DB concurrency: ${concurrency}`);
      
      const operations = [];
      const startTime = Date.now();
      
      for (let i = 0; i < concurrency; i++) {
        operations.push(this.performMixedDBOperations());
      }
      
      await Promise.all(operations);
      const totalTime = Date.now() - startTime;
      
      results[`concurrency_${concurrency}`] = {
        totalTime,
        operationsPerSecond: (concurrency / totalTime) * 1000
      };
    }
    
    return {
      results,
      recommendations: this.generateConcurrentLoadRecommendations(results)
    };
  }

  async performMixedDBOperations() {
    const operations = ['query', 'insert', 'update', 'delete'];
    const operation = operations[Math.floor(Math.random() * operations.length)];
    
    switch (operation) {
      case 'query':
        return await this.dbService.query('SELECT * FROM users LIMIT 5');
      case 'insert':
        return await this.dbService.insert('users', { name: `User_${Date.now()}` });
      case 'update':
        return await this.dbService.update('users', 1, { updated: new Date() });
      case 'delete':
        return await this.dbService.delete('temp_users', 1);
    }
  }

  generateQueryRecommendations(type, latency, throughput) {
    const recommendations = [];
    
    if (type === 'complex' && latency.avg > 500) {
      recommendations.push({
        type: 'optimization',
        severity: 'medium',
        message: 'Complex queries are slow. Consider adding indexes or query optimization.'
      });
    }
    
    if (throughput.operationsPerSecond < 100) {
      recommendations.push({
        type: 'performance',
        severity: 'high',
        message: `Low query throughput (${throughput.operationsPerSecond.toFixed(2)} ops/sec). Check database configuration.`
      });
    }
    
    return recommendations;
  }

  generateInsertRecommendations(latency, throughput) {
    const recommendations = [];
    
    if (latency.avg > 100) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: 'Insert operations are slow. Consider batch inserts or index optimization.'
      });
    }
    
    if (throughput.operationsPerSecond < 500) {
      recommendations.push({
        type: 'scalability',
        severity: 'medium',
        message: 'Low insert throughput. Consider connection pooling or write optimization.'
      });
    }
    
    return recommendations;
  }

  generateUpdateRecommendations(latency, throughput) {
    const recommendations = [];
    
    if (latency.avg > 50) {
      recommendations.push({
        type: 'optimization',
        severity: 'medium',
        message: 'Update operations are slow. Check for missing indexes on WHERE clauses.'
      });
    }
    
    return recommendations;
  }

  generateDeleteRecommendations(latency, throughput) {
    const recommendations = [];
    
    if (latency.avg > 30) {
      recommendations.push({
        type: 'optimization',
        severity: 'medium',
        message: 'Delete operations are slow. Ensure proper indexing on delete conditions.'
      });
    }
    
    return recommendations;
  }

  generateTransactionRecommendations(latency, throughput) {
    const recommendations = [];
    
    if (latency.avg > 200) {
      recommendations.push({
        type: 'performance',
        severity: 'high',
        message: 'Transactions are slow. Consider reducing transaction scope or optimizing queries within transactions.'
      });
    }
    
    return recommendations;
  }

  generateConcurrentLoadRecommendations(results) {
    const recommendations = [];
    const concurrencyLevels = Object.keys(results).map(k => parseInt(k.split('_')[1]));
    
    // Find point where performance starts degrading
    let previousThroughput = 0;
    let degradationPoint = null;
    
    concurrencyLevels.forEach(level => {
      const result = results[`concurrency_${level}`];
      
      if (previousThroughput > 0 && result.operationsPerSecond < previousThroughput * 0.8) {
        if (!degradationPoint) {
          degradationPoint = level;
        }
      }
      
      previousThroughput = result.operationsPerSecond;
    });
    
    if (degradationPoint) {
      recommendations.push({
        type: 'scalability',
        severity: 'medium',
        message: `Database performance degrades significantly at ${degradationPoint} concurrent connections. Consider connection pooling limits.`
      });
    }
    
    return recommendations;
  }
}

// Network and Storage I/O Benchmark
class NetworkStorageBenchmark extends SystemBenchmark {
  constructor(config) {
    super(config);
    this.networkService = config.networkService || this.createMockNetworkService();
    this.storageService = config.storageService || this.createMockStorageService();
  }

  createMockNetworkService() {
    return {
      request: async (url, options = {}) => {
        const latency = Math.random() * 200 + 50;
        await new Promise(resolve => setTimeout(resolve, latency));
        
        return {
          status: 200,
          data: { success: true },
          latency,
          size: options.size || 1024
        };
      },
      upload: async (data, size) => {
        const uploadTime = (size / 1024 / 1024) * 1000 + Math.random() * 100; // Simulate based on size
        await new Promise(resolve => setTimeout(resolve, uploadTime));
        
        return {
          success: true,
          size,
          uploadTime
        };
      },
      download: async (url, expectedSize) => {
        const downloadTime = (expectedSize / 1024 / 1024) * 800 + Math.random() * 100;
        await new Promise(resolve => setTimeout(resolve, downloadTime));
        
        return {
          success: true,
          size: expectedSize,
          downloadTime
        };
      }
    };
  }

  createMockStorageService() {
    return {
      read: async (path, size = 1024) => {
        const readTime = (size / 1024 / 1024) * 100 + Math.random() * 50;
        await new Promise(resolve => setTimeout(resolve, readTime));
        
        return {
          data: Buffer.alloc(size),
          readTime,
          iops: Math.floor(Math.random() * 1000) + 500
        };
      },
      write: async (path, data, size = 1024) => {
        const writeTime = (size / 1024 / 1024) * 120 + Math.random() * 60;
        await new Promise(resolve => setTimeout(resolve, writeTime));
        
        return {
          success: true,
          writeTime,
          iops: Math.floor(Math.random() * 800) + 300
        };
      },
      delete: async (path) => {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 20 + 5));
        return { success: true };
      }
    };
  }

  async run() {
    console.log('Running Network and Storage I/O Benchmark...');
    
    const results = {
      networkPerformance: await this.benchmarkNetwork(),
      storagePerformance: await this.benchmarkStorage(),
      ioMixedWorkload: await this.benchmarkMixedIOWorkload()
    };

    return {
      success: true,
      timestamp: Date.now(),
      results
    };
  }

  async benchmarkNetwork() {
    console.log('  Benchmarking Network Performance...');
    
    const results = {
      latency: await this.benchmarkNetworkLatency(),
      throughput: await this.benchmarkNetworkThroughput(),
      reliability: await this.benchmarkNetworkReliability()
    };
    
    return results;
  }

  async benchmarkNetworkLatency() {
    const requestOperation = () => this.networkService.request('http://api.novacron.local/health');
    
    const latency = await this.measureLatency(requestOperation, 50);
    
    return {
      latency,
      recommendations: this.generateNetworkLatencyRecommendations(latency)
    };
  }

  async benchmarkNetworkThroughput() {
    const sizes = [1024, 10240, 102400, 1048576]; // 1KB, 10KB, 100KB, 1MB
    const results = {};
    
    for (const size of sizes) {
      console.log(`    Testing network throughput with ${size} bytes...`);
      
      const uploadOperation = () => this.networkService.upload(Buffer.alloc(size), size);
      const downloadOperation = () => this.networkService.download('http://api.novacron.local/data', size);
      
      const uploadThroughput = await this.measureThroughput(uploadOperation, 10000);
      const downloadThroughput = await this.measureThroughput(downloadOperation, 10000);
      
      results[`size_${size}`] = {
        upload: {
          throughput: uploadThroughput,
          mbps: (size * uploadThroughput.operationsPerSecond) / 1024 / 1024
        },
        download: {
          throughput: downloadThroughput,
          mbps: (size * downloadThroughput.operationsPerSecond) / 1024 / 1024
        }
      };
    }
    
    return {
      results,
      recommendations: this.generateNetworkThroughputRecommendations(results)
    };
  }

  async benchmarkNetworkReliability() {
    let successCount = 0;
    let failureCount = 0;
    const totalRequests = 100;
    
    for (let i = 0; i < totalRequests; i++) {
      try {
        await this.networkService.request('http://api.novacron.local/test');
        successCount++;
      } catch (error) {
        failureCount++;
      }
    }
    
    const reliability = (successCount / totalRequests) * 100;
    
    return {
      successRate: reliability,
      totalRequests,
      successCount,
      failureCount,
      recommendations: this.generateNetworkReliabilityRecommendations(reliability)
    };
  }

  async benchmarkStorage() {
    console.log('  Benchmarking Storage Performance...');
    
    const results = {
      readPerformance: await this.benchmarkStorageRead(),
      writePerformance: await this.benchmarkStorageWrite(),
      iopsPerformance: await this.benchmarkStorageIOPS()
    };
    
    return results;
  }

  async benchmarkStorageRead() {
    const sizes = [4096, 65536, 1048576, 10485760]; // 4KB, 64KB, 1MB, 10MB
    const results = {};
    
    for (const size of sizes) {
      console.log(`    Testing storage read with ${size} bytes...`);
      
      const readOperation = () => this.storageService.read(`/tmp/benchmark_${Date.now()}`, size);
      
      const latency = await this.measureLatency(readOperation, 20);
      const throughput = await this.measureThroughput(readOperation, 10000);
      
      results[`size_${size}`] = {
        latency,
        throughput,
        mbps: (size * throughput.operationsPerSecond) / 1024 / 1024
      };
    }
    
    return {
      results,
      recommendations: this.generateStorageReadRecommendations(results)
    };
  }

  async benchmarkStorageWrite() {
    const sizes = [4096, 65536, 1048576, 10485760]; // 4KB, 64KB, 1MB, 10MB
    const results = {};
    
    for (const size of sizes) {
      console.log(`    Testing storage write with ${size} bytes...`);
      
      const writeOperation = () => this.storageService.write(`/tmp/benchmark_write_${Date.now()}`, Buffer.alloc(size), size);
      
      const latency = await this.measureLatency(writeOperation, 20);
      const throughput = await this.measureThroughput(writeOperation, 10000);
      
      results[`size_${size}`] = {
        latency,
        throughput,
        mbps: (size * throughput.operationsPerSecond) / 1024 / 1024
      };
    }
    
    return {
      results,
      recommendations: this.generateStorageWriteRecommendations(results)
    };
  }

  async benchmarkStorageIOPS() {
    console.log('    Testing storage IOPS...');
    
    const smallReadOperation = () => this.storageService.read(`/tmp/iops_test_${Date.now()}`, 4096);
    const smallWriteOperation = () => this.storageService.write(`/tmp/iops_write_${Date.now()}`, Buffer.alloc(4096), 4096);
    
    const readIOPS = await this.measureThroughput(smallReadOperation, 30000);
    const writeIOPS = await this.measureThroughput(smallWriteOperation, 30000);
    
    return {
      readIOPS: readIOPS.operationsPerSecond,
      writeIOPS: writeIOPS.operationsPerSecond,
      recommendations: this.generateIOPSRecommendations(readIOPS.operationsPerSecond, writeIOPS.operationsPerSecond)
    };
  }

  async benchmarkMixedIOWorkload() {
    console.log('  Benchmarking Mixed I/O Workload...');
    
    const mixedOperation = async () => {
      const operations = [];
      
      // Network operations
      operations.push(this.networkService.request('http://api.novacron.local/data'));
      
      // Storage operations
      operations.push(this.storageService.read('/tmp/mixed_test', 65536));
      operations.push(this.storageService.write('/tmp/mixed_write', Buffer.alloc(32768), 32768));
      
      await Promise.all(operations);
    };
    
    const latency = await this.measureLatency(mixedOperation, 30);
    const throughput = await this.measureThroughput(mixedOperation, 20000);
    
    return {
      latency,
      throughput,
      recommendations: this.generateMixedWorkloadRecommendations(latency, throughput)
    };
  }

  generateNetworkLatencyRecommendations(latency) {
    const recommendations = [];
    
    if (latency.avg > 200) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `High network latency detected (${latency.avg.toFixed(2)}ms avg). Check network configuration.`
      });
    }
    
    if (latency.p99 > 1000) {
      recommendations.push({
        type: 'reliability',
        severity: 'high',
        message: `Very high P99 network latency (${latency.p99.toFixed(2)}ms). Investigate network instability.`
      });
    }
    
    return recommendations;
  }

  generateNetworkThroughputRecommendations(results) {
    const recommendations = [];
    
    // Check if throughput scales with size
    const sizes = Object.keys(results).map(k => parseInt(k.split('_')[1]));
    const largeSizeResult = results[`size_${Math.max(...sizes)}`];
    
    if (largeSizeResult.upload.mbps < 10) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `Low upload bandwidth (${largeSizeResult.upload.mbps.toFixed(2)} Mbps). Consider network optimization.`
      });
    }
    
    if (largeSizeResult.download.mbps < 50) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `Low download bandwidth (${largeSizeResult.download.mbps.toFixed(2)} Mbps). Check network capacity.`
      });
    }
    
    return recommendations;
  }

  generateNetworkReliabilityRecommendations(reliability) {
    const recommendations = [];
    
    if (reliability < 99) {
      recommendations.push({
        type: 'reliability',
        severity: 'high',
        message: `Low network reliability (${reliability.toFixed(2)}%). Investigate network stability.`
      });
    } else if (reliability < 99.9) {
      recommendations.push({
        type: 'reliability',
        severity: 'medium',
        message: `Network reliability could be improved (${reliability.toFixed(2)}%). Consider redundancy.`
      });
    }
    
    return recommendations;
  }

  generateStorageReadRecommendations(results) {
    const recommendations = [];
    
    const sizes = Object.keys(results).map(k => parseInt(k.split('_')[1]));
    const largeRead = results[`size_${Math.max(...sizes)}`];
    
    if (largeRead.mbps < 100) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `Low storage read performance (${largeRead.mbps.toFixed(2)} MB/s). Consider SSD or storage optimization.`
      });
    }
    
    return recommendations;
  }

  generateStorageWriteRecommendations(results) {
    const recommendations = [];
    
    const sizes = Object.keys(results).map(k => parseInt(k.split('_')[1]));
    const largeWrite = results[`size_${Math.max(...sizes)}`];
    
    if (largeWrite.mbps < 50) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `Low storage write performance (${largeWrite.mbps.toFixed(2)} MB/s). Consider storage optimization.`
      });
    }
    
    return recommendations;
  }

  generateIOPSRecommendations(readIOPS, writeIOPS) {
    const recommendations = [];
    
    if (readIOPS < 1000) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `Low read IOPS (${readIOPS.toFixed(0)}). Consider SSD storage for better random I/O.`
      });
    }
    
    if (writeIOPS < 500) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `Low write IOPS (${writeIOPS.toFixed(0)}). Consider storage optimization or SSD.`
      });
    }
    
    return recommendations;
  }

  generateMixedWorkloadRecommendations(latency, throughput) {
    const recommendations = [];
    
    if (throughput.operationsPerSecond < 10) {
      recommendations.push({
        type: 'performance',
        severity: 'high',
        message: `Low mixed workload throughput (${throughput.operationsPerSecond.toFixed(2)} ops/sec). Optimize I/O scheduling.`
      });
    }
    
    if (latency.avg > 500) {
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `High mixed workload latency (${latency.avg.toFixed(2)}ms). Consider I/O prioritization.`
      });
    }
    
    return recommendations;
  }
}

// Auto-scaling Response Time Benchmark
class AutoScalingBenchmark extends SystemBenchmark {
  constructor(config) {
    super(config);
    this.scalingService = config.scalingService || this.createMockScalingService();
  }

  createMockScalingService() {
    let currentInstances = 2;
    
    return {
      getCurrentScale: () => currentInstances,
      scaleUp: async (targetInstances) => {
        const scaleTime = (targetInstances - currentInstances) * 30000 + Math.random() * 10000;
        await new Promise(resolve => setTimeout(resolve, scaleTime));
        currentInstances = targetInstances;
        return { success: true, newScale: currentInstances, scaleTime };
      },
      scaleDown: async (targetInstances) => {
        const scaleTime = (currentInstances - targetInstances) * 15000 + Math.random() * 5000;
        await new Promise(resolve => setTimeout(resolve, scaleTime));
        currentInstances = targetInstances;
        return { success: true, newScale: currentInstances, scaleTime };
      },
      autoScale: async (trigger) => {
        const decision = Math.random() > 0.5 ? 'up' : 'down';
        const newScale = decision === 'up' ? currentInstances + 1 : Math.max(1, currentInstances - 1);
        
        const result = decision === 'up' ? 
          await this.scaleUp(newScale) : 
          await this.scaleDown(newScale);
        
        return { ...result, trigger, decision };
      }
    };
  }

  async run() {
    console.log('Running Auto-scaling Response Time Benchmark...');
    
    const results = {
      scaleUpPerformance: await this.benchmarkScaleUp(),
      scaleDownPerformance: await this.benchmarkScaleDown(),
      autoScaleResponse: await this.benchmarkAutoScaleResponse(),
      scaleStressTest: await this.benchmarkScaleStressTest()
    };

    return {
      success: true,
      timestamp: Date.now(),
      results
    };
  }

  async benchmarkScaleUp() {
    console.log('  Benchmarking Scale-up Performance...');
    
    const scaleTargets = [3, 5, 8, 12, 20];
    const results = {};
    
    for (const target of scaleTargets) {
      const currentScale = this.scalingService.getCurrentScale();
      
      if (target > currentScale) {
        const scaleOperation = () => this.scalingService.scaleUp(target);
        
        const latency = await this.measureLatency(scaleOperation, 5);
        
        results[`scale_to_${target}`] = {
          latency,
          scaleIncrease: target - currentScale,
          timePerInstance: latency.avg / (target - currentScale),
          recommendations: this.generateScaleUpRecommendations(target - currentScale, latency)
        };
        
        // Reset to base scale for next test
        await this.scalingService.scaleDown(2);
      }
    }
    
    return results;
  }

  async benchmarkScaleDown() {
    console.log('  Benchmarking Scale-down Performance...');
    
    // First scale up to have instances to scale down
    await this.scalingService.scaleUp(10);
    
    const scaleTargets = [8, 5, 3, 2, 1];
    const results = {};
    
    for (const target of scaleTargets) {
      const currentScale = this.scalingService.getCurrentScale();
      
      if (target < currentScale) {
        const scaleOperation = () => this.scalingService.scaleDown(target);
        
        const latency = await this.measureLatency(scaleOperation, 5);
        
        results[`scale_to_${target}`] = {
          latency,
          scaleDecrease: currentScale - target,
          timePerInstance: latency.avg / (currentScale - target),
          recommendations: this.generateScaleDownRecommendations(currentScale - target, latency)
        };
      }
    }
    
    return results;
  }

  async benchmarkAutoScaleResponse() {
    console.log('  Benchmarking Auto-scale Response...');
    
    const triggers = ['cpu_high', 'memory_high', 'requests_high', 'cpu_low', 'requests_low'];
    const results = {};
    
    for (const trigger of triggers) {
      console.log(`    Testing auto-scale trigger: ${trigger}`);
      
      const autoScaleOperation = () => this.scalingService.autoScale(trigger);
      
      const latency = await this.measureLatency(autoScaleOperation, 10);
      
      results[trigger] = {
        latency,
        recommendations: this.generateAutoScaleRecommendations(trigger, latency)
      };
    }
    
    return results;
  }

  async benchmarkScaleStressTest() {
    console.log('  Benchmarking Scale Stress Test...');
    
    const operations = [];
    const startTime = Date.now();
    
    // Simulate rapid scaling events
    for (let i = 0; i < 20; i++) {
      const operation = async () => {
        const isScaleUp = Math.random() > 0.5;
        const currentScale = this.scalingService.getCurrentScale();
        
        if (isScaleUp) {
          const target = Math.min(20, currentScale + Math.floor(Math.random() * 3) + 1);
          return await this.scalingService.scaleUp(target);
        } else {
          const target = Math.max(1, currentScale - Math.floor(Math.random() * 2) - 1);
          return await this.scalingService.scaleDown(target);
        }
      };
      
      operations.push(operation());
      
      // Add some randomness to timing
      await new Promise(resolve => setTimeout(resolve, Math.random() * 2000 + 1000));
    }
    
    await Promise.all(operations);
    const totalTime = Date.now() - startTime;
    
    return {
      totalOperations: operations.length,
      totalTime,
      operationsPerMinute: (operations.length / totalTime) * 60000,
      recommendations: this.generateStressTestRecommendations(operations.length, totalTime)
    };
  }

  generateScaleUpRecommendations(instances, latency) {
    const recommendations = [];
    
    if (latency.avg > 120000) { // 2 minutes
      recommendations.push({
        type: 'performance',
        severity: 'high',
        message: `Slow scale-up performance (${(latency.avg / 1000).toFixed(1)}s for ${instances} instances). Optimize instance provisioning.`
      });
    }
    
    const timePerInstance = latency.avg / instances;
    if (timePerInstance > 60000) { // 1 minute per instance
      recommendations.push({
        type: 'scalability',
        severity: 'medium',
        message: `High per-instance scale-up time (${(timePerInstance / 1000).toFixed(1)}s). Consider pre-warmed instances.`
      });
    }
    
    return recommendations;
  }

  generateScaleDownRecommendations(instances, latency) {
    const recommendations = [];
    
    if (latency.avg > 60000) { // 1 minute
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: `Slow scale-down performance (${(latency.avg / 1000).toFixed(1)}s for ${instances} instances). Optimize instance termination.`
      });
    }
    
    return recommendations;
  }

  generateAutoScaleRecommendations(trigger, latency) {
    const recommendations = [];
    
    if (latency.avg > 180000) { // 3 minutes
      recommendations.push({
        type: 'responsiveness',
        severity: 'high',
        message: `Slow auto-scaling response to ${trigger} (${(latency.avg / 1000).toFixed(1)}s). Improve trigger sensitivity.`
      });
    }
    
    if (trigger.includes('_high') && latency.avg > 90000) { // 1.5 minutes for urgent triggers
      recommendations.push({
        type: 'performance',
        severity: 'high',
        message: `Urgent scaling trigger ${trigger} is too slow (${(latency.avg / 1000).toFixed(1)}s). Reduce decision time.`
      });
    }
    
    return recommendations;
  }

  generateStressTestRecommendations(operations, totalTime) {
    const recommendations = [];
    
    const operationsPerMinute = (operations / totalTime) * 60000;
    
    if (operationsPerMinute < 5) {
      recommendations.push({
        type: 'scalability',
        severity: 'high',
        message: `Low scaling throughput under stress (${operationsPerMinute.toFixed(2)} ops/min). System may struggle with rapid changes.`
      });
    }
    
    if (totalTime / operations > 30000) { // 30 seconds average
      recommendations.push({
        type: 'performance',
        severity: 'medium',
        message: 'High average scaling time under stress. Consider optimizing for concurrent scaling operations.'
      });
    }
    
    return recommendations;
  }
}

module.exports = {
  VMOperationsBenchmark,
  DatabaseBenchmark,
  NetworkStorageBenchmark,
  AutoScalingBenchmark,
  SystemBenchmark
};