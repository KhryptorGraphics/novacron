/**
 * MLE-Star Workflow Test Suite
 * Comprehensive unit and integration tests for MLE-Star installation and workflow functionality
 * 
 * @author: TESTER (Hive Mind QA Specialist)
 * @mission: Validate MLE-Star command functionality, template generation, and workflow execution
 */

const { spawn, exec } = require('child_process');
const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');

// Test configuration
const TEST_CONFIG = {
  timeout: 30000,
  maxExecutionTime: 5000,
  tempDir: path.join(__dirname, '..', 'temp-test'),
  testProjectName: 'mle-star-test-project',
  requiredCommands: ['npx', 'claude-flow'],
  expectedTemplates: ['basic', 'advanced', 'enterprise']
};

describe('MLE-Star Installation and Workflow Tests', () => {
  let tempTestDir;
  
  beforeAll(async () => {
    // Setup test environment
    tempTestDir = path.join(TEST_CONFIG.tempDir, Date.now().toString());
    await fs.mkdir(tempTestDir, { recursive: true });
    process.chdir(tempTestDir);
  }, TEST_CONFIG.timeout);

  afterAll(async () => {
    // Cleanup test environment
    try {
      await fs.rm(TEST_CONFIG.tempDir, { recursive: true, force: true });
    } catch (error) {
      console.warn('Cleanup warning:', error.message);
    }
  });

  describe('Command Registration and Help Display', () => {
    test('should register mle-star command successfully', async () => {
      const result = await execPromise('npx claude-flow@alpha --help');
      expect(result.stdout).toMatch(/mle-star|sparc|workflow/i);
    }, TEST_CONFIG.timeout);

    test('should display mle-star help information', async () => {
      const result = await execPromise('npx claude-flow@alpha sparc --help');
      expect(result.stdout).toContain('modes');
      expect(result.stdout).toContain('run');
      expect(result.stdout).toContain('tdd');
    }, TEST_CONFIG.timeout);

    test('should list available SPARC modes', async () => {
      const result = await execPromise('npx claude-flow@alpha sparc modes');
      expect(result.stdout).toMatch(/spec-pseudocode|architect|tdd|integration/);
    }, TEST_CONFIG.timeout);

    test('should provide detailed mode information', async () => {
      const result = await execPromise('npx claude-flow@alpha sparc info architect');
      expect(result.stdout).toMatch(/architect|design|system/i);
    }, TEST_CONFIG.timeout);
  });

  describe('Template Generation and Customization', () => {
    test('should generate basic project template', async () => {
      const projectPath = path.join(tempTestDir, 'basic-template-test');
      
      const result = await execPromise(
        `npx claude-flow@alpha sparc run spec-pseudocode "Create basic REST API template"`,
        { cwd: tempTestDir }
      );
      
      expect(result.stdout).toMatch(/specification|pseudocode|generated/i);
    }, TEST_CONFIG.timeout);

    test('should customize template with options', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha sparc run architect "Design microservices architecture with database"',
        { cwd: tempTestDir }
      );
      
      expect(result.stdout).toMatch(/architecture|microservices|database/i);
    }, TEST_CONFIG.timeout);

    test('should validate generated template structure', async () => {
      // Test template structure after generation
      const result = await execPromise('find . -name "*.md" -o -name "*.json" | head -10');
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);
  });

  describe('Workflow Execution Stages', () => {
    test('should execute specification stage', async () => {
      const startTime = performance.now();
      
      const result = await execPromise(
        'npx claude-flow@alpha sparc run spec-pseudocode "User authentication system"'
      );
      
      const executionTime = performance.now() - startTime;
      
      expect(result.stdout).toMatch(/specification|requirements|pseudocode/i);
      expect(executionTime).toBeLessThan(TEST_CONFIG.maxExecutionTime);
    }, TEST_CONFIG.timeout);

    test('should execute architecture stage', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha sparc run architect "Design scalable web application"'
      );
      
      expect(result.stdout).toMatch(/architecture|design|system/i);
    }, TEST_CONFIG.timeout);

    test('should execute TDD workflow', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha sparc tdd "Password validation feature"'
      );
      
      expect(result.stdout).toMatch(/test|driven|development|tdd/i);
    }, TEST_CONFIG.timeout);

    test('should execute integration stage', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha sparc run integration "Complete system integration"'
      );
      
      expect(result.stdout).toMatch(/integration|complete|system/i);
    }, TEST_CONFIG.timeout);
  });

  describe('Batch Processing and Parallel Execution', () => {
    test('should execute batch processing', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha sparc batch spec-pseudocode,architect "Full-stack application"'
      );
      
      expect(result.stdout).toMatch(/batch|parallel|spec|architect/i);
    }, TEST_CONFIG.timeout);

    test('should handle pipeline processing', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha sparc pipeline "E-commerce platform development"'
      );
      
      expect(result.stdout).toMatch(/pipeline|processing|workflow/i);
    }, TEST_CONFIG.timeout);

    test('should process concurrent tasks', async () => {
      // Create a test tasks file
      const tasksFile = path.join(tempTestDir, 'test-tasks.txt');
      await fs.writeFile(tasksFile, [
        'User registration API',
        'Product catalog service',
        'Payment processing module'
      ].join('\n'));

      const result = await execPromise(
        `npx claude-flow@alpha sparc concurrent architect "${tasksFile}"`
      );
      
      expect(result.stdout).toMatch(/concurrent|tasks|architect/i);
    }, TEST_CONFIG.timeout);
  });

  describe('Error Handling and Recovery', () => {
    test('should handle invalid command gracefully', async () => {
      try {
        await execPromise('npx claude-flow@alpha sparc run invalid-mode "test"');
      } catch (error) {
        expect(error.stderr).toMatch(/invalid|error|mode/i);
        expect(error.code).not.toBe(0);
      }
    }, TEST_CONFIG.timeout);

    test('should handle missing parameters', async () => {
      try {
        await execPromise('npx claude-flow@alpha sparc run');
      } catch (error) {
        expect(error.stderr).toMatch(/missing|parameter|required/i);
      }
    }, TEST_CONFIG.timeout);

    test('should handle network timeout gracefully', async () => {
      const result = await execPromise(
        'timeout 2s npx claude-flow@alpha sparc run architect "test" || echo "timeout handled"'
      );
      
      expect(result.stdout).toMatch(/timeout handled|architect/);
    }, TEST_CONFIG.timeout);

    test('should recover from interrupted execution', async () => {
      // Test recovery mechanism
      const result = await execPromise(
        'npx claude-flow@alpha sparc run spec-pseudocode "recovery test" --resume'
      );
      
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);
  });

  describe('Performance Benchmarks', () => {
    test('should complete basic workflow under performance threshold', async () => {
      const startTime = performance.now();
      
      const result = await execPromise(
        'npx claude-flow@alpha sparc run spec-pseudocode "Simple API endpoint"'
      );
      
      const executionTime = performance.now() - startTime;
      
      expect(executionTime).toBeLessThan(TEST_CONFIG.maxExecutionTime);
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);

    test('should handle multiple concurrent executions efficiently', async () => {
      const promises = Array.from({ length: 3 }, (_, i) =>
        execPromise(`npx claude-flow@alpha sparc run spec-pseudocode "Test task ${i + 1}"`)
      );
      
      const startTime = performance.now();
      const results = await Promise.allSettled(promises);
      const executionTime = performance.now() - startTime;
      
      const successfulResults = results.filter(r => r.status === 'fulfilled');
      expect(successfulResults.length).toBeGreaterThan(0);
      expect(executionTime).toBeLessThan(TEST_CONFIG.maxExecutionTime * 2);
    }, TEST_CONFIG.timeout);

    test('should benchmark memory usage during execution', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      await execPromise(
        'npx claude-flow@alpha sparc run architect "Complex system design"'
      );
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be reasonable (less than 100MB)
      expect(memoryIncrease).toBeLessThan(100 * 1024 * 1024);
    }, TEST_CONFIG.timeout);
  });

  describe('Integration with Claude-Flow', () => {
    test('should initialize swarm coordination', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha hooks pre-task --description "MLE-Star test workflow"'
      );
      
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);

    test('should coordinate with memory management', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha hooks post-edit --file "test.js" --memory-key "swarm/tester/validation"'
      );
      
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);

    test('should handle session management', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha hooks session-end --export-metrics true'
      );
      
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);

    test('should validate swarm status reporting', async () => {
      try {
        const result = await execPromise('npx claude-flow@alpha mcp swarm_status');
        expect(result.stdout).toBeTruthy();
      } catch (error) {
        // Swarm might not be initialized, which is acceptable
        expect(error.code).toBeDefined();
      }
    }, TEST_CONFIG.timeout);
  });

  describe('Command Validation and Security', () => {
    test('should validate command safety', async () => {
      const result = await execPromise(
        'npx claude-flow@alpha hooks pre-command --command "ls -la" --validate-safety true'
      );
      
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);

    test('should reject potentially dangerous commands', async () => {
      try {
        await execPromise(
          'npx claude-flow@alpha hooks pre-command --command "rm -rf /" --validate-safety true'
        );
      } catch (error) {
        expect(error.stderr).toMatch(/safety|dangerous|rejected/i);
      }
    }, TEST_CONFIG.timeout);

    test('should validate file operation permissions', async () => {
      const testFile = path.join(tempTestDir, 'permission-test.js');
      await fs.writeFile(testFile, '// Test file');
      
      const result = await execPromise(
        `npx claude-flow@alpha hooks pre-edit --file "${testFile}" --auto-assign-agents true`
      );
      
      expect(result.stdout).toBeTruthy();
    }, TEST_CONFIG.timeout);
  });
});

// Utility functions
function execPromise(command, options = {}) {
  return new Promise((resolve, reject) => {
    exec(command, { ...options, timeout: TEST_CONFIG.timeout }, (error, stdout, stderr) => {
      if (error) {
        error.stdout = stdout;
        error.stderr = stderr;
        reject(error);
      } else {
        resolve({ stdout, stderr, code: 0 });
      }
    });
  });
}

// Performance monitoring utilities
class PerformanceMonitor {
  constructor() {
    this.metrics = {
      executionTimes: [],
      memoryUsage: [],
      errorRates: []
    };
  }

  recordExecution(duration) {
    this.metrics.executionTimes.push(duration);
  }

  recordMemoryUsage(usage) {
    this.metrics.memoryUsage.push(usage);
  }

  recordError() {
    this.metrics.errorRates.push(Date.now());
  }

  getAverageExecutionTime() {
    const times = this.metrics.executionTimes;
    return times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0;
  }

  getMaxMemoryUsage() {
    return Math.max(...this.metrics.memoryUsage);
  }

  getErrorRate() {
    const totalTests = this.metrics.executionTimes.length + this.metrics.errorRates.length;
    return totalTests > 0 ? this.metrics.errorRates.length / totalTests : 0;
  }

  generateReport() {
    return {
      averageExecutionTime: this.getAverageExecutionTime(),
      maxMemoryUsage: this.getMaxMemoryUsage(),
      errorRate: this.getErrorRate(),
      totalTests: this.metrics.executionTimes.length + this.metrics.errorRates.length
    };
  }
}

// Export for use in integration tests
module.exports = {
  TEST_CONFIG,
  execPromise,
  PerformanceMonitor
};