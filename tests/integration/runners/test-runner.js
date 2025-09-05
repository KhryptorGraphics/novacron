/**
 * Integration Test Runner
 * 
 * Orchestrates the execution of all integration tests with features like:
 * - Test suite selection and filtering
 * - Parallel and sequential execution
 * - Comprehensive reporting
 * - Environment management
 * - Performance monitoring
 * - Failure analysis
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;
const os = require('os');

// Test utilities
const TestEnvironment = require('../utils/test-environment');
const ReportGenerator = require('../utils/report-generator');
const MetricsCollector = require('../utils/metrics-collector');

class IntegrationTestRunner {
  constructor(config = {}) {
    this.config = {
      // Test execution configuration
      testDir: config.testDir || path.join(__dirname, '..'),
      parallel: config.parallel !== undefined ? config.parallel : true,
      maxWorkers: config.maxWorkers || Math.min(os.cpus().length, 4),
      timeout: config.timeout || 600000, // 10 minutes
      
      // Test filtering
      testPattern: config.testPattern || '**/*.integration.test.js',
      excludePattern: config.excludePattern || '**/node_modules/**',
      categories: config.categories || ['system', 'mle-star', 'performance'],
      
      // Reporting
      generateReports: config.generateReports !== undefined ? config.generateReports : true,
      reportDir: config.reportDir || path.join(__dirname, '../reports'),
      reportFormats: config.reportFormats || ['html', 'json', 'junit'],
      
      // Environment
      setupEnvironment: config.setupEnvironment !== undefined ? config.setupEnvironment : true,
      cleanupAfterTests: config.cleanupAfterTests !== undefined ? config.cleanupAfterTests : true,
      
      // Performance monitoring
      monitorResources: config.monitorResources !== undefined ? config.monitorResources : true,
      collectMetrics: config.collectMetrics !== undefined ? config.collectMetrics : true,
      
      ...config
    };
    
    this.testEnvironment = null;
    this.metricsCollector = new MetricsCollector();
    this.reportGenerator = new ReportGenerator(this.config.reportDir);
    this.testResults = [];
    this.startTime = null;
  }

  /**
   * Run all integration tests
   */
  async runAll() {
    console.log('🚀 Starting Integration Test Suite');
    console.log('📋 Configuration:', JSON.stringify(this.config, null, 2));
    
    this.startTime = Date.now();
    
    try {
      // Setup test environment
      if (this.config.setupEnvironment) {
        await this.setupEnvironment();
      }
      
      // Discover and run tests
      const testSuites = await this.discoverTestSuites();
      await this.runTestSuites(testSuites);
      
      // Generate reports
      if (this.config.generateReports) {
        await this.generateReports();
      }
      
      // Print summary
      this.printSummary();
      
      const success = this.testResults.every(result => result.success);
      return { success, results: this.testResults };
      
    } catch (error) {
      console.error('❌ Integration test runner failed:', error);
      throw error;
    } finally {
      if (this.config.cleanupAfterTests) {
        await this.cleanupEnvironment();
      }
    }
  }

  /**
   * Run specific test categories
   */
  async runCategories(categories) {
    console.log(`🎯 Running test categories: ${categories.join(', ')}`);\n    
    const filteredConfig = {\n      ...this.config,\n      categories: categories\n    };\n    \n    const runner = new IntegrationTestRunner(filteredConfig);\n    return runner.runAll();\n  }\n\n  /**\n   * Run specific test files\n   */\n  async runFiles(testFiles) {\n    console.log(`📁 Running specific test files: ${testFiles.join(', ')}`);\n    \n    const filteredConfig = {\n      ...this.config,\n      testPattern: testFiles.join(','),\n      setupEnvironment: true\n    };\n    \n    const runner = new IntegrationTestRunner(filteredConfig);\n    return runner.runAll();\n  }\n\n  /**\n   * Setup test environment\n   */\n  async setupEnvironment() {\n    console.log('🏗️ Setting up test environment...');\n    \n    this.testEnvironment = new TestEnvironment({\n      useDocker: process.env.NOVACRON_USE_DOCKER === 'true',\n      logLevel: process.env.NOVACRON_LOG_LEVEL || 'info'\n    });\n    \n    await this.testEnvironment.setup();\n    console.log('✅ Test environment ready');\n  }\n\n  /**\n   * Discover test suites\n   */\n  async discoverTestSuites() {\n    console.log('🔍 Discovering test suites...');\n    \n    const testSuites = [];\n    \n    for (const category of this.config.categories) {\n      const categoryPath = path.join(this.config.testDir, category);\n      \n      try {\n        await fs.access(categoryPath);\n        const suites = await this.findTestFiles(categoryPath);\n        testSuites.push(...suites.map(suite => ({ ...suite, category })));\n      } catch (error) {\n        console.warn(`⚠️ Category not found: ${category}`);\n      }\n    }\n    \n    console.log(`📊 Found ${testSuites.length} test suites`);\n    testSuites.forEach(suite => {\n      console.log(`  - ${suite.category}/${suite.name}`);\n    });\n    \n    return testSuites;\n  }\n\n  /**\n   * Find test files in directory\n   */\n  async findTestFiles(directory) {\n    const testFiles = [];\n    \n    const files = await fs.readdir(directory, { withFileTypes: true });\n    \n    for (const file of files) {\n      const fullPath = path.join(directory, file.name);\n      \n      if (file.isDirectory()) {\n        const subFiles = await this.findTestFiles(fullPath);\n        testFiles.push(...subFiles);\n      } else if (file.name.endsWith('.integration.test.js')) {\n        testFiles.push({\n          name: file.name,\n          path: fullPath,\n          relativePath: path.relative(this.config.testDir, fullPath)\n        });\n      }\n    }\n    \n    return testFiles;\n  }\n\n  /**\n   * Run test suites\n   */\n  async runTestSuites(testSuites) {\n    console.log('🧪 Running test suites...');\n    \n    if (this.config.parallel) {\n      await this.runTestSuitesParallel(testSuites);\n    } else {\n      await this.runTestSuitesSequential(testSuites);\n    }\n  }\n\n  /**\n   * Run test suites in parallel\n   */\n  async runTestSuitesParallel(testSuites) {\n    console.log(`⚡ Running ${testSuites.length} test suites in parallel (max ${this.config.maxWorkers} workers)`);\n    \n    const workers = [];\n    const results = [];\n    \n    // Group test suites into batches\n    const batches = this.groupIntoBatches(testSuites, this.config.maxWorkers);\n    \n    for (const batch of batches) {\n      const batchPromises = batch.map(suite => this.runSingleTestSuite(suite));\n      const batchResults = await Promise.allSettled(batchPromises);\n      \n      batchResults.forEach((result, index) => {\n        if (result.status === 'fulfilled') {\n          results.push(result.value);\n        } else {\n          results.push({\n            suite: batch[index],\n            success: false,\n            error: result.reason.message,\n            duration: 0,\n            tests: []\n          });\n        }\n      });\n    }\n    \n    this.testResults = results;\n  }\n\n  /**\n   * Run test suites sequentially\n   */\n  async runTestSuitesSequential(testSuites) {\n    console.log(`🔄 Running ${testSuites.length} test suites sequentially`);\n    \n    const results = [];\n    \n    for (const suite of testSuites) {\n      try {\n        const result = await this.runSingleTestSuite(suite);\n        results.push(result);\n      } catch (error) {\n        results.push({\n          suite,\n          success: false,\n          error: error.message,\n          duration: 0,\n          tests: []\n        });\n      }\n    }\n    \n    this.testResults = results;\n  }\n\n  /**\n   * Run a single test suite\n   */\n  async runSingleTestSuite(suite) {\n    console.log(`🧪 Running ${suite.category}/${suite.name}...`);\n    \n    const startTime = Date.now();\n    \n    try {\n      const result = await this.executeJestTest(suite);\n      const duration = Date.now() - startTime;\n      \n      console.log(`${result.success ? '✅' : '❌'} ${suite.category}/${suite.name} (${duration}ms)`);\n      \n      return {\n        suite,\n        success: result.success,\n        duration,\n        tests: result.tests,\n        coverage: result.coverage,\n        output: result.output\n      };\n    } catch (error) {\n      const duration = Date.now() - startTime;\n      console.error(`❌ ${suite.category}/${suite.name} failed: ${error.message}`);\n      \n      return {\n        suite,\n        success: false,\n        duration,\n        error: error.message,\n        tests: []\n      };\n    }\n  }\n\n  /**\n   * Execute Jest test\n   */\n  async executeJestTest(suite) {\n    return new Promise((resolve, reject) => {\n      const jestArgs = [\n        suite.path,\n        '--verbose',\n        '--json',\n        `--testTimeout=${this.config.timeout}`,\n        '--forceExit',\n        '--detectOpenHandles'\n      ];\n      \n      if (process.env.NOVACRON_TEST_COVERAGE === 'true') {\n        jestArgs.push('--coverage');\n      }\n      \n      const jest = spawn('npx', ['jest', ...jestArgs], {\n        stdio: ['ignore', 'pipe', 'pipe'],\n        env: {\n          ...process.env,\n          NODE_ENV: 'test',\n          NOVACRON_TEST_MODE: 'integration'\n        }\n      });\n      \n      let stdout = '';\n      let stderr = '';\n      \n      jest.stdout.on('data', (data) => {\n        stdout += data.toString();\n      });\n      \n      jest.stderr.on('data', (data) => {\n        stderr += data.toString();\n      });\n      \n      jest.on('close', (code) => {\n        try {\n          // Parse Jest JSON output\n          const lines = stdout.split('\\n');\n          const jsonLine = lines.find(line => {\n            try {\n              const parsed = JSON.parse(line);\n              return parsed.testResults !== undefined;\n            } catch {\n              return false;\n            }\n          });\n          \n          if (jsonLine) {\n            const jestResult = JSON.parse(jsonLine);\n            resolve({\n              success: jestResult.success,\n              tests: this.parseJestTests(jestResult),\n              coverage: jestResult.coverageMap,\n              output: stdout + stderr\n            });\n          } else {\n            resolve({\n              success: code === 0,\n              tests: [],\n              output: stdout + stderr\n            });\n          }\n        } catch (error) {\n          reject(new Error(`Failed to parse Jest output: ${error.message}`));\n        }\n      });\n      \n      jest.on('error', (error) => {\n        reject(error);\n      });\n    });\n  }\n\n  /**\n   * Parse Jest test results\n   */\n  parseJestTests(jestResult) {\n    const tests = [];\n    \n    for (const testFile of jestResult.testResults) {\n      for (const testResult of testFile.assertionResults) {\n        tests.push({\n          name: testResult.fullName || testResult.title,\n          status: testResult.status,\n          duration: testResult.duration,\n          failureMessages: testResult.failureMessages || [],\n          location: testResult.location\n        });\n      }\n    }\n    \n    return tests;\n  }\n\n  /**\n   * Group items into batches\n   */\n  groupIntoBatches(items, batchSize) {\n    const batches = [];\n    \n    for (let i = 0; i < items.length; i += batchSize) {\n      batches.push(items.slice(i, i + batchSize));\n    }\n    \n    return batches;\n  }\n\n  /**\n   * Generate reports\n   */\n  async generateReports() {\n    console.log('📊 Generating test reports...');\n    \n    await fs.mkdir(this.config.reportDir, { recursive: true });\n    \n    const reportData = {\n      summary: this.generateSummary(),\n      results: this.testResults,\n      metrics: this.metricsCollector.getMetrics(),\n      environment: {\n        node: process.version,\n        platform: process.platform,\n        arch: process.arch,\n        memory: process.memoryUsage(),\n        env: {\n          NODE_ENV: process.env.NODE_ENV,\n          NOVACRON_TEST_MODE: process.env.NOVACRON_TEST_MODE,\n          NOVACRON_USE_DOCKER: process.env.NOVACRON_USE_DOCKER\n        }\n      },\n      timestamp: new Date().toISOString(),\n      duration: Date.now() - this.startTime\n    };\n    \n    // Generate different report formats\n    for (const format of this.config.reportFormats) {\n      await this.reportGenerator.generate(format, reportData);\n    }\n    \n    console.log(`✅ Reports generated in: ${this.config.reportDir}`);\n  }\n\n  /**\n   * Generate test summary\n   */\n  generateSummary() {\n    const totalSuites = this.testResults.length;\n    const successfulSuites = this.testResults.filter(r => r.success).length;\n    const failedSuites = totalSuites - successfulSuites;\n    \n    const allTests = this.testResults.flatMap(r => r.tests || []);\n    const totalTests = allTests.length;\n    const passedTests = allTests.filter(t => t.status === 'passed').length;\n    const failedTests = allTests.filter(t => t.status === 'failed').length;\n    const skippedTests = allTests.filter(t => t.status === 'skipped').length;\n    \n    const totalDuration = this.testResults.reduce((sum, r) => sum + r.duration, 0);\n    \n    return {\n      suites: {\n        total: totalSuites,\n        successful: successfulSuites,\n        failed: failedSuites,\n        successRate: totalSuites > 0 ? (successfulSuites / totalSuites) : 0\n      },\n      tests: {\n        total: totalTests,\n        passed: passedTests,\n        failed: failedTests,\n        skipped: skippedTests,\n        passRate: totalTests > 0 ? (passedTests / totalTests) : 0\n      },\n      duration: {\n        total: totalDuration,\n        average: totalSuites > 0 ? (totalDuration / totalSuites) : 0\n      }\n    };\n  }\n\n  /**\n   * Print test summary\n   */\n  printSummary() {\n    const summary = this.generateSummary();\n    const duration = Date.now() - this.startTime;\n    \n    console.log('\\n📊 Integration Test Summary');\n    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');\n    console.log(`⏱️  Total Duration: ${(duration / 1000).toFixed(2)}s`);\n    console.log(`📦 Test Suites: ${summary.suites.successful}/${summary.suites.total} passed (${(summary.suites.successRate * 100).toFixed(1)}%)`);\n    console.log(`🧪 Tests: ${summary.tests.passed}/${summary.tests.total} passed (${(summary.tests.passRate * 100).toFixed(1)}%)`);\n    console.log(`⏭️  Skipped: ${summary.tests.skipped}`);\n    \n    if (summary.suites.failed > 0) {\n      console.log('\\n❌ Failed Test Suites:');\n      this.testResults.filter(r => !r.success).forEach(result => {\n        console.log(`   - ${result.suite.category}/${result.suite.name}: ${result.error || 'Unknown error'}`);\n      });\n    }\n    \n    if (summary.tests.failed > 0) {\n      console.log('\\n❌ Failed Tests:');\n      this.testResults.forEach(result => {\n        const failedTests = (result.tests || []).filter(t => t.status === 'failed');\n        failedTests.forEach(test => {\n          console.log(`   - ${result.suite.category}/${result.suite.name}: ${test.name}`);\n          if (test.failureMessages.length > 0) {\n            console.log(`     ${test.failureMessages[0].split('\\n')[0]}`);\n          }\n        });\n      });\n    }\n    \n    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');\n    \n    const success = summary.suites.successRate === 1 && summary.tests.passRate === 1;\n    console.log(success ? '✅ All tests passed!' : '❌ Some tests failed.');\n  }\n\n  /**\n   * Cleanup test environment\n   */\n  async cleanupEnvironment() {\n    if (this.testEnvironment) {\n      console.log('🧹 Cleaning up test environment...');\n      await this.testEnvironment.cleanup();\n      console.log('✅ Test environment cleaned up');\n    }\n  }\n}\n\nmodule.exports = IntegrationTestRunner;