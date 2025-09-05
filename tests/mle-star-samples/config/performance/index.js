/**
 * NovaCron Performance Benchmarking Framework
 * Main entry point for the comprehensive performance monitoring and optimization system
 */

// Core framework
const PerformanceBenchmarkingFramework = require('./performance-framework');

// System benchmarks
const {
  VMOperationsBenchmark,
  DatabaseBenchmark,
  NetworkStorageBenchmark,
  AutoScalingBenchmark,
  SystemBenchmark
} = require('./benchmarks/system-benchmarks');

// MLE-Star benchmarks
const {
  WorkflowExecutionBenchmark,
  ResourceUtilizationBenchmark,
  InferencePerformanceBenchmark,
  MultiFrameworkBenchmark,
  MLEStarBenchmark
} = require('./benchmarks/mle-star-benchmarks');

// Resource optimizers
const {
  ResourceOptimizer,
  CacheOptimizer,
  MemoryOptimizer,
  CPUOptimizer,
  NetworkOptimizer,
  StorageOptimizer
} = require('./optimizers/resource-optimizers');

// Automated runners
const AutomatedBenchmarkRunner = require('./runners/automated-runners');

// Monitoring and analytics
const PerformanceMonitoringDashboard = require('./dashboards/monitoring-dashboard');
const {
  PerformanceMetricsCollector,
  SystemMetricsCollector,
  ApplicationMetricsCollector,
  DatabaseMetricsCollector,
  NetworkMetricsCollector,
  MLWorkflowMetricsCollector,
  CustomMetricsCollector
} = require('./monitors/metrics-collector');

// Analysis and recommendations
const {
  HistoricalTrendAnalyzer,
  TrendModel,
  SystemTrendModel,
  ApplicationTrendModel,
  DatabaseTrendModel,
  NetworkTrendModel,
  MLWorkflowTrendModel
} = require('./analyzers/trend-analyzer');

const {
  PerformanceRecommendationEngine,
  RecommendationRule,
  PerformanceKnowledgeBase,
  RecommendationLearningEngine
} = require('./analyzers/recommendation-engine');

// Environment profiles
const {
  EnvironmentProfileManager,
  EnvironmentProfile,
  DevelopmentProfile,
  TestingProfile,
  StagingProfile,
  ProductionProfile,
  HPCProfile,
  CloudProfile,
  EdgeProfile,
  MLTrainingProfile,
  MLInferenceProfile,
  CustomProfile,
  ProfileOptimizer
} = require('./profiles/environment-profiles');

class NovaCronPerformanceSystem {
  constructor(config = {}) {
    this.config = {
      environment: config.environment || 'production',
      autoStart: config.autoStart !== false,
      profilePath: config.profilePath || './profiles',
      ...config
    };

    // Core components
    this.framework = null;
    this.metricsCollector = null;
    this.dashboard = null;
    this.runner = null;
    this.trendAnalyzer = null;
    this.recommendationEngine = null;
    this.profileManager = null;
    this.currentProfile = null;

    // Component states
    this.isInitialized = false;
    this.isRunning = false;
  }

  async initialize() {
    if (this.isInitialized) {
      console.log('Performance system already initialized');
      return;
    }

    console.log('Initializing NovaCron Performance System...');

    try {
      // Initialize profile manager and load environment profile
      this.profileManager = new EnvironmentProfileManager({
        profilesPath: this.config.profilePath,
        defaultProfile: this.config.environment
      });

      const profileConfig = await this.profileManager.loadProfile(this.config.environment);
      this.currentProfile = profileConfig;

      // Initialize core framework with profile configuration
      this.framework = new PerformanceBenchmarkingFramework({
        ...this.config,
        ...profileConfig.framework
      });

      // Initialize metrics collector
      this.metricsCollector = new PerformanceMetricsCollector({
        ...profileConfig.monitoring,
        storageLocation: './performance-data/metrics'
      });

      // Initialize trend analyzer
      this.trendAnalyzer = new HistoricalTrendAnalyzer(this.metricsCollector, {
        ...profileConfig.monitoring,
        storageLocation: './performance-data/trends'
      });

      // Initialize recommendation engine
      this.recommendationEngine = new PerformanceRecommendationEngine({
        storageLocation: './performance-data/recommendations',
        ...profileConfig.alerts
      });

      // Initialize monitoring dashboard
      this.dashboard = new PerformanceMonitoringDashboard(this.framework, {
        port: 8080,
        wsPort: 8081,
        ...profileConfig.monitoring
      });

      // Initialize automated runner
      this.runner = new AutomatedBenchmarkRunner(this.framework, {
        scheduledRuns: true,
        autoOptimization: profileConfig.optimizers ? true : false,
        outputPath: './performance-data/benchmarks',
        alertThresholds: profileConfig.alerts?.thresholds || {}
      });

      this.isInitialized = true;
      console.log(`Performance system initialized with profile: ${this.config.environment}`);

      // Auto-start if configured
      if (this.config.autoStart) {
        await this.start();
      }

    } catch (error) {
      console.error('Error initializing performance system:', error);
      throw error;
    }
  }

  async start() {
    if (!this.isInitialized) {
      await this.initialize();
    }

    if (this.isRunning) {
      console.log('Performance system already running');
      return;
    }

    console.log('Starting NovaCron Performance System...');

    try {
      // Start metrics collection
      await this.metricsCollector.start();

      // Start trend analysis
      await this.trendAnalyzer.start();

      // Start recommendation engine
      await this.recommendationEngine.start();

      // Start monitoring dashboard
      await this.dashboard.start();

      // Start automated benchmark runner
      await this.runner.start();

      this.isRunning = true;
      console.log('Performance system started successfully');

      // Set up event handlers
      this.setupEventHandlers();

    } catch (error) {
      console.error('Error starting performance system:', error);
      throw error;
    }
  }

  async stop() {
    if (!this.isRunning) {
      console.log('Performance system not running');
      return;
    }

    console.log('Stopping NovaCron Performance System...');

    try {
      // Stop components in reverse order
      if (this.runner) await this.runner.stop();
      if (this.dashboard) await this.dashboard.stop();
      if (this.recommendationEngine) await this.recommendationEngine.stop();
      if (this.trendAnalyzer) await this.trendAnalyzer.stop();
      if (this.metricsCollector) await this.metricsCollector.stop();

      this.isRunning = false;
      console.log('Performance system stopped');

    } catch (error) {
      console.error('Error stopping performance system:', error);
      throw error;
    }
  }

  setupEventHandlers() {
    // Metrics collection events
    this.metricsCollector.on('metrics:collected', (metrics) => {
      // Forward to dashboard for real-time updates
      if (this.dashboard) {
        this.dashboard.emit('metrics:update', metrics);
      }
    });

    // Trend analysis events
    this.trendAnalyzer.on('anomalies:detected', (anomalies) => {
      console.log(`Detected ${anomalies.length} performance anomalies`);
      // Could trigger alerts or notifications here
    });

    this.trendAnalyzer.on('predictions:generated', (predictions) => {
      console.log('Performance predictions updated');
      // Could trigger proactive optimizations here
    });

    // Recommendation engine events
    this.recommendationEngine.on('recommendations:generated', (data) => {
      console.log(`Generated ${data.count} new performance recommendations`);
    });

    // Automated runner events
    this.runner.on('benchmark:completed', (result) => {
      console.log(`Completed benchmark: ${result.name}`);
    });

    this.runner.on('optimization:completed', (result) => {
      console.log(`Completed optimization: ${result.type}`);
    });

    // Dashboard events
    this.dashboard.on('dashboard:started', (info) => {
      console.log(`Performance dashboard available at http://localhost:${info.port}`);
    });
  }

  // Public API methods

  async runBenchmark(benchmarkName, config = {}) {
    if (!this.isInitialized) {
      throw new Error('Performance system not initialized');
    }

    return await this.framework.runBenchmark(benchmarkName, config);
  }

  async runBenchmarkSuite(suiteConfig) {
    if (!this.isInitialized) {
      throw new Error('Performance system not initialized');
    }

    return await this.framework.runBenchmarkSuite(suiteConfig);
  }

  async getMetrics(query = {}) {
    if (!this.metricsCollector) {
      throw new Error('Metrics collector not initialized');
    }

    return await this.metricsCollector.getMetrics(query);
  }

  async getTrendSummary(timeRange = '24h') {
    if (!this.trendAnalyzer) {
      throw new Error('Trend analyzer not initialized');
    }

    return await this.trendAnalyzer.getTrendSummary(timeRange);
  }

  async getRecommendations(filter = {}) {
    if (!this.recommendationEngine) {
      throw new Error('Recommendation engine not initialized');
    }

    return this.recommendationEngine.getRecommendations(filter);
  }

  async switchProfile(profileName) {
    console.log(`Switching to profile: ${profileName}`);

    // Load new profile
    const profileConfig = await this.profileManager.loadProfile(profileName);
    this.currentProfile = profileConfig;

    // Reconfigure components if running
    if (this.isRunning) {
      // Stop current components
      await this.stop();

      // Update configurations
      this.config.environment = profileName;

      // Restart with new profile
      await this.start();
    }

    console.log(`Switched to profile: ${profileName}`);
  }

  async createCustomProfile(name, baseProfile, customizations) {
    if (!this.profileManager) {
      throw new Error('Profile manager not initialized');
    }

    const baseConfig = await this.profileManager.loadProfile(baseProfile);
    const customConfig = { ...baseConfig, ...customizations };

    await this.profileManager.saveCustomProfile(name, customConfig);
    console.log(`Created custom profile: ${name}`);
  }

  getStatus() {
    return {
      initialized: this.isInitialized,
      running: this.isRunning,
      currentProfile: this.config.environment,
      components: {
        framework: !!this.framework,
        metricsCollector: this.metricsCollector?.getStatus(),
        dashboard: this.dashboard?.getStatus(),
        runner: this.runner?.getStatus(),
        trendAnalyzer: this.trendAnalyzer?.getStatus(),
        recommendationEngine: this.recommendationEngine?.getStatus()
      }
    };
  }

  // Static factory methods for quick setup

  static async createDevelopmentSystem(config = {}) {
    const system = new NovaCronPerformanceSystem({
      environment: 'development',
      ...config
    });

    await system.initialize();
    return system;
  }

  static async createProductionSystem(config = {}) {
    const system = new NovaCronPerformanceSystem({
      environment: 'production',
      ...config
    });

    await system.initialize();
    return system;
  }

  static async createMLTrainingSystem(config = {}) {
    const system = new NovaCronPerformanceSystem({
      environment: 'ml-training',
      ...config
    });

    await system.initialize();
    return system;
  }

  static async createCloudSystem(config = {}) {
    const system = new NovaCronPerformanceSystem({
      environment: 'cloud',
      ...config
    });

    await system.initialize();
    return system;
  }
}

// Export everything for modular usage
module.exports = {
  // Main system
  NovaCronPerformanceSystem,

  // Core framework
  PerformanceBenchmarkingFramework,

  // System benchmarks
  VMOperationsBenchmark,
  DatabaseBenchmark,
  NetworkStorageBenchmark,
  AutoScalingBenchmark,
  SystemBenchmark,

  // MLE-Star benchmarks
  WorkflowExecutionBenchmark,
  ResourceUtilizationBenchmark,
  InferencePerformanceBenchmark,
  MultiFrameworkBenchmark,
  MLEStarBenchmark,

  // Resource optimizers
  ResourceOptimizer,
  CacheOptimizer,
  MemoryOptimizer,
  CPUOptimizer,
  NetworkOptimizer,
  StorageOptimizer,

  // Automated systems
  AutomatedBenchmarkRunner,
  PerformanceMonitoringDashboard,

  // Metrics and monitoring
  PerformanceMetricsCollector,
  SystemMetricsCollector,
  ApplicationMetricsCollector,
  DatabaseMetricsCollector,
  NetworkMetricsCollector,
  MLWorkflowMetricsCollector,
  CustomMetricsCollector,

  // Analysis and recommendations
  HistoricalTrendAnalyzer,
  PerformanceRecommendationEngine,
  TrendModel,
  RecommendationRule,
  PerformanceKnowledgeBase,

  // Environment profiles
  EnvironmentProfileManager,
  EnvironmentProfile,
  DevelopmentProfile,
  TestingProfile,
  StagingProfile,
  ProductionProfile,
  HPCProfile,
  CloudProfile,
  EdgeProfile,
  MLTrainingProfile,
  MLInferenceProfile,
  CustomProfile,
  ProfileOptimizer
};