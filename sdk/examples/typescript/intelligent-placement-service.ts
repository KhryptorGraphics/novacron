#!/usr/bin/env node
/**
 * Intelligent VM Placement Service Example
 * 
 * This TypeScript example demonstrates:
 * - AI-powered intelligent placement decisions
 * - Real-time resource optimization
 * - Predictive scaling based on workload patterns
 * - Multi-cloud cost optimization
 * - Event-driven architecture with WebSocket streaming
 * - Comprehensive error handling and circuit breaker patterns
 */

import { 
  EnhancedNovaCronClient, 
  CloudProvider, 
  AIFeature,
  VMSpec,
  MigrationSpec,
  PlacementRecommendation 
} from '../src/enhanced-client';
import EventEmitter from 'events';
import { createLogger, Logger } from 'winston';

// Configure structured logging
const logger: Logger = createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ filename: 'placement-service.log' })
  ]
});

interface WorkloadPattern {
  id: string;
  name: string;
  cpuPattern: number[];
  memoryPattern: number[];
  predictedPeaks: Date[];
  seasonality: 'hourly' | 'daily' | 'weekly' | 'monthly';
}

interface PlacementRequest {
  id: string;
  workloadType: string;
  vmSpecs: VMSpec;
  constraints: {
    maxLatency?: number;
    complianceRegions?: string[];
    budgetLimit?: number;
    performance?: 'low' | 'medium' | 'high';
    redundancy?: boolean;
  };
  priority: 'low' | 'medium' | 'high' | 'critical';
  requestedAt: Date;
}

interface OptimizationMetrics {
  costSavings: number;
  performanceGain: number;
  latencyReduction: number;
  resourceUtilization: number;
  placementAccuracy: number;
}

class IntelligentPlacementService extends EventEmitter {
  private client: EnhancedNovaCronClient;
  private workloadPatterns: Map<string, WorkloadPattern> = new Map();
  private activeRequests: Map<string, PlacementRequest> = new Map();
  private placementHistory: Array<{
    request: PlacementRequest;
    recommendation: PlacementRecommendation;
    outcome: 'success' | 'failure' | 'partial';
    actualMetrics?: any;
  }> = [];

  constructor() {
    super();
    
    this.client = new EnhancedNovaCronClient({
      baseURL: process.env.NOVACRON_API_URL || 'https://api.novacron.io',
      apiToken: process.env.NOVACRON_API_TOKEN,
      enableAIFeatures: true,
      cloudProvider: CloudProvider.LOCAL,
      region: process.env.NOVACRON_REGION || 'us-west-2',
      redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
      cacheTTL: 300,
      enableMetrics: true,
      circuitBreakerThreshold: 5,
      circuitBreakerTimeout: 60000
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    // Handle placement requests
    this.on('placement-request', this.handlePlacementRequest.bind(this));
    
    // Handle workload pattern updates
    this.on('workload-pattern-update', this.updateWorkloadPattern.bind(this));
    
    // Handle optimization triggers
    this.on('optimization-trigger', this.triggerOptimization.bind(this));
    
    // Handle scaling events
    this.on('scaling-event', this.handleScalingEvent.bind(this));
  }

  async initialize(): Promise<void> {
    logger.info('Initializing Intelligent Placement Service...');

    try {
      // Test API connectivity
      const health = await this.client.healthCheck();
      logger.info('API health check passed', { status: health.status });

      // Load historical workload patterns
      await this.loadWorkloadPatterns();

      // Start real-time monitoring
      await this.startRealTimeMonitoring();

      // Start periodic optimization
      this.startPeriodicOptimization();

      logger.info('Intelligent Placement Service initialized successfully');
      this.emit('service-ready');

    } catch (error) {
      logger.error('Failed to initialize service', { error: error.message });
      throw error;
    }
  }

  private async loadWorkloadPatterns(): Promise<void> {
    logger.info('Loading historical workload patterns...');

    // In production, this would load from a database
    // For demo, we'll create some sample patterns
    const samplePatterns: WorkloadPattern[] = [
      {
        id: 'web-traffic',
        name: 'Web Application Traffic',
        cpuPattern: [20, 15, 10, 8, 12, 25, 45, 65, 80, 85, 75, 70, 60, 55, 50, 45, 55, 65, 70, 60, 45, 35, 30, 25],
        memoryPattern: [30, 28, 25, 22, 25, 35, 50, 70, 85, 90, 85, 80, 70, 65, 60, 55, 65, 75, 80, 70, 55, 45, 40, 35],
        predictedPeaks: [
          new Date(Date.now() + 2 * 60 * 60 * 1000), // 2 hours from now
          new Date(Date.now() + 26 * 60 * 60 * 1000)  // Tomorrow same time
        ],
        seasonality: 'daily'
      },
      {
        id: 'batch-processing',
        name: 'Batch Processing Jobs',
        cpuPattern: [5, 5, 5, 5, 5, 5, 95, 95, 95, 95, 10, 10, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        memoryPattern: [10, 10, 10, 10, 10, 10, 85, 90, 95, 90, 15, 15, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        predictedPeaks: [
          new Date(Date.now() + 22 * 60 * 60 * 1000) // Tomorrow 6 AM
        ],
        seasonality: 'daily'
      },
      {
        id: 'ml-training',
        name: 'Machine Learning Training',
        cpuPattern: [90, 90, 90, 90, 90, 90, 90, 90, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        memoryPattern: [95, 95, 95, 95, 95, 95, 95, 95, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
        predictedPeaks: [
          new Date(Date.now() + 12 * 60 * 60 * 1000) // Tonight midnight
        ],
        seasonality: 'weekly'
      }
    ];

    for (const pattern of samplePatterns) {
      this.workloadPatterns.set(pattern.id, pattern);
      logger.debug('Loaded workload pattern', { patternId: pattern.id, name: pattern.name });
    }

    logger.info(`Loaded ${this.workloadPatterns.size} workload patterns`);
  }

  private async startRealTimeMonitoring(): Promise<void> {
    logger.info('Starting real-time monitoring...');

    try {
      // Stream federated events for placement optimization
      const eventStream = this.client.streamFederatedEvents(
        ['vm.created', 'vm.failed', 'vm.performance_degraded', 'cluster.capacity_changed'],
        [CloudProvider.AWS, CloudProvider.GCP, CloudProvider.AZURE]
      );

      eventStream.on('connected', () => {
        logger.info('Real-time event stream connected');
      });

      eventStream.on('event', (event) => {
        this.handleRealtimeEvent(event);
      });

      eventStream.on('error', (error) => {
        logger.error('Event stream error', { error: error.message });
        // Implement reconnection logic
        setTimeout(() => this.startRealTimeMonitoring(), 5000);
      });

      eventStream.on('disconnected', () => {
        logger.warn('Event stream disconnected, attempting reconnect...');
        setTimeout(() => this.startRealTimeMonitoring(), 1000);
      });

    } catch (error) {
      logger.error('Failed to start real-time monitoring', { error: error.message });
      throw error;
    }
  }

  private handleRealtimeEvent(event: any): void {
    const { type, data, cloud_provider, timestamp } = event;

    logger.debug('Received real-time event', { 
      type, 
      cloudProvider: cloud_provider, 
      timestamp 
    });

    switch (type) {
      case 'vm.performance_degraded':
        this.handlePerformanceDegradation(data);
        break;
      
      case 'vm.failed':
        this.handleVMFailure(data);
        break;
      
      case 'cluster.capacity_changed':
        this.handleCapacityChange(data);
        break;
      
      case 'vm.created':
        this.updatePlacementOutcome(data);
        break;
    }
  }

  private async handlePerformanceDegradation(data: any): Promise<void> {
    const { vm_id, metrics } = data;
    
    logger.warn('Performance degradation detected', { 
      vmId: vm_id, 
      cpuUsage: metrics.cpu_usage,
      memoryUsage: metrics.memory_usage 
    });

    // Check if this VM can be optimized through migration
    const optimizationRecommendation = await this.analyzeVMOptimization(vm_id, metrics);
    
    if (optimizationRecommendation.shouldMigrate) {
      logger.info('Triggering automatic optimization', { 
        vmId: vm_id,
        recommendation: optimizationRecommendation 
      });
      
      this.emit('optimization-trigger', {
        type: 'performance_optimization',
        vmId: vm_id,
        recommendation: optimizationRecommendation
      });
    }
  }

  private async handleVMFailure(data: any): Promise<void> {
    const { vm_id, failure_reason, tags } = data;
    
    logger.error('VM failure detected', { vmId: vm_id, reason: failure_reason });

    // If this was an AI-placed VM, learn from the failure
    if (tags && tags.ai_placed === 'true') {
      await this.updatePlacementModel(vm_id, 'failure', failure_reason);
    }

    // Check if automatic replacement should be triggered
    if (tags && tags.auto_replace === 'true') {
      const replacementRequest: PlacementRequest = {
        id: `replacement-${vm_id}-${Date.now()}`,
        workloadType: tags.workload_type || 'unknown',
        vmSpecs: {
          name: `${tags.original_name}-replacement`,
          cpu_shares: parseInt(tags.cpu_shares) || 1024,
          memory_mb: parseInt(tags.memory_mb) || 2048,
          disk_size_gb: parseInt(tags.disk_size_gb) || 20
        },
        constraints: {
          // Avoid the failed node/region
          complianceRegions: tags.compliance_regions?.split(','),
          performance: tags.performance_level || 'medium'
        },
        priority: 'high',
        requestedAt: new Date()
      };

      this.emit('placement-request', replacementRequest);
    }
  }

  private async handleCapacityChange(data: any): Promise<void> {
    const { cluster_id, provider, region, available_capacity } = data;
    
    logger.info('Cluster capacity changed', { 
      clusterId: cluster_id, 
      provider, 
      region, 
      availableCapacity: available_capacity 
    });

    // Update internal cluster capacity cache
    // This would trigger re-evaluation of pending placement requests
    
    // Check if any pending requests can now be fulfilled
    await this.processPendingRequests();
  }

  private async updatePlacementOutcome(data: any): Promise<void> {
    const { vm_id, placement_info } = data;
    
    // Find corresponding placement request
    const requestId = placement_info?.request_id;
    if (requestId && this.activeRequests.has(requestId)) {
      const request = this.activeRequests.get(requestId)!;
      this.activeRequests.delete(requestId);

      // Record successful placement
      this.placementHistory.push({
        request,
        recommendation: placement_info.recommendation,
        outcome: 'success'
      });

      logger.info('Placement request fulfilled', { 
        requestId, 
        vmId: vm_id,
        targetNode: placement_info.target_node 
      });
    }
  }

  public async requestPlacement(request: PlacementRequest): Promise<string> {
    logger.info('Received placement request', { 
      requestId: request.id, 
      workloadType: request.workloadType,
      priority: request.priority 
    });

    // Store active request
    this.activeRequests.set(request.id, request);

    // Emit for processing
    this.emit('placement-request', request);

    return request.id;
  }

  private async handlePlacementRequest(request: PlacementRequest): Promise<void> {
    try {
      // Get workload pattern if available
      const workloadPattern = this.workloadPatterns.get(request.workloadType);
      
      // Prepare placement constraints with workload-aware optimizations
      const constraints = {
        ...request.constraints,
        workload_pattern: workloadPattern?.id,
        predicted_peaks: workloadPattern?.predictedPeaks,
        priority: request.priority
      };

      // Get AI-powered placement recommendation
      const recommendation = await this.client.getIntelligentPlacementRecommendation(
        request.vmSpecs,
        constraints
      );

      logger.info('AI placement recommendation received', {
        requestId: request.id,
        recommendedNode: recommendation.recommended_node,
        confidence: recommendation.confidence_score,
        reasoning: recommendation.reasoning
      });

      // Validate placement recommendation
      const validationResult = await this.validatePlacement(request, recommendation);
      
      if (!validationResult.isValid) {
        logger.warn('Placement validation failed', { 
          requestId: request.id,
          reason: validationResult.reason 
        });

        // Try alternative placement
        if (recommendation.alternative_nodes.length > 0) {
          const alternative = recommendation.alternative_nodes[0];
          logger.info('Trying alternative placement', { 
            requestId: request.id, 
            alternativeNode: alternative.node_id 
          });
          // Use alternative...
        }
        return;
      }

      // Execute placement
      await this.executePlacement(request, recommendation);

      logger.info('Placement executed successfully', { requestId: request.id });

    } catch (error) {
      logger.error('Placement request failed', { 
        requestId: request.id, 
        error: error.message 
      });

      // Update request status
      this.activeRequests.delete(request.id);
      
      this.emit('placement-failed', {
        requestId: request.id,
        error: error.message
      });
    }
  }

  private async validatePlacement(
    request: PlacementRequest, 
    recommendation: PlacementRecommendation
  ): Promise<{ isValid: boolean; reason?: string }> {
    
    // Check budget constraints
    if (request.constraints.budgetLimit) {
      const estimatedCost = await this.estimatePlacementCost(
        request.vmSpecs, 
        recommendation.recommended_node
      );
      
      if (estimatedCost > request.constraints.budgetLimit) {
        return { 
          isValid: false, 
          reason: `Estimated cost $${estimatedCost} exceeds budget $${request.constraints.budgetLimit}` 
        };
      }
    }

    // Check compliance requirements
    if (request.constraints.complianceRegions) {
      const nodeRegion = await this.getNodeRegion(recommendation.recommended_node);
      
      if (!request.constraints.complianceRegions.includes(nodeRegion)) {
        return { 
          isValid: false, 
          reason: `Node region ${nodeRegion} not in compliance regions` 
        };
      }
    }

    // Check performance requirements
    if (request.constraints.performance === 'high') {
      const nodePerformanceLevel = await this.getNodePerformanceLevel(
        recommendation.recommended_node
      );
      
      if (nodePerformanceLevel < 8) { // Scale of 1-10
        return { 
          isValid: false, 
          reason: `Node performance level ${nodePerformanceLevel} below high performance requirement` 
        };
      }
    }

    return { isValid: true };
  }

  private async executePlacement(
    request: PlacementRequest,
    recommendation: PlacementRecommendation
  ): Promise<void> {
    
    // Create VM with AI placement
    const vmSpec = {
      ...request.vmSpecs,
      tags: {
        ...request.vmSpecs.tags,
        ai_placed: 'true',
        request_id: request.id,
        workload_type: request.workloadType,
        placement_reasoning: recommendation.reasoning,
        confidence_score: recommendation.confidence_score.toString(),
        auto_replace: 'true'
      }
    };

    const vm = await this.client.createVMWithAIPlacement(
      vmSpec,
      true,
      {
        preferred_node: recommendation.recommended_node,
        ...request.constraints
      }
    );

    // Start monitoring the new VM
    this.startVMMonitoring(vm.id, request.workloadType);
  }

  private async startVMMonitoring(vmId: string, workloadType: string): Promise<void> {
    // Set up predictive scaling if workload pattern exists
    const pattern = this.workloadPatterns.get(workloadType);
    
    if (pattern && pattern.predictedPeaks.length > 0) {
      logger.info('Setting up predictive scaling', { vmId, workloadType });
      
      // Get predictive scaling forecast
      try {
        const forecast = await this.client.getPredictiveScalingForecast(vmId, 24);
        
        // Schedule pre-scaling for predicted peaks
        for (const peak of pattern.predictedPeaks) {
          if (peak > new Date()) {
            this.schedulePreScaling(vmId, peak, forecast);
          }
        }
      } catch (error) {
        logger.warn('Failed to get predictive scaling forecast', { 
          vmId, 
          error: error.message 
        });
      }
    }
  }

  private schedulePreScaling(vmId: string, peakTime: Date, forecast: any): void {
    const preScaleTime = new Date(peakTime.getTime() - 10 * 60 * 1000); // 10 minutes before
    const delay = preScaleTime.getTime() - Date.now();
    
    if (delay > 0) {
      setTimeout(async () => {
        logger.info('Executing pre-scaling for predicted peak', { vmId, peakTime });
        
        try {
          // Scale up resources before peak
          await this.client.updateVM(vmId, {
            cpu_shares: Math.floor(forecast.recommended_cpu * 1.5),
            memory_mb: Math.floor(forecast.recommended_memory * 1.5)
          });
          
          // Schedule scale-down after peak
          const scaleDownDelay = 60 * 60 * 1000; // 1 hour after peak
          setTimeout(async () => {
            await this.client.updateVM(vmId, {
              cpu_shares: forecast.recommended_cpu,
              memory_mb: forecast.recommended_memory
            });
            logger.info('Scaled down after peak', { vmId });
          }, scaleDownDelay);
          
        } catch (error) {
          logger.error('Pre-scaling failed', { vmId, error: error.message });
        }
      }, delay);
      
      logger.info('Scheduled pre-scaling', { vmId, preScaleTime });
    }
  }

  private startPeriodicOptimization(): void {
    // Run optimization every 30 minutes
    setInterval(async () => {
      logger.info('Running periodic optimization...');
      
      try {
        await this.runGlobalOptimization();
      } catch (error) {
        logger.error('Periodic optimization failed', { error: error.message });
      }
    }, 30 * 60 * 1000);
  }

  private async runGlobalOptimization(): Promise<void> {
    // Get cost optimization recommendations
    const costRecommendations = await this.client.getCostOptimizationRecommendations();
    
    logger.info(`Found ${costRecommendations.length} cost optimization opportunities`);
    
    let totalSavings = 0;
    
    for (const recommendation of costRecommendations) {
      // Auto-implement low-risk, high-impact optimizations
      if (recommendation.risk_level === 'low' && recommendation.monthly_savings > 200) {
        try {
          await this.implementOptimization(recommendation);
          totalSavings += recommendation.monthly_savings;
          logger.info('Implemented optimization', { 
            title: recommendation.title,
            savings: recommendation.monthly_savings 
          });
        } catch (error) {
          logger.error('Failed to implement optimization', { 
            title: recommendation.title,
            error: error.message 
          });
        }
      }
    }
    
    if (totalSavings > 0) {
      logger.info(`Global optimization completed, total savings: $${totalSavings}/month`);
    }

    // Run anomaly detection
    const anomalies = await this.client.detectAnomalies(undefined, 3600);
    
    if (anomalies.length > 0) {
      logger.warn(`Detected ${anomalies.length} anomalies`);
      
      for (const anomaly of anomalies) {
        await this.handleAnomaly(anomaly);
      }
    }
  }

  private async implementOptimization(recommendation: any): Promise<void> {
    switch (recommendation.type) {
      case 'rightsizing':
        await this.implementRightsizing(recommendation);
        break;
      
      case 'cross_cloud_migration':
        await this.implementCrossCloudMigration(recommendation);
        break;
      
      case 'scheduling_optimization':
        await this.implementSchedulingOptimization(recommendation);
        break;
      
      default:
        logger.info('Manual optimization required', { 
          type: recommendation.type,
          title: recommendation.title 
        });
    }
  }

  private async implementRightsizing(recommendation: any): Promise<void> {
    const { vm_id, recommended_specs } = recommendation;
    
    await this.client.updateVM(vm_id, recommended_specs);
    logger.info('Rightsizing implemented', { vmId: vm_id, specs: recommended_specs });
  }

  private async implementCrossCloudMigration(recommendation: any): Promise<void> {
    const { vm_id, target_provider, target_cluster, target_region } = recommendation;
    
    const migration = await this.client.createCrossCloudMigration(
      vm_id,
      target_cluster,
      CloudProvider[target_provider.toUpperCase()],
      target_region
    );
    
    logger.info('Cross-cloud migration initiated', { 
      vmId: vm_id,
      migrationId: migration.id,
      targetProvider: target_provider 
    });
  }

  private async implementSchedulingOptimization(recommendation: any): Promise<void> {
    const { vm_ids, schedule_type, schedule_config } = recommendation;
    
    // This would integrate with a scheduler service
    logger.info('Scheduling optimization implemented', { 
      vmCount: vm_ids.length,
      scheduleType: schedule_type 
    });
  }

  private async handleAnomaly(anomaly: any): Promise<void> {
    const { type, vm_id, severity, description, metrics } = anomaly;
    
    logger.warn('Handling anomaly', { type, vmId: vm_id, severity, description });
    
    switch (type) {
      case 'performance_degradation':
        await this.handlePerformanceDegradation({ vm_id, metrics });
        break;
      
      case 'resource_exhaustion':
        await this.handleResourceExhaustion(vm_id, metrics);
        break;
      
      case 'cost_spike':
        await this.handleCostSpike(vm_id, metrics);
        break;
      
      default:
        logger.info('Unknown anomaly type', { type });
    }
  }

  private async handleResourceExhaustion(vmId: string, metrics: any): Promise<void> {
    logger.warn('Resource exhaustion detected', { vmId, metrics });
    
    // Implement emergency scaling
    const currentSpecs = await this.client.getVM(vmId);
    const scalingFactor = 1.5;
    
    await this.client.updateVM(vmId, {
      cpu_shares: Math.floor(currentSpecs.config.cpu_shares * scalingFactor),
      memory_mb: Math.floor(currentSpecs.config.memory_mb * scalingFactor)
    });
    
    logger.info('Emergency scaling applied', { vmId, scalingFactor });
  }

  private async handleCostSpike(vmId: string, metrics: any): Promise<void> {
    logger.warn('Cost spike detected', { vmId, currentCost: metrics.hourly_cost });
    
    // Check if VM can be migrated to cheaper provider
    const costComparison = await this.client.getCrossCloudCosts(
      CloudProvider.AWS,
      CloudProvider.GCP,
      metrics.vm_specs
    );
    
    if (costComparison.potential_savings > metrics.hourly_cost * 0.3) {
      // Significant savings available, trigger migration
      logger.info('Triggering cost-saving migration', { 
        vmId, 
        potentialSavings: costComparison.potential_savings 
      });
      
      // This would create a migration request
    }
  }

  // Helper methods
  private async analyzeVMOptimization(vmId: string, metrics: any): Promise<any> {
    // Analyze if VM needs optimization
    return {
      shouldMigrate: metrics.cpu_usage > 90 || metrics.memory_usage > 90,
      reason: 'High resource utilization',
      targetSpecs: {
        cpu_shares: Math.ceil(metrics.current_specs.cpu_shares * 1.5),
        memory_mb: Math.ceil(metrics.current_specs.memory_mb * 1.5)
      }
    };
  }

  private async updatePlacementModel(vmId: string, outcome: string, details: string): Promise<void> {
    // Update AI model based on placement outcomes
    logger.debug('Updating placement model', { vmId, outcome, details });
  }

  private async processPendingRequests(): Promise<void> {
    // Process any pending placement requests
    logger.debug(`Processing ${this.activeRequests.size} pending requests`);
  }

  private async estimatePlacementCost(vmSpecs: VMSpec, nodeId: string): Promise<number> {
    // Estimate cost for placing VM on specific node
    return (vmSpecs.cpu_shares / 1024) * 0.05 + (vmSpecs.memory_mb / 1024) * 0.02; // Mock calculation
  }

  private async getNodeRegion(nodeId: string): Promise<string> {
    // Get region for specific node
    return 'us-west-2'; // Mock
  }

  private async getNodePerformanceLevel(nodeId: string): Promise<number> {
    // Get performance level for node (1-10 scale)
    return 8; // Mock high performance
  }

  public getMetrics(): OptimizationMetrics {
    const successfulPlacements = this.placementHistory.filter(p => p.outcome === 'success').length;
    const totalPlacements = this.placementHistory.length;
    
    return {
      costSavings: this.placementHistory.reduce((sum, p) => sum + (p.actualMetrics?.cost_savings || 0), 0),
      performanceGain: 15.5, // Mock
      latencyReduction: 23.2, // Mock  
      resourceUtilization: 78.5, // Mock
      placementAccuracy: totalPlacements > 0 ? (successfulPlacements / totalPlacements) * 100 : 0
    };
  }

  public async generateReport(): Promise<any> {
    const metrics = this.getMetrics();
    const requestMetrics = this.client.getRequestMetrics();
    const circuitBreakerStatus = this.client.getCircuitBreakerStatus();
    
    return {
      timestamp: new Date().toISOString(),
      service_status: 'active',
      workload_patterns: this.workloadPatterns.size,
      active_requests: this.activeRequests.size,
      placement_history: this.placementHistory.length,
      optimization_metrics: metrics,
      api_performance: {
        total_requests: Object.values(requestMetrics).reduce((sum, m) => sum + m.count, 0),
        avg_response_time: Object.values(requestMetrics).reduce((sum, m) => sum + m.avgDuration, 0) / Object.keys(requestMetrics).length,
        circuit_breakers_open: Object.values(circuitBreakerStatus).filter(s => s.isOpen).length
      }
    };
  }

  async shutdown(): Promise<void> {
    logger.info('Shutting down Intelligent Placement Service...');
    await this.client.close();
    logger.info('Service shutdown completed');
  }
}

// Demo usage
async function main() {
  const placementService = new IntelligentPlacementService();

  try {
    await placementService.initialize();

    // Example placement requests
    const webAppRequest: PlacementRequest = {
      id: 'req-web-app-001',
      workloadType: 'web-traffic',
      vmSpecs: {
        name: 'web-frontend',
        cpu_shares: 2048,
        memory_mb: 4096,
        disk_size_gb: 50
      },
      constraints: {
        maxLatency: 100, // ms
        performance: 'high',
        budgetLimit: 500, // monthly
        redundancy: true
      },
      priority: 'high',
      requestedAt: new Date()
    };

    const batchJobRequest: PlacementRequest = {
      id: 'req-batch-job-001',
      workloadType: 'batch-processing',
      vmSpecs: {
        name: 'batch-processor',
        cpu_shares: 8192,
        memory_mb: 16384,
        disk_size_gb: 200
      },
      constraints: {
        performance: 'medium',
        budgetLimit: 200,
        complianceRegions: ['us-west-2', 'us-east-1']
      },
      priority: 'medium',
      requestedAt: new Date()
    };

    const mlTrainingRequest: PlacementRequest = {
      id: 'req-ml-training-001',
      workloadType: 'ml-training',
      vmSpecs: {
        name: 'ml-trainer',
        cpu_shares: 16384,
        memory_mb: 32768,
        disk_size_gb: 1000
      },
      constraints: {
        performance: 'high',
        budgetLimit: 1000
      },
      priority: 'critical',
      requestedAt: new Date()
    };

    // Submit placement requests
    logger.info('Submitting placement requests...');
    
    const webAppId = await placementService.requestPlacement(webAppRequest);
    const batchJobId = await placementService.requestPlacement(batchJobRequest);
    const mlTrainingId = await placementService.requestPlacement(mlTrainingRequest);

    logger.info('Placement requests submitted', { 
      webAppId, 
      batchJobId, 
      mlTrainingId 
    });

    // Monitor service for a while
    logger.info('Monitoring placement service for 60 seconds...');
    
    // Generate periodic reports
    const reportInterval = setInterval(async () => {
      const report = await placementService.generateReport();
      logger.info('Service Report', report);
    }, 20000);

    await new Promise(resolve => setTimeout(resolve, 60000));
    
    clearInterval(reportInterval);
    
    // Final report
    const finalReport = await placementService.generateReport();
    logger.info('Final Service Report', finalReport);

  } catch (error) {
    logger.error('Service error', { error: error.message, stack: error.stack });
  } finally {
    await placementService.shutdown();
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

export { IntelligentPlacementService, PlacementRequest, WorkloadPattern, OptimizationMetrics };