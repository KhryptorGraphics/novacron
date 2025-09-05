/**
 * MLE-Star Workflow Integration Tests
 * 
 * Comprehensive integration tests for the complete MLE-Star workflow including:
 * - All 7 stages of MLE-Star methodology
 * - Multi-framework support (PyTorch, TensorFlow, Scikit-learn)
 * - Template generation and customization
 * - Notebook execution and validation
 * - Model deployment and monitoring
 * - End-to-end workflow validation
 */

const { describe, it, beforeAll, afterAll, beforeEach, afterEach, expect } = require('@jest/globals');
const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');

// Test utilities
const TestEnvironment = require('../../utils/test-environment');
const MLEStarClient = require('../../utils/mle-star-client');
const NotebookExecutor = require('../../utils/notebook-executor');
const ModelValidator = require('../../utils/model-validator');

describe('Integration: MLE-Star Workflow', () => {
  let testEnv;
  let mleStarClient;
  let notebookExecutor;
  let modelValidator;
  let testProjectPath;

  const MLE_STAR_STAGES = [
    'specification',
    'design',
    'implementation',
    'evaluation',
    'deployment',
    'monitoring',
    'maintenance'
  ];

  beforeAll(async () => {
    console.log('🚀 Starting MLE-Star Workflow Integration Tests...');
    
    // Initialize test environment
    testEnv = new TestEnvironment();
    await testEnv.setup();
    
    // Initialize MLE-Star client
    mleStarClient = new MLEStarClient({
      baseURL: process.env.NOVACRON_API_URL || 'http://localhost:8090',
      timeout: 60000
    });
    
    // Initialize notebook executor
    notebookExecutor = new NotebookExecutor();
    
    // Initialize model validator
    modelValidator = new ModelValidator();
    
    // Create test project directory
    testProjectPath = path.join(__dirname, '../../fixtures/mle-star-test-project');
    await fs.mkdir(testProjectPath, { recursive: true });
    
    // Wait for MLE-Star service to be ready
    await testEnv.waitForServices(['mle-star-service', 'jupyter-hub', 'model-registry']);
    
    console.log('✅ MLE-Star test environment initialized');
  });

  afterAll(async () => {
    console.log('🧹 Cleaning up MLE-Star test environment...');
    
    // Cleanup test projects
    try {
      await fs.rmdir(testProjectPath, { recursive: true });
    } catch (error) {
      console.warn('Warning: Failed to cleanup test project directory:', error.message);
    }
    
    await testEnv?.cleanup();
    
    console.log('✅ MLE-Star test environment cleaned up');
  });

  beforeEach(async () => {
    // Clean up any existing test data
    await testEnv.cleanupTestData();
  });

  describe('Complete 7-Stage MLE-Star Workflow', () => {
    it('should execute complete ML project lifecycle', async () => {
      const projectName = `integration-test-project-${Date.now()}`;
      const projectId = await executeCompleteWorkflow(projectName);
      
      // Verify all stages were completed
      const projectDetails = await mleStarClient.getProject(projectId);
      
      expect(projectDetails.status).toBe('completed');
      expect(projectDetails.stages).toHaveLength(7);
      
      for (const stage of MLE_STAR_STAGES) {
        const stageDetails = projectDetails.stages.find(s => s.name === stage);
        expect(stageDetails).toBeDefined();
        expect(stageDetails.status).toBe('completed');
        expect(stageDetails.artifacts).toBeDefined();
      }
      
      // Verify final model deployment
      expect(projectDetails.deployment).toBeDefined();
      expect(projectDetails.deployment.status).toBe('active');
      expect(projectDetails.deployment.endpoint).toBeDefined();
      
      // Cleanup
      await mleStarClient.deleteProject(projectId);
    }, 600000); // 10-minute timeout for complete workflow

    async function executeCompleteWorkflow(projectName) {
      console.log(`🏁 Starting complete MLE-Star workflow for: ${projectName}`);
      
      // Stage 1: Specification
      console.log('📋 Stage 1: Specification');
      const projectConfig = {
        name: projectName,
        type: 'classification',
        framework: 'pytorch',
        dataset: 'iris',
        target: 'species',
        metrics: ['accuracy', 'precision', 'recall', 'f1'],
        requirements: {
          accuracy: 0.95,
          latency: 100, // ms
          throughput: 1000 // requests/second
        }
      };
      
      const project = await mleStarClient.createProject(projectConfig);
      const projectId = project.id;
      
      // Stage 2: Design
      console.log('🎨 Stage 2: Design');
      const designConfig = {
        architecture: 'neural_network',
        layers: [
          { type: 'linear', input_size: 4, output_size: 16 },
          { type: 'relu' },
          { type: 'linear', input_size: 16, output_size: 8 },
          { type: 'relu' },
          { type: 'linear', input_size: 8, output_size: 3 },
          { type: 'softmax' }
        ],
        optimizer: 'adam',
        learning_rate: 0.001,
        loss_function: 'cross_entropy'
      };
      
      await mleStarClient.updateProjectStage(projectId, 'design', designConfig);
      
      // Stage 3: Implementation
      console.log('⚙️ Stage 3: Implementation');
      const implementationResult = await mleStarClient.generateCode(projectId, {
        template: 'pytorch-classification',
        customizations: designConfig
      });
      
      expect(implementationResult.success).toBe(true);
      expect(implementationResult.files).toContain('model.py');
      expect(implementationResult.files).toContain('train.py');
      expect(implementationResult.files).toContain('evaluate.py');
      
      // Stage 4: Evaluation
      console.log('📊 Stage 4: Evaluation');
      const trainingResult = await mleStarClient.trainModel(projectId, {
        epochs: 50,
        batch_size: 32,
        validation_split: 0.2
      });
      
      expect(trainingResult.success).toBe(true);
      expect(trainingResult.metrics.accuracy).toBeGreaterThan(0.90);
      
      const evaluationResult = await mleStarClient.evaluateModel(projectId);
      expect(evaluationResult.metrics.accuracy).toBeGreaterThan(projectConfig.requirements.accuracy);
      
      // Stage 5: Deployment
      console.log('🚀 Stage 5: Deployment');
      const deploymentResult = await mleStarClient.deployModel(projectId, {
        environment: 'staging',
        scaling: {
          min_replicas: 1,
          max_replicas: 3,
          cpu_threshold: 70
        }
      });
      
      expect(deploymentResult.success).toBe(true);
      expect(deploymentResult.endpoint).toBeDefined();
      
      // Test deployed model
      const predictionTest = await axios.post(deploymentResult.endpoint, {
        features: [5.1, 3.5, 1.4, 0.2] // Iris setosa sample
      });
      
      expect(predictionTest.status).toBe(200);
      expect(predictionTest.data.prediction).toBeDefined();
      
      // Stage 6: Monitoring
      console.log('📈 Stage 6: Monitoring');
      const monitoringResult = await mleStarClient.setupMonitoring(projectId, {
        metrics: ['latency', 'throughput', 'accuracy'],
        alerts: {
          latency_threshold: 200,
          accuracy_threshold: 0.90
        }
      });
      
      expect(monitoringResult.success).toBe(true);
      expect(monitoringResult.dashboard_url).toBeDefined();
      
      // Stage 7: Maintenance
      console.log('🔧 Stage 7: Maintenance');
      const maintenanceResult = await mleStarClient.setupMaintenance(projectId, {
        retraining_schedule: 'weekly',
        data_drift_detection: true,
        model_versioning: true
      });
      
      expect(maintenanceResult.success).toBe(true);
      
      console.log('✅ Complete MLE-Star workflow executed successfully');
      return projectId;
    }
  });

  describe('Multi-Framework Support', () => {
    const frameworks = ['pytorch', 'tensorflow', 'sklearn'];
    
    frameworks.forEach(framework => {
      it(`should support complete workflow with ${framework.toUpperCase()}`, async () => {
        const projectName = `${framework}-integration-test-${Date.now()}`;
        
        const projectConfig = {
          name: projectName,
          type: 'classification',
          framework: framework,
          dataset: 'iris',
          target: 'species'
        };
        
        const project = await mleStarClient.createProject(projectConfig);
        const projectId = project.id;
        
        // Test code generation for specific framework
        const codeGenResult = await mleStarClient.generateCode(projectId, {
          template: `${framework}-classification`
        });
        
        expect(codeGenResult.success).toBe(true);
        expect(codeGenResult.framework).toBe(framework);
        
        // Framework-specific file expectations
        const expectedFiles = getExpectedFilesForFramework(framework);
        for (const file of expectedFiles) {
          expect(codeGenResult.files).toContain(file);
        }
        
        // Test training with framework
        const trainingResult = await mleStarClient.trainModel(projectId, {
          epochs: 10, // Reduced for faster testing
          batch_size: 32
        });
        
        expect(trainingResult.success).toBe(true);
        expect(trainingResult.framework).toBe(framework);
        expect(trainingResult.model_path).toBeDefined();
        
        // Test model evaluation
        const evaluationResult = await mleStarClient.evaluateModel(projectId);
        expect(evaluationResult.success).toBe(true);
        expect(evaluationResult.metrics.accuracy).toBeGreaterThan(0.8);
        
        // Test deployment (framework-agnostic)
        const deploymentResult = await mleStarClient.deployModel(projectId, {
          environment: 'test'
        });
        
        expect(deploymentResult.success).toBe(true);
        expect(deploymentResult.endpoint).toBeDefined();
        
        // Test prediction
        const predictionResult = await axios.post(deploymentResult.endpoint, {
          features: [5.0, 3.0, 1.5, 0.2]
        });
        
        expect(predictionResult.status).toBe(200);
        expect(predictionResult.data.prediction).toBeDefined();
        
        // Cleanup
        await mleStarClient.deleteProject(projectId);
      }, 300000); // 5-minute timeout per framework
    });

    function getExpectedFilesForFramework(framework) {
      const commonFiles = ['requirements.txt', 'config.yaml', 'README.md'];
      
      switch (framework) {
        case 'pytorch':
          return [...commonFiles, 'model.py', 'train.py', 'evaluate.py', 'utils.py'];
        case 'tensorflow':
          return [...commonFiles, 'model.py', 'train.py', 'evaluate.py', 'preprocessing.py'];
        case 'sklearn':
          return [...commonFiles, 'model.py', 'train.py', 'evaluate.py', 'pipeline.py'];
        default:
          return commonFiles;
      }
    }
  });

  describe('Template Generation and Customization', () => {
    it('should generate customized templates for different project types', async () => {
      const projectTypes = [
        { type: 'classification', dataset: 'iris' },
        { type: 'regression', dataset: 'boston_housing' },
        { type: 'clustering', dataset: 'wine' },
        { type: 'time_series', dataset: 'stock_prices' }
      ];
      
      for (const projectType of projectTypes) {
        const projectName = `template-test-${projectType.type}-${Date.now()}`;
        
        const projectConfig = {
          name: projectName,
          type: projectType.type,
          framework: 'pytorch',
          dataset: projectType.dataset
        };
        
        const project = await mleStarClient.createProject(projectConfig);
        const projectId = project.id;
        
        // Generate template
        const templateResult = await mleStarClient.generateTemplate(projectId, {
          type: projectType.type,
          customizations: {
            include_visualization: true,
            include_data_validation: true,
            include_hyperparameter_tuning: true
          }
        });
        
        expect(templateResult.success).toBe(true);
        expect(templateResult.template_type).toBe(projectType.type);
        
        // Verify template structure
        expect(templateResult.structure).toBeDefined();
        expect(templateResult.structure.notebooks).toBeDefined();
        expect(templateResult.structure.src).toBeDefined();
        expect(templateResult.structure.data).toBeDefined();
        expect(templateResult.structure.models).toBeDefined();
        
        // Verify customizations were applied
        expect(templateResult.features.visualization).toBe(true);
        expect(templateResult.features.data_validation).toBe(true);
        expect(templateResult.features.hyperparameter_tuning).toBe(true);
        
        // Cleanup
        await mleStarClient.deleteProject(projectId);
      }
    });

    it('should support custom template modifications', async () => {
      const projectName = `custom-template-test-${Date.now()}`;
      
      const project = await mleStarClient.createProject({
        name: projectName,
        type: 'classification',
        framework: 'pytorch',
        dataset: 'custom'
      });
      
      const projectId = project.id;
      
      // Define custom template modifications
      const customizations = {
        model_architecture: {
          type: 'cnn',
          layers: [
            { type: 'conv2d', filters: 32, kernel_size: 3 },
            { type: 'relu' },
            { type: 'maxpool2d', pool_size: 2 },
            { type: 'flatten' },
            { type: 'dense', units: 128 },
            { type: 'dropout', rate: 0.5 },
            { type: 'dense', units: 10, activation: 'softmax' }
          ]
        },
        data_pipeline: {
          preprocessing: ['normalize', 'augment'],
          batch_size: 64,
          shuffle: true
        },
        training: {
          optimizer: 'adamw',
          learning_rate: 0.0001,
          scheduler: 'cosine_annealing',
          early_stopping: true
        }
      };
      
      const templateResult = await mleStarClient.generateCustomTemplate(projectId, customizations);
      
      expect(templateResult.success).toBe(true);
      expect(templateResult.customizations_applied).toBe(true);
      
      // Verify custom architecture was included
      const modelCode = await mleStarClient.getGeneratedCode(projectId, 'model.py');
      expect(modelCode.content).toContain('conv2d');
      expect(modelCode.content).toContain('maxpool2d');
      expect(modelCode.content).toContain('dropout');
      
      // Verify custom training configuration
      const trainCode = await mleStarClient.getGeneratedCode(projectId, 'train.py');
      expect(trainCode.content).toContain('adamw');
      expect(trainCode.content).toContain('cosine_annealing');
      expect(trainCode.content).toContain('early_stopping');
      
      // Cleanup
      await mleStarClient.deleteProject(projectId);
    });
  });

  describe('Notebook Execution and Validation', () => {
    it('should execute generated notebooks successfully', async () => {
      const projectName = `notebook-test-${Date.now()}`;
      
      const project = await mleStarClient.createProject({
        name: projectName,
        type: 'classification',
        framework: 'pytorch',
        dataset: 'iris'
      });
      
      const projectId = project.id;
      
      // Generate project with notebooks
      await mleStarClient.generateCode(projectId, {
        include_notebooks: true,
        notebook_types: ['eda', 'training', 'evaluation']
      });
      
      // Get generated notebooks
      const notebooks = await mleStarClient.getProjectNotebooks(projectId);
      expect(notebooks.length).toBeGreaterThan(0);
      
      // Execute each notebook
      for (const notebook of notebooks) {
        console.log(`Executing notebook: ${notebook.name}`);
        
        const executionResult = await notebookExecutor.execute(notebook.path, {
          timeout: 300000, // 5-minute timeout
          kernel: 'python3'
        });
        
        expect(executionResult.success).toBe(true);
        expect(executionResult.errors).toHaveLength(0);
        
        // Verify notebook outputs
        if (notebook.type === 'eda') {
          expect(executionResult.outputs).toContain('data_shape');
          expect(executionResult.outputs).toContain('data_description');
        } else if (notebook.type === 'training') {
          expect(executionResult.outputs).toContain('training_loss');
          expect(executionResult.outputs).toContain('model_saved');
        } else if (notebook.type === 'evaluation') {
          expect(executionResult.outputs).toContain('accuracy_score');
          expect(executionResult.outputs).toContain('classification_report');
        }
      }
      
      // Cleanup
      await mleStarClient.deleteProject(projectId);
    });

    it('should validate notebook cell execution order and dependencies', async () => {
      const projectName = `notebook-validation-test-${Date.now()}`;
      
      const project = await mleStarClient.createProject({
        name: projectName,
        type: 'regression',
        framework: 'sklearn',
        dataset: 'boston_housing'
      });
      
      const projectId = project.id;
      
      await mleStarClient.generateCode(projectId, {
        include_notebooks: true
      });
      
      const notebooks = await mleStarClient.getProjectNotebooks(projectId);
      const mainNotebook = notebooks.find(nb => nb.type === 'main');
      
      expect(mainNotebook).toBeDefined();
      
      // Analyze notebook dependencies
      const dependencyAnalysis = await notebookExecutor.analyzeDependencies(mainNotebook.path);
      
      expect(dependencyAnalysis.valid).toBe(true);
      expect(dependencyAnalysis.dependency_violations).toHaveLength(0);
      
      // Test execution with dependency validation
      const validationResult = await notebookExecutor.executeWithValidation(mainNotebook.path, {
        validate_dependencies: true,
        check_variable_usage: true
      });
      
      expect(validationResult.success).toBe(true);
      expect(validationResult.dependency_errors).toHaveLength(0);
      expect(validationResult.variable_errors).toHaveLength(0);
      
      // Cleanup
      await mleStarClient.deleteProject(projectId);
    });
  });

  describe('Model Deployment and Monitoring', () => {
    it('should deploy models to different environments', async () => {
      const environments = ['development', 'staging', 'production'];
      
      // Create and train a model
      const project = await mleStarClient.createProject({
        name: `deployment-test-${Date.now()}`,
        type: 'classification',
        framework: 'pytorch',
        dataset: 'iris'
      });
      
      const projectId = project.id;
      
      // Quick training
      await mleStarClient.trainModel(projectId, {
        epochs: 5,
        batch_size: 32
      });
      
      // Deploy to each environment
      const deployments = [];
      
      for (const env of environments) {
        const deploymentConfig = {
          environment: env,
          replicas: env === 'production' ? 3 : 1,
          resources: {
            cpu: env === 'production' ? '1000m' : '500m',
            memory: env === 'production' ? '2Gi' : '1Gi'
          },
          autoscaling: env === 'production' ? {
            min_replicas: 2,
            max_replicas: 10,
            target_cpu: 70
          } : null
        };
        
        const deploymentResult = await mleStarClient.deployModel(projectId, deploymentConfig);
        
        expect(deploymentResult.success).toBe(true);
        expect(deploymentResult.environment).toBe(env);
        expect(deploymentResult.endpoint).toBeDefined();
        
        deployments.push(deploymentResult);
        
        // Test deployed endpoint
        const testResult = await axios.post(deploymentResult.endpoint + '/predict', {
          features: [5.1, 3.5, 1.4, 0.2]
        });
        
        expect(testResult.status).toBe(200);
        expect(testResult.data.prediction).toBeDefined();
      }
      
      // Verify all deployments are tracked
      const projectDeployments = await mleStarClient.getProjectDeployments(projectId);
      expect(projectDeployments.length).toBe(3);
      
      // Cleanup
      for (const deployment of deployments) {
        await mleStarClient.deleteDeployment(deployment.deployment_id);
      }
      await mleStarClient.deleteProject(projectId);
    });

    it('should set up comprehensive model monitoring', async () => {
      const project = await mleStarClient.createProject({
        name: `monitoring-test-${Date.now()}`,
        type: 'classification',
        framework: 'tensorflow',
        dataset: 'iris'
      });
      
      const projectId = project.id;
      
      // Train and deploy model
      await mleStarClient.trainModel(projectId, { epochs: 5 });
      const deployment = await mleStarClient.deployModel(projectId, {
        environment: 'staging'
      });
      
      // Set up monitoring
      const monitoringConfig = {
        metrics: [
          'request_rate',
          'response_time',
          'error_rate',
          'prediction_accuracy',
          'data_drift',
          'model_performance'
        ],
        alerts: [
          {
            metric: 'response_time',
            condition: 'greater_than',
            threshold: 1000, // ms
            severity: 'warning'
          },
          {
            metric: 'error_rate',
            condition: 'greater_than',
            threshold: 0.05, // 5%
            severity: 'critical'
          },
          {
            metric: 'prediction_accuracy',
            condition: 'less_than',
            threshold: 0.85,
            severity: 'warning'
          }
        ],
        dashboards: ['performance', 'business_metrics', 'data_quality']
      };
      
      const monitoringResult = await mleStarClient.setupMonitoring(
        deployment.deployment_id,
        monitoringConfig
      );
      
      expect(monitoringResult.success).toBe(true);
      expect(monitoringResult.monitoring_id).toBeDefined();
      expect(monitoringResult.dashboard_urls).toBeDefined();
      
      // Generate some test traffic
      const testRequests = Array.from({ length: 100 }, () => ({
        features: [
          Math.random() * 8 + 4,   // sepal_length
          Math.random() * 4 + 2,   // sepal_width  
          Math.random() * 7 + 1,   // petal_length
          Math.random() * 3 + 0.1  // petal_width
        ]
      }));
      
      await Promise.all(
        testRequests.map(request =>
          axios.post(deployment.endpoint + '/predict', request)
            .catch(() => {}) // Ignore individual failures
        )
      );
      
      // Wait for metrics to be collected
      await new Promise(resolve => setTimeout(resolve, 30000));
      
      // Check monitoring metrics
      const metrics = await mleStarClient.getMonitoringMetrics(
        monitoringResult.monitoring_id,
        {
          time_range: '30m'
        }
      );
      
      expect(metrics.request_count).toBeGreaterThan(0);
      expect(metrics.average_response_time).toBeDefined();
      expect(metrics.error_rate).toBeLessThan(0.1);
      
      // Cleanup
      await mleStarClient.deleteMonitoring(monitoringResult.monitoring_id);
      await mleStarClient.deleteDeployment(deployment.deployment_id);
      await mleStarClient.deleteProject(projectId);
    });
  });

  describe('End-to-End Workflow Validation', () => {
    it('should handle complex multi-model project', async () => {
      const projectName = `complex-e2e-test-${Date.now()}`;
      
      const project = await mleStarClient.createProject({
        name: projectName,
        type: 'ensemble',
        models: [
          { type: 'classification', framework: 'pytorch' },
          { type: 'classification', framework: 'tensorflow' },
          { type: 'classification', framework: 'sklearn' }
        ],
        dataset: 'iris',
        ensemble_method: 'voting'
      });
      
      const projectId = project.id;
      
      // Train all models
      const trainingResults = await mleStarClient.trainEnsemble(projectId, {
        epochs: 10,
        parallel_training: true
      });
      
      expect(trainingResults.success).toBe(true);
      expect(trainingResults.models_trained).toBe(3);
      
      // Evaluate ensemble
      const evaluationResult = await mleStarClient.evaluateEnsemble(projectId);
      expect(evaluationResult.ensemble_accuracy).toBeGreaterThan(0.95);
      
      // Deploy ensemble
      const deploymentResult = await mleStarClient.deployEnsemble(projectId, {
        environment: 'staging',
        load_balancing: 'round_robin'
      });
      
      expect(deploymentResult.success).toBe(true);
      expect(deploymentResult.endpoint).toBeDefined();
      
      // Test ensemble predictions
      const predictionResult = await axios.post(deploymentResult.endpoint + '/predict', {
        features: [5.1, 3.5, 1.4, 0.2]
      });
      
      expect(predictionResult.status).toBe(200);
      expect(predictionResult.data.ensemble_prediction).toBeDefined();
      expect(predictionResult.data.individual_predictions).toHaveLength(3);
      
      // Cleanup
      await mleStarClient.deleteProject(projectId);
    }, 900000); // 15-minute timeout for complex workflow

    it('should handle workflow with external data sources', async () => {
      // This test would integrate with external data sources
      // For now, we'll simulate it
      const projectName = `external-data-test-${Date.now()}`;
      
      const project = await mleStarClient.createProject({
        name: projectName,
        type: 'classification',
        framework: 'pytorch',
        data_source: {
          type: 'sql',
          connection_string: 'mock://test-database',
          query: 'SELECT * FROM iris_data'
        }
      });
      
      const projectId = project.id;
      
      // Test data ingestion
      const ingestionResult = await mleStarClient.ingestData(projectId);
      expect(ingestionResult.success).toBe(true);
      expect(ingestionResult.records_ingested).toBeGreaterThan(0);
      
      // Continue with normal workflow
      await mleStarClient.trainModel(projectId, { epochs: 5 });
      const deployment = await mleStarClient.deployModel(projectId, {
        environment: 'test'
      });
      
      expect(deployment.success).toBe(true);
      
      // Cleanup
      await mleStarClient.deleteProject(projectId);
    });
  });
});