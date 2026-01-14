/**
 * Integration Tests for MLE-Star Workflow
 * Tests the complete MLE-Star automation system
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const assert = require('assert');

// Import MLE-Star components
const MLEStarWorkflow = require('../src/mle-star/index');
const MLEStarCommands = require('../src/mle-star/commands');

describe('MLE-Star Workflow Integration Tests', function() {
    this.timeout(30000); // 30 second timeout for integration tests
    
    let testProjectPath;
    let workflow;
    let commands;
    
    before(function() {
        // Setup test environment
        testProjectPath = path.join(__dirname, 'temp_mle_star_test');
        
        // Clean up any existing test directory
        if (fs.existsSync(testProjectPath)) {
            fs.rmSync(testProjectPath, { recursive: true, force: true });
        }
        
        // Create test directory
        fs.mkdirSync(testProjectPath, { recursive: true });
        
        // Initialize workflow with test configuration
        const testConfig = {
            projectPath: testProjectPath,
            mlFramework: 'scikit-learn',
            experimentName: 'test-experiment',
            logLevel: 'error' // Reduce logging during tests
        };
        
        workflow = new MLEStarWorkflow(testConfig);
        commands = new MLEStarCommands();
    });
    
    after(function() {
        // Cleanup test directory
        if (fs.existsSync(testProjectPath)) {
            fs.rmSync(testProjectPath, { recursive: true, force: true });
        }
    });
    
    describe('Workflow Initialization', function() {
        it('should initialize MLE-Star project structure', async function() {
            const result = await workflow.initializeProject();
            
            assert.strictEqual(result.status, 'success');
            
            // Check directory structure
            const expectedDirs = [
                'data/raw', 'data/processed', 'data/external',
                'models', 'notebooks', 'src/data', 'src/features',
                'src/models', 'src/visualization', 'tests', 'configs',
                'outputs/models', 'outputs/figures', 'outputs/reports'
            ];
            
            for (const dir of expectedDirs) {
                const dirPath = path.join(testProjectPath, dir);
                assert(fs.existsSync(dirPath), `Directory ${dir} should exist`);
            }
            
            // Check essential files
            const expectedFiles = [
                'README.md', 'requirements.txt', 'configs/config.yaml', 'src/main.py'
            ];
            
            for (const file of expectedFiles) {
                const filePath = path.join(testProjectPath, file);
                assert(fs.existsSync(filePath), `File ${file} should exist`);
            }
        });
        
        it('should validate workflow status', function() {
            const status = workflow.getStatus();
            
            assert.strictEqual(status.framework, 'scikit-learn');
            assert.strictEqual(status.project_path, testProjectPath);
            assert.strictEqual(status.experiment_name, 'test-experiment');
            assert(Array.isArray(status.available_stages));
            assert.strictEqual(status.available_stages.length, 7);
        });
    });
    
    describe('Individual Stage Execution', function() {
        it('should execute model design stage', async function() {
            const result = await workflow.executeStage('model_design');
            
            assert.strictEqual(result.status, 'success');
            assert.strictEqual(result.stage, 'model_design');
            assert(Array.isArray(result.artifacts));
            assert(result.artifacts.length > 0);
            
            // Check created artifacts
            for (const artifact of result.artifacts) {
                const artifactPath = path.join(testProjectPath, artifact);
                assert(fs.existsSync(artifactPath), `Artifact ${artifact} should exist`);
            }
        });
        
        it('should execute learning pipeline stage', async function() {
            const result = await workflow.executeStage('learning_pipeline');
            
            assert.strictEqual(result.status, 'success');
            assert.strictEqual(result.stage, 'learning_pipeline');
            assert(Array.isArray(result.artifacts));
            
            // Verify data pipeline artifact
            const dataPipelinePath = path.join(testProjectPath, 'src/data/data_pipeline.py');
            assert(fs.existsSync(dataPipelinePath), 'Data pipeline should be created');
        });
        
        it('should execute evaluation setup stage', async function() {
            const result = await workflow.executeStage('evaluation_setup');
            
            assert.strictEqual(result.status, 'success');
            assert.strictEqual(result.stage, 'evaluation_setup');
            
            // Verify evaluation artifacts
            const evaluationPath = path.join(testProjectPath, 'src/models/evaluation.py');
            assert(fs.existsSync(evaluationPath), 'Evaluation module should be created');
        });
        
        it('should execute systematic testing stage', async function() {
            const result = await workflow.executeStage('systematic_testing');
            
            assert.strictEqual(result.status, 'success');
            assert.strictEqual(result.stage, 'systematic_testing');
            
            // Verify test files
            const testFiles = ['test_model.py', 'test_data_pipeline.py', 'test_features.py'];
            for (const testFile of testFiles) {
                const testPath = path.join(testProjectPath, 'tests', testFile);
                assert(fs.existsSync(testPath), `Test file ${testFile} should be created`);
            }
        });
        
        it('should execute training optimization stage', async function() {
            const result = await workflow.executeStage('training_optimization');
            
            assert.strictEqual(result.status, 'success');
            assert.strictEqual(result.stage, 'training_optimization');
            
            // Verify optimization artifacts
            const optimizationPath = path.join(testProjectPath, 'src/models/train_optimize.py');
            assert(fs.existsSync(optimizationPath), 'Training optimization script should be created');
        });
        
        it('should execute analysis validation stage', async function() {
            const result = await workflow.executeStage('analysis_validation');
            
            assert.strictEqual(result.status, 'success');
            assert.strictEqual(result.stage, 'analysis_validation');
            
            // Verify analysis artifacts
            const analysisNotebook = path.join(testProjectPath, 'notebooks/05_model_analysis.ipynb');
            assert(fs.existsSync(analysisNotebook), 'Analysis notebook should be created');
        });
        
        it('should execute refinement deployment stage', async function() {
            const result = await workflow.executeStage('refinement_deployment');
            
            assert.strictEqual(result.status, 'success');
            assert.strictEqual(result.stage, 'refinement_deployment');
            
            // Verify deployment artifacts
            const deploymentScript = path.join(testProjectPath, 'src/models/deploy_model.py');
            const apiScript = path.join(testProjectPath, 'src/api/model_api.py');
            
            assert(fs.existsSync(deploymentScript), 'Deployment script should be created');
            assert(fs.existsSync(apiScript), 'API script should be created');
        });
    });
    
    describe('Complete Workflow Execution', function() {
        it('should run complete MLE-Star workflow', async function() {
            const result = await workflow.runFullWorkflow({ continueOnError: true });
            
            assert.strictEqual(result.status, 'completed');
            assert(result.results);
            assert(result.summary);
            
            // Check that all stages were executed
            const expectedStages = workflow.stages;
            for (const stage of expectedStages) {
                assert(result.results[stage], `Stage ${stage} should be in results`);
                assert.strictEqual(result.results[stage].status, 'success', `Stage ${stage} should succeed`);
            }
            
            // Check summary metrics
            assert.strictEqual(result.summary.total_stages, 7);
            assert.strictEqual(result.summary.successful_stages, 7);
            assert.strictEqual(result.summary.failed_stages, 0);
            assert.strictEqual(result.summary.success_rate, '100%');
            assert.strictEqual(result.summary.completion_status, 'fully_completed');
        });
    });
    
    describe('Command Line Interface', function() {
        it('should handle init command', async function() {
            const result = await commands.handleCommand('init', [], {
                framework: 'pytorch',
                path: path.join(testProjectPath, 'cli_test'),
                name: 'cli-test-experiment'
            });
            
            assert.strictEqual(result.status, 'success');
            
            // Verify CLI initialization created proper structure
            const cliTestPath = path.join(testProjectPath, 'cli_test');
            assert(fs.existsSync(cliTestPath), 'CLI test directory should be created');
        });
        
        it('should handle status command', async function() {
            const result = await commands.handleCommand('status', [], {});
            
            assert(result.framework);
            assert(result.project_path);
            assert(result.experiment_name);
            assert(Array.isArray(result.available_stages));
        });
        
        it('should handle validate command', async function() {
            const result = await commands.handleCommand('validate', [], {});
            
            assert(result.validations);
            assert(Array.isArray(result.validations));
            assert(typeof result.all_passed === 'boolean');
        });
        
        it('should handle list-templates command', async function() {
            const result = await commands.handleCommand('list-templates', [], {});
            
            assert(result.templates);
            assert(Array.isArray(result.templates));
            assert(typeof result.count === 'number');
        });
        
        it('should handle help command', async function() {
            const result = commands.helpCommand();
            
            assert(result.commands);
            assert(Array.isArray(result.commands));
        });
        
        it('should handle error for unknown command', async function() {
            const result = await commands.handleCommand('unknown_command', [], {});
            
            assert.strictEqual(result.status, 'error');
            assert(result.message.includes('Unknown command'));
        });
    });
    
    describe('Template System', function() {
        it('should create files from templates', async function() {
            const templateFile = 'test_template.txt';
            const outputFile = 'test_output.txt';
            
            // Create a simple test template
            const templatePath = path.join(workflow.templatePath, templateFile);
            const templateDir = path.dirname(templatePath);
            
            if (!fs.existsSync(templateDir)) {
                fs.mkdirSync(templateDir, { recursive: true });
            }
            
            fs.writeFileSync(templatePath, 'Test template: {{experimentName}}');
            
            // Test template creation
            const success = await workflow.createFromTemplate(templateFile, outputFile, {
                experimentName: 'test-experiment'
            });
            
            assert.strictEqual(success, true);
            
            // Check output file
            const outputPath = path.join(testProjectPath, outputFile);
            assert(fs.existsSync(outputPath), 'Output file should be created');
            
            const content = fs.readFileSync(outputPath, 'utf8');
            assert.strictEqual(content, 'Test template: test-experiment');
        });
        
        it('should handle missing template gracefully', async function() {
            const success = await workflow.createFromTemplate('nonexistent_template.txt', 'output.txt');
            assert.strictEqual(success, false);
        });
    });
    
    describe('Environment Validation', function() {
        it('should validate Python environment', async function() {
            const validations = await workflow.validateEnvironment();
            
            assert(Array.isArray(validations));
            
            // Check for Python validation
            const pythonValidation = validations.find(v => v.check === 'python');
            assert(pythonValidation, 'Python validation should be present');
            
            // Check for Jupyter validation
            const jupyterValidation = validations.find(v => v.check === 'jupyter');
            assert(jupyterValidation, 'Jupyter validation should be present');
        });
    });
    
    describe('Error Handling', function() {
        it('should handle invalid stage name', async function() {
            try {
                await workflow.executeStage('invalid_stage');
                assert.fail('Should throw error for invalid stage');
            } catch (error) {
                assert(error.message.includes('Invalid stage'));
            }
        });
        
        it('should handle workflow errors gracefully', async function() {
            // Create a workflow with invalid configuration
            const invalidWorkflow = new MLEStarWorkflow({
                projectPath: '/invalid/path/that/cannot/be/created',
                mlFramework: 'scikit-learn'
            });
            
            try {
                await invalidWorkflow.initializeProject();
                assert.fail('Should throw error for invalid path');
            } catch (error) {
                assert(error.message); // Should have an error message
            }
        });
    });
    
    describe('Configuration Handling', function() {
        it('should use default configuration values', function() {
            const defaultWorkflow = new MLEStarWorkflow();
            const status = defaultWorkflow.getStatus();
            
            assert.strictEqual(status.framework, 'pytorch'); // default framework
            assert(status.project_path);
            assert(status.experiment_name);
        });
        
        it('should override default configuration', function() {
            const customConfig = {
                mlFramework: 'tensorflow',
                experimentName: 'custom-experiment',
                dataPath: './custom-data'
            };
            
            const customWorkflow = new MLEStarWorkflow(customConfig);
            const status = customWorkflow.getStatus();
            
            assert.strictEqual(status.framework, 'tensorflow');
            assert.strictEqual(status.experiment_name, 'custom-experiment');
        });
    });
    
    describe('Logging and Monitoring', function() {
        it('should log workflow progress', async function() {
            // Capture console output
            const originalLog = console.log;
            const logs = [];
            console.log = (...args) => {
                logs.push(args.join(' '));
            };
            
            try {
                const verboseWorkflow = new MLEStarWorkflow({
                    projectPath: path.join(testProjectPath, 'verbose_test'),
                    logLevel: 'info'
                });
                
                await verboseWorkflow.initializeProject();
                
                // Restore console.log
                console.log = originalLog;
                
                // Check that logs were generated
                assert(logs.length > 0, 'Should generate log messages');
                
                // Check for specific log messages
                const initLogs = logs.filter(log => log.includes('Initializing MLE-Star project'));
                assert(initLogs.length > 0, 'Should log initialization message');
                
            } finally {
                console.log = originalLog;
            }
        });
    });
});

describe('MLE-Star Template Generation', function() {
    let commands;
    let testTemplateDir;
    
    before(function() {
        commands = new MLEStarCommands();
        testTemplateDir = path.join(__dirname, 'temp_templates');
        
        if (fs.existsSync(testTemplateDir)) {
            fs.rmSync(testTemplateDir, { recursive: true, force: true });
        }
        fs.mkdirSync(testTemplateDir, { recursive: true });
    });
    
    after(function() {
        if (fs.existsSync(testTemplateDir)) {
            fs.rmSync(testTemplateDir, { recursive: true, force: true });
        }
    });
    
    it('should generate notebook template', async function() {
        const result = await commands.createTemplateCommand(
            ['notebook', 'test_notebook'],
            { path: testTemplateDir }
        );
        
        assert.strictEqual(result.type, 'notebook');
        assert.strictEqual(result.name, 'test_notebook');
        assert(result.template_path);
        
        // Verify template content
        const templateContent = fs.readFileSync(result.template_path, 'utf8');
        const notebook = JSON.parse(templateContent);
        
        assert(notebook.cells);
        assert(Array.isArray(notebook.cells));
        assert(notebook.metadata);
        assert.strictEqual(notebook.nbformat, 4);
    });
    
    it('should generate script template', async function() {
        const result = await commands.createTemplateCommand(
            ['script', 'test_script'],
            { path: testTemplateDir }
        );
        
        assert.strictEqual(result.type, 'script');
        assert.strictEqual(result.name, 'test_script');
        
        const templateContent = fs.readFileSync(result.template_path, 'utf8');
        assert(templateContent.includes('#!/usr/bin/env python3'));
        assert(templateContent.includes('def main():'));
        assert(templateContent.includes('if __name__ == "__main__":'));
    });
    
    it('should generate config template', async function() {
        const result = await commands.createTemplateCommand(
            ['config', 'test_config'],
            { path: testTemplateDir }
        );
        
        assert.strictEqual(result.type, 'config');
        assert.strictEqual(result.name, 'test_config');
        
        const templateContent = fs.readFileSync(result.template_path, 'utf8');
        assert(templateContent.includes('experiment:'));
        assert(templateContent.includes('data:'));
        assert(templateContent.includes('model:'));
    });
    
    it('should generate test template', async function() {
        const result = await commands.createTemplateCommand(
            ['test', 'test_unit'],
            { path: testTemplateDir }
        );
        
        assert.strictEqual(result.type, 'test');
        assert.strictEqual(result.name, 'test_unit');
        
        const templateContent = fs.readFileSync(result.template_path, 'utf8');
        assert(templateContent.includes('import unittest'));
        assert(templateContent.includes('class Test'));
        assert(templateContent.includes('def test_'));
    });
});

// Performance and stress tests
describe('MLE-Star Performance Tests', function() {
    this.timeout(60000); // 1 minute timeout
    
    it('should handle multiple concurrent workflows', async function() {
        const numWorkflows = 3;
        const workflows = [];
        
        // Create multiple workflow instances
        for (let i = 0; i < numWorkflows; i++) {
            const workflow = new MLEStarWorkflow({
                projectPath: path.join(__dirname, `temp_concurrent_${i}`),
                mlFramework: 'scikit-learn',
                experimentName: `concurrent-test-${i}`,
                logLevel: 'error'
            });
            workflows.push(workflow);
        }
        
        try {
            // Run all workflows concurrently
            const promises = workflows.map(workflow => workflow.initializeProject());
            const results = await Promise.all(promises);
            
            // Verify all workflows completed successfully
            for (const result of results) {
                assert.strictEqual(result.status, 'success');
            }
            
        } finally {
            // Cleanup
            for (let i = 0; i < numWorkflows; i++) {
                const testPath = path.join(__dirname, `temp_concurrent_${i}`);
                if (fs.existsSync(testPath)) {
                    fs.rmSync(testPath, { recursive: true, force: true });
                }
            }
        }
    });
    
    it('should complete full workflow within reasonable time', async function() {
        const startTime = Date.now();
        
        const workflow = new MLEStarWorkflow({
            projectPath: path.join(__dirname, 'temp_performance_test'),
            mlFramework: 'scikit-learn',
            logLevel: 'error'
        });
        
        try {
            await workflow.initializeProject();
            const result = await workflow.runFullWorkflow();
            
            const endTime = Date.now();
            const duration = endTime - startTime;
            
            assert.strictEqual(result.status, 'completed');
            assert(duration < 30000, `Workflow took too long: ${duration}ms`); // Should complete in under 30 seconds
            
        } finally {
            const testPath = path.join(__dirname, 'temp_performance_test');
            if (fs.existsSync(testPath)) {
                fs.rmSync(testPath, { recursive: true, force: true });
            }
        }
    });
});

// Run tests if this file is executed directly
if (require.main === module) {
    console.log('Running MLE-Star Workflow Integration Tests...');
    // This would normally be handled by a test runner like Mocha
}