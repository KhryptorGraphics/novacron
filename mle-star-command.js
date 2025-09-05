#!/usr/bin/env node
/**
 * MLE-Star Command for Claude-Flow Automation
 * Direct integration without complex plugin system
 */

const path = require('path');
const MLEStarWorkflow = require('./src/mle-star/index.js');

// Parse command line arguments
const args = process.argv.slice(2);
let options = {
    framework: 'pytorch',
    projectPath: process.cwd(),
    name: 'ml-experiment'
};

// Parse command line arguments
let action = 'help';
for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    switch (arg) {
        case '--init':
            action = 'init';
            break;
        case '--run':
            action = 'run';
            break;
        case '--validate':
            action = 'validate';
            break;
        case '--stage':
            action = 'stage';
            options.stage = args[++i];
            break;
        case '--framework':
            options.framework = args[++i];
            break;
        case '--project-path':
            options.projectPath = args[++i];
            break;
        case '--name':
            options.name = args[++i];
            break;
        case '--help':
        case '-h':
            action = 'help';
            break;
    }
}

function showHelp() {
    console.log(`🚀 MLE-Star Workflow - Machine Learning Engineering with Systematic Training, Analysis, and Refinement

USAGE:
  node mle-star-command.js [options]
  npx claude-flow@alpha automation mle-star [options] (when integrated)

OPTIONS:
  --init                        Initialize new MLE-Star project
  --run                         Run complete MLE-Star workflow
  --stage <stage>              Run specific workflow stage
  --validate                   Validate environment and dependencies
  --framework <framework>      ML framework (pytorch|tensorflow|scikit-learn)
  --project-path <path>        Project directory path
  --name <name>                Experiment name
  --help, -h                   Show this help

EXAMPLES:
  node mle-star-command.js --init --framework pytorch
  node mle-star-command.js --run --name my-experiment
  node mle-star-command.js --stage model_design
  node mle-star-command.js --validate

WORKFLOW STAGES:
  - model_design:         M: Model Design and Architecture
  - learning_pipeline:    L: Learning Pipeline Setup
  - evaluation_setup:     E: Evaluation and Metrics  
  - systematic_testing:   S: Systematic Testing
  - training_optimization: T: Training Optimization
  - analysis_validation:  A: Analysis and Validation
  - refinement_deployment: R: Refinement and Deployment

🎯 MLE-Star Benefits:
  • Systematic ML development methodology
  • Multi-framework support (PyTorch, TensorFlow, Scikit-learn)
  • Comprehensive templates and automation
  • Production-ready code generation
  • Experiment tracking and reproducibility`);
}

async function runCommand() {
    try {
        if (action === 'help') {
            showHelp();
            return;
        }

        console.log('🚀 MLE-Star Workflow Starting...');
        console.log(`   Framework: ${options.framework}`);
        console.log(`   Project: ${options.projectPath}`);
        console.log(`   Experiment: ${options.name}\n`);

        const workflow = new MLEStarWorkflow({
            mlFramework: options.framework,
            projectPath: options.projectPath,
            experimentName: options.name
        });

        switch (action) {
            case 'init':
                await workflow.initializeProject();
                console.log('✅ MLE-Star project initialized successfully');
                break;
            case 'run':
                await workflow.runFullWorkflow();
                console.log('✅ MLE-Star workflow completed successfully');
                break;
            case 'stage':
                if (!options.stage) {
                    console.error('❌ Stage name required. Use --stage <stage_name>');
                    process.exit(1);
                }
                await workflow.executeStage(options.stage);
                console.log(`✅ MLE-Star stage '${options.stage}' completed`);
                break;
            case 'validate':
                await workflow.validateEnvironment();
                console.log('✅ Environment validation completed');
                break;
            default:
                showHelp();
        }
    } catch (error) {
        console.error('❌ MLE-Star Error:', error.message);
        if (options.verbose) {
            console.error(error.stack);
        }
        process.exit(1);
    }
}

runCommand();