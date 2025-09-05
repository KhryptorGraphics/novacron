#!/usr/bin/env node
/**
 * MLE-Star Registration Script for Claude-Flow
 * Registers the MLE-Star workflow with claude-flow automation system
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

console.log('🚀 Registering MLE-Star workflow with claude-flow automation system...\n');

try {
    // Check if claude-flow is available
    execSync('npx claude-flow@alpha --version', { stdio: 'pipe' });
    console.log('✅ Claude-Flow found');

    // Create symlink or registration for MLE-Star
    const mleStarPath = path.resolve(__dirname, 'src/mle-star');
    const pluginPath = path.join(mleStarPath, 'plugin.js');
    
    if (fs.existsSync(pluginPath)) {
        console.log('✅ MLE-Star plugin found');
        
        // Load and test the plugin
        const plugin = require(pluginPath);
        const result = plugin.init();
        
        if (result.success) {
            console.log('✅ MLE-Star plugin initialized successfully');
            console.log(`   Commands: ${result.commands.join(', ')}`);
            console.log(`   Stages: ${result.stages.slice(0, 3).join(', ')}...`);
        }
    } else {
        throw new Error('MLE-Star plugin not found');
    }

    // Create a wrapper script for the automation command
    const wrapperScript = `#!/usr/bin/env node
const path = require('path');
const mleStarPath = path.join(__dirname, 'src/mle-star');
const MLEStarWorkflow = require(path.join(mleStarPath, 'index.js'));

// Parse command line arguments
const args = process.argv.slice(2);
const options = {};

// Simple argument parsing
for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--init') options.init = true;
    else if (arg === '--run') options.run = true;
    else if (arg === '--validate') options.validate = true;
    else if (arg === '--analyze') options.analyze = true;
    else if (arg === '--framework') options.framework = args[++i];
    else if (arg === '--stage') options.stage = args[++i];
    else if (arg === '--deploy') options.deploy = args[++i];
    else if (arg === '--project-path') options.projectPath = args[++i];
    else if (arg === '--name') options.name = args[++i];
    else if (arg === '--help') {
        console.log(\`🚀 MLE-Star Workflow - Machine Learning Engineering with Systematic Training, Analysis, and Refinement

USAGE:
  npx claude-flow@alpha automation mle-star [options]

OPTIONS:
  --init                        Initialize new MLE-Star project
  --run                         Run complete MLE-Star workflow
  --stage <stage>              Run specific workflow stage
  --validate                   Validate environment and dependencies
  --analyze                    Analyze project and generate report
  --deploy <service>           Deploy model (api|batch|stream)
  --framework <framework>      ML framework (pytorch|tensorflow|scikit-learn)
  --project-path <path>        Project directory path
  --name <name>                Experiment name

EXAMPLES:
  npx claude-flow@alpha automation mle-star --init --framework pytorch
  npx claude-flow@alpha automation mle-star --run --name my-experiment
  npx claude-flow@alpha automation mle-star --stage model_design
  npx claude-flow@alpha automation mle-star --validate
  npx claude-flow@alpha automation mle-star --analyze

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
  • Experiment tracking and reproducibility\`);
        process.exit(0);
    }
}

// Execute MLE-Star workflow
async function runMleStarCommand() {
    try {
        const workflow = new MLEStarWorkflow({
            mlFramework: options.framework || 'pytorch',
            projectPath: options.projectPath || process.cwd(),
            experimentName: options.name || 'ml-experiment',
        });

        if (options.init) {
            await workflow.initializeProject();
            console.log('✅ MLE-Star project initialized successfully');
        } else if (options.run) {
            await workflow.runCompleteWorkflow();
            console.log('✅ MLE-Star workflow completed successfully');
        } else if (options.stage) {
            await workflow.runStage(options.stage);
            console.log(\`✅ MLE-Star stage '\${options.stage}' completed\`);
        } else if (options.validate) {
            await workflow.validateEnvironment();
            console.log('✅ Environment validation completed');
        } else if (options.analyze) {
            const report = await workflow.analyzeProject();
            console.log('✅ Project analysis completed');
            console.log(JSON.stringify(report, null, 2));
        } else if (options.deploy) {
            await workflow.deployModel(options.deploy);
            console.log(\`✅ Model deployed as \${options.deploy} service\`);
        } else {
            // Show help by default
            process.argv.push('--help');
            runMleStarCommand();
        }
    } catch (error) {
        console.error('❌ MLE-Star Error:', error.message);
        process.exit(1);
    }
}

runMleStarCommand();
`;

    fs.writeFileSync('./mle-star-wrapper.js', wrapperScript);
    fs.chmodSync('./mle-star-wrapper.js', 0o755);
    console.log('✅ MLE-Star wrapper script created');

    console.log('\n🎉 MLE-Star registration completed!\n');
    console.log('📋 Available commands:');
    console.log('  npx claude-flow@alpha automation mle-star --help');
    console.log('  npx claude-flow@alpha automation mle-star --init --framework pytorch');
    console.log('  npx claude-flow@alpha automation mle-star --validate');
    console.log('  node ./mle-star-wrapper.js --help');
    console.log('\n🚀 MLE-Star is now ready to use!');

} catch (error) {
    console.error('❌ Registration failed:', error.message);
    console.error('\n🔧 Troubleshooting:');
    console.error('1. Ensure claude-flow@alpha is installed: npm install -g claude-flow@alpha');
    console.error('2. Check that src/mle-star/ directory exists with all required files');
    console.error('3. Verify Node.js version >= 14.0.0');
    process.exit(1);
}