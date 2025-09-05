/**
 * Claude-Flow MLE-Star Integration Module
 * Registers MLE-Star commands with claude-flow automation system
 */

const MLEStarWorkflow = require('./index.js');
const { program } = require('commander');

// MLE-Star automation integration for claude-flow
const mleStarIntegration = {
    
    // Register MLE-Star with claude-flow automation system  
    register: () => {
        // Add mle-star command to automation system
        program
            .command('mle-star')
            .description('🚀 MLE-Star Workflow - Machine Learning Engineering with Systematic Training, Analysis, and Refinement')
            .option('-f, --framework <framework>', 'ML framework (pytorch|tensorflow|scikit-learn)', 'pytorch')
            .option('-p, --project-path <path>', 'Project directory path', process.cwd())
            .option('-n, --name <name>', 'Experiment name', 'ml-experiment')
            .option('-d, --data-path <path>', 'Data directory path', './data')
            .option('-o, --output-path <path>', 'Output directory path', './outputs')
            .option('--init', 'Initialize new MLE-Star project')
            .option('--run', 'Run complete MLE-Star workflow')
            .option('--stage <stage>', 'Run specific workflow stage')
            .option('--validate', 'Validate environment and dependencies')
            .option('--analyze', 'Analyze project and generate report')
            .option('--deploy <service>', 'Deploy model (api|batch|stream)')
            .action(async (options) => {
                try {
                    const workflow = new MLEStarWorkflow({
                        mlFramework: options.framework,
                        projectPath: options.projectPath,
                        experimentName: options.name,
                        dataPath: options.dataPath,
                        outputPath: options.outputPath
                    });

                    if (options.init) {
                        await workflow.initializeProject();
                        console.log('✅ MLE-Star project initialized successfully');
                    } else if (options.run) {
                        await workflow.runCompleteWorkflow();
                        console.log('✅ MLE-Star workflow completed successfully');
                    } else if (options.stage) {
                        await workflow.runStage(options.stage);
                        console.log(`✅ MLE-Star stage '${options.stage}' completed`);
                    } else if (options.validate) {
                        await workflow.validateEnvironment();
                        console.log('✅ Environment validation completed');
                    } else if (options.analyze) {
                        const report = await workflow.analyzeProject();
                        console.log('✅ Project analysis completed');
                        console.log(JSON.stringify(report, null, 2));
                    } else if (options.deploy) {
                        await workflow.deployModel(options.deploy);
                        console.log(`✅ Model deployed as ${options.deploy} service`);
                    } else {
                        // Default: show help
                        program.commands.find(cmd => cmd.name() === 'mle-star').help();
                    }
                } catch (error) {
                    console.error('❌ MLE-Star Error:', error.message);
                    process.exit(1);
                }
            });

        return true;
    },

    // Get command help for automation system
    getHelp: () => {
        return {
            name: 'mle-star',
            description: 'MLE-Star Workflow - Machine Learning Engineering with Systematic Training, Analysis, and Refinement',
            usage: 'claude-flow automation mle-star [options]',
            options: [
                { flag: '-f, --framework <framework>', description: 'ML framework (pytorch|tensorflow|scikit-learn)' },
                { flag: '-p, --project-path <path>', description: 'Project directory path' },
                { flag: '-n, --name <name>', description: 'Experiment name' },
                { flag: '--init', description: 'Initialize new MLE-Star project' },
                { flag: '--run', description: 'Run complete MLE-Star workflow' },
                { flag: '--stage <stage>', description: 'Run specific workflow stage' },
                { flag: '--validate', description: 'Validate environment and dependencies' },
                { flag: '--analyze', description: 'Analyze project and generate report' },
                { flag: '--deploy <service>', description: 'Deploy model (api|batch|stream)' }
            ],
            examples: [
                'claude-flow automation mle-star --init --framework pytorch',
                'claude-flow automation mle-star --run --name my-experiment',
                'claude-flow automation mle-star --stage model_design',
                'claude-flow automation mle-star --validate',
                'claude-flow automation mle-star --analyze --output report.json',
                'claude-flow automation mle-star --deploy api'
            ]
        };
    },

    // Available workflow stages
    stages: [
        'model_design',        // M: Model Design and Architecture
        'learning_pipeline',   // L: Learning Pipeline Setup  
        'evaluation_setup',    // E: Evaluation and Metrics
        'systematic_testing',  // S: Systematic Testing
        'training_optimization', // T: Training Optimization
        'analysis_validation', // A: Analysis and Validation
        'refinement_deployment' // R: Refinement and Deployment
    ]
};

module.exports = mleStarIntegration;