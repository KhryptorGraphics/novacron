/**
 * MLE-Star Plugin for Claude-Flow Automation System
 * Extends automation commands with ML engineering capabilities
 */

const mleStarIntegration = require('./claude-flow-integration.js');

// Plugin registration for claude-flow automation system
const mleStarPlugin = {
    name: 'mle-star',
    version: '1.0.0',
    description: 'Machine Learning Engineering with Systematic Training, Analysis, and Refinement',
    
    // Initialize plugin
    init: (claudeFlow) => {
        console.log('🚀 Initializing MLE-Star plugin for claude-flow...');
        
        // Register MLE-Star automation commands
        mleStarIntegration.register();
        
        return {
            success: true,
            commands: ['mle-star'],
            stages: mleStarIntegration.stages
        };
    },

    // Plugin commands
    commands: {
        'mle-star': {
            description: mleStarIntegration.getHelp().description,
            usage: mleStarIntegration.getHelp().usage,
            options: mleStarIntegration.getHelp().options,
            examples: mleStarIntegration.getHelp().examples,
            handler: mleStarIntegration.register
        }
    },

    // Plugin metadata
    metadata: {
        category: 'Machine Learning',
        tags: ['ml', 'engineering', 'automation', 'workflow', 'systematic'],
        frameworks: ['pytorch', 'tensorflow', 'scikit-learn'],
        languages: ['python'],
        platforms: ['linux', 'macos', 'windows']
    }
};

// Export plugin for claude-flow discovery
module.exports = mleStarPlugin;

// Auto-register if loaded directly
if (require.main === module) {
    console.log('🔧 MLE-Star plugin loaded directly - registering with claude-flow...');
    mleStarPlugin.init();
}