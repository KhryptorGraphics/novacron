#!/usr/bin/env node
/**
 * Direct test of MLE-Star functionality
 */

const path = require('path');
const MLEStarWorkflow = require('./src/mle-star/index.js');

console.log('🧪 Testing MLE-Star Workflow Direct Integration...\n');

async function testMleStarDirect() {
    try {
        console.log('✅ MLE-Star module loaded successfully');
        
        const workflow = new MLEStarWorkflow({
            mlFramework: 'pytorch',
            projectPath: './test-ml-project',
            experimentName: 'test-experiment',
            logLevel: 'info'
        });
        
        console.log('✅ MLE-Star workflow instance created');
        console.log('   Framework:', workflow.config.mlFramework);
        console.log('   Project:', workflow.config.projectPath);
        console.log('   Experiment:', workflow.config.experimentName);
        
        // Test environment validation
        console.log('\n🔍 Testing environment validation...');
        const validation = await workflow.validateEnvironment();
        console.log('✅ Environment validation completed');
        
        // Test project analysis
        console.log('\n📊 Testing project analysis...');
        const analysis = await workflow.analyzeProject();
        console.log('✅ Project analysis completed');
        console.log('   Analysis:', JSON.stringify(analysis, null, 2).substring(0, 200) + '...');
        
        console.log('\n🎉 MLE-Star Direct Test: SUCCESS!');
        console.log('\n📋 Available methods:');
        const methods = Object.getOwnPropertyNames(Object.getPrototypeOf(workflow))
            .filter(name => name !== 'constructor' && typeof workflow[name] === 'function');
        methods.forEach(method => console.log(`   - ${method}()`));
        
        return true;
        
    } catch (error) {
        console.error('❌ MLE-Star Direct Test Failed:', error.message);
        console.error('   Stack:', error.stack.split('\n')[0]);
        return false;
    }
}

// Run the test
testMleStarDirect().then(success => {
    if (success) {
        console.log('\n✅ MLE-Star is ready for claude-flow integration!');
    } else {
        console.log('\n❌ MLE-Star needs fixes before claude-flow integration');
        process.exit(1);
    }
});