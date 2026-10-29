#!/usr/bin/env node
/**
 * MLE-Star Automation Extension for Claude-Flow
 * This extends claude-flow automation with mle-star command
 */

const { spawn } = require('child_process');
const path = require('path');

// Get the directory where this script is located
const scriptDir = path.dirname(__filename);
const mleStarCommand = path.join(scriptDir, 'mle-star-command.js');

// Function to execute MLE-Star command
function executeMleStarCommand(args) {
    return new Promise((resolve, reject) => {
        const child = spawn('node', [mleStarCommand, ...args], {
            stdio: 'inherit',
            cwd: scriptDir
        });

        child.on('close', (code) => {
            if (code === 0) {
                resolve();
            } else {
                reject(new Error(`MLE-Star command failed with code ${code}`));
            }
        });

        child.on('error', (error) => {
            reject(error);
        });
    });
}

// Parse arguments - claude-flow automation mle-star [options]
const args = process.argv.slice(2);

// Remove 'mle-star' if it's the first argument (from claude-flow automation mle-star)
if (args[0] === 'mle-star') {
    args.shift();
}

// Execute the MLE-Star command
executeMleStarCommand(args).catch(error => {
    console.error('❌ MLE-Star Automation Error:', error.message);
    process.exit(1);
});
