#!/bin/bash
# MLE-Star Alias Script for Claude-Flow
# Usage: ./mle-star.sh [options]

# Change to script directory
cd "$(dirname "$0")"

# Execute MLE-Star command
node mle-star-command.js "$@"
