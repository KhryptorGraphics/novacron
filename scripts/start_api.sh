#!/bin/bash
# Script to start the NovaCron API service

set -e

echo "Starting NovaCron API service..."

# Set environment variables
export CONFIG_PATH="$(pwd)/config/novacron/api.yaml"

# Start the API service
cd backend/services/api
python3 main.py --host 0.0.0.0 --port 8090 --config "$CONFIG_PATH" --debug

echo "API service started on port 8090"
