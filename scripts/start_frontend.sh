#!/bin/bash
# Script to start the NovaCron frontend

set -e

echo "Starting NovaCron frontend..."

# Start the frontend
cd frontend
npm run dev

echo "Frontend started on port 3000"
