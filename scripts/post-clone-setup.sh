#!/bin/bash

# Post-clone setup script for NovaCron
# This script should be run after cloning the repository

echo "🚀 Setting up NovaCron after clone..."

# Check if we're in the right directory
if [ ! -f "package.json" ] || [ ! -d "backend" ]; then
    echo "❌ Error: This script must be run from the NovaCron project root"
    exit 1
fi

# Run the reassembly script
if [ -f "scripts/reassemble-files.sh" ]; then
    echo "📦 Reassembling large files..."
    ./scripts/reassemble-files.sh
fi

# Install backend dependencies
if [ -f "go.mod" ]; then
    echo "📚 Installing backend dependencies..."
    go mod download
fi

# Install frontend dependencies
if [ -f "frontend/package.json" ]; then
    echo "📚 Installing frontend dependencies..."
    cd frontend && npm install && cd ..
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs
mkdir -p data
mkdir -p temp

# Set up environment files
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    echo "⚙️ Creating .env file from example..."
    cp .env.example .env
    echo "⚠️ Please update .env with your configuration"
fi

echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Update .env with your configuration"
echo "  2. Run 'docker-compose up -d' to start services"
echo "  3. Access the application at http://localhost:15566"
