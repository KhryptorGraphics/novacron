#!/bin/bash

# NovaCron Quick Install Script
# Run this immediately after cloning

echo "╔══════════════════════════════════════╗"
echo "║     NovaCron Automatic Setup         ║"
echo "╚══════════════════════════════════════╝"

# 1. Reassemble large files
if [ -d ".splits" ]; then
    echo ""
    echo "📦 Reassembling large files..."
    if [ -f "scripts/reassemble-files.sh" ]; then
        ./scripts/reassemble-files.sh
        if [ $? -eq 0 ]; then
            echo "🧹 Cleaning up split files..."
            rm -rf .splits
        fi
    fi
fi

# 2. Install git hooks
if [ -d ".git" ] && [ -d ".githooks" ]; then
    echo ""
    echo "🔧 Installing git hooks..."
    git config core.hooksPath .githooks
    echo "✅ Git hooks installed"
fi

# 3. Check for dependencies
echo ""
echo "📚 Checking dependencies..."

# Check Node.js
if command -v node &> /dev/null; then
    echo "✅ Node.js $(node --version) found"
else
    echo "❌ Node.js not found. Please install Node.js 18+"
fi

# Check npm
if command -v npm &> /dev/null; then
    echo "✅ npm $(npm --version) found"
else
    echo "❌ npm not found. Please install npm"
fi

# Check Go
if command -v go &> /dev/null; then
    echo "✅ Go $(go version | awk '{print $3}') found"
else
    echo "❌ Go not found. Please install Go 1.21+"
fi

echo ""
echo "╔══════════════════════════════════════╗"
echo "║        Setup Complete! 🎉            ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "📝 Next steps:"
echo "  1. Install dependencies:"
echo "     npm install"
echo "     cd frontend && npm install"
echo ""
echo "  2. Start development:"
echo "     npm run dev"
echo ""
echo "  3. Or use Docker:"
echo "     docker-compose up -d"
echo ""
