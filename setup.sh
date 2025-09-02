#!/bin/bash

# NovaCron Initial Setup Script
# This script automatically runs after cloning to reassemble large files

echo "🚀 NovaCron Initial Setup"
echo "========================="

# Reassemble large files first
if [ -f "scripts/reassemble-files.sh" ] && [ -d ".splits" ]; then
    echo "📦 Reassembling large files..."
    ./scripts/reassemble-files.sh
    
    # Clean up splits after successful reassembly
    if [ $? -eq 0 ]; then
        echo "🧹 Cleaning up split files..."
        rm -rf .splits
    fi
fi

# Install git hooks for future operations
if [ -d ".git" ]; then
    echo "🔧 Installing git hooks..."
    mkdir -p .git/hooks
    
    # Create post-checkout hook
    cat > .git/hooks/post-checkout << 'HOOK'
#!/bin/bash
# Automatically reassemble files after checkout
if [ -f "scripts/reassemble-files.sh" ] && [ -d ".splits" ]; then
    echo "📦 Reassembling large files after checkout..."
    ./scripts/reassemble-files.sh
fi
HOOK
    chmod +x .git/hooks/post-checkout
    
    # Create post-merge hook
    cp .git/hooks/post-checkout .git/hooks/post-merge
fi

echo "✅ Setup complete! Large files have been reassembled."
echo ""
echo "📝 Next steps:"
echo "  1. Install dependencies: npm install && cd frontend && npm install"
echo "  2. Start development: npm run dev"
