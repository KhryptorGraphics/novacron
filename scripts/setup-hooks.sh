#!/bin/bash
# Install Git hooks for automatic large file reassembly

set -e

HOOKS_DIR=".git/hooks"

echo "Installing Git hooks..."

# Copy hooks
cp .github/hooks/post-checkout "$HOOKS_DIR/post-checkout"
cp .github/hooks/post-merge "$HOOKS_DIR/post-merge"

# Make executable
chmod +x "$HOOKS_DIR/post-checkout"
chmod +x "$HOOKS_DIR/post-merge"

echo "✓ Git hooks installed successfully"
echo ""
echo "Hooks installed:"
echo "  - post-checkout: Runs after git checkout/clone"
echo "  - post-merge: Runs after git pull/merge"
echo ""
echo "Large files will be automatically reassembled when you:"
echo "  - Clone the repository"
echo "  - Switch branches"
echo "  - Pull changes"
