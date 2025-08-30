#!/bin/bash
# Script to split large files for GitHub storage
# Files are split into 40MB chunks for safety (GitHub limit is 100MB)

set -e

echo "==================================="
echo "Splitting Large Files for GitHub"
echo "==================================="

# Configuration
CHUNK_SIZE="40M"
SPLIT_DIR=".github/split-files"

# Create split directory
mkdir -p "$SPLIT_DIR"

# Function to split a file
split_file() {
    local file_path="$1"
    local file_name=$(basename "$file_path")
    local file_dir=$(dirname "$file_path")
    local relative_path="${file_path#./}"
    
    echo "Processing: $relative_path"
    
    # Create directory structure in split folder
    local split_subdir="$SPLIT_DIR/${relative_path%/*}"
    mkdir -p "$split_subdir"
    
    # Split the file
    split -b "$CHUNK_SIZE" -d "$file_path" "$SPLIT_DIR/${relative_path}.part"
    
    # Create metadata file
    local metadata_file="$SPLIT_DIR/${relative_path}.metadata"
    echo "original_path: $relative_path" > "$metadata_file"
    echo "original_size: $(stat -c%s "$file_path" 2>/dev/null || stat -f%z "$file_path" 2>/dev/null)" >> "$metadata_file"
    echo "chunks: $(ls "$SPLIT_DIR/${relative_path}.part"* | wc -l)" >> "$metadata_file"
    echo "md5sum: $(md5sum "$file_path" | cut -d' ' -f1)" >> "$metadata_file"
    
    # Remove original large file from git tracking
    git rm --cached "$file_path" 2>/dev/null || true
    
    # Add to .gitignore if not already there
    if ! grep -q "^${relative_path}$" .gitignore 2>/dev/null; then
        echo "$relative_path" >> .gitignore
    fi
    
    echo "✓ Split into $(ls "$SPLIT_DIR/${relative_path}.part"* | wc -l) parts"
}

# Split large webpack cache files
if [ -f "frontend/.next/cache/webpack/client-production/0.pack" ]; then
    split_file "frontend/.next/cache/webpack/client-production/0.pack"
fi

if [ -f "frontend/.next/cache/webpack/server-production/0.pack" ]; then
    split_file "frontend/.next/cache/webpack/server-production/0.pack"
fi

# Split other large files
if [ -f "acli.exe" ]; then
    split_file "acli.exe"
fi

# Also split large node_modules files if needed
if [ -f "frontend/node_modules/@next/swc-linux-x64-gnu/next-swc.linux-x64-gnu.node" ]; then
    split_file "frontend/node_modules/@next/swc-linux-x64-gnu/next-swc.linux-x64-gnu.node"
fi

if [ -f "frontend/node_modules/@next/swc-linux-x64-musl/next-swc.linux-x64-musl.node" ]; then
    split_file "frontend/node_modules/@next/swc-linux-x64-musl/next-swc.linux-x64-musl.node"
fi

echo ""
echo "==================================="
echo "Large files have been split!"
echo "==================================="
echo ""
echo "Split files are stored in: $SPLIT_DIR"
echo "Original files have been added to .gitignore"
echo ""
echo "Run ./scripts/reassemble-files.sh after cloning to restore files"