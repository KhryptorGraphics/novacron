#!/bin/bash
# Split large files for GitHub compatibility
# Files are split into 50MB chunks and reassembled automatically

set -e

CHUNK_SIZE="50M"
SPLIT_DIR=".github/large-files"

echo "Creating split directory..."
mkdir -p "$SPLIT_DIR"

# List of large files to split
LARGE_FILES=(
    "src/neural/knowledge_base_20251011_124231.json"
    "docs/neural_analysis_report_20251011_124231.json"
)

for file in "${LARGE_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Splitting $file..."
        filename=$(basename "$file")

        # Split the file
        split -b "$CHUNK_SIZE" -d "$file" "$SPLIT_DIR/${filename}.part"

        # Create checksum
        sha256sum "$file" > "$SPLIT_DIR/${filename}.sha256"

        echo "✓ Split $file into chunks"
    else
        echo "⚠ File not found: $file"
    fi
done

echo "✓ All large files split successfully"
