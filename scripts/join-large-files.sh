#!/bin/bash
# Reassemble split large files after clone
# This script runs automatically via Git hooks

set -e

SPLIT_DIR=".github/large-files"

echo "Reassembling large files..."

# Find all unique file prefixes
for checksum_file in "$SPLIT_DIR"/*.sha256; do
    [ -f "$checksum_file" ] || continue

    filename=$(basename "$checksum_file" .sha256)
    target_file=""

    # Determine target path based on filename
    case "$filename" in
        knowledge_base_*)
            target_file="src/neural/$filename"
            ;;
        neural_analysis_report_*)
            target_file="docs/$filename"
            ;;
    esac

    if [ -z "$target_file" ]; then
        echo "⚠ Unknown file pattern: $filename"
        continue
    fi

    # Skip if file already exists and is valid
    if [ -f "$target_file" ]; then
        if sha256sum -c "$checksum_file" --status 2>/dev/null; then
            echo "✓ $target_file already exists and is valid"
            continue
        fi
    fi

    echo "Reassembling $filename..."

    # Create target directory
    mkdir -p "$(dirname "$target_file")"

    # Join the parts
    cat "$SPLIT_DIR/${filename}.part"* > "$target_file"

    # Verify checksum
    if sha256sum -c "$checksum_file" --status; then
        echo "✓ Successfully reassembled and verified $target_file"
    else
        echo "✗ Checksum verification failed for $target_file"
        rm -f "$target_file"
        exit 1
    fi
done

echo "✓ All large files reassembled successfully"
