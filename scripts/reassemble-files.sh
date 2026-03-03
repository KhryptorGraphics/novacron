#!/bin/bash
# Script to reassemble split files after cloning from GitHub

set -e

echo "==================================="
echo "Reassembling Split Files"
echo "==================================="

SPLIT_DIR=".github/split-files"

# Check if split directory exists
if [ ! -d "$SPLIT_DIR" ]; then
    echo "No split files found. Repository appears complete."
    exit 0
fi

# Function to reassemble a file
reassemble_file() {
    local metadata_file="$1"
    local base_path="${metadata_file%.metadata}"
    local relative_path=$(grep "^original_path:" "$metadata_file" | cut -d' ' -f2)
    local original_size=$(grep "^original_size:" "$metadata_file" | cut -d' ' -f2)
    local expected_md5=$(grep "^md5sum:" "$metadata_file" | cut -d' ' -f2)
    
    echo "Reassembling: $relative_path"
    
    # Create directory if it doesn't exist
    local target_dir=$(dirname "$relative_path")
    mkdir -p "$target_dir"
    
    # Combine all parts
    cat "${base_path}.part"* > "$relative_path"
    
    # Verify size
    local actual_size=$(stat -c%s "$relative_path" 2>/dev/null || stat -f%z "$relative_path" 2>/dev/null)
    if [ "$actual_size" != "$original_size" ]; then
        echo "⚠️  Warning: Size mismatch for $relative_path (expected: $original_size, got: $actual_size)"
    fi
    
    # Verify MD5 if md5sum is available
    if command -v md5sum &> /dev/null; then
        local actual_md5=$(md5sum "$relative_path" | cut -d' ' -f1)
        if [ "$actual_md5" != "$expected_md5" ]; then
            echo "⚠️  Warning: MD5 mismatch for $relative_path"
        else
            echo "✓ MD5 verified"
        fi
    fi
    
    echo "✓ Reassembled: $relative_path"
}

# Process all metadata files
find "$SPLIT_DIR" -name "*.metadata" | while read metadata_file; do
    reassemble_file "$metadata_file"
done

echo ""
echo "==================================="
echo "File Reassembly Complete!"
echo "==================================="
echo ""
echo "All split files have been reassembled."
echo "You can now run the application normally."
echo ""
echo "Optional: Remove split files to save space:"
echo "  rm -rf $SPLIT_DIR"