#!/bin/bash

# Script to reassemble split files after cloning repository
# This should be run automatically after git clone

echo "🔄 Reassembling large files..."

# Check if splits directory exists
if [ ! -d ".splits" ]; then
    echo "ℹ️ No split files found. Repository is ready to use."
    exit 0
fi

# Find all .original files and reassemble them
find .splits -name "*.original" | while read original_file; do
    # Get the target path
    target_path=$(cat "$original_file")
    base_name=$(basename "$target_path")
    dir_name=$(dirname "$original_file")
    
    echo "🔧 Reassembling $target_path..."
    
    # Create target directory if it doesn't exist
    mkdir -p "$(dirname "$target_path")"
    
    # Concatenate all part files
    cat "$dir_name/${base_name}.part-"* > "$target_path"
    
    # Verify MD5 checksum if available
    if [ -f "$dir_name/${base_name}.md5" ]; then
        echo "🔍 Verifying checksum..."
        original_md5=$(cat "$dir_name/${base_name}.md5" | awk '{print $1}')
        new_md5=$(md5sum "$target_path" | awk '{print $1}')
        
        if [ "$original_md5" = "$new_md5" ]; then
            echo "✅ Checksum verified for $target_path"
        else
            echo "❌ Checksum mismatch for $target_path!"
            echo "   Expected: $original_md5"
            echo "   Got: $new_md5"
        fi
    fi
done

echo "✨ Reassembly complete!"
echo ""
echo "📝 You can now safely remove the .splits directory if desired:"
echo "   rm -rf .splits"
