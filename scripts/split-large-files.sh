#!/bin/bash

# Script to split large files for GitHub
# Files larger than 50MB will be split into 40MB chunks

echo "🔍 Splitting large files for GitHub..."

# Create splits directory
mkdir -p .splits

# Array of files to split (excluding .git and node_modules)
declare -a files_to_split=(
    "frontend/.next/cache/webpack/client-production/5.pack"
    "acli.exe"
)

for file in "${files_to_split[@]}"; do
    if [ -f "$file" ]; then
        echo "📦 Splitting $file..."
        
        # Get base name and directory
        base_name=$(basename "$file")
        dir_name=$(dirname "$file")
        
        # Create split directory structure
        split_dir=".splits/$dir_name"
        mkdir -p "$split_dir"
        
        # Split the file into 40MB chunks
        split -b 40M "$file" "$split_dir/${base_name}.part-"
        
        # Create metadata file
        echo "$file" > "$split_dir/${base_name}.original"
        md5sum "$file" > "$split_dir/${base_name}.md5"
        
        echo "✅ Split $file into chunks in $split_dir"
    else
        echo "⚠️ File not found: $file"
    fi
done

echo "✨ Splitting complete!"
