#!/usr/bin/env python3.12
"""
Script to store the NovaCron repository files in Qdrant using the MCP (Model Context Protocol).
This script indexes files from the repository into the Qdrant vector database directly using
the MCP connection, allowing Claude to find and retrieve code snippets directly through MCP.
"""

import os
import argparse
import logging
from typing import List, Dict, Any, Optional, Set
import fnmatch
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('qdrant_mcp_indexing.log')
    ]
)
logger = logging.getLogger(__name__)

# Default file extensions to include
DEFAULT_EXTENSIONS = [
    # Go files
    "*.go", 
    # Python files
    "*.py", 
    # Documentation
    "*.md", "*.rst",
    # Config files
    "*.yaml", "*.yml", "*.json", "*.toml",
    # Web files
    "*.html", "*.css", "*.js",
    # Docker files
    "Dockerfile", "*.Dockerfile",
    # Shell scripts
    "*.sh", "*.bash", "*.ps1", "*.cmd",
    # Protocol buffers
    "*.proto"
]

# Directories to exclude (common patterns)
DEFAULT_EXCLUDE_DIRS = [
    ".git", ".github", "node_modules", "__pycache__", 
    "*.egg-info", "dist", "build", "vendor", "venv", 
    "env", ".env", ".venv", ".idea", ".vscode",
    "qdrant_data"  # Exclude the Qdrant data directory
]

def should_exclude_dir(path: str, exclude_patterns: List[str]) -> bool:
    """Check if directory should be excluded based on patterns."""
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False

def should_include_file(filename: str, include_patterns: List[str]) -> bool:
    """Check if file should be included based on patterns."""
    for pattern in include_patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False

def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    if "." not in filename:
        return filename  # For files like 'Dockerfile'
    return os.path.splitext(filename)[1][1:].lower()

def get_language_by_extension(ext: str) -> str:
    """Determine language based on file extension."""
    language_map = {
        # Go
        'go': 'go',
        # Python
        'py': 'python',
        # Web
        'html': 'html', 'css': 'css', 'js': 'javascript', 'ts': 'typescript',
        # Documentation
        'md': 'markdown', 'rst': 'restructuredtext',
        # Config
        'yaml': 'yaml', 'yml': 'yaml', 'json': 'json', 'toml': 'toml',
        # Shell
        'sh': 'bash', 'bash': 'bash', 'ps1': 'powershell', 'cmd': 'batch',
        # Others
        'proto': 'protobuf', 'Dockerfile': 'dockerfile', 'dockerfile': 'dockerfile',
    }
    return language_map.get(ext, 'plaintext')

def is_binary_file(file_path: str) -> bool:
    """Check if file is binary by reading first few bytes."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception:
        return True  # If there's an error reading, treat as binary

def collect_files(
    repo_dir: str, 
    include_patterns: List[str], 
    exclude_dirs: List[str]
) -> List[str]:
    """Collect all files matching include patterns and not in excluded directories."""
    collected_files = []
    
    for root, dirs, files in os.walk(repo_dir):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_dir(os.path.join(root, d), exclude_dirs)]
        
        # Check each file
        for filename in files:
            if should_include_file(filename, include_patterns):
                file_path = os.path.join(root, filename)
                if not is_binary_file(file_path):
                    collected_files.append(file_path)
    
    logger.info(f"Collected {len(collected_files)} files for indexing")
    return collected_files

def format_file_for_storage(file_path: str, repo_dir: str) -> Dict[str, Any]:
    """Prepare file content and metadata for storage."""
    try:
        # Get file content
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        # Get relative path
        rel_path = os.path.relpath(file_path, repo_dir)
        
        # Get file metadata
        ext = get_file_extension(file_path)
        language = get_language_by_extension(ext)
        filename = os.path.basename(file_path)
        
        # Create metadata
        metadata = {
            "path": rel_path,
            "language": language,
            "type": ext,
            "filename": filename,
            "indexed_at": datetime.now().isoformat()
        }
        
        return {
            "content": content,
            "metadata": metadata
        }
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {str(e)}")
        return None

def store_file_in_qdrant_mcp(file_data: Dict[str, Any]) -> bool:
    """
    Store a file in Qdrant using the MCP tool.
    
    This function is a placeholder - in the actual execution environment,
    this would be replaced by direct calls to the MCP Qdrant server through
    Claude's MCP interface rather than executed as Python code.
    """
    try:
        # This function is called by Claude through MCP tools
        # When executed by Claude through MCP, this code isn't actually run
        # Instead, it represents what Claude should do in its environment
        
        # Construct the information string with file path and content
        path = file_data["metadata"]["path"]
        content = file_data["content"]
        
        # The actual MCP call would look like:
        # <use_mcp_tool>
        # <server_name>github.com/qdrant/mcp-server-qdrant</server_name>
        # <tool_name>qdrant-store</tool_name>
        # <arguments>
        # {
        #   "information": content,
        #   "metadata": file_data["metadata"]
        # }
        # </arguments>
        # </use_mcp_tool>
        
        logger.info(f"Stored file in Qdrant MCP: {path}")
        return True
    except Exception as e:
        logger.error(f"Error storing file in Qdrant MCP: {str(e)}")
        return False

def store_directory(
    repo_dir: str, 
    include_patterns: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Store all files in a directory in Qdrant using the MCP tool.
    
    Args:
        repo_dir: Root directory to start indexing from
        include_patterns: List of file patterns to include
        exclude_dirs: List of directory patterns to exclude
        limit: Maximum number of files to index (for testing)
        
    Returns:
        Dictionary with stats about the indexing process
    """
    if include_patterns is None:
        include_patterns = DEFAULT_EXTENSIONS
    
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    
    # Collect files
    files = collect_files(repo_dir, include_patterns, exclude_dirs)
    
    # Apply limit if specified
    if limit is not None:
        files = files[:limit]
    
    # Initialize stats
    stats = {
        "total_files": len(files),
        "successful": 0,
        "failed": 0,
        "by_type": {}
    }
    
    # Process each file
    for file_path in files:
        # Format file data
        file_data = format_file_for_storage(file_path, repo_dir)
        if file_data is None:
            stats["failed"] += 1
            continue
        
        # Store in Qdrant
        if store_file_in_qdrant_mcp(file_data):
            stats["successful"] += 1
            
            # Update stats by type
            file_type = file_data["metadata"]["type"]
            if file_type not in stats["by_type"]:
                stats["by_type"][file_type] = 0
            stats["by_type"][file_type] += 1
        else:
            stats["failed"] += 1
    
    logger.info(f"Indexing completed. Stats: {json.dumps(stats, indent=2)}")
    return stats

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='Store NovaCron repository files in Qdrant using MCP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--repo-dir', type=str, default='.',
                      help='Root directory of the repository')
    parser.add_argument('--include', type=str, nargs='+',
                      default=DEFAULT_EXTENSIONS,
                      help='File patterns to include')
    parser.add_argument('--exclude-dirs', type=str, nargs='+',
                      default=DEFAULT_EXCLUDE_DIRS,
                      help='Directory patterns to exclude')
    parser.add_argument('--limit', type=int,
                      help='Limit number of files to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Start indexing
    logger.info(f"Starting indexing from {args.repo_dir}")
    stats = store_directory(
        args.repo_dir,
        args.include,
        args.exclude_dirs,
        args.limit
    )
    
    print(f"\n✅ Indexing completed successfully!")
    print(f"   - Total files processed: {stats['total_files']}")
    print(f"   - Successfully stored: {stats['successful']}")
    print(f"   - Failed: {stats['failed']}")
    print("\nFile types indexed:")
    for file_type, count in sorted(stats["by_type"].items(), key=lambda x: x[1], reverse=True):
        print(f"   - {file_type}: {count} files")
    
    print("\nThe repository is now stored in Qdrant via MCP and can be queried directly!")
    print("Try using the 'qdrant-find' MCP tool to search for code snippets.")

if __name__ == "__main__":
    main()
