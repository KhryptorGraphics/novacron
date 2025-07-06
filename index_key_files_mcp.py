"""
Script to index key project files using the Qdrant MCP server
This indexes selected important files to demonstrate the MCP code memory functionality
"""
import os
import logging
import sys
from typing import List, Dict, Any, Optional
from qdrant_mcp_utils import store_code_in_qdrant

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_file_content(file_path: str) -> str:
    """Read content from a file, with UTF-8 encoding."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Try with a different encoding if UTF-8 fails
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return ""

def index_important_files():
    """Index a selection of important files from the project."""
    # List of key files to index
    key_files = [
        # Core scheduler files
        "backend/core/scheduler/scheduler.go",
        "backend/core/scheduler/resource_aware_scheduler.go",
        "backend/core/scheduler/enhanced_resource_scheduler.go",
        "backend/core/scheduler/network_aware_scheduler.go",
        
        # Cloud providers
        "backend/core/cloud/provider_interface.go",
        
        # Monitoring
        "backend/core/monitoring/collectors.go",
        "backend/core/monitoring/vm_telemetry_collector.go",
        "backend/core/monitoring/alert.go",
        
        # Auth
        "backend/core/auth/auth.go",
        "backend/core/auth/auth_manager.go",
        "backend/core/auth/user.go",
        "backend/core/auth/role.go",
        
        # VM management
        "backend/core/vm/vm_manager.go",
        
        # Storage
        "backend/core/storage/storage_interface.go",
        "backend/core/storage/distributed_storage.go",
        
        # Network
        "backend/core/network/network_manager.go",
        
        # Documentation
        "README.md",
        "PROJECT_MASTERPLAN.md",
        "SCHEDULER_ENHANCEMENT_PLAN.md",
        "MONITORING_IMPLEMENTATION.md",
    ]
    
    # Index each file
    success_count = 0
    error_count = 0
    for file_path in key_files:
        logger.info(f"Indexing file: {file_path}")
        file_content = read_file_content(file_path)
        
        if not file_content:
            logger.warning(f"Empty or unreadable file: {file_path}")
            error_count += 1
            continue
        
        # Get file tags based on path
        tags = []
        if "scheduler" in file_path:
            tags.append("scheduler")
        if "cloud" in file_path:
            tags.append("cloud")
            if "provider" in file_path:
                tags.append("provider")
        if "monitoring" in file_path:
            tags.append("monitoring")
        if "auth" in file_path:
            tags.append("auth")
            tags.append("security")
        if "vm" in file_path:
            tags.append("vm")
            if "migration" in file_path:
                tags.append("migration")
        if "storage" in file_path:
            tags.append("storage")
        if "network" in file_path:
            tags.append("network")
        if file_path.endswith(".md"):
            tags.append("documentation")
            
        # Store in Qdrant with metadata
        result = store_code_in_qdrant(
            file_path=file_path,
            content=file_content,
            metadata={
                "tags": tags,
                "importance": "high",
                "indexed_at": "manual_index"
            }
        )
        
        if result.get("success", False):
            logger.info(f"Successfully indexed: {file_path}")
            success_count += 1
        else:
            logger.error(f"Failed to index {file_path}: {result.get('error', 'Unknown error')}")
            error_count += 1
    
    # Summary
    logger.info(f"Indexing complete: {success_count} files indexed successfully, {error_count} failures")
    return success_count, error_count

if __name__ == "__main__":
    logger.info("Starting indexing of key project files")
    try:
        success_count, error_count = index_important_files()
        if error_count > 0:
            sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {str(e)}")
        sys.exit(1)
