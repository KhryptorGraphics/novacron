#!/usr/bin/env python3.12
"""
MCP Server for Qdrant-based code memory integration with Claude
"""

import os
import sys
import json
import argparse
from mcp_server_qdrant import Server, Resource, Error

# Import our utilities
from qdrant_mcp_utils import query_code_memory

def create_server(args):
    # Initialize server
    server = Server()
    
    @server.tool("find", "Look up code in the project codebase. Use this tool when you need to: 1) Find related code, 2) Understand implementation details, or 3) Get context about specific functionality")
    def find(query: str, path_filter: str = None, extension: str = None, limit: int = 5, full_content: bool = False):
        """
        Search the codebase for relevant files and code snippets.
        
        Args:
            query: What you're looking for in natural language
            path_filter: Optional filter for paths (e.g. 'backend/core/')
            extension: Optional file extension to filter by (e.g. 'go', 'md')
            limit: Maximum number of results to return (default: 5)
            full_content: Whether to include full file contents (default: False)
            
        Returns:
            Formatted string with search results
        """
        try:
            results = query_code_memory(
                query=query, 
                path_filter=path_filter, 
                ext_filter=extension, 
                limit=limit, 
                include_content=full_content
            )
            return results
        except Exception as e:
            return f"Error searching code memory: {str(e)}"
    
    @server.tool("store", "Keep the memory for later use, when you are asked to remember something.")
    def store(information: str, metadata: dict = None):
        """
        Store information in memory for later retrieval.
        NOTE: This is a placeholder - in this implementation, we're not actually
        storing new information as the database is pre-populated with code files.
        
        Args:
            information: Text information to store
            metadata: Optional metadata for the stored information
            
        Returns:
            Confirmation message
        """
        return "Information acknowledged. Note that this is a code memory database pre-populated with project files, and doesn't store arbitrary information."
    
    # Start the server
    server.start()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qdrant MCP Server for code memory")
    args = parser.parse_args()
    create_server(args)
