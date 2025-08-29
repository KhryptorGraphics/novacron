#!/usr/bin/env python3.12
"""
NovaCron Code Memory Demo - Using Qdrant vector database to remember project details.

This script demonstrates how to query the project information stored in the Qdrant
vector database. It showcases both direct queries and using the Qdrant MCP client.
"""

import argparse
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText

# Try to import the MCP server if available
try:
    from mcp_server_qdrant import Server, Resource, Tool
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    print("MCP server for Qdrant not found. Using direct Qdrant client only.")

# Constants
COLLECTION_NAME = "novacron_files"  # Same as in indexer
DEFAULT_LIMIT = 5

def query_direct_qdrant(query_text, limit=DEFAULT_LIMIT):
    """Query the Qdrant database directly using the Python client."""
    try:
        # Connect to Qdrant
        client = QdrantClient(host="localhost", port=6333)
        
        # Check if collection exists
        collections = client.get_collections().collections
        collection_names = [c.name for c in collections]
        if COLLECTION_NAME not in collection_names:
            print(f"Collection '{COLLECTION_NAME}' not found in Qdrant.")
            print("Available collections:", collection_names)
            return []
            
        # Query using the search method with the text payload field
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_text=query_text,
            query_filter=None,  # No additional filtering
            limit=limit
        )
        
        return search_result
    except Exception as e:
        print(f"Error querying Qdrant directly: {e}")
        return []

def query_qdrant_mcp(query_text):
    """Query the Qdrant database using the MCP server."""
    if not HAS_MCP:
        print("MCP server for Qdrant not available.")
        return None
        
    try:
        # Use the MCP client to query
        result = Server().call_tool("qdrant-find", {"query": query_text})
        return result
    except Exception as e:
        print(f"Error querying Qdrant via MCP: {e}")
        return None

def display_result(result, use_mcp=False):
    """Display the search result in a readable format."""
    if not result:
        print("No results found.")
        return
        
    if use_mcp:
        # MCP result format
        print("\nResults from MCP Qdrant server:")
        print("-" * 50)
        for entry in result:
            print(f"Content: {entry['content']}")
            print("-" * 50)
    else:
        # Direct client result format
        print("\nResults from direct Qdrant query:")
        print("-" * 50)
        for idx, item in enumerate(result, start=1):
            print(f"Result {idx}:")
            if hasattr(item, 'payload') and 'content' in item.payload:
                print(f"Content: {item.payload['content'][:500]}...")
            elif hasattr(item, 'payload') and 'path' in item.payload:
                print(f"File: {item.payload['path']}")
            print(f"Score: {item.score}")
            print("-" * 50)

def list_module_components(module_name):
    """List components and details of a specific module."""
    query = f"What components are part of the {module_name} in the NovaCron project?"
    
    # First try using MCP if available
    if HAS_MCP:
        mcp_result = query_qdrant_mcp(query)
        if mcp_result:
            display_result(mcp_result, use_mcp=True)
            return
    
    # Fall back to direct query
    direct_result = query_direct_qdrant(query)
    display_result(direct_result)

def main():
    parser = argparse.ArgumentParser(description="Query the NovaCron project code memory")
    parser.add_argument("-q", "--query", help="Free text query to search for")
    parser.add_argument("-m", "--module", help="Get info about a specific module (scheduler, migration, etc.)")
    parser.add_argument("-l", "--limit", type=int, default=DEFAULT_LIMIT, help=f"Maximum number of results (default: {DEFAULT_LIMIT})")
    parser.add_argument("--direct", action="store_true", help="Force using direct Qdrant client instead of MCP")
    
    args = parser.parse_args()
    
    if not args.query and not args.module:
        # If no arguments, display available modules for quick access
        print("No query specified. Here are some modules you can explore with -m:")
        print("  - scheduler")
        print("  - migration")
        print("  - policy")
        print("  - workload")
        print("  - cloud")
        print("  - auth")
        print("  - security")
        print("  - monitoring")
        print("  - analytics")
        print("  - storage")
        print("  - network")
        print("  - vm")
        print("\nOr use -q to perform a free text search.")
        sys.exit(0)
        
    if args.module:
        list_module_components(args.module)
    elif args.query:
        if HAS_MCP and not args.direct:
            mcp_result = query_qdrant_mcp(args.query)
            display_result(mcp_result, use_mcp=True)
        else:
            direct_result = query_direct_qdrant(args.query, args.limit)
            display_result(direct_result)

if __name__ == "__main__":
    main()
