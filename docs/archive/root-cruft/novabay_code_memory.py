#!/usr/bin/env python3.12
"""
NovaBay Code Memory - Simple utilities for accessing and working with the code memory system.
This module provides a simplified interface for working with the code memory system from Python.
"""

import argparse
import os
import sys
import json
from typing import List, Dict, Any, Optional, Union

# Import code memory utilities
from code_memory import CodeMemory

class NovaBayCodeMemory:
    """A simple interface to the NovaBay code memory system."""
    
    def __init__(self):
        """Initialize the NovaBay code memory."""
        self.code_memory = CodeMemory()
    
    def search(self, query: str, path_filter: Optional[str] = None, 
               ext_filter: Optional[str] = None, limit: int = 10,
               show_content: bool = False) -> List[Dict[str, Any]]:
        """Search the codebase for the given query."""
        results = self.code_memory.search(
            query=query,
            path_filter=path_filter,
            ext_filter=ext_filter,
            limit=limit
        )
        
        # Display results
        self.code_memory.display_results(results, show_content=show_content)
        
        return results
    
    def find_implementation(self, component_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find the implementation of a specific component."""
        query = f"implementation of {component_name}"
        return self.search(query, limit=limit)
    
    def find_usage(self, component_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find where a component is used in the codebase."""
        query = f"usage of {component_name} OR {component_name} being used"
        return self.search(query, limit=limit)
    
    def find_examples(self, component_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find examples of a component in the codebase."""
        query = f"example of {component_name}"
        ext_filter = "go"  # Focus on Go examples
        return self.search(query, ext_filter=ext_filter, limit=limit)
    
    def find_docs(self, component_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find documentation for a component."""
        query = f"documentation for {component_name}"
        ext_filter = "md"  # Focus on markdown docs
        return self.search(query, ext_filter=ext_filter, limit=limit)
    
    def find_interfaces(self, domain: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Find interfaces in a specific domain."""
        query = f"interface in {domain} OR {domain} interface"
        return self.search(query, limit=limit)
    
    def find_schedulers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Find scheduler implementations."""
        query = "scheduler implementation"
        path_filter = "backend/core/scheduler"
        return self.search(query, path_filter=path_filter, limit=limit)
    
    def find_migration(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Find migration-related code."""
        query = "VM migration implementation"
        return self.search(query, limit=limit)
    
    def find_cloud_providers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Find cloud provider implementations."""
        query = "cloud provider implementation"
        path_filter = "backend/core/cloud"
        return self.search(query, path_filter=path_filter, limit=limit)


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="NovaBay Code Memory - Search and understand the codebase")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the codebase")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-p", "--path", help="Filter by path")
    search_parser.add_argument("-e", "--ext", help="Filter by extension")
    search_parser.add_argument("-l", "--limit", type=int, default=10, help="Result limit")
    search_parser.add_argument("-c", "--content", action="store_true", help="Show full content")
    
    # Implementation finder
    impl_parser = subparsers.add_parser("implementation", help="Find component implementation")
    impl_parser.add_argument("component", help="Component name")
    impl_parser.add_argument("-l", "--limit", type=int, default=5, help="Result limit")
    
    # Usage finder
    usage_parser = subparsers.add_parser("usage", help="Find component usage")
    usage_parser.add_argument("component", help="Component name")
    usage_parser.add_argument("-l", "--limit", type=int, default=5, help="Result limit")
    
    # Examples finder
    examples_parser = subparsers.add_parser("examples", help="Find component examples")
    examples_parser.add_argument("component", help="Component name")
    examples_parser.add_argument("-l", "--limit", type=int, default=5, help="Result limit")
    
    # Documentation finder
    docs_parser = subparsers.add_parser("docs", help="Find component documentation")
    docs_parser.add_argument("component", help="Component name")
    docs_parser.add_argument("-l", "--limit", type=int, default=5, help="Result limit")
    
    # Shortcut commands
    schedulers_parser = subparsers.add_parser("schedulers", help="Find scheduler implementations")
    schedulers_parser.add_argument("-l", "--limit", type=int, default=5, help="Result limit")
    
    migration_parser = subparsers.add_parser("migration", help="Find migration-related code")
    migration_parser.add_argument("-l", "--limit", type=int, default=5, help="Result limit")
    
    cloud_parser = subparsers.add_parser("cloud", help="Find cloud provider implementations")
    cloud_parser.add_argument("-l", "--limit", type=int, default=5, help="Result limit")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create code memory
    code_memory = NovaBayCodeMemory()
    
    # Execute the command
    if args.command == "search":
        code_memory.search(
            query=args.query, 
            path_filter=args.path, 
            ext_filter=args.ext, 
            limit=args.limit, 
            show_content=args.content
        )
    elif args.command == "implementation":
        code_memory.find_implementation(args.component, limit=args.limit)
    elif args.command == "usage":
        code_memory.find_usage(args.component, limit=args.limit)
    elif args.command == "examples":
        code_memory.find_examples(args.component, limit=args.limit)
    elif args.command == "docs":
        code_memory.find_docs(args.component, limit=args.limit)
    elif args.command == "schedulers":
        code_memory.find_schedulers(limit=args.limit)
    elif args.command == "migration":
        code_memory.find_migration(limit=args.limit)
    elif args.command == "cloud":
        code_memory.find_cloud_providers(limit=args.limit)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
