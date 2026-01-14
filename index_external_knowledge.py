#!/usr/bin/env python3.12
"""
Command-line tool for indexing external knowledge sources into the NovaCron code memory system.
This script provides a convenient way to fetch and index knowledge from GitHub, Stack Overflow,
and Google Knowledge Graph.
"""

import os
import sys
import argparse
import time
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from external_knowledge import (
    GitHubConnector, 
    StackExchangeConnector, 
    GoogleKnowledgeGraphConnector, 
    ExternalKnowledgeIndexer,
    index_related_content
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('external_knowledge_indexing.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

def setup_api_keys() -> bool:
    """
    Check for required API keys and guide user to set them if missing.
    
    Returns:
        bool: True if all keys are available or optional, False otherwise
    """
    keys_status = {
        "GITHUB_TOKEN": os.environ.get("GITHUB_TOKEN"),
        "STACK_API_KEY": os.environ.get("STACK_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY")
    }
    
    missing_keys = [key for key, value in keys_status.items() if not value]
    
    if missing_keys:
        print("\n⚠️  Warning: Some API keys are missing. This may limit functionality.")
        print("The following keys are not set:")
        
        for key in missing_keys:
            print(f"  - {key}")
            
        print("\nYou can set these keys using one of the following methods:")
        print("1. Create a .env file with the following content:")
        for key in missing_keys:
            print(f"   {key}=your_{key.lower()}_here")
            
        print("\n2. Set environment variables directly:")
        if sys.platform == "win32":
            for key in missing_keys:
                print(f"   set {key}=your_{key.lower()}_here")
            print("   Or use set_openai_key.ps1 or set_openai_key.py scripts")
        else:
            for key in missing_keys:
                print(f"   export {key}=your_{key.lower()}_here")
            print("   Or use the set_openai_key.sh script")
            
        print("\nNote: GitHub API can be used without a token, but rate limits will be strict.")
        print("      Stack Overflow API can be used without a key, but with limitations.")
        print("      Google Knowledge Graph API requires a key.")
        
        # Always return True since we can proceed with limited functionality
        return True
    
    print("✅ All API keys are available.")
    return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Index external knowledge sources into NovaCron code memory.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index all sources with default settings:
  python index_external_knowledge.py --all
  
  # Index only GitHub repositories:
  python index_external_knowledge.py --github openstack/nova kubernetes/kubernetes
  
  # Index Stack Overflow questions about VM migration:
  python index_external_knowledge.py --stackoverflow "vm migration" --tags virtualization,cloud
  
  # Index both GitHub and Stack Overflow with a specific topic:
  python index_external_knowledge.py --github openstack/nova --stackoverflow "cloud orchestration" --tags openstack
  
  # Use a specific topic for all sources:
  python index_external_knowledge.py --all --topic "virtual machine live migration"
  
  # Set a lower limit per source:
  python index_external_knowledge.py --all --limit 5
"""
    )
    
    # Main options
    parser.add_argument('--all', action='store_true', 
                      help='Index all available knowledge sources')
    parser.add_argument('--topic', type=str, default="cloud orchestration virtual machine migration",
                      help='Main topic to search for across all sources')
    parser.add_argument('--limit', type=int, default=10,
                      help='Maximum number of items to index per source')
    
    # GitHub options
    parser.add_argument('--github', nargs='+', metavar='REPO',
                      help='GitHub repositories to index (format: owner/repo)')
    
    # Stack Overflow options
    parser.add_argument('--stackoverflow', type=str, metavar='QUERY',
                      help='Query for Stack Overflow questions')
    parser.add_argument('--tags', type=str, 
                      help='Comma-separated list of tags for Stack Overflow search')
    
    # Knowledge Graph options
    parser.add_argument('--knowledge-graph', type=str, metavar='QUERY',
                      help='Query for Google Knowledge Graph')
    parser.add_argument('--types', type=str,
                      help='Comma-separated list of entity types for Knowledge Graph')
    
    # General options
    parser.add_argument('--verify', action='store_true',
                      help='Verify API connections without indexing')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose output')
    
    return parser.parse_args()

def verify_connections(args):
    """Verify API connections without indexing."""
    print("\n🔍 Verifying API connections...\n")
    
    # Check GitHub connection
    print("GitHub API:")
    try:
        github = GitHubConnector()
        if args.github or args.all:
            # Try a simple repository search
            repos = github.search_repositories("openstack nova", limit=1)
            if repos:
                print("  ✅ Successfully connected to GitHub API")
                print(f"     Found repository: {repos[0]['full_name']}")
            else:
                print("  ⚠️  Connected to GitHub API but search returned no results")
        else:
            print("  ⏭️  Skipped (not requested)")
    except Exception as e:
        print(f"  ❌ Failed to connect to GitHub API: {str(e)}")
    
    # Check Stack Overflow connection
    print("\nStack Overflow API:")
    try:
        stack = StackExchangeConnector()
        if args.stackoverflow or args.all:
            # Try a simple question search
            questions = stack.search_questions("virtual machine migration", limit=1)
            if questions:
                print("  ✅ Successfully connected to Stack Overflow API")
                print(f"     Found question: {questions[0]['title']}")
            else:
                print("  ⚠️  Connected to Stack Overflow API but search returned no results")
        else:
            print("  ⏭️  Skipped (not requested)")
    except Exception as e:
        print(f"  ❌ Failed to connect to Stack Overflow API: {str(e)}")
    
    # Check Knowledge Graph connection
    print("\nGoogle Knowledge Graph API:")
    try:
        kg = GoogleKnowledgeGraphConnector()
        if (args.knowledge_graph or args.all) and kg.api_key:
            # Try a simple entity search
            entities = kg.search_entities("virtual machine", limit=1)
            if entities:
                print("  ✅ Successfully connected to Knowledge Graph API")
                print(f"     Found entity: {entities[0]['name']}")
            else:
                print("  ⚠️  Connected to Knowledge Graph API but search returned no results")
        elif not kg.api_key:
            print("  ❌ Google Knowledge Graph API key not provided")
        else:
            print("  ⏭️  Skipped (not requested)")
    except Exception as e:
        print(f"  ❌ Failed to connect to Knowledge Graph API: {str(e)}")
    
    print("\nVerification complete.")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Check API keys
    setup_api_keys()
    
    # Verify connections if requested
    if args.verify:
        verify_connections(args)
        return
    
    # Prepare indexing parameters
    github_repos = args.github if args.github else None
    so_query = args.stackoverflow if args.stackoverflow else args.topic
    so_tags = args.tags.split(',') if args.tags else ["virtualization", "cloud", "vm-migration"]
    kg_query = args.knowledge_graph if args.knowledge_graph else args.topic
    kg_types = args.types.split(',') if args.types else None
    
    # Only index GitHub if specified or --all is used
    if github_repos or args.all:
        if args.all and not github_repos:
            # Default repositories if --all is used
            github_repos = [
                "openstack/nova",
                "kubernetes/kubernetes",
                "moby/moby",
                "qemu/qemu"
            ]
        
        print(f"\n📚 Indexing documentation from {len(github_repos)} GitHub repositories...")
        indexer = ExternalKnowledgeIndexer()
        count = 0
        for repo in github_repos:
            print(f"  🔍 Processing {repo}...")
            repo_count = indexer.index_github_repo_docs(repo)
            count += repo_count
            print(f"  ✅ Indexed {repo_count} documents from {repo}")
            
        print(f"📋 GitHub indexing complete. Total: {count} documents indexed.")
            
    # Only index Stack Overflow if specified or --all is used
    if args.stackoverflow or args.all:
        print(f"\n❓ Indexing Stack Overflow questions for '{so_query}' with tags: {', '.join(so_tags)}...")
        indexer = ExternalKnowledgeIndexer()
        count = indexer.index_stack_overflow_questions(so_query, tags=so_tags, limit=args.limit)
        print(f"📋 Stack Overflow indexing complete. Total: {count} questions indexed.")
            
    # Only index Knowledge Graph if specified or --all is used
    if (args.knowledge_graph or args.all) and os.environ.get("GOOGLE_API_KEY"):
        print(f"\n🌐 Indexing Knowledge Graph entities for '{kg_query}'...")
        indexer = ExternalKnowledgeIndexer()
        count = indexer.index_knowledge_graph_entities(kg_query, types=kg_types, limit=args.limit)
        print(f"📋 Knowledge Graph indexing complete. Total: {count} entities indexed.")
    elif (args.knowledge_graph or args.all) and not os.environ.get("GOOGLE_API_KEY"):
        print("\n⚠️  Skipping Knowledge Graph indexing: API key not provided")
            
    print("\n✨ Indexing complete! External knowledge has been added to the code memory system.")
    print("You can now search this knowledge using the qdrant_code_memory_demo.py script:")
    print("  python qdrant_code_memory_demo.py -q \"your query\"")
    print("Or use the web interface:")
    print("  cd web_interface && python app.py")

if __name__ == "__main__":
    main()
