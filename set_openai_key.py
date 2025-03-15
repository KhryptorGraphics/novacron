#!/usr/bin/env python3
"""
Script to set the OpenAI API key for Qdrant and code memory integration.
This script will update both environment variables and the .env file.
"""
import os
import sys
import argparse
import dotenv
import openai
import requests
from pathlib import Path

# Set up argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Set OpenAI API key for NovaCron")
    parser.add_argument("api_key", help="Your OpenAI API key (starts with 'sk-')")
    parser.add_argument("--test", action="store_true", help="Test the API key with OpenAI")
    parser.add_argument("--qdrant-test", action="store_true", help="Test Qdrant connection")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant HTTP port (default: 6333)")
    return parser.parse_args()

def set_openai_key(api_key):
    """Set the OpenAI API key in environment and .env file."""
    # Set for current session
    os.environ["OPENAI_API_KEY"] = api_key
    print(f"✅ Set OPENAI_API_KEY environment variable for current session")
    
    # Update .env file
    env_path = Path(".env")
    if env_path.exists():
        # Load existing .env file
        dotenv.load_dotenv(env_path)
        
    # Update .env file with new key
    dotenv.set_key(env_path, "OPENAI_API_KEY", api_key)
    print(f"✅ Updated OPENAI_API_KEY in .env file: {env_path.absolute()}")
    
    return True

def test_openai_key(api_key):
    """Test the OpenAI API key with a simple request."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input="Hello, world",
            model="text-embedding-ada-002"
        )
        if response.data and len(response.data) > 0:
            print(f"✅ OpenAI API key is valid! Successfully generated embeddings.")
            print(f"   Model: text-embedding-ada-002")
            print(f"   Embedding dimensions: {len(response.data[0].embedding)}")
            return True
    except Exception as e:
        print(f"❌ Error testing OpenAI API key: {e}")
    return False

def test_qdrant_connection(host, port):
    """Test connection to Qdrant."""
    try:
        url = f"http://{host}:{port}/collections"
        response = requests.get(url)
        if response.status_code == 200:
            collections = response.json().get("result", {}).get("collections", [])
            print(f"✅ Successfully connected to Qdrant at {host}:{port}")
            if collections:
                print(f"   Found {len(collections)} collections:")
                for collection in collections:
                    print(f"   - {collection.get('name')}")
            else:
                print("   No collections found.")
            return True
        else:
            print(f"❌ Failed to connect to Qdrant. Status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error connecting to Qdrant: {e}")
    return False

def main():
    args = parse_args()
    
    # Validate API key format
    if not args.api_key.startswith("sk-"):
        print("❌ Error: API key should start with 'sk-'")
        return 1
    
    # Set the API key
    success = set_openai_key(args.api_key)
    if not success:
        return 1
    
    # Test the API key if requested
    if args.test:
        print("\nTesting OpenAI API key...")
        if not test_openai_key(args.api_key):
            return 1
    
    # Test Qdrant connection if requested
    if args.qdrant_test:
        print("\nTesting Qdrant connection...")
        if not test_qdrant_connection(args.qdrant_host, args.qdrant_port):
            return 1
    
    # Print usage instructions
    print("\n--- How to use the API key ---")
    print("1. In Python scripts:")
    print("   import os")
    print("   from dotenv import load_dotenv")
    print("   load_dotenv()  # Load environment variables from .env file")
    print("   api_key = os.getenv('OPENAI_API_KEY')")
    print("")
    print("2. For Go tools:")
    print("   - The key is automatically used from the environment variable")
    print("   - To run the indexer: cd tools/indexer && go run main.go")
    print("")
    print("3. For new terminal sessions:")
    print("   - Linux/Mac: source .env")
    print("   - Windows PowerShell: foreach($line in Get-Content .env) { if($line -match '^(.+)=(.+)$') { Set-Item -Path Env:$($matches[1]) -Value $matches[2] }}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
