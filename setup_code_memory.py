#!/usr/bin/env python3
"""
Initialize Qdrant and index project files for code memory.
This script:
1. Starts Qdrant using Docker Compose
2. Waits for Qdrant to be ready
3. Indexes project files using the existing Go indexer
"""

import os
import subprocess
import time
import requests
import sys

def run_command(command, cwd=None):
    """Run a command and return its output"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, cwd=cwd, 
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                           text=True, check=False)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        print(f"STDERR: {result.stderr}")
    
    return result.stdout.strip()

def start_qdrant():
    """Start Qdrant using Docker Compose"""
    print("Starting Qdrant...")
    run_command("docker-compose -f qdrant-docker-compose.yml up -d")
    
    # Wait for Qdrant to be ready
    print("Waiting for Qdrant to be ready...")
    retries = 30
    while retries > 0:
        try:
            response = requests.get("http://localhost:6333/")
            if response.status_code == 200:
                print("Qdrant is ready.")
                return True
            
            print(f"Qdrant returned status code {response.status_code}, retrying...")
        except requests.exceptions.ConnectionError:
            print("Qdrant is not ready yet, retrying...")
        
        time.sleep(1)
        retries -= 1
    
    print("Failed to connect to Qdrant after multiple retries")
    return False

def index_files():
    """Index project files using the Go indexer"""
    print("Indexing project files...")
    
    # Set OpenAI API key environment variable if provided
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not set. Using mock embeddings.")
    
    # Change to the indexer directory
    indexer_dir = "tools/indexer"
    
    # Run the indexer
    output = run_command("go run main.go ../../", cwd=indexer_dir)
    print(output)
    
    print("Indexing complete.")

def main():
    """Main function"""
    print("Setting up code memory...")
    
    # Start Qdrant
    if not start_qdrant():
        print("Failed to start Qdrant. Exiting.")
        sys.exit(1)
    
    # Index files
    index_files()
    
    print("Code memory setup complete!")
    print("You can now use the utility to search and query project files.")

if __name__ == "__main__":
    main()
