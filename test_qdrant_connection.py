#!/usr/bin/env python3.12
"""
Simple test script to verify the Qdrant connection.
"""

import requests
import sys
import json

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333  # Main Qdrant HTTP port
COLLECTION_NAME = "novacron_files"

def test_qdrant_connection():
    """Test the connection to Qdrant."""
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    
    print(f"Testing connection to Qdrant at {base_url}...")
    
    try:
        # Test Qdrant server connection
        response = requests.get(f"{base_url}/")
        if response.status_code != 200:
            print(f"Error connecting to Qdrant server: {response.status_code}")
            return False
        
        print("✅ Connected to Qdrant server successfully!")
        
        # Check if collection exists
        response = requests.get(f"{base_url}/collections/{COLLECTION_NAME}")
        if response.status_code != 200:
            print(f"Error: Collection '{COLLECTION_NAME}' does not exist.")
            print("Run 'python setup_code_memory.py' to create collection and index files.")
            return False
        
        print(f"✅ Collection '{COLLECTION_NAME}' exists!")
        
        # Get collection info
        collection_info = response.json()
        vector_size = collection_info["result"]["config"]["params"]["vectors"]["size"]
        print(f"Collection vector size: {vector_size}")
        
        # Count points in collection
        response = requests.post(
            f"{base_url}/collections/{COLLECTION_NAME}/points/count",
            json={}
        )
        
        if response.status_code != 200:
            print(f"Error counting points: {response.status_code}")
            return False
        
        count = response.json()["result"]["count"]
        print(f"Collection contains {count} indexed files.")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to Qdrant. Make sure it's running.")
        print("Run 'python setup_code_memory.py' to start Qdrant and index files.")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_qdrant_connection()
    sys.exit(0 if success else 1)
