#!/usr/bin/env python3
"""
Create a collection in Qdrant for storing the project files.
This script:
1. Connects to Qdrant
2. Creates a collection for code memory
3. Verifies the collection was created successfully
"""

import sys
import requests
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "novacron_files"
VECTOR_DIM = 1536  # OpenAI ada-002 dimension

def create_collection():
    """Create a new collection in Qdrant for code memory."""
    base_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    
    # Check if Qdrant is accessible
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code != 200:
            logger.error(f"Error connecting to Qdrant server: {response.status_code}")
            return False
        logger.info("Successfully connected to Qdrant")
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {str(e)}")
        return False
    
    # Check if collection already exists
    try:
        response = requests.get(f"{base_url}/collections/{COLLECTION_NAME}")
        if response.status_code == 200:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists")
            return True
    except Exception:
        # Collection doesn't exist, will create it
        pass
    
    # Create collection payload
    payload = {
        "name": COLLECTION_NAME,
        "vectors": {
            "size": VECTOR_DIM,
            "distance": "Cosine"
        }
    }
    
    # Create collection
    try:
        response = requests.put(
            f"{base_url}/collections/{COLLECTION_NAME}",
            json=payload
        )
        
        if response.status_code == 200:
            logger.info(f"Successfully created collection '{COLLECTION_NAME}'")
            
            # Create payload field index for path (optional, improves search performance)
            index_payload = {
                "field_name": "path",
                "field_schema": "keyword",
                "wait": True
            }
            
            response = requests.put(
                f"{base_url}/collections/{COLLECTION_NAME}/index",
                json=index_payload
            )
            
            if response.status_code == 200:
                logger.info(f"Created index on 'path' field")
            else:
                logger.warning(f"Failed to create index on 'path' field: {response.text}")
            
            # Create payload field index for extension (optional)
            index_payload = {
                "field_name": "extension",
                "field_schema": "keyword",
                "wait": True
            }
            
            response = requests.put(
                f"{base_url}/collections/{COLLECTION_NAME}/index",
                json=index_payload
            )
            
            if response.status_code == 200:
                logger.info(f"Created index on 'extension' field")
            else:
                logger.warning(f"Failed to create index on 'extension' field: {response.text}")
            
            # Create payload field index for content (optional, for text search)
            index_payload = {
                "field_name": "content",
                "field_schema": "text",
                "wait": True
            }
            
            response = requests.put(
                f"{base_url}/collections/{COLLECTION_NAME}/index",
                json=index_payload
            )
            
            if response.status_code == 200:
                logger.info(f"Created index on 'content' field")
            else:
                logger.warning(f"Failed to create index on 'content' field: {response.text}")
            
            return True
        else:
            logger.error(f"Failed to create collection: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error creating collection: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Creating Qdrant collection for code memory...")
    if create_collection():
        logger.info("Collection created successfully")
    else:
        logger.error("Failed to create collection")
        sys.exit(1)
