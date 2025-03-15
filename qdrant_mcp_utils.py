#!/usr/bin/env python3
"""
Utilities for integrating the Qdrant-based code memory with Claude's MCP.
"""

import os
import requests
import json
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Union

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333  # Standard Qdrant port
COLLECTION_NAME = "novacron_files"
VECTOR_DIM = 1536  # OpenAI ada-002 dimension

class QdrantMemory:
    """Class to interact with Qdrant for code memory."""

    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT, collection: str = COLLECTION_NAME):
        """Initialize the Qdrant memory client."""
        self.base_url = f"http://{host}:{port}"
        self.collection = collection
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        # Check Qdrant connection
        self._check_qdrant_connection()

    def _check_qdrant_connection(self) -> bool:
        """Check if Qdrant is accessible and collection exists."""
        try:
            # Check server connection
            response = requests.get(f"{self.base_url}/")
            if response.status_code != 200:
                print(f"Error connecting to Qdrant server: {response.status_code}")
                return False

            # Check collection exists
            response = requests.get(f"{self.base_url}/collections/{self.collection}")
            if response.status_code != 200:
                print(f"Error: Collection '{self.collection}' does not exist.")
                print("Run 'python setup_code_memory.py' to create collection and index files.")
                return False

            return True
        except Exception as e:
            print(f"Error checking Qdrant connection: {e}")
            return False
    
    def get_collections(self) -> List[str]:
        """Get list of available collections."""
        try:
            response = requests.get(f"{self.base_url}/collections")
            if response.status_code != 200:
                print(f"Error getting collections: {response.status_code}")
                return []
            
            collections_data = response.json()
            return [c["name"] for c in collections_data.get("result", {}).get("collections", [])]
        except Exception as e:
            print(f"Error getting collections: {e}")
            return []
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection."""
        try:
            response = requests.get(f"{self.base_url}/collections/{self.collection}")
            if response.status_code != 200:
                print(f"Error getting collection info: {response.status_code}")
                return {}
            
            return response.json().get("result", {})
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
            
    def get_indexed_file_types(self) -> List[str]:
        """Get list of file types/extensions that have been indexed."""
        try:
            # Get sample points to determine file types
            response = requests.post(
                f"{self.base_url}/collections/{self.collection}/points/scroll",
                json={"limit": 100, "with_payload": True}
            )
            
            if response.status_code != 200:
                print(f"Error getting points: {response.status_code}")
                return []
            
            points = response.json().get("result", {}).get("points", [])
            extensions = set()
            
            for point in points:
                if "payload" in point and "extension" in point["payload"]:
                    ext = point["payload"]["extension"]
                    if ext:
                        # Remove dot prefix if present
                        if ext.startswith("."):
                            ext = ext[1:]
                        extensions.add(ext)
            
            return list(extensions)
        except Exception as e:
            print(f"Error getting file types: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI API or a mock if API key unavailable."""
        # If no API key, create a deterministic mock embedding
        if not self.openai_api_key:
            # Create a deterministic hash of the text
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()

            # Create a vector with 1536 dimensions (OpenAI embedding size)
            mock_embedding = []
            for i in range(VECTOR_DIM):
                # Use the hash to seed the vector
                byte_idx = i % len(hash_bytes)
                bit_idx = (i // len(hash_bytes)) % 8
                bit_value = (hash_bytes[byte_idx] >> bit_idx) & 1
                mock_embedding.append(float(bit_value))

            # Normalize the vector
            norm = np.linalg.norm(mock_embedding)
            return [x / norm for x in mock_embedding]

        # Use OpenAI API to get the embedding
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.openai_api_key}"
        }

        payload = {
            "input": text,
            "model": "text-embedding-ada-002"
        }

        response = requests.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            print(f"Error from OpenAI API: {response.text}")
            return self.get_embedding("Error getting embedding")  # Fallback to mock

        data = response.json()
        return data["data"][0]["embedding"]
    
    def store(self, file_path: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Store code in Qdrant with metadata."""
        try:
            # Create a unique point ID based on the file path
            point_id = hashlib.md5(file_path.encode()).hexdigest()
            
            # Get embedding for content
            embedding = self.get_embedding(content)
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
                
            # Add basic file metadata
            if "path" not in metadata:
                metadata["path"] = file_path
                
            # Get file extension
            _, extension = os.path.splitext(file_path)
            if "extension" not in metadata:
                metadata["extension"] = extension
                
            # Store content
            metadata["content"] = content
            
            # Create the point
            point = {
                "id": point_id,
                "vector": embedding,
                "payload": metadata
            }
            
            # Send upsert request to Qdrant
            response = requests.put(
                f"{self.base_url}/collections/{self.collection}/points",
                json={"points": [point]}
            )
            
            if response.status_code != 200:
                print(f"Error storing in Qdrant: {response.text}")
                return {"success": False, "error": response.text}
                
            return {"success": True, "id": point_id}
            
        except Exception as e:
            print(f"Error storing in Qdrant: {str(e)}")
            return {"success": False, "error": str(e)}

    def search(self, query: str, limit: int = 10, path_filter: Optional[str] = None,
               ext_filter: Optional[str] = None, threshold: float = 0.65) -> List[Dict[str, Any]]:
        """Search for code using a query."""
        # Get embedding for query
        embedding = self.get_embedding(query)

        # Build search payload
        payload = {
            "vector": embedding,
            "limit": limit,
            "with_payload": True,
            "score_threshold": threshold
        }

        # Add filters if provided
        filter_conditions = []
        if path_filter:
            filter_conditions.append({
                "key": "path",
                "match": {
                    "text": path_filter
                }
            })

        if ext_filter:
            # Ensure extension starts with a dot
            if not ext_filter.startswith("."):
                ext_filter = f".{ext_filter}"

            filter_conditions.append({
                "key": "extension",
                "match": {
                    "keyword": ext_filter
                }
            })

        if filter_conditions:
            payload["filter"] = {
                "must": filter_conditions
            }

        # Send search request
        try:
            response = requests.post(
                f"{self.base_url}/collections/{self.collection}/points/search",
                json=payload
            )

            if response.status_code != 200:
                print(f"Error searching Qdrant: {response.text}")
                return []

            return response.json()["result"]
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def create_excerpt(self, content: str, query: str, max_length: int = 150) -> str:
        """Create an excerpt from content highlighting relevant parts."""
        import re

        # Convert to lowercase for case-insensitive matching
        content_lower = content.lower()
        query_lower = query.lower()

        # Extract query keywords (ignore common words)
        stop_words = {"the", "a", "an", "in", "of", "to", "for", "with", "on", "at", "from", "by"}
        query_words = [word for word in re.findall(r'\w+', query_lower)
                    if word not in stop_words and len(word) > 2]

        # Find best match position
        best_pos = 0
        best_score = -1

        for i in range(len(content_lower)):
            score = sum(1 for word in query_words
                    if word in content_lower[i:i+len(word)*3])

            if score > best_score:
                best_score = score
                best_pos = i

        # If no good match found, start at beginning
        if best_score == 0:
            best_pos = 0

        # Find start position (prefer beginning of line)
        start = best_pos
        for i in range(start, max(0, start-100), -1):
            if content[i] == '\n':
                start = i + 1  # Start after newline
                break

        # Find end position (try to end at line break or punctuation)
        end = min(start + max_length, len(content))
        for i in range(end, min(len(content), end + 50)):
            if content[i] == '\n':
                end = i
                break

        excerpt = content[start:end]

        # Add ellipsis if truncated
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."

        return excerpt

    def format_results(self, results: List[Dict[str, Any]], query: str,
                       max_results: int = 5, include_content: bool = False) -> str:
        """Format search results for display in MCP."""
        if not results:
            return "No code memory results found."

        # Limit number of results
        results = results[:max_results]

        formatted_output = f"Found {len(results)} results in code memory:\n\n"

        for i, result in enumerate(results):
            score = result["score"]
            payload = result["payload"]
            path = payload["path"]
            content = payload["content"]

            # Format score as percentage
            score_pct = score * 100

            formatted_output += f"{i+1}. **{path}** (Score: {score_pct:.1f}%)\n"

            # Create excerpt
            excerpt = self.create_excerpt(content, query)
            formatted_output += f"```\n{excerpt}\n```\n\n"

            # Include full content if requested
            if include_content:
                formatted_output += f"Full content:\n```\n{content}\n```\n\n"

        return formatted_output

# For use with MCP
def query_code_memory(query: str, path_filter: Optional[str] = None,
                     ext_filter: Optional[str] = None, limit: int = 5,
                     include_content: bool = False) -> str:
    """Query the code memory database and return formatted results."""
    memory = QdrantMemory()
    results = memory.search(query, limit=limit, path_filter=path_filter, ext_filter=ext_filter)
    return memory.format_results(results, query, max_results=limit, include_content=include_content)

def store_code_in_qdrant(file_path: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Store code file in Qdrant with metadata.
    
    Args:
        file_path: Path to the file (used for identification)
        content: Content of the file
        metadata: Additional metadata to store with the file
        
    Returns:
        Dict with the result of the operation
    """
    memory = QdrantMemory()
    return memory.store(file_path, content, metadata)

if __name__ == "__main__":
    # Simple CLI test
    import sys
    if len(sys.argv) < 2:
        print("Usage: python qdrant_mcp_utils.py 'your search query'")
        sys.exit(1)

    query = sys.argv[1]
    memory = QdrantMemory()
    results = memory.search(query)
    print(memory.format_results(results, query))
