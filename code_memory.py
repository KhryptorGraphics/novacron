#!/usr/bin/env python3.12
"""
Code Memory - Search and Query Project Files
This utility allows you to search and understand code across the NovaCron project.
"""

import argparse
import json
import os
import sys
import requests
from typing import Dict, List, Any, Optional

# Qdrant connection settings
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333  # Main Qdrant HTTP port
COLLECTION_NAME = "novacron_files"
SEARCH_LIMIT = 10
SCORE_THRESHOLD = 0.65

class CodeMemory:
    def __init__(self, host: str = QDRANT_HOST, port: int = QDRANT_PORT):
        """Initialize the CodeMemory client"""
        self.base_url = f"http://{host}:{port}"
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not set. Using mock embeddings.")
    
    def _check_qdrant_connection(self) -> bool:
        """Check if Qdrant is accessible"""
        try:
            response = requests.get(f"{self.base_url}/collections/{COLLECTION_NAME}")
            return response.status_code == 200
        except Exception as e:
            print(f"Error connecting to Qdrant: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI API or mock if no API key"""
        # If no API key, create a deterministic mock embedding
        if not self.openai_api_key:
            import hashlib
            import numpy as np
            
            # Create a deterministic hash of the text
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            
            # Create a vector with 1536 dimensions (OpenAI embedding size)
            mock_embedding = []
            for i in range(1536):
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
            sys.exit(1)
        
        data = response.json()
        return data["data"][0]["embedding"]
    
    def search(self, query: str, limit: int = SEARCH_LIMIT, 
               path_filter: Optional[str] = None, ext_filter: Optional[str] = None,
               score_threshold: float = SCORE_THRESHOLD) -> List[Dict]:
        """Search for code using a natural language query"""
        # Check connection
        if not self._check_qdrant_connection():
            print("Error: Cannot connect to Qdrant. Make sure it's running.")
            print("Run 'python setup_code_memory.py' to start Qdrant and index files.")
            sys.exit(1)
        
        # Get embedding for query
        embedding = self.get_embedding(query)
        
        # Build search payload
        payload = {
            "vector": embedding,
            "limit": limit,
            "with_payload": True,
            "score_threshold": score_threshold
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
                f"{self.base_url}/collections/{COLLECTION_NAME}/points/search",
                json=payload
            )
            
            if response.status_code != 200:
                print(f"Error searching Qdrant: {response.text}")
                sys.exit(1)
            
            return response.json()["result"]
        except Exception as e:
            print(f"Error during search: {e}")
            sys.exit(1)
    
    def create_excerpt(self, content: str, query: str, max_length: int = 150) -> str:
        """Create an excerpt from content highlighting relevant parts"""
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
        
        # Highlight query terms
        for word in query_words:
            if len(word) < 3:  # Skip very short words
                continue
                
            # Highlight matching terms with bold formatting
            excerpt = re.sub(
                f'(?i)({re.escape(word)})',
                r'**\1**',
                excerpt
            )
        
        return excerpt
    
    def display_results(self, results: List[Dict], show_content: bool = False) -> None:
        """Display search results in a readable format"""
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, result in enumerate(results):
            score = result["score"]
            payload = result["payload"]
            path = payload["path"]
            content = payload["content"]
            
            # Format the score as percentage
            score_pct = score * 100
            
            # Determine language for syntax highlighting based on file extension
            extension = os.path.splitext(path)[1]
            lang = extension[1:] if extension else "text"
            
            print(f"{i+1}. {path} (Score: {score_pct:.1f}%)")
            
            # Create and display excerpt
            excerpt = self.create_excerpt(content, query)
            print(f"\n{excerpt}\n")
            
            # Show full content if requested
            if show_content:
                print(f"```{lang}")
                print(content)
                print("```\n")
            
            print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Code Memory - Search your project files.")
    parser.add_argument("-q", "--query", required=True, help="Search query")
    parser.add_argument("-p", "--path", help="Filter results by path prefix")
    parser.add_argument("-e", "--ext", help="Filter results by file extension")
    parser.add_argument("-c", "--content", action="store_true", help="Show full file content in results")
    parser.add_argument("-l", "--limit", type=int, default=SEARCH_LIMIT, help="Maximum number of results to return")
    
    args = parser.parse_args()
    
    # Get query
    query = args.query
    
    # Perform search
    code_memory = CodeMemory()
    results = code_memory.search(
        query=query, 
        limit=args.limit,
        path_filter=args.path,
        ext_filter=args.ext
    )
    
    # Display results
    code_memory.display_results(results, show_content=args.content)

if __name__ == "__main__":
    main()
