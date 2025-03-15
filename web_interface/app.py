#!/usr/bin/env python3
"""
Web interface for the NovaCron Code Memory system.
This is a simple Flask application that allows searching the code memory database.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify

# Add parent directory to path to import the Qdrant utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdrant_mcp_utils import QdrantMemory

app = Flask(__name__)

@app.route('/')
def index():
    """Render the home page with search interface."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests and return results."""
    try:
        # Get search parameters from request
        data = request.json
        query = data.get('query', '')
        path_filter = data.get('path_filter', '')
        ext_filter = data.get('ext_filter', '')
        limit = int(data.get('limit', 10))
        include_content = data.get('include_content', False)
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
            
        # Initialize Qdrant memory and perform search
        memory = QdrantMemory()
        results = memory.search(
            query=query, 
            limit=limit,
            path_filter=path_filter if path_filter else None,
            ext_filter=ext_filter if ext_filter else None
        )
        
        # Format results for display
        formatted_results = []
        for i, result in enumerate(results):
            score = result["score"]
            payload = result["payload"]
            path = payload["path"]
            content = payload["content"]
            
            # Create excerpt
            excerpt = memory.create_excerpt(content, query)
            
            formatted_result = {
                'index': i + 1,
                'path': path,
                'score': round(score * 100, 1),
                'excerpt': excerpt,
                'content': content if include_content else None
            }
            formatted_results.append(formatted_result)
        
        # Get stats about the collection
        collection_info = memory.get_collection_info()
        vectors_count = collection_info.get('vectors_count', 'Unknown')
        file_types = memory.get_indexed_file_types()
        
        return jsonify({
            'results': formatted_results,
            'stats': {
                'query': query,
                'result_count': len(formatted_results),
                'vectors_count': vectors_count,
                'file_types': file_types
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats():
    """Get statistics about the Qdrant collection."""
    try:
        memory = QdrantMemory()
        collection_info = memory.get_collection_info()
        
        return jsonify({
            'collection_name': memory.collection,
            'vectors_count': collection_info.get('vectors_count', 'Unknown'),
            'file_types': memory.get_indexed_file_types()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
