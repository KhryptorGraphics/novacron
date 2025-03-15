# NovaCron Code Memory System

This document describes the Code Memory system implemented using Qdrant vector database to store and search project files and documentation.

## Overview

The Code Memory system provides the following capabilities:

1. **Semantic Code Search**: Search code and documentation using natural language queries
2. **Project Knowledge Base**: Store project files in a queryable vector database
3. **MCP Integration**: Use the Qdrant MCP server to query code memory from Claude
4. **Context Retrieval**: Get context about project components for enhanced understanding

## System Components

The system consists of the following components:

- **Qdrant Database**: Vector database for storing and searching code embeddings
- **Indexing Tools**: Python scripts for creating and populating the database
- **Utility Library**: Functions for interacting with the Qdrant database
- **MCP Integration**: Qdrant MCP server for Claude to use code memory
- **Query Tools**: Tools for searching and retrieving information from code memory

## Setup Instructions

### 1. Start Qdrant Server

Qdrant runs in a Docker container:

```bash
docker-compose -f qdrant-docker-compose.yml up -d
```

### 2. Create Collection

The collection needs to be created before indexing files:

```bash
python create_collection.py
```

This creates a collection named `novacron_files` with the appropriate vector size and indexes.

### 3. Index Files

The system provides two ways to index files:

**Option 1: Using the Python indexer**

```bash
python index_key_files_mcp.py
```

This indexes key project files like scheduler implementations, cloud providers, and documentation.

**Option 2: Using the Go indexer**

```bash
cd tools/indexer
go run main.go ../../
```

This recursively indexes all project files.

## Using the Code Memory System

### Searching Code Memory

#### Direct Query (Python)

```bash
python qdrant_code_memory_demo.py -q "scheduler implementation"
```

#### Module Information

```bash
python qdrant_code_memory_demo.py -m scheduler
```

#### Using MCP

Claude can query the code memory directly using the Qdrant MCP server:

```
What components are part of the scheduler in NovaCron?
```

## Code Memory APIs

### Python API

The `qdrant_mcp_utils.py` module provides functions for interacting with the code memory:

- `query_code_memory(query, path_filter, ext_filter, limit)`: Search for code using natural language
- `store_code_in_qdrant(file_path, content, metadata)`: Store code in the database

### MCP API

The Qdrant MCP server provides the following tools:

- `qdrant-store`: Store information in the code memory
- `qdrant-find`: Search the code memory with natural language queries

## Architecture

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│ NovaCron   │     │ Index      │     │ Qdrant     │
│ Project    │────▶│ Scripts    │────▶│ Database   │
│ Files      │     │            │     │            │
└────────────┘     └────────────┘     └────────────┘
                                           │
                                           ▼
┌────────────┐     ┌────────────┐     ┌────────────┐
│ Claude     │     │ Qdrant     │     │ Search     │
│ Assistant  │◀───▶│ MCP Server │◀───▶│ API        │
│            │     │            │     │            │
└────────────┘     └────────────┘     └────────────┘
```

## Indexed Content

The system currently has indexed:

- Core scheduler components
- Cloud provider implementations
- Monitoring subsystem
- Authentication and authorization
- VM management
- Storage interfaces
- Network components
- Project documentation

## Future Enhancements

Potential enhancements to the code memory system:

1. **Automated indexing**: Integrate with CI/CD to keep code memory up-to-date
2. **Web interface**: Add a web-based search interface
3. **Change tracking**: Track changes in code over time
4. **Chunking strategies**: Improve how files are segmented for better context retrieval
5. **Integration with issue tracking**: Link code memory with issue tracking systems

## Troubleshooting

### Collection Doesn't Exist

If you encounter errors about the collection not existing:

```
Error: Collection 'novacron_files' does not exist.
```

Run `python create_collection.py` to create the collection.

### OpenAI API Key Issues

If embedding creation fails due to OpenAI API issues, the system will fall back to mock embeddings. Set the `OPENAI_API_KEY` environment variable to use real embeddings:

```bash
# For Windows
python set_openai_key.py YOUR_API_KEY

# For Linux/macOS
source set_openai_key.sh YOUR_API_KEY
