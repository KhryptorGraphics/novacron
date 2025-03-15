# OpenAI and Qdrant Integration for NovaCron

This guide explains how to set up and use the OpenAI API with Qdrant for code indexing and semantic search in the NovaCron project.

## Overview

The NovaCron project uses two key components for code memory and semantic search:

1. **Qdrant** - A vector database for storing code embeddings
2. **OpenAI API** - For generating high-quality text embeddings

## Requirements

- OpenAI API key (obtain from [OpenAI platform](https://platform.openai.com/))
- Qdrant running (locally via Docker or as a service)
- Python 3.7+ with required packages
- Go 1.16+ for the indexer tool

## Setting Up OpenAI API Key

We've provided multiple ways to set up your OpenAI API key:

### Option 1: Using the Python Script (Recommended)

```bash
# Install required dependencies
pip install python-dotenv openai requests

# Set the API key
python set_openai_key.py your-openai-api-key

# Test the key and Qdrant connection
python set_openai_key.py your-openai-api-key --test --qdrant-test
```

### Option 2: Using Shell Script (Linux/macOS)

```bash
# Make the script executable (if not already)
chmod +x set_openai_key.sh

# Set the API key
./set_openai_key.sh your-openai-api-key
```

### Option 3: Using PowerShell Script (Windows)

```powershell
# Set the API key
.\set_openai_key.ps1 your-openai-api-key
```

## Indexing Code with Qdrant

### Using Go Indexer

The Go-based indexer automatically uses your OpenAI API key from the environment variables:

```bash
# Navigate to the indexer directory
cd tools/indexer

# Test Qdrant connection
go run test_qdrant.go

# Run the indexer to embed code into Qdrant
go run main.go
```

You can specify a different starting directory:

```bash
go run main.go path/to/directory
```

### Using Python Code Memory

The Python code memory utilities provide a higher-level interface to the Qdrant database:

```bash
# Index code in a directory
python setup_code_memory.py

# Run a search
python novabay_code_memory.py search "scheduler implementation"
```

## Using Code Memory

### Direct Command Line Interface

NovaBay provides a CLI for code memory searches:

```bash
# Basic search
python novabay_code_memory.py search "VM migration implementation"

# Find implementation of a component
python novabay_code_memory.py implementation "resource_aware_scheduler"

# Find examples
python novabay_code_memory.py examples "cloud provider"

# Find documentation
python novabay_code_memory.py docs "migration"
```

### From Python Code

```python
from novabay_code_memory import NovaBayCodeMemory

# Initialize code memory
cm = NovaBayCodeMemory()

# Search for code
results = cm.search("network aware scheduler implementation")

# Find specific implementations
scheduler_results = cm.find_schedulers()
migration_results = cm.find_migration()
```

## Troubleshooting

### API Key Issues

- Ensure your OpenAI API key is valid and has not expired
- Check that the key is correctly set in the environment variable
- Verify that the `.env` file exists and contains the correct key

### Qdrant Connection Issues

- Check that Qdrant is running: `docker ps | grep qdrant`
- Start Qdrant if needed: `docker-compose -f qdrant-docker-compose.yml up -d`
- Verify connection: `python test_qdrant_connection.py`

### Embedding Issues

- The OpenAI `text-embedding-ada-002` model has a token limit of 8,000 tokens
- Large files are automatically chunked to avoid token limits
- Check for any rate limiting errors in the logs

## Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction)
- [NovaCron Code Memory Documentation](CODE_MEMORY.md)
