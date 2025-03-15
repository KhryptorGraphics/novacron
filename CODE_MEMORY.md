# NovaCron Code Memory System

The Code Memory system provides a vector database for storing, searching, and retrieving code snippets and documentation from throughout the NovaCron project. It uses Qdrant as the vector database and OpenAI's text-embedding-ada-002 model for generating semantic embeddings.

## Components

- **Qdrant Vector Database**: Stores code snippets with semantic embeddings
- **OpenAI Integration**: Generates embeddings for code understanding
- **Python Utilities**: For indexing, searching, and managing the code memory
- **CI/CD Integration**: Keeps code memory up-to-date with the latest code
- **Web Interface**: Provides an intuitive UI for exploring the code base

## Setup

### 1. Start Qdrant

Start the Qdrant server using Docker Compose:

```bash
docker-compose -f qdrant-docker-compose.yml up -d
```

This will start Qdrant on port 6333 (API) and 6334 (web UI).

### 2. Set OpenAI API Key

Set your OpenAI API key as an environment variable:

```bash
# For Linux/macOS
source set_openai_key.sh YOUR_API_KEY

# For Windows PowerShell
.\set_openai_key.ps1 YOUR_API_KEY

# For Windows Command Prompt
set_openai_key.py YOUR_API_KEY
```

Or create a `.env` file with:

```
OPENAI_API_KEY=your_api_key_here
```

### 3. Create Collection

Create the vector collection in Qdrant:

```bash
python create_collection.py
```

### 4. Index Code Files

Index the key project files:

```bash
python index_key_files_mcp.py
```

For more specific indexing needs, you can also use the Go-based indexer:

```bash
cd tools/indexer
go run main.go
```

## Usage

### Command Line Usage

Search for code directly from the command line:

```bash
python qdrant_code_memory_demo.py -q "scheduler workload analysis"
```

Or with file type filters:

```bash
python qdrant_code_memory_demo.py -q "constraint solving" -e go
```

### MCP Integration with Claude

The system integrates with Claude's MCP (Model Context Protocol) to provide code memory access directly from Claude.

Example Claude prompt:

```
Can you help me understand how the workload analyzer functions in NovaCron?
```

Claude can now use the `qdrant-find` MCP tool to search the code memory and provide relevant code snippets.

### Web Interface

You can use the web interface to explore the code memory system:

1. Start the web server:

```bash
cd web_interface
pip install flask
python app.py
```

2. Open your browser at http://localhost:5000

The web interface lets you:
- Search code with semantic understanding
- Filter by file path or extension
- View syntax-highlighted code snippets
- See full file contents when needed
- Browse statistics about indexed files

## Automated Updates

The code memory system is automatically updated through a GitHub Actions workflow that runs:
- On every push to the main branch
- On pull requests to main
- Daily at midnight UTC
- Manually when triggered

The workflow:
1. Starts a Qdrant instance
2. Updates the collection with the latest code
3. Generates a report with indexing statistics
4. Uploads the report as an artifact

You can view the workflow configuration in `.github/workflows/update-code-memory.yml`

## Advanced Usage

### Adding External Documentation

To include external documentation, add it to the vector database:

```python
from qdrant_mcp_utils import store_code_in_qdrant

store_code_in_qdrant(
    "docs/external/design_patterns.md",
    "# Design Patterns\n\nThis document describes...",
    metadata={"source": "external", "category": "documentation"}
)
```

### Customizing the Search

You can customize how searches work by modifying the search parameters:

```python
from qdrant_mcp_utils import QdrantMemory

memory = QdrantMemory()
results = memory.search(
    "virtual machine migration",
    limit=20,
    path_filter="backend/core/vm",
    threshold=0.60  # Lower threshold for more results
)
```

## Contributing

To enhance the code memory system:

1. Add new utility functions in `qdrant_mcp_utils.py`
2. Update the indexing logic in `index_key_files_mcp.py`
3. Improve the web interface in the `web_interface/` directory
4. Update the CI/CD workflow in `.github/workflows/update-code-memory.yml`

## Troubleshooting

If you encounter issues:

1. Check Qdrant is running: `curl http://localhost:6333/healthz`
2. Verify the collection exists: `curl http://localhost:6333/collections/novacron_files`
3. Test with a simple query: `python test_qdrant_connection.py`
4. Check for proper OpenAI API key configuration
