# NovaCron Code Memory System

The Code Memory system provides a vector database for storing, searching, and retrieving code snippets and documentation from throughout the NovaCron project. It uses Qdrant as the vector database and OpenAI's text-embedding-ada-002 model for generating semantic embeddings.

## Components

- **Qdrant Vector Database**: Stores code snippets with semantic embeddings
- **OpenAI Integration**: Generates embeddings for code understanding
- **Python Utilities**: For indexing, searching, and managing the code memory
- **CI/CD Integration**: Keeps code memory up-to-date with the latest code
- **Web Interface**: Provides an intuitive UI for exploring the code base
- **External Knowledge Integration**: Incorporates documentation and knowledge from GitHub, Stack Overflow, and Google Knowledge Graph

## Setup

### 1. Start Qdrant

Start the Qdrant server using Docker Compose:

```bash
docker-compose -f qdrant-docker-compose.yml up -d
```

This will start Qdrant on port 6333 (API) and 6334 (web UI).

### 2. Set Environment Variables

Copy the `.env.template` file to `.env` and fill in your API keys:

```bash
cp .env.template .env
# Edit the .env file with your API keys
```

At minimum, you need to set your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

For external knowledge integration, you may also want to set:

```
GITHUB_TOKEN=your_github_personal_access_token
STACK_API_KEY=your_stack_exchange_api_key
GOOGLE_API_KEY=your_google_api_key
```

Alternatively, you can use the provided scripts:

```bash
# For Linux/macOS
source set_openai_key.sh YOUR_API_KEY

# For Windows PowerShell
.\set_openai_key.ps1 YOUR_API_KEY

# For Windows Command Prompt
set_openai_key.py YOUR_API_KEY
```

### 3. Install Requirements

Install the required Python packages:

```bash
pip install -r code_memory_requirements.txt
```

### 4. Create Collection

Create the vector collection in Qdrant:

```bash
python create_collection.py
```

### 5. Index Code Files

Index the key project files:

```bash
python index_key_files_mcp.py
```

For more specific indexing needs, you can also use the Go-based indexer:

```bash
cd tools/indexer
go run main.go
```

### 6. (Optional) Index External Knowledge

To enhance the code memory with external knowledge, use the `index_external_knowledge.py` script:

```bash
# Index all external sources (GitHub, Stack Overflow, Knowledge Graph)
python index_external_knowledge.py --all

# Only index specific GitHub repositories
python index_external_knowledge.py --github openstack/nova kubernetes/kubernetes

# Only index Stack Overflow with specific tags
python index_external_knowledge.py --stackoverflow "vm migration" --tags virtualization,cloud
```

Run with `--help` to see all available options:

```bash
python index_external_knowledge.py --help
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

### Direct MCP Integration with Claude

The system can be used directly through Claude's MCP (Model Context Protocol). This allows Claude to search and store code directly in Qdrant without going through the web interface:

```
# Claude can search the code repository with
<use_mcp_tool>
<server_name>github.com/qdrant/mcp-server-qdrant</server_name>
<tool_name>qdrant-find</tool_name>
<arguments>
{
  "query": "virtual machine migration implementation"
}
</arguments>
</use_mcp_tool>
```

To index the entire repository into Qdrant via MCP, you can use the provided helper script:

```bash
python store_repo_in_qdrant_mcp.py
```

This script will:
1. Scan all files in the repository
2. Format them for storage in Qdrant
3. Store each file using the `qdrant-store` MCP tool
4. Track statistics on indexed file types

You can also specify which files to include/exclude:

```bash
python store_repo_in_qdrant_mcp.py --include "*.go" "*.py" --exclude-dirs "vendor" "node_modules"
```

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

### External Knowledge Integration

The system can be enhanced with external knowledge from:

1. **GitHub repositories** - Documentation, README files, and code examples from relevant projects
2. **Stack Overflow** - Questions and answers related to cloud orchestration, VM migration, etc.
3. **Google Knowledge Graph** - Structured knowledge about technical concepts

Example script to index multiple sources:

```python
from external_knowledge import index_related_content

# Index content related to a specific topic
index_related_content(
    topic="cloud orchestration virtual machine migration",
    technology_tags=["virtualization", "cloud", "vm-migration", "openstack"],
    github_repos=[
        "openstack/nova",
        "kubernetes/kubernetes",
        "moby/moby",
        "qemu/qemu"
    ],
    limit_per_source=10
)
```

You can also use the connectors directly for more control:

```python
from external_knowledge import GitHubConnector, ExternalKnowledgeIndexer

# Initialize connectors
github = GitHubConnector()  # Uses GITHUB_TOKEN from environment if available
indexer = ExternalKnowledgeIndexer(github_connector=github)

# Search for relevant repositories
repos = github.search_repositories("cloud orchestration", limit=5)

# Index documents from each repository
for repo in repos:
    indexer.index_github_repo_docs(repo["full_name"])
```

### Adding Custom Documentation

To include custom external documentation, add it to the vector database:

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
