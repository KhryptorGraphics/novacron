#!/usr/bin/env python3.12
"""
MCP server for Qdrant-based code memory integration.
"""

import argparse
import json
import sys
from typing import Any, Callable, Dict, Optional

from qdrant_mcp_utils import query_code_memory


class SimpleMCPServer:
    """Minimal stdio MCP server with no third-party runtime dependency."""

    def __init__(self) -> None:
        self.tools: Dict[str, Dict[str, Any]] = {}

    def tool(self, name: str, description: str, input_schema: Dict[str, Any]) -> Callable:
        def decorator(func: Callable) -> Callable:
            self.tools[name] = {
                "description": description,
                "inputSchema": input_schema,
                "handler": func,
            }
            return func

        return decorator

    def start(self) -> None:
        while True:
            message = self._read_message()
            if message is None:
                return
            if "id" not in message:
                continue

            response = self._handle_request(message)
            self._write_message(response)

    def _handle_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        method = message.get("method")
        request_id = message.get("id")
        params = message.get("params") or {}

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "novacron-qdrant-memory", "version": "1.0.0"},
                    },
                }

            if method == "ping":
                return {"jsonrpc": "2.0", "id": request_id, "result": {}}

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "tools": [
                            {
                                "name": name,
                                "description": tool["description"],
                                "inputSchema": tool["inputSchema"],
                            }
                            for name, tool in self.tools.items()
                        ]
                    },
                }

            if method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                if name not in self.tools:
                    raise ValueError(f"Unknown tool: {name}")

                result = self.tools[name]["handler"](**arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": str(result)}]},
                }

            raise ValueError(f"Unsupported method: {method}")
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    def _read_message(self) -> Optional[Dict[str, Any]]:
        headers = {}
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            if line in (b"\r\n", b"\n"):
                break
            key, _, value = line.decode("ascii").partition(":")
            headers[key.lower()] = value.strip()

        content_length = int(headers.get("content-length", "0"))
        if content_length <= 0:
            return None

        raw = sys.stdin.buffer.read(content_length)
        return json.loads(raw.decode("utf-8"))

    def _write_message(self, message: Dict[str, Any]) -> None:
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()


def create_server(args: argparse.Namespace) -> SimpleMCPServer:
    server = SimpleMCPServer()

    @server.tool(
        "find",
        "Look up code in the project codebase.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query."},
                "path_filter": {"type": "string", "description": "Optional path prefix filter."},
                "extension": {"type": "string", "description": "Optional file extension filter."},
                "limit": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                "full_content": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    )
    def find(query: str, path_filter: str = None, extension: str = None,
             limit: int = 5, full_content: bool = False) -> str:
        return query_code_memory(
            query=query,
            path_filter=path_filter,
            ext_filter=extension,
            limit=limit,
            include_content=full_content,
        )

    @server.tool(
        "store",
        "Acknowledge memory storage requests.",
        {
            "type": "object",
            "properties": {
                "information": {"type": "string"},
                "metadata": {"type": "object"},
            },
            "required": ["information"],
        },
    )
    def store(information: str, metadata: dict = None) -> str:
        return (
            "Information acknowledged. This code memory database is pre-populated "
            "with project files and does not store arbitrary information."
        )

    if not getattr(args, "no_start", False):
        server.start()

    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qdrant MCP server for code memory")
    parser.add_argument("--no-start", action="store_true", help=argparse.SUPPRESS)
    create_server(parser.parse_args())
