#!/usr/bin/env python3.12
"""
Mock Hypervisor Service for NovaCron

This script provides a simple mock hypervisor service that responds to API requests
from the NovaCron API service. It's used for testing purposes only.
"""

import argparse
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("mock-hypervisor")

# API Models
class VMCreateRequest(BaseModel):
    name: str
    vcpus: int
    memory_mb: int
    disk_mb: int
    image: str
    network_config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

class VM(BaseModel):
    id: str
    name: str
    state: str
    node_id: str
    created_at: str
    updated_at: str
    spec: Dict[str, Any]
    tags: List[str] = []

# Create FastAPI app
app = FastAPI(
    title="Mock Hypervisor",
    description="Mock Hypervisor Service for NovaCron",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage
vms: Dict[str, VM] = {}

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# VM endpoints
@app.get("/vms")
async def list_vms():
    return list(vms.values())

@app.post("/vms", status_code=201)
async def create_vm(request: VMCreateRequest):
    vm_id = f"vm-{uuid.uuid4()}"
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Create VM
    vm = VM(
        id=vm_id,
        name=request.name,
        state="creating",
        node_id="mock-node-1",
        created_at=now,
        updated_at=now,
        spec={
            "vcpus": request.vcpus,
            "memory_mb": request.memory_mb,
            "disk_mb": request.disk_mb,
            "image": request.image,
            "network_config": request.network_config or {},
        },
        tags=request.tags,
    )

    # Store VM
    vms[vm_id] = vm

    logger.info(f"Created VM {vm_id} with name {request.name}")

    # Return VM
    return vm

@app.get("/vms/{vm_id}")
async def get_vm(vm_id: str):
    if vm_id not in vms:
        raise HTTPException(status_code=404, detail=f"VM {vm_id} not found")

    return vms[vm_id]

@app.delete("/vms/{vm_id}", status_code=204)
async def delete_vm(vm_id: str):
    if vm_id not in vms:
        raise HTTPException(status_code=404, detail=f"VM {vm_id} not found")

    # Delete VM
    del vms[vm_id]

    logger.info(f"Deleted VM {vm_id}")

    return None

@app.post("/vms/{vm_id}/start")
async def start_vm(vm_id: str):
    if vm_id not in vms:
        raise HTTPException(status_code=404, detail=f"VM {vm_id} not found")

    # Update VM state
    vms[vm_id].state = "running"
    vms[vm_id].updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger.info(f"Started VM {vm_id}")

    return {"status": "success"}

@app.post("/vms/{vm_id}/stop")
async def stop_vm(vm_id: str):
    if vm_id not in vms:
        raise HTTPException(status_code=404, detail=f"VM {vm_id} not found")

    # Update VM state
    vms[vm_id].state = "stopped"
    vms[vm_id].updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger.info(f"Stopped VM {vm_id}")

    return {"status": "success"}

# Node endpoints
@app.get("/nodes")
async def list_nodes():
    return [
        {
            "id": "mock-node-1",
            "name": "Mock Node 1",
            "state": "running",
            "address": "127.0.0.1",
            "port": 9000,
            "cpu_cores": 8,
            "memory_mb": 16384,
            "disk_gb": 500,
            "created_at": "2025-04-10T00:00:00Z",
            "updated_at": "2025-04-10T00:00:00Z",
        }
    ]

@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    if node_id != "mock-node-1":
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    return {
        "id": "mock-node-1",
        "name": "Mock Node 1",
        "state": "running",
        "address": "127.0.0.1",
        "port": 9000,
        "cpu_cores": 8,
        "memory_mb": 16384,
        "disk_gb": 500,
        "created_at": "2025-04-10T00:00:00Z",
        "updated_at": "2025-04-10T00:00:00Z",
    }

# Main function
def main():
    parser = argparse.ArgumentParser(description="Mock Hypervisor Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    logger.info(f"Starting Mock Hypervisor on {args.host}:{args.port}")
    uvicorn.run(
        "mock_hypervisor:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )

if __name__ == "__main__":
    main()
