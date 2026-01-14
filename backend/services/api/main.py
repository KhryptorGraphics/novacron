#!/usr/bin/env python3.12
"""
NovaCron API Service

This service provides a REST API for interacting with NovaCron.
It communicates with the core hypervisor service and exposes endpoints
for VM management, migration, monitoring, and administration.
"""

import argparse
import logging
import os
import sys
import yaml
import json
from typing import Dict, List, Optional, Union, Any

from fastapi import FastAPI, HTTPException, Depends, Security, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import httpx
import uvicorn
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("novacron-api")

# API Models
class VMCreateRequest(BaseModel):
    name: str
    memory_mb: int
    vcpus: int
    disk_mb: int
    image: str
    network_config: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v or len(v) < 3:
            raise ValueError('Name must be at least 3 characters')
        return v
    
    @validator('memory_mb', 'vcpus', 'disk_mb')
    def resources_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Resource allocation must be positive')
        return v

class VMOperationRequest(BaseModel):
    vm_id: str
    force: bool = False

class MigrationRequest(BaseModel):
    vm_id: str
    target_node: str
    migration_type: str = "cold"  # cold, warm, live
    bandwidth_limit: Optional[int] = None  # MB/s
    force: bool = False

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load configuration, establish connections
    logger.info("Starting NovaCron API Service")
    load_config()
    
    # Initialize HTTP client for communicating with hypervisor
    app.state.http_client = httpx.AsyncClient(
        base_url=app.state.config["hypervisor"]["url"],
        timeout=30.0,
    )
    
    yield
    
    # Shutdown: Close connections, clean up resources
    logger.info("Shutting down NovaCron API Service")
    await app.state.http_client.aclose()

app = FastAPI(
    title="NovaCron API",
    description="REST API for NovaCron VM Management and Orchestration",
    version="0.1.0",
    lifespan=lifespan,
)

# Security
security = HTTPBearer(auto_error=False)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be configured from settings in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
def load_config():
    config_path = os.environ.get("CONFIG_PATH", "/etc/novacron/api.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            app.state.config = config
            logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Use default configuration
        app.state.config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8090,
            },
            "hypervisor": {
                "url": "http://localhost:8090",
            },
            "auth": {
                "enabled": False,
            },
            "cors": {
                "allowed_origins": ["*"],
            },
        }

# Auth dependency
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Security(security)):
    if not app.state.config["auth"]["enabled"]:
        return {"id": "anonymous", "role": "admin"}
        
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # In a real implementation, validate the token and get user info
    # For now, just return a mock user
    return {"id": "user1", "role": "admin"}

# Error handling
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )

# Health check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# API Routes

# VM endpoints
@app.get("/api/v1/vms")
async def list_vms(current_user: Dict = Depends(get_current_user)):
    """List all VMs"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.get("/vms")
        if response.status_code != 200:
            logger.error(f"Failed to list VMs: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to list VMs")
        return response.json()

@app.post("/api/v1/vms")
async def create_vm(vm_request: VMCreateRequest, current_user: Dict = Depends(get_current_user)):
    """Create a new VM"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.post("/vms", json=vm_request.dict())
        if response.status_code != 201:
            logger.error(f"Failed to create VM: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to create VM")
        return response.json()

@app.get("/api/v1/vms/{vm_id}")
async def get_vm(vm_id: str, current_user: Dict = Depends(get_current_user)):
    """Get VM details"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.get(f"/vms/{vm_id}")
        if response.status_code != 200:
            logger.error(f"Failed to get VM {vm_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to get VM")
        return response.json()

@app.delete("/api/v1/vms/{vm_id}")
async def delete_vm(vm_id: str, force: bool = False, current_user: Dict = Depends(get_current_user)):
    """Delete a VM"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.delete(f"/vms/{vm_id}", params={"force": force})
        if response.status_code != 204:
            logger.error(f"Failed to delete VM {vm_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to delete VM")
        return {"status": "success"}

@app.post("/api/v1/vms/{vm_id}/start")
async def start_vm(vm_id: str, current_user: Dict = Depends(get_current_user)):
    """Start a VM"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.post(f"/vms/{vm_id}/start")
        if response.status_code != 200:
            logger.error(f"Failed to start VM {vm_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to start VM")
        return {"status": "success"}

@app.post("/api/v1/vms/{vm_id}/stop")
async def stop_vm(vm_id: str, force: bool = False, current_user: Dict = Depends(get_current_user)):
    """Stop a VM"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.post(f"/vms/{vm_id}/stop", params={"force": force})
        if response.status_code != 200:
            logger.error(f"Failed to stop VM {vm_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to stop VM")
        return {"status": "success"}

@app.post("/api/v1/vms/{vm_id}/restart")
async def restart_vm(vm_id: str, force: bool = False, current_user: Dict = Depends(get_current_user)):
    """Restart a VM"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.post(f"/vms/{vm_id}/restart", params={"force": force})
        if response.status_code != 200:
            logger.error(f"Failed to restart VM {vm_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to restart VM")
        return {"status": "success"}

# Migration endpoints
@app.post("/api/v1/migrations")
async def create_migration(
    migration_request: MigrationRequest, 
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Create a new migration"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.post("/migrations", json=migration_request.dict())
        if response.status_code != 201:
            logger.error(f"Failed to create migration: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to create migration")
        return response.json()

@app.get("/api/v1/migrations")
async def list_migrations(current_user: Dict = Depends(get_current_user)):
    """List all migrations"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.get("/migrations")
        if response.status_code != 200:
            logger.error(f"Failed to list migrations: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to list migrations")
        return response.json()

@app.get("/api/v1/migrations/{migration_id}")
async def get_migration(migration_id: str, current_user: Dict = Depends(get_current_user)):
    """Get migration details"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.get(f"/migrations/{migration_id}")
        if response.status_code != 200:
            logger.error(f"Failed to get migration {migration_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to get migration")
        return response.json()

@app.post("/api/v1/migrations/{migration_id}/cancel")
async def cancel_migration(migration_id: str, current_user: Dict = Depends(get_current_user)):
    """Cancel a migration"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.post(f"/migrations/{migration_id}/cancel")
        if response.status_code != 200:
            logger.error(f"Failed to cancel migration {migration_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to cancel migration")
        return {"status": "success"}

# Node endpoints
@app.get("/api/v1/nodes")
async def list_nodes(current_user: Dict = Depends(get_current_user)):
    """List all nodes"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.get("/nodes")
        if response.status_code != 200:
            logger.error(f"Failed to list nodes: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to list nodes")
        return response.json()

@app.get("/api/v1/nodes/{node_id}")
async def get_node(node_id: str, current_user: Dict = Depends(get_current_user)):
    """Get node details"""
    async with httpx.AsyncClient(base_url=app.state.config["hypervisor"]["url"]) as client:
        response = await client.get(f"/nodes/{node_id}")
        if response.status_code != 200:
            logger.error(f"Failed to get node {node_id}: {response.text}")
            raise HTTPException(status_code=response.status_code, detail="Failed to get node")
        return response.json()

# Main function
def main():
    parser = argparse.ArgumentParser(description="NovaCron API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8090, help="Port to bind to")
    parser.add_argument("--config", type=str, default="/etc/novacron/api.yaml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    os.environ["CONFIG_PATH"] = args.config
    
    logger.info(f"Starting NovaCron API on {args.host}:{args.port}")
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )

if __name__ == "__main__":
    main()
