#!/usr/bin/env python3
"""
NovaCron WebSocket Service

This service provides real-time updates for NovaCron events via WebSocket.
It connects to the hypervisor's event stream and forwards events to connected clients.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import yaml
import time
from typing import Dict, List, Optional, Set, Any

import httpx
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("novacron-websocket")

app = FastAPI(
    title="NovaCron WebSocket Service",
    description="Real-time WebSocket service for NovaCron events",
    version="0.1.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Will be configured from settings in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state
class AppState:
    def __init__(self):
        self.config = {}
        self.connections: Dict[str, Set[WebSocket]] = {
            "all": set(),
            "vm": set(),
            "migration": set(),
            "node": set(),
            "system": set(),
        }
        self.event_queue = asyncio.Queue()
        self.running = True
        self.hypervisor_connected = False
        self.event_task = None
        
app_state = AppState()

# Configuration
def load_config():
    config_path = os.environ.get("CONFIG_PATH", "/etc/novacron/websocket.yaml")
    try:
        with open(config_path, "r") as f:
            app_state.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        # Use default configuration
        app_state.config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8091,
            },
            "hypervisor": {
                "url": "http://localhost:8080",
                "events_endpoint": "/events",
            },
            "auth": {
                "enabled": False,
            },
            "events": {
                "buffer_size": 100,
                "heartbeat_interval": 30,
            },
        }

# WebSocket connection manager
class ConnectionManager:
    async def connect(self, websocket: WebSocket, topics: List[str]):
        await websocket.accept()
        
        # Add to all connections
        app_state.connections["all"].add(websocket)
        
        # Add to topic-specific connections
        for topic in topics:
            if topic in app_state.connections:
                app_state.connections[topic].add(websocket)
        
        logger.debug(f"Client connected. Active connections: {len(app_state.connections['all'])}")
    
    async def disconnect(self, websocket: WebSocket):
        # Remove from all connection sets
        for connections in app_state.connections.values():
            if websocket in connections:
                connections.remove(websocket)
        
        logger.debug(f"Client disconnected. Active connections: {len(app_state.connections['all'])}")
    
    async def broadcast(self, message: str, topic: str = "all"):
        if topic not in app_state.connections:
            logger.warning(f"Unknown topic: {topic}")
            return
        
        disconnected = set()
        for connection in app_state.connections[topic]:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.debug(f"Failed to send message: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)

connection_manager = ConnectionManager()

# Event consumer
async def event_consumer():
    logger.info("Starting event consumer")
    
    while app_state.running:
        try:
            # Get event from queue
            event = await app_state.event_queue.get()
            
            # Determine the topic based on event type
            topic = "all"
            if "type" in event:
                event_type = event["type"].lower()
                if "vm" in event_type:
                    topic = "vm"
                elif "migration" in event_type:
                    topic = "migration"
                elif "node" in event_type:
                    topic = "node"
                elif "system" in event_type:
                    topic = "system"
            
            # Broadcast to appropriate topics
            await connection_manager.broadcast(json.dumps(event), topic)
            await connection_manager.broadcast(json.dumps(event), "all")
            
            # Mark task as done
            app_state.event_queue.task_done()
        
        except asyncio.CancelledError:
            logger.info("Event consumer task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in event consumer: {e}", exc_info=True)
            await asyncio.sleep(1)  # Avoid tight loop in case of errors

# Hypervisor event stream
async def hypervisor_event_stream():
    logger.info("Starting hypervisor event stream")
    
    reconnect_delay = 1  # seconds
    max_reconnect_delay = 60  # seconds
    
    while app_state.running:
        try:
            hypervisor_url = app_state.config["hypervisor"]["url"]
            events_endpoint = app_state.config["hypervisor"]["events_endpoint"]
            url = f"{hypervisor_url}{events_endpoint}"
            
            logger.info(f"Connecting to hypervisor event stream: {url}")
            
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        logger.error(f"Failed to connect to hypervisor event stream: {response.status_code}")
                        await asyncio.sleep(reconnect_delay)
                        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)
                        continue
                    
                    app_state.hypervisor_connected = True
                    reconnect_delay = 1  # Reset reconnect delay on successful connection
                    
                    # Process events
                    async for line in response.aiter_lines():
                        if not app_state.running:
                            break
                        
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            event = json.loads(line)
                            await app_state.event_queue.put(event)
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in event stream: {line}")
                        
        except asyncio.CancelledError:
            logger.info("Hypervisor event stream task cancelled")
            break
        except Exception as e:
            app_state.hypervisor_connected = False
            logger.error(f"Error in hypervisor event stream: {e}", exc_info=True)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# Heartbeat task
async def heartbeat():
    heartbeat_interval = app_state.config.get("events", {}).get("heartbeat_interval", 30)
    logger.info(f"Starting heartbeat (interval: {heartbeat_interval}s)")
    
    while app_state.running:
        try:
            # Create heartbeat event
            event = {
                "type": "heartbeat",
                "timestamp": time.time(),
                "hypervisor_connected": app_state.hypervisor_connected,
                "connections": {
                    topic: len(connections) for topic, connections in app_state.connections.items()
                }
            }
            
            # Broadcast to all clients
            await connection_manager.broadcast(json.dumps(event), "all")
            
            # Wait for next heartbeat
            await asyncio.sleep(heartbeat_interval)
            
        except asyncio.CancelledError:
            logger.info("Heartbeat task cancelled")
            break
        except Exception as e:
            logger.error(f"Error in heartbeat: {e}", exc_info=True)
            await asyncio.sleep(heartbeat_interval)

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, topics: str = "all"):
    """WebSocket endpoint for real-time updates

    Args:
        topics: Comma-separated list of topics to subscribe to (default: "all")
               Available topics: all, vm, migration, node, system
    """
    topic_list = [topic.strip() for topic in topics.split(",")]
    
    # Validate topics
    valid_topics = set(app_state.connections.keys())
    invalid_topics = set(topic_list) - valid_topics
    if invalid_topics:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA, reason=f"Invalid topics: {', '.join(invalid_topics)}")
        return
    
    await connection_manager.connect(websocket, topic_list)
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "timestamp": time.time(),
            "topics": topic_list,
            "hypervisor_connected": app_state.hypervisor_connected,
        }))
        
        # Keep connection alive until client disconnects
        while True:
            data = await websocket.receive_text()
            # Echo back with timestamp
            await websocket.send_text(json.dumps({
                "type": "echo",
                "timestamp": time.time(),
                "data": data,
            }))
    except WebSocketDisconnect:
        logger.debug("WebSocket disconnected")
    finally:
        await connection_manager.disconnect(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "hypervisor_connected": app_state.hypervisor_connected,
        "connections": {
            topic: len(connections) for topic, connections in app_state.connections.items()
        }
    }

# Startup and shutdown
@app.on_event("startup")
async def startup_event():
    # Load configuration
    load_config()
    
    # Start event consumer
    app_state.event_consumer_task = asyncio.create_task(event_consumer())
    
    # Start hypervisor event stream
    app_state.hypervisor_task = asyncio.create_task(hypervisor_event_stream())
    
    # Start heartbeat
    app_state.heartbeat_task = asyncio.create_task(heartbeat())
    
    logger.info("WebSocket service started")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down WebSocket service")
    
    # Stop all tasks
    app_state.running = False
    
    if app_state.event_consumer_task:
        app_state.event_consumer_task.cancel()
        try:
            await app_state.event_consumer_task
        except asyncio.CancelledError:
            pass
    
    if app_state.hypervisor_task:
        app_state.hypervisor_task.cancel()
        try:
            await app_state.hypervisor_task
        except asyncio.CancelledError:
            pass
    
    if app_state.heartbeat_task:
        app_state.heartbeat_task.cancel()
        try:
            await app_state.heartbeat_task
        except asyncio.CancelledError:
            pass
    
    # Close all WebSocket connections
    for connections in app_state.connections.values():
        for websocket in connections:
            try:
                await websocket.close()
            except:
                pass
    
    logger.info("WebSocket service shutdown complete")

# Main function
def main():
    parser = argparse.ArgumentParser(description="NovaCron WebSocket Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8091, help="Port to bind to")
    parser.add_argument("--config", type=str, default="/etc/novacron/websocket.yaml", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    os.environ["CONFIG_PATH"] = args.config
    
    logger.info(f"Starting NovaCron WebSocket Service on {args.host}:{args.port}")
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=args.debug,
        log_level="debug" if args.debug else "info",
    )

if __name__ == "__main__":
    main()
