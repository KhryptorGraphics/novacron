"""
DWCP Client Implementation

Provides the main client class for connecting to and interacting with DWCP servers.
"""

import asyncio
import json
import ssl
import struct
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, Optional, Union

from .exceptions import (
    AuthenticationError,
    ConnectionError,
    DWCPError,
    InvalidOperationError,
    TimeoutError,
)


class MessageType(IntEnum):
    """DWCP message types"""
    AUTH = 0x01
    VM = 0x02
    STREAM = 0x03
    MIGRATION = 0x04
    HEALTH = 0x05
    METRICS = 0x06
    CONFIG = 0x07
    SNAPSHOT = 0x08


class VMOperation(IntEnum):
    """VM operation types"""
    CREATE = 0x10
    START = 0x11
    STOP = 0x12
    DESTROY = 0x13
    STATUS = 0x14
    MIGRATE = 0x15
    SNAPSHOT = 0x16
    RESTORE = 0x17


@dataclass
class ClientConfig:
    """Client configuration"""
    address: str
    port: int = 9000
    api_key: Optional[str] = None
    tls_enabled: bool = True
    ssl_context: Optional[ssl.SSLContext] = None
    connect_timeout: float = 30.0
    request_timeout: float = 60.0
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    keepalive: bool = True
    keepalive_period: float = 30.0
    max_streams: int = 100
    buffer_size: int = 65536

    @classmethod
    def default(cls) -> "ClientConfig":
        """Create default configuration"""
        return cls(address="localhost")


@dataclass
class Message:
    """DWCP protocol message"""
    version: int = 3
    msg_type: int = 0
    timestamp: int = 0
    request_id: str = ""
    payload: bytes = b""

    def marshal(self) -> bytes:
        """Serialize message to bytes"""
        # Reserve space for length
        buf = bytearray(4)

        # Version and type
        buf.append(self.version)
        buf.append(self.msg_type)

        # Timestamp (8 bytes)
        buf.extend(struct.pack(">Q", self.timestamp))

        # Request ID
        rid_bytes = self.request_id.encode("utf-8")
        buf.extend(struct.pack(">H", len(rid_bytes)))
        buf.extend(rid_bytes)

        # Payload
        buf.extend(struct.pack(">I", len(self.payload)))
        buf.extend(self.payload)

        # Write total length
        msg_len = len(buf) - 4
        struct.pack_into(">I", buf, 0, msg_len)

        return bytes(buf)

    @classmethod
    def unmarshal(cls, data: bytes) -> "Message":
        """Deserialize message from bytes"""
        if len(data) < 15:
            raise DWCPError("Message too short")

        version = data[0]
        msg_type = data[1]
        timestamp = struct.unpack(">Q", data[2:10])[0]

        rid_len = struct.unpack(">H", data[10:12])[0]
        if len(data) < 12 + rid_len + 4:
            raise DWCPError("Invalid message format")

        request_id = data[12:12 + rid_len].decode("utf-8")

        payload_len = struct.unpack(">I", data[12 + rid_len:16 + rid_len])[0]
        if len(data) < 16 + rid_len + payload_len:
            raise DWCPError("Invalid payload length")

        payload = data[16 + rid_len:16 + rid_len + payload_len]

        return cls(
            version=version,
            msg_type=msg_type,
            timestamp=timestamp,
            request_id=request_id,
            payload=payload,
        )


@dataclass
class ClientMetrics:
    """Client metrics"""
    connections_total: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    errors_total: int = 0
    last_connected: Optional[float] = None


class Stream:
    """Bidirectional streaming connection"""

    def __init__(self, stream_id: str, client: "Client"):
        self.id = stream_id
        self.client = client
        self.data_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.error_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.closed = False
        self._lock = asyncio.Lock()

    async def send(self, data: bytes) -> None:
        """Send data on the stream"""
        async with self._lock:
            if self.closed:
                raise DWCPError("Stream closed")

            await self.data_queue.put(data)

    async def receive(self) -> bytes:
        """Receive data from the stream"""
        if self.closed:
            raise DWCPError("Stream closed")

        # Check for errors
        if not self.error_queue.empty():
            error = await self.error_queue.get()
            raise error

        return await self.data_queue.get()

    async def close(self) -> None:
        """Close the stream"""
        async with self._lock:
            if self.closed:
                return

            self.closed = True

            # Remove from client's streams
            if self.id in self.client._streams:
                del self.client._streams[self.id]


class Client:
    """DWCP client for server communication"""

    PROTOCOL_VERSION = 3

    def __init__(self, config: ClientConfig):
        """Initialize client with configuration"""
        if not config.address:
            raise DWCPError("Address is required")

        if config.port == 0:
            config.port = 9000

        self.config = config
        self._address = (config.address, config.port)
        self._api_key = config.api_key

        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._connected = False
        self._authenticated = False

        self._streams: Dict[str, Stream] = {}
        self._response_handlers: Dict[str, asyncio.Queue] = {}
        self._message_handlers: Dict[int, Callable] = {}

        self.metrics = ClientMetrics()

        self._read_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Connect to DWCP server"""
        async with self._lock:
            if self._connected:
                return

            # Retry logic
            last_error = None
            for attempt in range(self.config.retry_attempts + 1):
                if attempt > 0:
                    await asyncio.sleep(self.config.retry_backoff * attempt)

                try:
                    if self.config.tls_enabled:
                        ssl_context = self.config.ssl_context
                        if ssl_context is None:
                            ssl_context = ssl.create_default_context()
                            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_3

                        self._reader, self._writer = await asyncio.wait_for(
                            asyncio.open_connection(
                                self.config.address,
                                self.config.port,
                                ssl=ssl_context,
                            ),
                            timeout=self.config.connect_timeout,
                        )
                    else:
                        self._reader, self._writer = await asyncio.wait_for(
                            asyncio.open_connection(
                                self.config.address,
                                self.config.port,
                            ),
                            timeout=self.config.connect_timeout,
                        )

                    break
                except Exception as e:
                    last_error = e

            if self._reader is None or self._writer is None:
                raise ConnectionError(f"Failed to connect: {last_error}")

            self._connected = True

            # Start message reader
            self._read_task = asyncio.create_task(self._read_loop())

            # Authenticate if API key provided
            if self._api_key:
                await self._authenticate()

            self.metrics.connections_total += 1
            self.metrics.last_connected = time.time()

    async def _authenticate(self) -> None:
        """Authenticate with API key"""
        auth_req = {
            "api_key": self._api_key,
            "version": self.PROTOCOL_VERSION,
        }

        resp = await self._send_request(MessageType.AUTH, auth_req)
        auth_resp = json.loads(resp)

        if not auth_resp.get("success"):
            raise AuthenticationError("Authentication failed")

        self._authenticated = True

    async def disconnect(self) -> None:
        """Disconnect from DWCP server"""
        async with self._lock:
            if not self._connected:
                return

            # Close all streams
            for stream in list(self._streams.values()):
                await stream.close()

            # Cancel read task
            if self._read_task:
                self._read_task.cancel()
                try:
                    await self._read_task
                except asyncio.CancelledError:
                    pass

            # Close connection
            if self._writer:
                self._writer.close()
                await self._writer.wait_closed()

            self._connected = False
            self._authenticated = False
            self._reader = None
            self._writer = None

    async def _send_request(
        self,
        msg_type: int,
        payload: Any,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Send request and wait for response"""
        if not self._connected:
            raise ConnectionError("Not connected to DWCP server")

        # Serialize payload
        payload_bytes = json.dumps(payload).encode("utf-8")

        # Build message
        msg = Message(
            version=self.PROTOCOL_VERSION,
            msg_type=msg_type,
            timestamp=int(time.time()),
            request_id=str(uuid.uuid4()),
            payload=payload_bytes,
        )

        # Serialize message
        msg_bytes = msg.marshal()

        # Create response queue
        response_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._response_handlers[msg.request_id] = response_queue

        try:
            # Send message
            self._writer.write(msg_bytes)
            await self._writer.drain()

            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(msg_bytes)

            # Wait for response
            if timeout is None:
                timeout = self.config.request_timeout

            resp = await asyncio.wait_for(response_queue.get(), timeout=timeout)
            return resp
        except asyncio.TimeoutError:
            raise TimeoutError("Request timeout")
        finally:
            del self._response_handlers[msg.request_id]

    async def _read_loop(self) -> None:
        """Continuously read messages from connection"""
        try:
            while self._connected and self._reader:
                # Read message length
                length_bytes = await self._reader.readexactly(4)
                msg_len = struct.unpack(">I", length_bytes)[0]

                # Read message data
                msg_data = await self._reader.readexactly(msg_len)

                self.metrics.messages_received += 1
                self.metrics.bytes_received += msg_len

                # Parse message
                msg = Message.unmarshal(msg_data)

                # Handle message
                await self._handle_message(msg)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.metrics.errors_total += 1
            # Connection error, disconnect
            await self.disconnect()

    async def _handle_message(self, msg: Message) -> None:
        """Handle incoming message"""
        # Check for response handler
        if msg.request_id in self._response_handlers:
            await self._response_handlers[msg.request_id].put(msg.payload)
            return

        # Check for custom message handler
        if msg.msg_type in self._message_handlers:
            handler = self._message_handlers[msg.msg_type]
            asyncio.create_task(handler(msg))

    async def new_stream(self) -> Stream:
        """Create a new bidirectional stream"""
        if len(self._streams) >= self.config.max_streams:
            raise DWCPError("Max streams reached")

        stream_id = str(uuid.uuid4())
        stream = Stream(stream_id, self)
        self._streams[stream_id] = stream

        return stream

    @property
    def is_connected(self) -> bool:
        """Check if client is connected"""
        return self._connected

    @property
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        return self._authenticated

    def get_metrics(self) -> ClientMetrics:
        """Get current metrics"""
        return self.metrics

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.disconnect()
