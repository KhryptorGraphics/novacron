"""
DWCP Python SDK

A comprehensive Python SDK for the Distributed Worker Control Protocol (DWCP) v3.
Provides Pythonic interfaces for VM management, migration, monitoring, and more.
"""

__version__ = "3.0.0"
__author__ = "NovaCron Platform Team"
__license__ = "Apache-2.0"

from .client import Client, ClientConfig
from .vm import VMClient, VMConfig, VM, VMState, MigrationOptions
from .exceptions import (
    DWCPError,
    ConnectionError,
    AuthenticationError,
    VMNotFoundError,
    TimeoutError,
    InvalidOperationError,
)

__all__ = [
    "Client",
    "ClientConfig",
    "VMClient",
    "VMConfig",
    "VM",
    "VMState",
    "MigrationOptions",
    "DWCPError",
    "ConnectionError",
    "AuthenticationError",
    "VMNotFoundError",
    "TimeoutError",
    "InvalidOperationError",
]
