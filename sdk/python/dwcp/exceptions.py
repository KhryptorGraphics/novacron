"""
DWCP Exceptions

Custom exception classes for DWCP SDK.
"""


class DWCPError(Exception):
    """Base exception for DWCP SDK"""
    pass


class ConnectionError(DWCPError):
    """Connection-related errors"""
    pass


class AuthenticationError(DWCPError):
    """Authentication failures"""
    pass


class VMNotFoundError(DWCPError):
    """VM not found"""
    pass


class TimeoutError(DWCPError):
    """Operation timeout"""
    pass


class InvalidOperationError(DWCPError):
    """Invalid operation"""
    pass


class MigrationError(DWCPError):
    """Migration-related errors"""
    pass


class SnapshotError(DWCPError):
    """Snapshot-related errors"""
    pass
