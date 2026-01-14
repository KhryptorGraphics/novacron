"""
NovaCron SDK Exceptions
"""


class NovaCronException(Exception):
    """Base exception for NovaCron SDK"""
    pass


class AuthenticationError(NovaCronException):
    """Authentication failed"""
    pass


class ResourceNotFoundError(NovaCronException):
    """Resource not found"""
    pass


class ValidationError(NovaCronException):
    """Request validation failed"""
    pass


class APIError(NovaCronException):
    """General API error"""
    pass


class NetworkError(NovaCronException):
    """Network connectivity error"""
    pass


class TimeoutError(NovaCronException):
    """Request timeout"""
    pass