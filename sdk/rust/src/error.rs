//! Error types for DWCP SDK

use thiserror::Error;

/// Result type alias
pub type Result<T> = std::result::Result<T, DWCPError>;

/// DWCP error types
#[derive(Error, Debug)]
pub enum DWCPError {
    /// Connection error
    #[error("Connection error: {0}")]
    Connection(String),

    /// Authentication error
    #[error("Authentication failed: {0}")]
    Authentication(String),

    /// VM not found
    #[error("VM not found: {0}")]
    VMNotFound(String),

    /// Timeout error
    #[error("Operation timeout: {0}")]
    Timeout(String),

    /// Invalid operation
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    /// Protocol error
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Migration error
    #[error("Migration error: {0}")]
    Migration(String),

    /// Snapshot error
    #[error("Snapshot error: {0}")]
    Snapshot(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// Other error
    #[error("Error: {0}")]
    Other(String),
}
