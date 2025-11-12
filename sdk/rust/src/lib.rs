//! DWCP Rust SDK
//!
//! A high-performance Rust SDK for the Distributed Worker Control Protocol (DWCP) v3.
//! Provides zero-cost abstractions and async/await support for VM management.
//!
//! # Features
//!
//! - Full async/await support with Tokio runtime
//! - Zero-cost abstractions with no runtime overhead
//! - Type-safe API with compile-time guarantees
//! - Efficient binary protocol implementation
//! - TLS support with native-tls
//! - Stream-based operations for real-time updates
//! - Comprehensive error handling
//!
//! # Example
//!
//! ```no_run
//! use dwcp::{Client, ClientConfig, VMConfig};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let config = ClientConfig::new("localhost")
//!         .with_port(9000)
//!         .with_api_key("your-api-key");
//!
//!     let client = Client::new(config).await?;
//!
//!     let vm_config = VMConfig::builder()
//!         .name("my-vm")
//!         .memory(2 * 1024 * 1024 * 1024) // 2GB
//!         .cpus(2)
//!         .disk(20 * 1024 * 1024 * 1024) // 20GB
//!         .image("ubuntu-22.04")
//!         .build()?;
//!
//!     let vm = client.vm().create(vm_config).await?;
//!     println!("Created VM: {} ({})", vm.name, vm.id);
//!
//!     client.vm().start(&vm.id).await?;
//!     println!("VM started successfully");
//!
//!     Ok(())
//! }
//! ```

pub mod client;
pub mod error;
pub mod protocol;
pub mod vm;

pub use client::{Client, ClientConfig, ClientMetrics};
pub use error::{DWCPError, Result};
pub use vm::{
    Affinity, MigrationOptions, MigrationState, MigrationStatus, NetworkConfig, NetworkInterface,
    Snapshot, SnapshotOptions, VMClient, VMConfig, VMConfigBuilder, VMEvent, VMMetrics, VMState, VM,
};

/// Protocol version
pub const PROTOCOL_VERSION: u8 = 3;

/// Default server port
pub const DEFAULT_PORT: u16 = 9000;
