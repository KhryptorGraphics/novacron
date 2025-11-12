/**
 * DWCP TypeScript SDK
 *
 * TypeScript/JavaScript SDK for the Distributed Worker Control Protocol (DWCP) v3.
 * Supports both Node.js and browser environments.
 */

export {
  // Client
  Client,
  ClientConfig,
  ClientMetrics,
  Message,
  Stream,
  defaultConfig,
  PROTOCOL_VERSION,
  DEFAULT_PORT,

  // Enums
  MessageType,
  VMOperation,
  ErrorCode,

  // Errors
  DWCPError,
  ConnectionError,
  AuthenticationError,
  VMNotFoundError,
  TimeoutError,
} from './client';

export {
  // VM Client
  VMClient,

  // VM types
  VM,
  VMConfig,
  VMState,
  VMEvent,
  VMMetrics,

  // Network
  NetworkConfig,
  NetworkInterface,

  // Affinity
  Affinity,

  // Migration
  MigrationOptions,
  MigrationStatus,
  MigrationState,

  // Snapshots
  Snapshot,
  SnapshotOptions,
} from './vm';

// Version
export const VERSION = '3.0.0';
