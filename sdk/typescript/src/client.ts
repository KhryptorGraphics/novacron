/**
 * DWCP TypeScript SDK - Client Implementation
 *
 * Provides a comprehensive TypeScript client for the Distributed Worker Control Protocol (DWCP) v3.
 * Supports both Node.js and browser environments with full type safety.
 */

import { EventEmitter } from 'events';
import * as net from 'net';
import * as tls from 'tls';
import { v4 as uuidv4 } from 'uuid';

// Protocol constants
export const PROTOCOL_VERSION = 3;
export const DEFAULT_PORT = 9000;

// Message types
export enum MessageType {
  AUTH = 0x01,
  VM = 0x02,
  STREAM = 0x03,
  MIGRATION = 0x04,
  HEALTH = 0x05,
  METRICS = 0x06,
  CONFIG = 0x07,
  SNAPSHOT = 0x08,
}

// VM operations
export enum VMOperation {
  CREATE = 0x10,
  START = 0x11,
  STOP = 0x12,
  DESTROY = 0x13,
  STATUS = 0x14,
  MIGRATE = 0x15,
  SNAPSHOT = 0x16,
  RESTORE = 0x17,
}

// Error codes
export enum ErrorCode {
  AUTH = 0x1000,
  INVALID_MSG = 0x1001,
  VM_NOT_FOUND = 0x1002,
  RESOURCE_LIMIT = 0x1003,
  MIGRATION = 0x1004,
}

// Custom errors
export class DWCPError extends Error {
  constructor(message: string, public code?: number) {
    super(message);
    this.name = 'DWCPError';
  }
}

export class ConnectionError extends DWCPError {
  constructor(message: string) {
    super(message);
    this.name = 'ConnectionError';
  }
}

export class AuthenticationError extends DWCPError {
  constructor(message: string) {
    super(message, ErrorCode.AUTH);
    this.name = 'AuthenticationError';
  }
}

export class VMNotFoundError extends DWCPError {
  constructor(message: string) {
    super(message, ErrorCode.VM_NOT_FOUND);
    this.name = 'VMNotFoundError';
  }
}

export class TimeoutError extends DWCPError {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}

// Client configuration
export interface ClientConfig {
  address: string;
  port?: number;
  apiKey?: string;
  tlsEnabled?: boolean;
  tlsOptions?: tls.ConnectionOptions;
  connectTimeout?: number;
  requestTimeout?: number;
  retryAttempts?: number;
  retryBackoff?: number;
  keepAlive?: boolean;
  keepAlivePeriod?: number;
  maxStreams?: number;
  bufferSize?: number;
}

// Default configuration
export const defaultConfig: Partial<ClientConfig> = {
  port: DEFAULT_PORT,
  tlsEnabled: true,
  connectTimeout: 30000,
  requestTimeout: 60000,
  retryAttempts: 3,
  retryBackoff: 1000,
  keepAlive: true,
  keepAlivePeriod: 30000,
  maxStreams: 100,
  bufferSize: 65536,
};

// Message structure
export interface Message {
  version: number;
  type: MessageType;
  timestamp: number;
  requestId: string;
  payload: Buffer;
}

// Client metrics
export interface ClientMetrics {
  connectionsTotal: number;
  messagesSent: number;
  messagesReceived: number;
  bytesSent: number;
  bytesReceived: number;
  errorsTotal: number;
  lastConnected?: Date;
}

// Stream class
export class Stream extends EventEmitter {
  private closed = false;

  constructor(
    public readonly id: string,
    private client: Client
  ) {
    super();
  }

  async send(data: Buffer): Promise<void> {
    if (this.closed) {
      throw new DWCPError('Stream closed');
    }

    this.emit('data', data);
  }

  async receive(): Promise<Buffer> {
    return new Promise((resolve, reject) => {
      if (this.closed) {
        reject(new DWCPError('Stream closed'));
        return;
      }

      const onData = (data: Buffer) => {
        this.off('data', onData);
        this.off('error', onError);
        resolve(data);
      };

      const onError = (error: Error) => {
        this.off('data', onData);
        this.off('error', onError);
        reject(error);
      };

      this.once('data', onData);
      this.once('error', onError);
    });
  }

  close(): void {
    if (this.closed) {
      return;
    }

    this.closed = true;
    this.emit('close');
    this.removeAllListeners();
  }

  isClosed(): boolean {
    return this.closed;
  }
}

// Main client class
export class Client extends EventEmitter {
  private socket?: net.Socket | tls.TLSSocket;
  private config: Required<ClientConfig>;
  private connected = false;
  private authenticated = false;
  private streams = new Map<string, Stream>();
  private responseHandlers = new Map<string, (data: Buffer) => void>();
  private metrics: ClientMetrics = {
    connectionsTotal: 0,
    messagesSent: 0,
    messagesReceived: 0,
    bytesSent: 0,
    bytesReceived: 0,
    errorsTotal: 0,
  };

  constructor(config: ClientConfig) {
    super();

    if (!config.address) {
      throw new DWCPError('Address is required');
    }

    this.config = {
      ...defaultConfig,
      ...config,
    } as Required<ClientConfig>;
  }

  async connect(): Promise<void> {
    if (this.connected) {
      return;
    }

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= this.config.retryAttempts; attempt++) {
      if (attempt > 0) {
        await this.sleep(this.config.retryBackoff * attempt);
      }

      try {
        await this.connectAttempt();
        this.metrics.connectionsTotal++;
        this.metrics.lastConnected = new Date();
        return;
      } catch (error) {
        lastError = error as Error;
      }
    }

    throw new ConnectionError(`Failed to connect: ${lastError?.message}`);
  }

  private connectAttempt(): Promise<void> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new TimeoutError('Connection timeout'));
      }, this.config.connectTimeout);

      const onConnect = async () => {
        clearTimeout(timeout);
        this.connected = true;
        this.startReadLoop();

        // Authenticate if API key provided
        if (this.config.apiKey) {
          try {
            await this.authenticate();
            resolve();
          } catch (error) {
            this.disconnect();
            reject(error);
          }
        } else {
          resolve();
        }
      };

      const onError = (error: Error) => {
        clearTimeout(timeout);
        reject(error);
      };

      if (this.config.tlsEnabled) {
        this.socket = tls.connect(
          {
            host: this.config.address,
            port: this.config.port,
            ...this.config.tlsOptions,
          },
          onConnect
        );
      } else {
        this.socket = net.connect(
          {
            host: this.config.address,
            port: this.config.port,
          },
          onConnect
        );
      }

      this.socket.once('error', onError);

      if (this.config.keepAlive) {
        this.socket.setKeepAlive(true, this.config.keepAlivePeriod);
      }
    });
  }

  private async authenticate(): Promise<void> {
    const authReq = {
      api_key: this.config.apiKey,
      version: PROTOCOL_VERSION,
    };

    const resp = await this.sendRequest(MessageType.AUTH, authReq);
    const authResp = JSON.parse(resp.toString());

    if (!authResp.success) {
      throw new AuthenticationError('Authentication failed');
    }

    this.authenticated = true;
  }

  disconnect(): void {
    if (!this.connected) {
      return;
    }

    // Close all streams
    for (const stream of this.streams.values()) {
      stream.close();
    }
    this.streams.clear();

    // Close socket
    if (this.socket) {
      this.socket.destroy();
      this.socket = undefined;
    }

    this.connected = false;
    this.authenticated = false;
    this.emit('disconnect');
  }

  async sendRequest(
    msgType: MessageType,
    payload: any,
    timeout?: number
  ): Promise<Buffer> {
    if (!this.connected || !this.socket) {
      throw new ConnectionError('Not connected to DWCP server');
    }

    // Serialize payload
    const payloadBuffer = Buffer.from(JSON.stringify(payload));

    // Build message
    const message: Message = {
      version: PROTOCOL_VERSION,
      type: msgType,
      timestamp: Date.now(),
      requestId: uuidv4(),
      payload: payloadBuffer,
    };

    // Serialize message
    const msgBuffer = this.marshalMessage(message);

    // Send message
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.responseHandlers.delete(message.requestId);
        reject(new TimeoutError('Request timeout'));
      }, timeout || this.config.requestTimeout);

      // Register response handler
      this.responseHandlers.set(message.requestId, (data: Buffer) => {
        clearTimeout(timer);
        this.responseHandlers.delete(message.requestId);
        resolve(data);
      });

      // Write to socket
      this.socket!.write(msgBuffer, (error) => {
        if (error) {
          clearTimeout(timer);
          this.responseHandlers.delete(message.requestId);
          this.metrics.errorsTotal++;
          reject(error);
        } else {
          this.metrics.messagesSent++;
          this.metrics.bytesSent += msgBuffer.length;
        }
      });
    });
  }

  private startReadLoop(): void {
    if (!this.socket) {
      return;
    }

    let buffer = Buffer.alloc(0);

    this.socket.on('data', (chunk: Buffer) => {
      buffer = Buffer.concat([buffer, chunk]);

      while (buffer.length >= 4) {
        // Read message length
        const msgLen = buffer.readUInt32BE(0);

        if (buffer.length < 4 + msgLen) {
          // Wait for more data
          break;
        }

        // Extract message
        const msgData = buffer.slice(4, 4 + msgLen);
        buffer = buffer.slice(4 + msgLen);

        this.metrics.messagesReceived++;
        this.metrics.bytesReceived += msgLen;

        try {
          const message = this.unmarshalMessage(msgData);
          this.handleMessage(message);
        } catch (error) {
          this.metrics.errorsTotal++;
          this.emit('error', error);
        }
      }
    });

    this.socket.on('error', (error) => {
      this.metrics.errorsTotal++;
      this.emit('error', error);
      this.disconnect();
    });

    this.socket.on('close', () => {
      this.disconnect();
    });
  }

  private handleMessage(message: Message): void {
    // Check for response handler
    const handler = this.responseHandlers.get(message.requestId);
    if (handler) {
      handler(message.payload);
      return;
    }

    // Emit as event
    this.emit('message', message);
  }

  private marshalMessage(message: Message): Buffer {
    const buffers: Buffer[] = [];

    // Reserve space for length (4 bytes)
    buffers.push(Buffer.alloc(4));

    // Version and type
    const header = Buffer.alloc(2);
    header.writeUInt8(message.version, 0);
    header.writeUInt8(message.type, 1);
    buffers.push(header);

    // Timestamp
    const timestamp = Buffer.alloc(8);
    timestamp.writeBigUInt64BE(BigInt(message.timestamp), 0);
    buffers.push(timestamp);

    // Request ID
    const requestIdBuffer = Buffer.from(message.requestId, 'utf8');
    const requestIdLen = Buffer.alloc(2);
    requestIdLen.writeUInt16BE(requestIdBuffer.length, 0);
    buffers.push(requestIdLen);
    buffers.push(requestIdBuffer);

    // Payload
    const payloadLen = Buffer.alloc(4);
    payloadLen.writeUInt32BE(message.payload.length, 0);
    buffers.push(payloadLen);
    buffers.push(message.payload);

    // Combine all buffers
    const result = Buffer.concat(buffers);

    // Write total length
    result.writeUInt32BE(result.length - 4, 0);

    return result;
  }

  private unmarshalMessage(data: Buffer): Message {
    if (data.length < 15) {
      throw new DWCPError('Message too short');
    }

    let offset = 0;

    const version = data.readUInt8(offset);
    offset += 1;

    const type = data.readUInt8(offset);
    offset += 1;

    const timestamp = Number(data.readBigUInt64BE(offset));
    offset += 8;

    const requestIdLen = data.readUInt16BE(offset);
    offset += 2;

    const requestId = data.toString('utf8', offset, offset + requestIdLen);
    offset += requestIdLen;

    const payloadLen = data.readUInt32BE(offset);
    offset += 4;

    const payload = data.slice(offset, offset + payloadLen);

    return {
      version,
      type,
      timestamp,
      requestId,
      payload,
    };
  }

  async newStream(): Promise<Stream> {
    if (this.streams.size >= this.config.maxStreams) {
      throw new DWCPError('Max streams reached');
    }

    const streamId = uuidv4();
    const stream = new Stream(streamId, this);

    this.streams.set(streamId, stream);

    stream.once('close', () => {
      this.streams.delete(streamId);
    });

    return stream;
  }

  isConnected(): boolean {
    return this.connected;
  }

  isAuthenticated(): boolean {
    return this.authenticated;
  }

  getMetrics(): ClientMetrics {
    return { ...this.metrics };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }
}
