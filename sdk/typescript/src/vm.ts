/**
 * DWCP TypeScript SDK - VM Management
 *
 * Provides VM lifecycle management, migration, snapshots, and monitoring.
 */

import { Client, MessageType, VMOperation } from './client';

// VM state enumeration
export enum VMState {
  CREATING = 'creating',
  STARTING = 'starting',
  RUNNING = 'running',
  STOPPING = 'stopping',
  STOPPED = 'stopped',
  MIGRATING = 'migrating',
  FAILED = 'failed',
  UNKNOWN = 'unknown',
}

// Network interface configuration
export interface NetworkInterface {
  name: string;
  type: string;
  mac?: string;
  bridge?: string;
  vlan?: number;
  ipAddress?: string;
  netmask?: string;
  bandwidth?: number;
}

// Network configuration
export interface NetworkConfig {
  mode: string;
  interfaces: NetworkInterface[];
  dns?: string[];
  gateway?: string;
  mtu?: number;
}

// Node affinity
export interface Affinity {
  nodeSelector?: Record<string, string>;
  requiredNodes?: string[];
  preferredNodes?: string[];
  antiAffinityVMs?: string[];
  requireSameHost?: string[];
}

// VM configuration
export interface VMConfig {
  name: string;
  memory: number;
  cpus: number;
  disk: number;
  image: string;
  network?: NetworkConfig;
  cloudInit?: string;
  labels?: Record<string, string>;
  annotations?: Record<string, string>;
  priority?: number;
  affinity?: Affinity;

  // Advanced features
  enableGPU?: boolean;
  gpuType?: string;
  enableSRIOV?: boolean;
  enableTPM?: boolean;
  enableSecureBoot?: boolean;
  hostDevices?: string[];

  // Performance tuning
  cpuPinning?: number[];
  numaNodes?: number[];
  hugePages?: boolean;
  ioThreads?: number;

  // Resource limits
  memoryMax?: number;
  cpuQuota?: number;
  diskIOPSLimit?: number;
  networkBandwidth?: number;
}

// VM metrics
export interface VMMetrics {
  cpuUsage: number;
  memoryUsed: number;
  memoryAvailable: number;
  diskRead: number;
  diskWrite: number;
  networkRx: number;
  networkTx: number;
  timestamp: Date;
}

// VM representation
export interface VM {
  id: string;
  name: string;
  state: VMState;
  config: VMConfig;
  node: string;
  createdAt: Date;
  updatedAt: Date;
  startedAt?: Date;
  stoppedAt?: Date;
  metrics?: VMMetrics;
  labels?: Record<string, string>;
  annotations?: Record<string, string>;
}

// VM event
export interface VMEvent {
  type: string;
  vm: VM;
  timestamp: Date;
  message: string;
}

// Migration options
export interface MigrationOptions {
  live?: boolean;
  offline?: boolean;
  maxDowntime?: number;
  bandwidth?: number;
  compression?: boolean;
  autoConverge?: boolean;
  postCopy?: boolean;
  parallel?: number;
  verifyChecksum?: boolean;
  encryptTransport?: boolean;
}

// Migration state
export enum MigrationState {
  PREPARING = 'preparing',
  RUNNING = 'running',
  COMPLETING = 'completing',
  COMPLETED = 'completed',
  FAILED = 'failed',
  CANCELLED = 'cancelled',
}

// Migration status
export interface MigrationStatus {
  id: string;
  vmId: string;
  sourceNode: string;
  targetNode: string;
  state: MigrationState;
  progress: number;
  bytesTotal: number;
  bytesSent: number;
  throughput: number;
  downtime: number;
  startedAt: Date;
  completedAt?: Date;
  error?: string;
}

// Snapshot options
export interface SnapshotOptions {
  includeMemory?: boolean;
  description?: string;
  quiesce?: boolean;
}

// Snapshot
export interface Snapshot {
  id: string;
  vmId: string;
  name: string;
  description: string;
  size: number;
  createdAt: Date;
  parent?: string;
  children?: string[];
}

// VM client
export class VMClient {
  constructor(private client: Client) {}

  async create(config: VMConfig): Promise<VM> {
    const req = {
      operation: VMOperation.CREATE,
      request: {
        config,
      },
    };

    const resp = await this.client.sendRequest(MessageType.VM, req);
    const data = JSON.parse(resp.toString());

    return this.parseVM(data.vm);
  }

  async start(vmId: string): Promise<void> {
    const req = {
      operation: VMOperation.START,
      vm_id: vmId,
    };

    await this.client.sendRequest(MessageType.VM, req);
  }

  async stop(vmId: string, force: boolean = false): Promise<void> {
    const req = {
      operation: VMOperation.STOP,
      vm_id: vmId,
      force,
    };

    await this.client.sendRequest(MessageType.VM, req);
  }

  async destroy(vmId: string): Promise<void> {
    const req = {
      operation: VMOperation.DESTROY,
      vm_id: vmId,
    };

    await this.client.sendRequest(MessageType.VM, req);
  }

  async get(vmId: string): Promise<VM> {
    const req = {
      operation: VMOperation.STATUS,
      vm_id: vmId,
    };

    const resp = await this.client.sendRequest(MessageType.VM, req);
    const data = JSON.parse(resp.toString());

    return this.parseVM(data);
  }

  async list(filters?: Record<string, string>): Promise<VM[]> {
    const req = {
      operation: VMOperation.STATUS,
      filters: filters || {},
    };

    const resp = await this.client.sendRequest(MessageType.VM, req);
    const data = JSON.parse(resp.toString());

    return data.map((vm: any) => this.parseVM(vm));
  }

  async *watch(vmId: string): AsyncIterableIterator<VMEvent> {
    const stream = await this.client.newStream();

    // Send watch request
    const req = {
      operation: 'watch',
      vm_id: vmId,
    };

    await stream.send(Buffer.from(JSON.stringify(req)));

    try {
      while (!stream.isClosed()) {
        const data = await stream.receive();
        const event = JSON.parse(data.toString());

        yield {
          type: event.type,
          vm: this.parseVM(event.vm),
          timestamp: new Date(event.timestamp),
          message: event.message,
        };
      }
    } finally {
      stream.close();
    }
  }

  async migrate(
    vmId: string,
    targetNode: string,
    options?: MigrationOptions
  ): Promise<MigrationStatus> {
    const req = {
      vm_id: vmId,
      target_node: targetNode,
      options: options || {},
    };

    const resp = await this.client.sendRequest(MessageType.MIGRATION, req);
    const data = JSON.parse(resp.toString());

    return this.parseMigrationStatus(data);
  }

  async getMigrationStatus(migrationId: string): Promise<MigrationStatus> {
    const req = {
      operation: 'status',
      migration_id: migrationId,
    };

    const resp = await this.client.sendRequest(MessageType.MIGRATION, req);
    const data = JSON.parse(resp.toString());

    return this.parseMigrationStatus(data);
  }

  async snapshot(
    vmId: string,
    snapshotName: string,
    options?: SnapshotOptions
  ): Promise<Snapshot> {
    const req = {
      vm_id: vmId,
      name: snapshotName,
      options: options || {},
    };

    const resp = await this.client.sendRequest(MessageType.SNAPSHOT, req);
    const data = JSON.parse(resp.toString());

    return this.parseSnapshot(data);
  }

  async listSnapshots(vmId: string): Promise<Snapshot[]> {
    const req = {
      operation: 'list',
      vm_id: vmId,
    };

    const resp = await this.client.sendRequest(MessageType.SNAPSHOT, req);
    const data = JSON.parse(resp.toString());

    return data.map((s: any) => this.parseSnapshot(s));
  }

  async restoreSnapshot(vmId: string, snapshotId: string): Promise<void> {
    const req = {
      operation: VMOperation.RESTORE,
      vm_id: vmId,
      snapshot_id: snapshotId,
    };

    await this.client.sendRequest(MessageType.SNAPSHOT, req);
  }

  async deleteSnapshot(snapshotId: string): Promise<void> {
    const req = {
      operation: 'delete',
      snapshot_id: snapshotId,
    };

    await this.client.sendRequest(MessageType.SNAPSHOT, req);
  }

  async getMetrics(vmId: string, duration: string = '5m'): Promise<VMMetrics> {
    const req = {
      vm_id: vmId,
      duration,
    };

    const resp = await this.client.sendRequest(MessageType.METRICS, req);
    const data = JSON.parse(resp.toString());

    return this.parseMetrics(data);
  }

  async *streamMetrics(
    vmId: string,
    interval: string = '1s'
  ): AsyncIterableIterator<VMMetrics> {
    const stream = await this.client.newStream();

    const req = {
      operation: 'stream_metrics',
      vm_id: vmId,
      interval,
    };

    await stream.send(Buffer.from(JSON.stringify(req)));

    try {
      while (!stream.isClosed()) {
        const data = await stream.receive();
        const metrics = JSON.parse(data.toString());
        yield this.parseMetrics(metrics);
      }
    } finally {
      stream.close();
    }
  }

  // Helper methods

  private parseVM(data: any): VM {
    return {
      id: data.id,
      name: data.name,
      state: data.state as VMState,
      config: data.config,
      node: data.node,
      createdAt: new Date(data.created_at),
      updatedAt: new Date(data.updated_at),
      startedAt: data.started_at ? new Date(data.started_at) : undefined,
      stoppedAt: data.stopped_at ? new Date(data.stopped_at) : undefined,
      metrics: data.metrics ? this.parseMetrics(data.metrics) : undefined,
      labels: data.labels,
      annotations: data.annotations,
    };
  }

  private parseMetrics(data: any): VMMetrics {
    return {
      cpuUsage: data.cpu_usage,
      memoryUsed: data.memory_used,
      memoryAvailable: data.memory_available,
      diskRead: data.disk_read,
      diskWrite: data.disk_write,
      networkRx: data.network_rx,
      networkTx: data.network_tx,
      timestamp: new Date(data.timestamp),
    };
  }

  private parseMigrationStatus(data: any): MigrationStatus {
    return {
      id: data.id,
      vmId: data.vm_id,
      sourceNode: data.source_node,
      targetNode: data.target_node,
      state: data.state as MigrationState,
      progress: data.progress,
      bytesTotal: data.bytes_total,
      bytesSent: data.bytes_sent,
      throughput: data.throughput,
      downtime: data.downtime,
      startedAt: new Date(data.started_at),
      completedAt: data.completed_at ? new Date(data.completed_at) : undefined,
      error: data.error,
    };
  }

  private parseSnapshot(data: any): Snapshot {
    return {
      id: data.id,
      vmId: data.vm_id,
      name: data.name,
      description: data.description,
      size: data.size,
      createdAt: new Date(data.created_at),
      parent: data.parent,
      children: data.children,
    };
  }
}
