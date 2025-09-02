import { apiClient } from './client';

export interface StoragePool {
  id: string;
  name: string;
  type: string;
  path: string;
  total_space: number;
  used_space: number;
  created_at: string;
  updated_at: string;
  tags: string[];
  metadata: { [key: string]: string };
}

export interface StorageVolume {
  id: string;
  name: string;
  pool_id: string;
  format: string;
  capacity: number;
  allocation: number;
  path: string;
  created_at: string;
  updated_at: string;
  tags: string[];
  metadata: { [key: string]: string };
}

export interface StorageMetrics {
  total_capacity_bytes: number;
  used_capacity_bytes: number;
  available_capacity_bytes: number;
  total_volumes: number;
  active_volumes: number;
  pool_metrics: { [key: string]: any };
  last_updated: string;
}

export interface CreateStoragePoolRequest {
  name: string;
  type: string;
  path: string;
  tags?: string[];
  metadata?: { [key: string]: string };
}

export interface CreateStorageVolumeRequest {
  name: string;
  pool_id: string;
  format: string;
  capacity: number;
  tags?: string[];
  metadata?: { [key: string]: string };
}

export interface ResizeVolumeRequest {
  new_capacity: number;
}

export interface CloneVolumeRequest {
  name: string;
  tags?: string[];
  metadata?: { [key: string]: string };
}

export interface AttachVolumeRequest {
  vm_id: string;
  device?: string;
  read_only?: boolean;
}

export interface DetachVolumeRequest {
  vm_id: string;
  force?: boolean;
}

export const storageApi = {
  // Storage Pools
  async getStoragePools(): Promise<StoragePool[]> {
    const response = await apiClient.get('/storage/pools');
    return response.data;
  },

  async createStoragePool(data: CreateStoragePoolRequest): Promise<StoragePool> {
    const response = await apiClient.post('/storage/pools', data);
    return response.data;
  },

  async getStoragePool(poolId: string): Promise<StoragePool> {
    const response = await apiClient.get(`/storage/pools/${poolId}`);
    return response.data;
  },

  async deleteStoragePool(poolId: string): Promise<void> {
    await apiClient.delete(`/storage/pools/${poolId}`);
  },

  // Storage Volumes
  async getStorageVolumes(poolId?: string): Promise<StorageVolume[]> {
    const params = poolId ? { pool_id: poolId } : {};
    const response = await apiClient.get('/storage/volumes', { params });
    return response.data;
  },

  async createStorageVolume(data: CreateStorageVolumeRequest): Promise<StorageVolume> {
    const response = await apiClient.post('/storage/volumes', data);
    return response.data;
  },

  async getStorageVolume(volumeId: string): Promise<StorageVolume> {
    const response = await apiClient.get(`/storage/volumes/${volumeId}`);
    return response.data;
  },

  async deleteStorageVolume(volumeId: string): Promise<void> {
    await apiClient.delete(`/storage/volumes/${volumeId}`);
  },

  async resizeStorageVolume(volumeId: string, data: ResizeVolumeRequest): Promise<StorageVolume> {
    const response = await apiClient.post(`/storage/volumes/${volumeId}/resize`, data);
    return response.data;
  },

  async cloneStorageVolume(volumeId: string, data: CloneVolumeRequest): Promise<StorageVolume> {
    const response = await apiClient.post(`/storage/volumes/${volumeId}/clone`, data);
    return response.data;
  },

  async attachVolume(volumeId: string, data: AttachVolumeRequest): Promise<{ status: string }> {
    const response = await apiClient.post(`/storage/volumes/${volumeId}/attach`, data);
    return response.data;
  },

  async detachVolume(volumeId: string, data: DetachVolumeRequest): Promise<{ status: string }> {
    const response = await apiClient.post(`/storage/volumes/${volumeId}/detach`, data);
    return response.data;
  },

  // Storage Metrics
  async getStorageMetrics(): Promise<StorageMetrics> {
    const response = await apiClient.get('/storage/metrics');
    return response.data;
  },

  // Utility functions
  formatBytes(bytes: number): string {
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  },

  formatUsagePercentage(used: number, total: number): number {
    if (total === 0) return 0;
    return Math.round((used / total) * 100);
  },

  getStorageTypeIcon(type: string): string {
    switch (type.toLowerCase()) {
      case 'local':
        return '💾';
      case 'ceph':
      case 'distributed':
        return '🗄️';
      case 'nfs':
        return '🌐';
      case 'iscsi':
        return '🔗';
      default:
        return '💿';
    }
  },

  getVolumeFormatColor(format: string): string {
    switch (format.toLowerCase()) {
      case 'qcow2':
        return 'bg-blue-100 text-blue-800';
      case 'raw':
        return 'bg-gray-100 text-gray-800';
      case 'vmdk':
        return 'bg-green-100 text-green-800';
      case 'vhd':
        return 'bg-purple-100 text-purple-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  }
};