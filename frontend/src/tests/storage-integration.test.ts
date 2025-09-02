/**
 * Storage API Integration Tests
 * Tests the integration between frontend and backend storage APIs
 */

import { storageApi } from '../lib/api/storage';

// Mock fetch for testing
global.fetch = jest.fn();

describe('Storage API Integration', () => {
  beforeEach(() => {
    (fetch as jest.Mock).mockClear();
  });

  describe('Storage Pools', () => {
    it('should fetch storage pools successfully', async () => {
      const mockPools = [
        {
          id: 'pool-1',
          name: 'Test Pool',
          type: 'local',
          path: '/var/lib/novacron/volumes',
          total_space: 1000000000,
          used_space: 500000000,
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
          tags: ['test'],
          metadata: { status: 'healthy' }
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: mockPools })
      });

      const pools = await storageApi.getStoragePools();
      
      expect(fetch).toHaveBeenCalledWith('http://localhost:8090/api/storage/pools', expect.any(Object));
      expect(pools).toEqual(mockPools);
    });

    it('should create storage pool successfully', async () => {
      const newPool = {
        name: 'New Pool',
        type: 'ceph',
        path: 'rbd',
        tags: ['production'],
        metadata: { cluster: 'main' }
      };

      const createdPool = { id: 'pool-2', ...newPool };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: createdPool })
      });

      const result = await storageApi.createStoragePool(newPool);
      
      expect(fetch).toHaveBeenCalledWith('http://localhost:8090/api/storage/pools', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(newPool)
      }));
      expect(result).toEqual(createdPool);
    });
  });

  describe('Storage Volumes', () => {
    it('should fetch storage volumes successfully', async () => {
      const mockVolumes = [
        {
          id: 'vol-1',
          name: 'test-volume',
          pool_id: 'pool-1',
          format: 'qcow2',
          capacity: 1000000000,
          allocation: 500000000,
          path: '/var/lib/novacron/volumes/vol-1.img',
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
          tags: ['test'],
          metadata: { vm_name: 'test-vm' }
        }
      ];

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: mockVolumes })
      });

      const volumes = await storageApi.getStorageVolumes();
      
      expect(fetch).toHaveBeenCalledWith('http://localhost:8090/api/storage/volumes', expect.any(Object));
      expect(volumes).toEqual(mockVolumes);
    });

    it('should attach volume to VM successfully', async () => {
      const attachRequest = {
        vm_id: 'vm-1',
        device: 'vda',
        read_only: false
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: { status: 'attached' } })
      });

      const result = await storageApi.attachVolume('vol-1', attachRequest);
      
      expect(fetch).toHaveBeenCalledWith('http://localhost:8090/api/storage/volumes/vol-1/attach', expect.objectContaining({
        method: 'POST',
        body: JSON.stringify(attachRequest)
      }));
      expect(result).toEqual({ status: 'attached' });
    });
  });

  describe('Storage Metrics', () => {
    it('should fetch storage metrics successfully', async () => {
      const mockMetrics = {
        total_capacity_bytes: 10000000000,
        used_capacity_bytes: 5000000000,
        available_capacity_bytes: 5000000000,
        total_volumes: 10,
        active_volumes: 8,
        pool_metrics: {},
        last_updated: '2024-01-01T00:00:00Z'
      };

      (fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ data: mockMetrics })
      });

      const metrics = await storageApi.getStorageMetrics();
      
      expect(fetch).toHaveBeenCalledWith('http://localhost:8090/api/storage/metrics', expect.any(Object));
      expect(metrics).toEqual(mockMetrics);
    });
  });

  describe('Utility Functions', () => {
    it('should format bytes correctly', () => {
      expect(storageApi.formatBytes(0)).toBe('0 Bytes');
      expect(storageApi.formatBytes(1024)).toBe('1 KB');
      expect(storageApi.formatBytes(1048576)).toBe('1 MB');
      expect(storageApi.formatBytes(1073741824)).toBe('1 GB');
      expect(storageApi.formatBytes(1099511627776)).toBe('1 TB');
    });

    it('should format usage percentage correctly', () => {
      expect(storageApi.formatUsagePercentage(0, 100)).toBe(0);
      expect(storageApi.formatUsagePercentage(50, 100)).toBe(50);
      expect(storageApi.formatUsagePercentage(100, 100)).toBe(100);
      expect(storageApi.formatUsagePercentage(0, 0)).toBe(0);
    });

    it('should return correct storage type icons', () => {
      expect(storageApi.getStorageTypeIcon('local')).toBe('💾');
      expect(storageApi.getStorageTypeIcon('ceph')).toBe('🗄️');
      expect(storageApi.getStorageTypeIcon('nfs')).toBe('🌐');
      expect(storageApi.getStorageTypeIcon('iscsi')).toBe('🔗');
      expect(storageApi.getStorageTypeIcon('unknown')).toBe('💿');
    });
  });
});