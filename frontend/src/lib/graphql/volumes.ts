import authService from '@/lib/auth';
import { buildApiUrl } from '@/lib/api/origin';

export type VolumeTier = 'HOT' | 'WARM' | 'COLD';

export interface VolumeRecord {
  id: string;
  name: string;
  size: number;
  tier: VolumeTier;
  vmId?: string;
  createdAt: string;
  updatedAt: string;
}

interface GraphQLResponse<T> {
  data?: T;
  errors?: Array<{ message: string }>;
}

async function executeGraphQL<T>(query: string, variables?: Record<string, unknown>): Promise<T> {
  const token = authService.getToken();
  const response = await fetch(buildApiUrl('/graphql'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify({ query, variables }),
  });

  const payload = (await response.json()) as GraphQLResponse<T>;
  if (!response.ok || payload.errors?.length) {
    const message = payload.errors?.map((error) => error.message).join('; ') || 'GraphQL request failed';
    throw new Error(message);
  }

  if (!payload.data) {
    throw new Error('GraphQL response did not include data');
  }

  return payload.data;
}

export async function listVolumes(): Promise<VolumeRecord[]> {
  const data = await executeGraphQL<{ volumes: VolumeRecord[] }>(
    'query Volumes { volumes { id name size tier vmId createdAt updatedAt } }',
  );
  return Array.isArray(data.volumes) ? data.volumes : [];
}

export async function createVolume(input: {
  name: string;
  size: number;
  tier: VolumeTier;
  vmId?: string;
}): Promise<VolumeRecord> {
  const data = await executeGraphQL<{ createVolume: VolumeRecord }>(
    'mutation CreateVolume($input: CreateVolumeInput!) { createVolume(input: $input) { id name size tier vmId createdAt updatedAt } }',
    { input },
  );
  return data.createVolume;
}

export async function changeVolumeTier(id: string, tier: VolumeTier): Promise<VolumeRecord> {
  const data = await executeGraphQL<{ changeVolumeTier: VolumeRecord }>(
    'mutation ChangeVolumeTier($id: ID!, $tier: StorageTier!) { changeVolumeTier(id: $id, tier: $tier) { id name size tier vmId createdAt updatedAt } }',
    { id, tier },
  );
  return data.changeVolumeTier;
}
