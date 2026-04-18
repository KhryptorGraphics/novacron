'use client';

import { useCallback, useEffect, useState } from 'react';

import { useToast } from '@/components/ui/use-toast';
import {
  changeVolumeTier as changeVolumeTierRequest,
  createVolume as createVolumeRequest,
  listVolumes,
  type VolumeRecord,
  type VolumeTier,
} from '@/lib/graphql/volumes';

export function useVolumes() {
  const { toast } = useToast();
  const [volumes, setVolumes] = useState<VolumeRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [changingTier, setChangingTier] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const nextVolumes = await listVolumes();
      setVolumes(nextVolumes);
    } catch (requestError) {
      setError(
        requestError instanceof Error ? requestError.message : 'Failed to load volumes from the canonical GraphQL API.',
      );
      setVolumes([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const createVolume = useCallback(
    async (input: { name: string; size: number; tier: VolumeTier; vmId?: string }) => {
      setCreating(true);
      setError(null);

      try {
        await createVolumeRequest(input);
        toast({
          title: 'Volume created',
          description: `${input.name} was created on the canonical GraphQL surface.`,
        });
        await refresh();
      } catch (requestError) {
        const message =
          requestError instanceof Error ? requestError.message : 'Failed to create volume.';
        setError(message);
        toast({
          title: 'Create volume failed',
          description: message,
          variant: 'destructive',
        });
      } finally {
        setCreating(false);
      }
    },
    [refresh, toast],
  );

  const changeVolumeTier = useCallback(
    async (id: string, tier: VolumeTier) => {
      setChangingTier(id);
      setError(null);

      try {
        await changeVolumeTierRequest(id, tier);
        toast({
          title: 'Tier updated',
          description: `Volume ${id} was moved to the ${tier} tier.`,
        });
        await refresh();
      } catch (requestError) {
        const message =
          requestError instanceof Error ? requestError.message : 'Failed to change volume tier.';
        setError(message);
        toast({
          title: 'Tier update failed',
          description: message,
          variant: 'destructive',
        });
      } finally {
        setChangingTier(null);
      }
    },
    [refresh, toast],
  );

  return {
    volumes,
    loading,
    creating,
    changingTier,
    error,
    refresh,
    createVolume,
    changeVolumeTier,
  };
}
