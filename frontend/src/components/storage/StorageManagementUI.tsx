'use client';

import { useMemo, useState } from 'react';
import { Database, HardDrive, Layers, Loader2, Sparkles } from 'lucide-react';

import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useVolumes } from '@/hooks/useVolumes';

const TIER_OPTIONS = ['HOT', 'WARM', 'COLD'] as const;

function formatGiB(size: number) {
  return `${size.toLocaleString()} GiB`;
}

export default function StorageManagementUI() {
  const { volumes, loading, error, creating, changingTier, createVolume, changeVolumeTier } = useVolumes();
  const [formState, setFormState] = useState({
    name: '',
    size: '100',
    tier: 'HOT',
    vmId: '',
  });

  const summary = useMemo(() => {
    const totalProvisioned = volumes.reduce((sum, volume) => sum + volume.size, 0);
    const attachedCount = volumes.filter((volume) => volume.vmId).length;
    const tierBreakdown = TIER_OPTIONS.map((tier) => ({
      tier,
      count: volumes.filter((volume) => volume.tier === tier).length,
    }));

    return {
      totalVolumes: volumes.length,
      totalProvisioned,
      attachedCount,
      tierBreakdown,
    };
  }, [volumes]);

  const handleCreateVolume = async () => {
    const parsedSize = Number.parseInt(formState.size, 10);
    await createVolume({
      name: formState.name.trim(),
      size: parsedSize,
      tier: formState.tier,
      vmId: formState.vmId.trim() || undefined,
    });
    setFormState({
      name: '',
      size: '100',
      tier: 'HOT',
      vmId: '',
    });
  };

  return (
    <div className="space-y-6 p-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Storage</h1>
          <p className="text-muted-foreground">
            Volume-only release surface backed by the canonical GraphQL contract.
          </p>
        </div>
        <Badge variant="outline">Volume Operations Only</Badge>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Storage surface unavailable</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Alert>
        <Sparkles className="h-4 w-4" />
        <AlertTitle>Release-candidate scope</AlertTitle>
        <AlertDescription>
          Routed storage supports only volume listing, creation, and tier changes. Pools, snapshots, backups,
          deletion, and performance charts are intentionally out of scope.
        </AlertDescription>
      </Alert>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Layers className="h-4 w-4 text-muted-foreground" />
              Volumes
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{summary.totalVolumes}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <HardDrive className="h-4 w-4 text-muted-foreground" />
              Provisioned
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{formatGiB(summary.totalProvisioned)}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Database className="h-4 w-4 text-muted-foreground" />
              Attached
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{summary.attachedCount}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Tier Breakdown</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {summary.tierBreakdown.map((entry) => (
              <div key={entry.tier} className="flex items-center justify-between text-sm">
                <span>{entry.tier}</span>
                <Badge variant="secondary">{entry.count}</Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Create Volume</CardTitle>
          <CardDescription>
            Calls the supported `createVolume` GraphQL mutation on the canonical backend.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="space-y-2">
            <Label htmlFor="volume-name">Name</Label>
            <Input
              id="volume-name"
              value={formState.name}
              onChange={(event) => setFormState((current) => ({ ...current, name: event.target.value }))}
              placeholder="analytics-cache"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="volume-size">Size (GiB)</Label>
            <Input
              id="volume-size"
              type="number"
              min="1"
              value={formState.size}
              onChange={(event) => setFormState((current) => ({ ...current, size: event.target.value }))}
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="volume-tier">Tier</Label>
            <Select
              value={formState.tier}
              onValueChange={(value) => setFormState((current) => ({ ...current, tier: value }))}
            >
              <SelectTrigger id="volume-tier">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TIER_OPTIONS.map((tier) => (
                  <SelectItem key={tier} value={tier}>
                    {tier}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="volume-vm">VM ID (optional)</Label>
            <Input
              id="volume-vm"
              value={formState.vmId}
              onChange={(event) => setFormState((current) => ({ ...current, vmId: event.target.value }))}
              placeholder="vm-42"
            />
          </div>
          <div className="md:col-span-2 xl:col-span-4">
            <Button
              onClick={handleCreateVolume}
              disabled={creating || !formState.name.trim() || Number.parseInt(formState.size, 10) <= 0}
            >
              {creating ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
              Create Volume
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Volumes</CardTitle>
          <CardDescription>Lists the canonical `volumes` query result and supports tier changes only.</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center text-sm text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Loading canonical volumes…
            </div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Name</TableHead>
                  <TableHead>Size</TableHead>
                  <TableHead>Tier</TableHead>
                  <TableHead>VM</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {volumes.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="text-center text-muted-foreground">
                      No volumes found on the canonical backend.
                    </TableCell>
                  </TableRow>
                ) : (
                  volumes.map((volume) => (
                    <TableRow key={volume.id}>
                      <TableCell className="font-medium">{volume.name}</TableCell>
                      <TableCell>{formatGiB(volume.size)}</TableCell>
                      <TableCell>
                        <Badge variant="secondary">{volume.tier}</Badge>
                      </TableCell>
                      <TableCell>{volume.vmId || 'Unattached'}</TableCell>
                      <TableCell>{new Date(volume.createdAt).toLocaleString()}</TableCell>
                      <TableCell className="text-right">
                        <div className="flex justify-end gap-2">
                          <Select
                            value={volume.tier}
                            onValueChange={(tier) => {
                              if (tier !== volume.tier) {
                                void changeVolumeTier(volume.id, tier);
                              }
                            }}
                            disabled={changingTier === volume.id}
                          >
                            <SelectTrigger className="w-28">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {TIER_OPTIONS.map((tier) => (
                                <SelectItem key={tier} value={tier}>
                                  {tier}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
