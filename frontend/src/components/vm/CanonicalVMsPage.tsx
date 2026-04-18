'use client';

import Link from 'next/link';
import { Loader2, Power, Square } from 'lucide-react';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/components/ui/use-toast';
import { buildApiV1Url } from '@/lib/api/origin';
import { useVMs } from '@/lib/api/hooks/useVMs';
import { useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';

function statusClass(state: string) {
  switch (state) {
    case 'running':
      return 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400';
    case 'stopped':
      return 'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400';
    case 'creating':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400';
    default:
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-400';
  }
}

async function postVmAction(vmId: string, action: 'start' | 'stop') {
  const token = typeof window !== 'undefined' ? window.localStorage.getItem('novacron_token') : null;
  const response = await fetch(buildApiV1Url(`/vms/${vmId}/${action}`), {
    method: 'POST',
    headers: {
      Accept: 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Failed to ${action} VM`);
  }
}

export default function CanonicalVMsPage() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [q, setQ] = useState('');
  const [state, setState] = useState<string>('all');
  const [busyId, setBusyId] = useState<string | null>(null);
  const { items, isLoading, error } = useVMs({
    q: q || undefined,
    state: state === 'all' ? undefined : state,
    page: 1,
    pageSize: 50,
  });

  const handleAction = async (vmId: string, action: 'start' | 'stop') => {
    try {
      setBusyId(vmId);
      await postVmAction(vmId, action);
      toast({
        title: `VM ${action} requested`,
        description: `Canonical ${action} action submitted for ${vmId}.`,
      });
      await queryClient.invalidateQueries({ queryKey: ['vms'] });
    } catch (requestError) {
      toast({
        title: `VM ${action} failed`,
        description: requestError instanceof Error ? requestError.message : `Failed to ${action} VM.`,
        variant: 'destructive',
      });
    } finally {
      setBusyId(null);
    }
  };

  const total = items.length;
  const running = items.filter((vm) => vm.state === 'running').length;
  const stopped = items.filter((vm) => vm.state === 'stopped').length;

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
        <div>
          <h1 className="text-3xl font-bold">Virtual Machines</h1>
          <p className="text-muted-foreground">
            Canonical VM inventory backed by `/api/v1/vms*`.
          </p>
        </div>
        <div className="flex gap-2">
          <Badge variant="outline">{total} total</Badge>
          <Badge variant="secondary">{running} running</Badge>
          <Badge variant="secondary">{stopped} stopped</Badge>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardDescription>Use the canonical VM query surface to narrow the list.</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4 md:flex-row">
          <Input
            placeholder="Search VMs by name"
            value={q}
            onChange={(event) => setQ(event.target.value)}
            className="md:max-w-sm"
          />
          <Select value={state} onValueChange={setState}>
            <SelectTrigger className="md:max-w-[180px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All states</SelectItem>
              <SelectItem value="running">Running</SelectItem>
              <SelectItem value="stopped">Stopped</SelectItem>
              <SelectItem value="creating">Creating</SelectItem>
            </SelectContent>
          </Select>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Inventory</CardTitle>
          <CardDescription>Unsupported websocket-only actions were removed from the routed release surface.</CardDescription>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center text-sm text-muted-foreground">
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Loading canonical VM inventory…
            </div>
          ) : error ? (
            <div className="text-sm text-destructive">
              {error instanceof Error ? error.message : 'Failed to load virtual machines.'}
            </div>
          ) : items.length === 0 ? (
            <div className="text-sm text-muted-foreground">No virtual machines matched the current filters.</div>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>ID</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead>State</TableHead>
                  <TableHead>Node</TableHead>
                  <TableHead>Created</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.map((vm) => (
                  <TableRow key={vm.id}>
                    <TableCell className="font-mono text-xs">{vm.id}</TableCell>
                    <TableCell>
                      <Link href={`/core/vms/${vm.id}`} className="hover:underline">
                        {vm.name}
                      </Link>
                    </TableCell>
                    <TableCell>
                      <Badge className={statusClass(vm.state)}>{vm.state}</Badge>
                    </TableCell>
                    <TableCell>{vm.node_id || '-'}</TableCell>
                    <TableCell>{new Date(vm.created_at).toLocaleString()}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          size="sm"
                          variant="secondary"
                          disabled={busyId === vm.id || vm.state === 'running'}
                          onClick={() => handleAction(vm.id, 'start')}
                        >
                          {busyId === vm.id ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Power className="mr-2 h-4 w-4" />}
                          Start
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          disabled={busyId === vm.id || vm.state === 'stopped'}
                          onClick={() => handleAction(vm.id, 'stop')}
                        >
                          {busyId === vm.id ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Square className="mr-2 h-4 w-4" />}
                          Stop
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
