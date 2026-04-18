'use client';

import { useEffect, useMemo, useState } from 'react';
import { Link2, Loader2, Network, Plus, Trash2, Unplug } from 'lucide-react';

import AuthGuard from '@/components/auth/AuthGuard';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/components/ui/use-toast';
import { networkApi, type CanonicalNetwork, type CanonicalVmInterface } from '@/lib/api/networks';
import { useVMs } from '@/lib/api/hooks/useVMs';

type InterfaceMap = Record<string, CanonicalVmInterface[]>;

const emptyNetworkForm = {
  name: '',
  type: 'bridged',
  subnet: '',
  gateway: '',
};

const emptyInterfaceForm = {
  vmId: '',
  networkId: '',
  name: '',
  macAddress: '',
  ipAddress: '',
};

export default function NetworkPage() {
  const { toast } = useToast();
  const [networks, setNetworks] = useState<CanonicalNetwork[]>([]);
  const [interfacesByVm, setInterfacesByVm] = useState<InterfaceMap>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [networkDialogOpen, setNetworkDialogOpen] = useState(false);
  const [interfaceDialogOpen, setInterfaceDialogOpen] = useState(false);
  const [networkForm, setNetworkForm] = useState(emptyNetworkForm);
  const [interfaceForm, setInterfaceForm] = useState(emptyInterfaceForm);
  const [saving, setSaving] = useState(false);
  const { items: vms } = useVMs({ page: 1, pageSize: 50 });

  const loadNetworkState = async () => {
    setLoading(true);
    setError(null);

    try {
      const networkList = await networkApi.listNetworks();
      setNetworks(networkList);

      if (vms.length > 0) {
        const interfaceEntries = await Promise.all(
          vms.map(async (vm) => {
            try {
              return [vm.id, await networkApi.listVmInterfaces(vm.id)] as const;
            } catch {
              return [vm.id, []] as const;
            }
          }),
        );

        setInterfacesByVm(Object.fromEntries(interfaceEntries));
      } else {
        setInterfacesByVm({});
      }
    } catch (loadError) {
      setError(loadError instanceof Error ? loadError.message : 'Failed to load canonical network inventory.');
      setNetworks([]);
      setInterfacesByVm({});
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadNetworkState();
  }, [vms.length]);

  const stats = useMemo(() => {
    const interfaceCount = Object.values(interfacesByVm).reduce((sum, current) => sum + current.length, 0);
    const attachedNetworks = new Set(
      Object.values(interfacesByVm)
        .flat()
        .map((entry) => entry.network_id)
        .filter(Boolean),
    );

    return {
      totalNetworks: networks.length,
      activeNetworks: networks.filter((entry) => entry.status === 'active').length,
      totalInterfaces: interfaceCount,
      utilizedNetworks: attachedNetworks.size,
    };
  }, [interfacesByVm, networks]);

  const createNetwork = async () => {
    setSaving(true);
    try {
      const createdNetwork = await networkApi.createNetwork({
        name: networkForm.name,
        type: networkForm.type,
        subnet: networkForm.subnet,
        gateway: networkForm.gateway || undefined,
      });
      setNetworks((current) => [createdNetwork, ...current]);
      setNetworkForm(emptyNetworkForm);
      setNetworkDialogOpen(false);
      toast({
        title: 'Network created',
        description: `${createdNetwork.name} is now available on the canonical network surface.`,
      });
    } catch (createError) {
      toast({
        title: 'Network creation failed',
        description: createError instanceof Error ? createError.message : 'Failed to create network.',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const deleteNetwork = async (selectedNetwork: CanonicalNetwork) => {
    if (!window.confirm(`Delete network ${selectedNetwork.name}?`)) {
      return;
    }

    try {
      await networkApi.deleteNetwork(selectedNetwork.id);
      setNetworks((current) => current.filter((entry) => entry.id !== selectedNetwork.id));
      toast({
        title: 'Network deleted',
        description: `${selectedNetwork.name} was removed from the canonical inventory.`,
      });
    } catch (deleteError) {
      toast({
        title: 'Network deletion failed',
        description: deleteError instanceof Error ? deleteError.message : 'Failed to delete network.',
        variant: 'destructive',
      });
    }
  };

  const attachInterface = async () => {
    setSaving(true);
    try {
      const attachedInterface = await networkApi.attachVmInterface(interfaceForm.vmId, {
        network_id: interfaceForm.networkId || undefined,
        name: interfaceForm.name,
        mac_address: interfaceForm.macAddress,
        ip_address: interfaceForm.ipAddress || undefined,
      });

      setInterfacesByVm((current) => ({
        ...current,
        [interfaceForm.vmId]: [...(current[interfaceForm.vmId] || []), attachedInterface],
      }));
      setInterfaceForm(emptyInterfaceForm);
      setInterfaceDialogOpen(false);
      toast({
        title: 'Interface attached',
        description: `${attachedInterface.name} was attached through the canonical network API.`,
      });
    } catch (attachError) {
      toast({
        title: 'Interface attachment failed',
        description: attachError instanceof Error ? attachError.message : 'Failed to attach interface.',
        variant: 'destructive',
      });
    } finally {
      setSaving(false);
    }
  };

  const deleteInterface = async (vmId: string, entry: CanonicalVmInterface) => {
    if (!window.confirm(`Detach interface ${entry.name} from VM ${vmId}?`)) {
      return;
    }

    try {
      await networkApi.deleteVmInterface(vmId, entry.id);
      setInterfacesByVm((current) => ({
        ...current,
        [vmId]: (current[vmId] || []).filter((candidate) => candidate.id !== entry.id),
      }));
      toast({
        title: 'Interface detached',
        description: `${entry.name} was removed from VM ${vmId}.`,
      });
    } catch (deleteError) {
      toast({
        title: 'Interface detach failed',
        description: deleteError instanceof Error ? deleteError.message : 'Failed to delete interface.',
        variant: 'destructive',
      });
    }
  };

  return (
    <AuthGuard>
      <div className="container mx-auto space-y-6 p-6">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight">Network</h1>
            <p className="text-muted-foreground">
              Canonical network inventory and VM interface management backed by `/api/v1/networks*`.
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => setInterfaceDialogOpen(true)}>
              <Link2 className="mr-2 h-4 w-4" />
              Attach Interface
            </Button>
            <Button onClick={() => setNetworkDialogOpen(true)}>
              <Plus className="mr-2 h-4 w-4" />
              Add Network
            </Button>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Networks</CardTitle>
              <Network className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalNetworks}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Active</CardTitle>
              <Badge variant="secondary">{stats.activeNetworks}</Badge>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">Inventory only. Topology and QoS analytics stay deferred.</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Interfaces</CardTitle>
              <Link2 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalInterfaces}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Utilized Networks</CardTitle>
              <Badge variant="outline">{stats.utilizedNetworks}</Badge>
            </CardHeader>
            <CardContent>
              <div className="text-sm text-muted-foreground">Based on current canonical VM interface attachments.</div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Networks</CardTitle>
            <CardDescription>Only live inventory and interface operations remain on this route.</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center text-sm text-muted-foreground">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading canonical network inventory…
              </div>
            ) : error ? (
              <div className="text-sm text-destructive">{error}</div>
            ) : networks.length === 0 ? (
              <div className="text-sm text-muted-foreground">No canonical networks have been created yet.</div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Name</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Subnet</TableHead>
                    <TableHead>Gateway</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {networks.map((entry) => (
                    <TableRow key={entry.id}>
                      <TableCell className="font-medium">{entry.name}</TableCell>
                      <TableCell>
                        <Badge variant="outline">{entry.type}</Badge>
                      </TableCell>
                      <TableCell>{entry.subnet}</TableCell>
                      <TableCell>{entry.gateway || '-'}</TableCell>
                      <TableCell>
                        <Badge variant={entry.status === 'active' ? 'secondary' : 'outline'}>{entry.status}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <Button size="sm" variant="outline" onClick={() => deleteNetwork(entry)}>
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>VM Interfaces</CardTitle>
            <CardDescription>Attach or detach interfaces from the current canonical VM inventory.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {vms.length === 0 ? (
              <div className="text-sm text-muted-foreground">No VMs are available for interface attachment.</div>
            ) : (
              vms.map((vm) => {
                const vmInterfaces = interfacesByVm[vm.id] || [];
                return (
                  <div key={vm.id} className="rounded-lg border p-4">
                    <div className="mb-3 flex items-center justify-between">
                      <div>
                        <div className="font-medium">{vm.name}</div>
                        <div className="text-xs text-muted-foreground">{vm.id}</div>
                      </div>
                      <Badge variant="outline">{vmInterfaces.length} interface(s)</Badge>
                    </div>
                    {vmInterfaces.length === 0 ? (
                      <div className="text-sm text-muted-foreground">No interfaces attached.</div>
                    ) : (
                      <div className="space-y-2">
                        {vmInterfaces.map((entry) => (
                          <div key={entry.id} className="flex flex-col gap-2 rounded-md border p-3 md:flex-row md:items-center md:justify-between">
                            <div className="text-sm">
                              <div className="font-medium">{entry.name}</div>
                              <div className="text-muted-foreground">
                                MAC {entry.mac_address} · IP {entry.ip_address || 'unassigned'} · Network {entry.network_id || 'detached'}
                              </div>
                            </div>
                            <Button size="sm" variant="outline" onClick={() => deleteInterface(vm.id, entry)}>
                              <Unplug className="mr-2 h-4 w-4" />
                              Detach
                            </Button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </CardContent>
        </Card>

        <Dialog open={networkDialogOpen} onOpenChange={setNetworkDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Network</DialogTitle>
              <DialogDescription>Add a canonical network inventory record.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="network-name">Name</Label>
                <Input
                  id="network-name"
                  value={networkForm.name}
                  onChange={(event) => setNetworkForm((current) => ({ ...current, name: event.target.value }))}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="network-type">Type</Label>
                <Select value={networkForm.type} onValueChange={(value) => setNetworkForm((current) => ({ ...current, type: value }))}>
                  <SelectTrigger id="network-type">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bridged">Bridged</SelectItem>
                    <SelectItem value="isolated">Isolated</SelectItem>
                    <SelectItem value="nat">NAT</SelectItem>
                    <SelectItem value="host-only">Host Only</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="network-subnet">Subnet</Label>
                <Input
                  id="network-subnet"
                  value={networkForm.subnet}
                  onChange={(event) => setNetworkForm((current) => ({ ...current, subnet: event.target.value }))}
                  placeholder="192.168.10.0/24"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="network-gateway">Gateway</Label>
                <Input
                  id="network-gateway"
                  value={networkForm.gateway}
                  onChange={(event) => setNetworkForm((current) => ({ ...current, gateway: event.target.value }))}
                  placeholder="192.168.10.1"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setNetworkDialogOpen(false)}>Cancel</Button>
              <Button onClick={createNetwork} disabled={saving || !networkForm.name || !networkForm.subnet}>
                {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Create Network
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        <Dialog open={interfaceDialogOpen} onOpenChange={setInterfaceDialogOpen}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Attach VM Interface</DialogTitle>
              <DialogDescription>Create a canonical VM interface attachment.</DialogDescription>
            </DialogHeader>
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="interface-vm">VM</Label>
                <Select value={interfaceForm.vmId} onValueChange={(value) => setInterfaceForm((current) => ({ ...current, vmId: value }))}>
                  <SelectTrigger id="interface-vm">
                    <SelectValue placeholder="Select VM" />
                  </SelectTrigger>
                  <SelectContent>
                    {vms.map((vm) => (
                      <SelectItem key={vm.id} value={vm.id}>
                        {vm.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="interface-network">Network</Label>
                <Select
                  value={interfaceForm.networkId}
                  onValueChange={(value) => setInterfaceForm((current) => ({ ...current, networkId: value }))}
                >
                  <SelectTrigger id="interface-network">
                    <SelectValue placeholder="Select network" />
                  </SelectTrigger>
                  <SelectContent>
                    {networks.map((entry) => (
                      <SelectItem key={entry.id} value={entry.id}>
                        {entry.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="interface-name">Interface Name</Label>
                <Input
                  id="interface-name"
                  value={interfaceForm.name}
                  onChange={(event) => setInterfaceForm((current) => ({ ...current, name: event.target.value }))}
                  placeholder="eth1"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="interface-mac">MAC Address</Label>
                <Input
                  id="interface-mac"
                  value={interfaceForm.macAddress}
                  onChange={(event) => setInterfaceForm((current) => ({ ...current, macAddress: event.target.value }))}
                  placeholder="00:16:3e:12:34:56"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="interface-ip">IP Address</Label>
                <Input
                  id="interface-ip"
                  value={interfaceForm.ipAddress}
                  onChange={(event) => setInterfaceForm((current) => ({ ...current, ipAddress: event.target.value }))}
                  placeholder="192.168.10.15"
                />
              </div>
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setInterfaceDialogOpen(false)}>Cancel</Button>
              <Button
                onClick={attachInterface}
                disabled={saving || !interfaceForm.vmId || !interfaceForm.name || !interfaceForm.macAddress}
              >
                {saving ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
                Attach Interface
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </AuthGuard>
  );
}
