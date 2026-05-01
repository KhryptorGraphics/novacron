"use client";

import { useEffect, useState } from "react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  AlertCircle,
  CheckCircle,
  Network,
  Plus,
  RefreshCw,
  Router,
  Shield,
  Trash2,
} from "lucide-react";
import { apiClient } from "@/lib/api/client";
import { cn } from "@/lib/utils";

type NetworkRecord = {
  id: string;
  name: string;
  type: "bridged" | "overlay" | "isolated" | string;
  subnet: string;
  gateway?: string | null;
  status: string;
  created_at?: string;
  updated_at?: string;
};

type CreateNetworkForm = {
  name: string;
  type: "bridged" | "overlay" | "isolated";
  subnet: string;
  gateway: string;
};

const emptyNetworkForm: CreateNetworkForm = {
  name: "",
  type: "bridged",
  subnet: "",
  gateway: "",
};

function formatDate(value?: string) {
  if (!value) return "unknown";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function statusClass(status: string) {
  switch (status.toLowerCase()) {
    case "active":
      return "bg-green-100 text-green-800";
    case "configuring":
      return "bg-yellow-100 text-yellow-800";
    case "inactive":
      return "bg-gray-100 text-gray-800";
    default:
      return "";
  }
}

export function NetworkConfigurationPanel() {
  const [networks, setNetworks] = useState<NetworkRecord[]>([]);
  const [selectedNetworkId, setSelectedNetworkId] = useState<string | null>(null);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [newNetwork, setNewNetwork] = useState<CreateNetworkForm>(emptyNetworkForm);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const selectedNetwork = networks.find((network) => network.id === selectedNetworkId) || networks[0] || null;

  const loadNetworks = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await apiClient.get<NetworkRecord[]>("/api/v1/networks");
      setNetworks(Array.isArray(response) ? response : []);
      if (!selectedNetworkId && Array.isArray(response) && response.length > 0) {
        setSelectedNetworkId(response[0].id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load networks.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadNetworks();
    // loadNetworks intentionally closes over current selectedNetworkId only for first selection.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const createNetwork = async () => {
    const name = newNetwork.name.trim();
    const subnet = newNetwork.subnet.trim();
    if (!name || !subnet) {
      setError("Network name and subnet are required.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const created = await apiClient.post<NetworkRecord>("/api/v1/networks", {
        name,
        type: newNetwork.type,
        subnet,
        gateway: newNetwork.gateway.trim(),
      });
      setNetworks((current) => [created, ...current]);
      setSelectedNetworkId(created.id);
      setNewNetwork(emptyNetworkForm);
      setCreateDialogOpen(false);
      setNotice("Network created through the canonical network API.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create network.");
    } finally {
      setLoading(false);
    }
  };

  const deleteNetwork = async (networkId: string) => {
    if (!window.confirm("Delete this network through the canonical network API?")) {
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await apiClient.delete(`/api/v1/networks/${networkId}`);
      setNetworks((current) => current.filter((network) => network.id !== networkId));
      if (selectedNetworkId === networkId) {
        setSelectedNetworkId(null);
      }
      setNotice("Network deleted through the canonical network API.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete network.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-bold">Network Configuration</h1>
          <p className="text-muted-foreground">
            Manage virtual networks through the canonical NovaCron network API
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={loadNetworks} disabled={loading}>
            <RefreshCw className={cn("mr-2 h-4 w-4", loading && "animate-spin")} />
            Refresh
          </Button>
          <Button onClick={() => setCreateDialogOpen(true)} disabled={loading}>
            <Plus className="mr-2 h-4 w-4" />
            Create Network
          </Button>
        </div>
      </div>

      <Alert className="border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950">
        <CheckCircle className="h-4 w-4 text-green-700 dark:text-green-300" />
        <AlertDescription className="text-green-800 dark:text-green-200">
          Virtual networks are backed by live `/api/v1/networks` contracts. Firewall and load balancer controls stay disabled until their canonical APIs exist.
        </AlertDescription>
      </Alert>

      {notice && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertDescription>{notice}</AlertDescription>
        </Alert>
      )}

      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Virtual Networks</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{networks.length}</div>
            <p className="text-xs text-muted-foreground">
              {networks.filter((network) => network.status === "active").length} active
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Firewall Rules</CardTitle>
            <Shield className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">pending</div>
            <p className="text-xs text-muted-foreground">No canonical rule API yet</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Load Balancers</CardTitle>
            <Router className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">pending</div>
            <p className="text-xs text-muted-foreground">No canonical load-balancer API yet</p>
          </CardContent>
        </Card>
      </div>

      <Tabs defaultValue="networks" className="space-y-4">
        <TabsList>
          <TabsTrigger value="networks">Virtual Networks</TabsTrigger>
          <TabsTrigger value="firewall">Firewall</TabsTrigger>
          <TabsTrigger value="load-balancers">Load Balancers</TabsTrigger>
        </TabsList>

        <TabsContent value="networks" className="space-y-4">
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Networks</CardTitle>
                <CardDescription>Live records from `/api/v1/networks`</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {networks.length === 0 && (
                    <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
                      {loading ? "Loading networks..." : "No networks returned by the canonical API."}
                    </div>
                  )}

                  {networks.map((network) => (
                    <button
                      key={network.id}
                      type="button"
                      onClick={() => setSelectedNetworkId(network.id)}
                      className={cn(
                        "w-full rounded-lg border p-4 text-left transition-colors hover:bg-muted/60",
                        selectedNetwork?.id === network.id && "border-primary bg-muted"
                      )}
                    >
                      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                        <div className="space-y-1">
                          <div className="flex items-center gap-2">
                            <span className="font-semibold">{network.name}</span>
                            <Badge variant="outline">{network.type}</Badge>
                            <Badge className={statusClass(network.status)}>{network.status}</Badge>
                          </div>
                          <div className="font-mono text-sm text-muted-foreground">{network.subnet}</div>
                          <div className="text-sm text-muted-foreground">
                            Gateway: {network.gateway || "not configured"}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(event) => {
                            event.stopPropagation();
                            deleteNetwork(network.id);
                          }}
                          disabled={loading}
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Network Details</CardTitle>
                <CardDescription>{selectedNetwork?.id || "Select a network"}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedNetwork ? (
                  <>
                    <Detail label="Name" value={selectedNetwork.name} />
                    <Detail label="Type" value={selectedNetwork.type} />
                    <Detail label="Subnet" value={selectedNetwork.subnet} mono />
                    <Detail label="Gateway" value={selectedNetwork.gateway || "not configured"} mono />
                    <Detail label="Status" value={selectedNetwork.status} />
                    <Detail label="Created" value={formatDate(selectedNetwork.created_at)} />
                    <Detail label="Updated" value={formatDate(selectedNetwork.updated_at)} />
                  </>
                ) : (
                  <div className="text-sm text-muted-foreground">No network selected.</div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="firewall">
          <PendingContractCard
            title="Firewall Rules"
            description="Firewall and security-group editing is disabled until a canonical network policy API is available."
          />
        </TabsContent>

        <TabsContent value="load-balancers">
          <PendingContractCard
            title="Load Balancers"
            description="Load balancer topology and backend pool management is disabled until a canonical load-balancer API is available."
          />
        </TabsContent>
      </Tabs>

      <Dialog open={createDialogOpen} onOpenChange={setCreateDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Network</DialogTitle>
            <DialogDescription>
              Creates a network through `POST /api/v1/networks`.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="network-name">Name</Label>
              <Input
                id="network-name"
                value={newNetwork.name}
                onChange={(event) => setNewNetwork((current) => ({ ...current, name: event.target.value }))}
                placeholder="Production network"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="network-subnet">Subnet</Label>
              <Input
                id="network-subnet"
                value={newNetwork.subnet}
                onChange={(event) => setNewNetwork((current) => ({ ...current, subnet: event.target.value }))}
                placeholder="10.0.0.0/24"
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="network-gateway">Gateway</Label>
              <Input
                id="network-gateway"
                value={newNetwork.gateway}
                onChange={(event) => setNewNetwork((current) => ({ ...current, gateway: event.target.value }))}
                placeholder="10.0.0.1"
              />
            </div>
            <div className="space-y-2">
              <Label>Type</Label>
              <Select
                value={newNetwork.type}
                onValueChange={(value) => setNewNetwork((current) => ({
                  ...current,
                  type: value as CreateNetworkForm["type"],
                }))}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="bridged">Bridged</SelectItem>
                  <SelectItem value="overlay">Overlay</SelectItem>
                  <SelectItem value="isolated">Isolated</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateDialogOpen(false)} disabled={loading}>
              Cancel
            </Button>
            <Button onClick={createNetwork} disabled={loading}>
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function Detail({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex justify-between gap-4">
      <span className="text-sm text-muted-foreground">{label}</span>
      <span className={cn("text-right text-sm font-medium", mono && "font-mono")}>{value}</span>
    </div>
  );
}

function PendingContractCard({ title, description }: { title: string; description: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>Canonical contract pending</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border border-dashed p-6 text-sm text-muted-foreground">
          {description}
        </div>
      </CardContent>
    </Card>
  );
}
