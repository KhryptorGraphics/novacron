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
import { networkApi, type FirewallRule, type LoadBalancer } from "@/lib/api/networks";
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
  const [firewallRules, setFirewallRules] = useState<FirewallRule[]>([]);
  const [loadBalancers, setLoadBalancers] = useState<LoadBalancer[]>([]);
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
      const [response, rules, balancers] = await Promise.all([
        networkApi.listNetworks(),
        networkApi.listFirewallRules(),
        networkApi.listLoadBalancers(),
      ]);
      setNetworks(Array.isArray(response) ? response : []);
      setFirewallRules(Array.isArray(rules) ? rules : []);
      setLoadBalancers(Array.isArray(balancers) ? balancers : []);
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
      const created = await networkApi.createNetwork({
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
      await networkApi.deleteNetwork(networkId);
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

  const createFirewallRule = async () => {
    if (!selectedNetwork) {
      setError("Select a network before creating a firewall rule.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const created = await networkApi.createFirewallRule({
        name: `${selectedNetwork.name} allow tcp`,
        networkId: selectedNetwork.id,
        direction: "inbound",
        action: "allow",
        protocol: "tcp",
        source: "0.0.0.0/0",
        destination: selectedNetwork.subnet,
        port: "any",
        priority: 100,
        enabled: true,
      });
      setFirewallRules((current) => [created, ...current]);
      setNotice("Firewall rule created through the canonical network policy API.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create firewall rule.");
    } finally {
      setLoading(false);
    }
  };

  const deleteFirewallRule = async (ruleId: string) => {
    setLoading(true);
    setError(null);
    try {
      await networkApi.deleteFirewallRule(ruleId);
      setFirewallRules((current) => current.filter((rule) => rule.id !== ruleId));
      setNotice("Firewall rule deleted through the canonical network policy API.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete firewall rule.");
    } finally {
      setLoading(false);
    }
  };

  const createLoadBalancer = async () => {
    if (!selectedNetwork) {
      setError("Select a network before creating a load balancer.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const created = await networkApi.createLoadBalancer({
        name: `${selectedNetwork.name} edge`,
        networkId: selectedNetwork.id,
        vip: selectedNetwork.gateway || "",
        port: 80,
        algorithm: "round_robin",
        type: "layer4",
      });
      setLoadBalancers((current) => [created, ...current]);
      setNotice("Load balancer created through the canonical load-balancer API.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create load balancer.");
    } finally {
      setLoading(false);
    }
  };

  const deleteLoadBalancer = async (loadBalancerId: string) => {
    setLoading(true);
    setError(null);
    try {
      await networkApi.deleteLoadBalancer(loadBalancerId);
      setLoadBalancers((current) => current.filter((lb) => lb.id !== loadBalancerId));
      setNotice("Load balancer deleted through the canonical load-balancer API.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to delete load balancer.");
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
          Virtual networks, firewall rules, and load balancers are backed by live canonical network contracts.
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
            <div className="text-2xl font-bold">{firewallRules.length}</div>
            <p className="text-xs text-muted-foreground">Canonical policy records</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Load Balancers</CardTitle>
            <Router className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loadBalancers.length}</div>
            <p className="text-xs text-muted-foreground">Canonical balancer records</p>
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
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Firewall Rules</CardTitle>
                <CardDescription>Live records from `/api/v1/network/firewall-rules`</CardDescription>
              </div>
              <Button onClick={createFirewallRule} disabled={loading || !selectedNetwork}>
                <Plus className="mr-2 h-4 w-4" />
                Add Rule
              </Button>
            </CardHeader>
            <CardContent className="space-y-3">
              {firewallRules.length === 0 && (
                <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
                  {loading ? "Loading firewall rules..." : "No firewall rules returned by the canonical API."}
                </div>
              )}
              {firewallRules.map((rule) => (
                <div key={rule.id} className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <div className="font-medium">{rule.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {rule.direction} {rule.action} {rule.protocol} {rule.source} to {rule.destination}:{rule.port}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{rule.status}</Badge>
                    <Button variant="ghost" size="sm" onClick={() => deleteFirewallRule(rule.id)} disabled={loading}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="load-balancers">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>Load Balancers</CardTitle>
                <CardDescription>Live records from `/api/v1/network/load-balancers`</CardDescription>
              </div>
              <Button onClick={createLoadBalancer} disabled={loading || !selectedNetwork}>
                <Plus className="mr-2 h-4 w-4" />
                Add Load Balancer
              </Button>
            </CardHeader>
            <CardContent className="space-y-3">
              {loadBalancers.length === 0 && (
                <div className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
                  {loading ? "Loading load balancers..." : "No load balancers returned by the canonical API."}
                </div>
              )}
              {loadBalancers.map((balancer) => (
                <div key={balancer.id} className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <div className="font-medium">{balancer.name}</div>
                    <div className="text-sm text-muted-foreground">
                      {balancer.type} {balancer.algorithm} on {balancer.vip || "unassigned"}:{balancer.port}
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline">{balancer.status}</Badge>
                    <Button variant="ghost" size="sm" onClick={() => deleteLoadBalancer(balancer.id)} disabled={loading}>
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ))}
            </CardContent>
          </Card>
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
