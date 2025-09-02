'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Network,
  Globe,
  Shield,
  Wifi,
  Router,
  Server,
  Link,
  Settings,
  Plus,
  Trash2,
  Edit,
  Save,
  X,
  AlertCircle,
  CheckCircle,
  Activity,
  Layers,
  Share2,
  Lock,
  Unlock,
  GitBranch,
  Cloud,
  Database,
  Zap,
  Filter,
  ArrowRight,
  ArrowLeft,
  ArrowUpDown,
  MoreVertical,
  Copy,
  Download,
  Upload
} from 'lucide-react';
import { Diagram, Draggable, Portal } from '@/components/ui/diagram';

interface VirtualNetwork {
  id: string;
  name: string;
  cidr: string;
  vlan: number;
  type: 'overlay' | 'bridged' | 'isolated';
  status: 'active' | 'inactive' | 'configuring';
  subnets: Subnet[];
  connectedVMs: number;
  gateway: string;
  dhcp: boolean;
  dns: string[];
  mtu: number;
}

interface Subnet {
  id: string;
  name: string;
  cidr: string;
  available: number;
  used: number;
  gateway: string;
}

interface FirewallRule {
  id: string;
  name: string;
  enabled: boolean;
  direction: 'inbound' | 'outbound';
  action: 'allow' | 'deny';
  protocol: 'tcp' | 'udp' | 'icmp' | 'any';
  source: string;
  destination: string;
  port: string;
  priority: number;
  hits: number;
}

interface LoadBalancer {
  id: string;
  name: string;
  type: 'layer4' | 'layer7';
  algorithm: 'round-robin' | 'least-connections' | 'ip-hash';
  status: 'active' | 'inactive' | 'degraded';
  vip: string;
  port: number;
  backends: Backend[];
  healthCheck: {
    enabled: boolean;
    interval: number;
    timeout: number;
    unhealthyThreshold: number;
  };
  ssl: boolean;
  persistence: 'none' | 'source-ip' | 'cookie';
}

interface Backend {
  id: string;
  ip: string;
  port: number;
  weight: number;
  status: 'healthy' | 'unhealthy' | 'draining';
  activeConnections: number;
}

interface NetworkFlow {
  source: string;
  destination: string;
  protocol: string;
  port: number;
  bytes: number;
  packets: number;
  timestamp: string;
}

const NetworkConfigurationPanel: React.FC = () => {
  const [networks, setNetworks] = useState<VirtualNetwork[]>([]);
  const [firewallRules, setFirewallRules] = useState<FirewallRule[]>([]);
  const [loadBalancers, setLoadBalancers] = useState<LoadBalancer[]>([]);
  const [networkFlows, setNetworkFlows] = useState<NetworkFlow[]>([]);
  const [selectedNetwork, setSelectedNetwork] = useState<VirtualNetwork | null>(null);
  const [selectedLB, setSelectedLB] = useState<LoadBalancer | null>(null);
  const [createNetworkDialog, setCreateNetworkDialog] = useState(false);
  const [createRuleDialog, setCreateRuleDialog] = useState(false);
  const [createLBDialog, setCreateLBDialog] = useState(false);
  const [topologyView, setTopologyView] = useState('logical');

  // Form states
  const [newNetwork, setNewNetwork] = useState({
    name: '',
    cidr: '10.0.0.0/24',
    vlan: 100,
    type: 'overlay',
    dhcp: true,
    dns: ['8.8.8.8', '8.8.4.4'],
    mtu: 1500
  });

  const [newRule, setNewRule] = useState({
    name: '',
    direction: 'inbound',
    action: 'allow',
    protocol: 'tcp',
    source: 'any',
    destination: 'any',
    port: '',
    priority: 100
  });

  // Mock data
  useEffect(() => {
    const mockNetworks: VirtualNetwork[] = [
      {
        id: 'net-001',
        name: 'Production Network',
        cidr: '10.0.0.0/16',
        vlan: 100,
        type: 'overlay',
        status: 'active',
        subnets: [
          { id: 'sub-001', name: 'Web Tier', cidr: '10.0.1.0/24', available: 200, used: 54, gateway: '10.0.1.1' },
          { id: 'sub-002', name: 'App Tier', cidr: '10.0.2.0/24', available: 200, used: 32, gateway: '10.0.2.1' },
          { id: 'sub-003', name: 'DB Tier', cidr: '10.0.3.0/24', available: 200, used: 12, gateway: '10.0.3.1' }
        ],
        connectedVMs: 98,
        gateway: '10.0.0.1',
        dhcp: true,
        dns: ['10.0.0.2', '10.0.0.3'],
        mtu: 1500
      },
      {
        id: 'net-002',
        name: 'Development Network',
        cidr: '172.16.0.0/16',
        vlan: 200,
        type: 'bridged',
        status: 'active',
        subnets: [
          { id: 'sub-004', name: 'Dev Subnet', cidr: '172.16.1.0/24', available: 200, used: 25, gateway: '172.16.1.1' }
        ],
        connectedVMs: 25,
        gateway: '172.16.0.1',
        dhcp: true,
        dns: ['8.8.8.8', '8.8.4.4'],
        mtu: 1500
      },
      {
        id: 'net-003',
        name: 'DMZ Network',
        cidr: '192.168.1.0/24',
        vlan: 300,
        type: 'isolated',
        status: 'active',
        subnets: [],
        connectedVMs: 5,
        gateway: '192.168.1.1',
        dhcp: false,
        dns: ['1.1.1.1'],
        mtu: 1500
      }
    ];

    const mockFirewallRules: FirewallRule[] = [
      {
        id: 'rule-001',
        name: 'Allow HTTP',
        enabled: true,
        direction: 'inbound',
        action: 'allow',
        protocol: 'tcp',
        source: 'any',
        destination: '10.0.1.0/24',
        port: '80',
        priority: 100,
        hits: 125432
      },
      {
        id: 'rule-002',
        name: 'Allow HTTPS',
        enabled: true,
        direction: 'inbound',
        action: 'allow',
        protocol: 'tcp',
        source: 'any',
        destination: '10.0.1.0/24',
        port: '443',
        priority: 101,
        hits: 543211
      },
      {
        id: 'rule-003',
        name: 'Block SSH from Internet',
        enabled: true,
        direction: 'inbound',
        action: 'deny',
        protocol: 'tcp',
        source: '0.0.0.0/0',
        destination: '10.0.0.0/16',
        port: '22',
        priority: 50,
        hits: 8923
      },
      {
        id: 'rule-004',
        name: 'Allow Internal Traffic',
        enabled: true,
        direction: 'inbound',
        action: 'allow',
        protocol: 'any',
        source: '10.0.0.0/8',
        destination: '10.0.0.0/8',
        port: 'any',
        priority: 200,
        hits: 9876543
      }
    ];

    const mockLoadBalancers: LoadBalancer[] = [
      {
        id: 'lb-001',
        name: 'Web Load Balancer',
        type: 'layer7',
        algorithm: 'round-robin',
        status: 'active',
        vip: '10.0.0.100',
        port: 443,
        backends: [
          { id: 'be-001', ip: '10.0.1.10', port: 443, weight: 1, status: 'healthy', activeConnections: 234 },
          { id: 'be-002', ip: '10.0.1.11', port: 443, weight: 1, status: 'healthy', activeConnections: 189 },
          { id: 'be-003', ip: '10.0.1.12', port: 443, weight: 1, status: 'unhealthy', activeConnections: 0 }
        ],
        healthCheck: {
          enabled: true,
          interval: 30,
          timeout: 5,
          unhealthyThreshold: 3
        },
        ssl: true,
        persistence: 'cookie'
      },
      {
        id: 'lb-002',
        name: 'Database Load Balancer',
        type: 'layer4',
        algorithm: 'least-connections',
        status: 'active',
        vip: '10.0.3.100',
        port: 5432,
        backends: [
          { id: 'be-004', ip: '10.0.3.10', port: 5432, weight: 2, status: 'healthy', activeConnections: 45 },
          { id: 'be-005', ip: '10.0.3.11', port: 5432, weight: 1, status: 'healthy', activeConnections: 23 }
        ],
        healthCheck: {
          enabled: true,
          interval: 10,
          timeout: 3,
          unhealthyThreshold: 2
        },
        ssl: false,
        persistence: 'source-ip'
      }
    ];

    const mockFlows: NetworkFlow[] = [
      { source: '10.0.1.10', destination: '10.0.3.10', protocol: 'TCP', port: 5432, bytes: 1024000, packets: 1500, timestamp: '2024-03-21T10:30:00Z' },
      { source: '192.168.1.5', destination: '10.0.1.10', protocol: 'TCP', port: 443, bytes: 512000, packets: 800, timestamp: '2024-03-21T10:31:00Z' },
      { source: '10.0.2.15', destination: '10.0.3.11', protocol: 'TCP', port: 5432, bytes: 2048000, packets: 2300, timestamp: '2024-03-21T10:32:00Z' }
    ];

    setNetworks(mockNetworks);
    setFirewallRules(mockFirewallRules);
    setLoadBalancers(mockLoadBalancers);
    setNetworkFlows(mockFlows);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': case 'healthy': return 'bg-green-500';
      case 'inactive': case 'unhealthy': return 'bg-red-500';
      case 'configuring': case 'degraded': case 'draining': return 'bg-yellow-500';
      default: return 'bg-gray-500';
    }
  };

  const handleCreateNetwork = () => {
    console.log('Creating network:', newNetwork);
    setCreateNetworkDialog(false);
  };

  const handleCreateRule = () => {
    console.log('Creating firewall rule:', newRule);
    setCreateRuleDialog(false);
  };

  const handleToggleRule = (ruleId: string) => {
    setFirewallRules(prev => prev.map(rule => 
      rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
    ));
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold">Network Configuration</h1>
          <p className="text-muted-foreground">Manage virtual networks, firewall rules, and load balancers</p>
        </div>
        <div className="flex gap-2">
          <Button onClick={() => setCreateNetworkDialog(true)}>
            <Plus className="mr-2 h-4 w-4" />
            Create Network
          </Button>
          <Button variant="outline" onClick={() => setCreateRuleDialog(true)}>
            <Shield className="mr-2 h-4 w-4" />
            Add Firewall Rule
          </Button>
        </div>
      </div>

      {/* Network Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Virtual Networks</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{networks.length}</div>
            <p className="text-xs text-muted-foreground">
              {networks.filter(n => n.status === 'active').length} active
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Connected VMs</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {networks.reduce((sum, n) => sum + n.connectedVMs, 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              Across {networks.length} networks
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
            <p className="text-xs text-muted-foreground">
              {firewallRules.filter(r => r.enabled).length} enabled
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Load Balancers</CardTitle>
            <Layers className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{loadBalancers.length}</div>
            <p className="text-xs text-muted-foreground">
              {loadBalancers.filter(lb => lb.status === 'active').length} active
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="networks" className="space-y-4">
        <TabsList>
          <TabsTrigger value="networks">Virtual Networks</TabsTrigger>
          <TabsTrigger value="firewall">Firewall Rules</TabsTrigger>
          <TabsTrigger value="loadbalancer">Load Balancers</TabsTrigger>
          <TabsTrigger value="topology">Network Topology</TabsTrigger>
          <TabsTrigger value="flows">Traffic Flows</TabsTrigger>
        </TabsList>

        <TabsContent value="networks" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Network List */}
            <div className="lg:col-span-2">
              <Card>
                <CardHeader>
                  <CardTitle>Virtual Networks</CardTitle>
                  <CardDescription>Configure and manage virtual network segments</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {networks.map((network) => (
                    <div
                      key={network.id}
                      className={`border rounded-lg p-4 hover:bg-accent cursor-pointer transition-colors ${
                        selectedNetwork?.id === network.id ? 'ring-2 ring-primary' : ''
                      }`}
                      onClick={() => setSelectedNetwork(network)}
                    >
                      <div className="space-y-3">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <Network className="h-4 w-4 text-muted-foreground" />
                            <h3 className="font-semibold">{network.name}</h3>
                            <Badge variant="outline" className={`${getStatusColor(network.status)} text-white`}>
                              {network.status}
                            </Badge>
                            <Badge variant="outline">{network.type}</Badge>
                          </div>
                          <div className="flex gap-1">
                            <Button size="icon" variant="ghost">
                              <Settings className="h-4 w-4" />
                            </Button>
                            <Button size="icon" variant="ghost">
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>

                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div>
                            <p className="text-muted-foreground">CIDR</p>
                            <p className="font-medium font-mono">{network.cidr}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">VLAN</p>
                            <p className="font-medium">{network.vlan}</p>
                          </div>
                          <div>
                            <p className="text-muted-foreground">Connected VMs</p>
                            <p className="font-medium">{network.connectedVMs}</p>
                          </div>
                        </div>

                        {network.subnets.length > 0 && (
                          <div className="space-y-2">
                            <p className="text-sm font-medium">Subnets</p>
                            <div className="space-y-1">
                              {network.subnets.map((subnet) => (
                                <div key={subnet.id} className="flex items-center justify-between text-sm p-2 bg-muted rounded">
                                  <span className="font-mono">{subnet.name}</span>
                                  <span className="text-muted-foreground">{subnet.cidr}</span>
                                  <Badge variant="outline" className="text-xs">
                                    {subnet.used}/{subnet.available} IPs
                                  </Badge>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </CardContent>
              </Card>
            </div>

            {/* Network Details */}
            <div>
              {selectedNetwork ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Network Details</CardTitle>
                    <CardDescription>{selectedNetwork.name}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">Gateway</span>
                        <span className="text-sm font-medium font-mono">{selectedNetwork.gateway}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">DHCP</span>
                        <Badge variant={selectedNetwork.dhcp ? 'default' : 'outline'}>
                          {selectedNetwork.dhcp ? 'Enabled' : 'Disabled'}
                        </Badge>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-muted-foreground">MTU</span>
                        <span className="text-sm font-medium">{selectedNetwork.mtu}</span>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground mb-1">DNS Servers</p>
                        <div className="space-y-1">
                          {selectedNetwork.dns.map((dns, index) => (
                            <div key={index} className="text-sm font-mono bg-muted p-1 rounded">
                              {dns}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Button className="w-full" variant="outline">
                        <Edit className="mr-2 h-4 w-4" />
                        Edit Configuration
                      </Button>
                      <Button className="w-full" variant="outline">
                        <GitBranch className="mr-2 h-4 w-4" />
                        Create Subnet
                      </Button>
                      <Button className="w-full" variant="outline">
                        <Link className="mr-2 h-4 w-4" />
                        Connect VM
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="flex items-center justify-center h-[400px] text-muted-foreground">
                    Select a network to view details
                  </CardContent>
                </Card>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="firewall" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Firewall Rules</CardTitle>
                  <CardDescription>Configure inbound and outbound traffic rules</CardDescription>
                </div>
                <Button onClick={() => setCreateRuleDialog(true)}>
                  <Plus className="mr-2 h-4 w-4" />
                  Add Rule
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {firewallRules
                  .sort((a, b) => a.priority - b.priority)
                  .map((rule) => (
                    <div key={rule.id} className="flex items-center justify-between p-4 border rounded-lg">
                      <div className="flex items-center gap-4 flex-1">
                        <Switch
                          checked={rule.enabled}
                          onCheckedChange={() => handleToggleRule(rule.id)}
                        />
                        <div className="flex items-center gap-2">
                          {rule.direction === 'inbound' ? (
                            <ArrowRight className="h-4 w-4 text-blue-500" />
                          ) : (
                            <ArrowLeft className="h-4 w-4 text-green-500" />
                          )}
                          <Badge variant={rule.action === 'allow' ? 'default' : 'destructive'}>
                            {rule.action}
                          </Badge>
                        </div>
                        <div className="space-y-1 flex-1">
                          <div className="flex items-center gap-2">
                            <p className="font-medium">{rule.name}</p>
                            <Badge variant="outline" className="text-xs">
                              Priority: {rule.priority}
                            </Badge>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-muted-foreground">
                            <span className="font-mono">{rule.source}</span>
                            <ArrowRight className="h-3 w-3" />
                            <span className="font-mono">{rule.destination}</span>
                            <span>{rule.protocol.toUpperCase()}/{rule.port}</span>
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-sm font-medium">{rule.hits.toLocaleString()}</p>
                          <p className="text-xs text-muted-foreground">hits</p>
                        </div>
                        <div className="flex gap-1">
                          <Button size="icon" variant="ghost">
                            <Edit className="h-4 w-4" />
                          </Button>
                          <Button size="icon" variant="ghost">
                            <Copy className="h-4 w-4" />
                          </Button>
                          <Button size="icon" variant="ghost">
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="loadbalancer" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {loadBalancers.map((lb) => (
              <Card key={lb.id}>
                <CardHeader>
                  <div className="flex justify-between items-start">
                    <div>
                      <CardTitle>{lb.name}</CardTitle>
                      <CardDescription>
                        {lb.type === 'layer7' ? 'Application (L7)' : 'Network (L4)'} • {lb.algorithm}
                      </CardDescription>
                    </div>
                    <Badge className={getStatusColor(lb.status)}>
                      {lb.status}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-muted-foreground">Virtual IP</p>
                      <p className="font-medium font-mono">{lb.vip}:{lb.port}</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Persistence</p>
                      <p className="font-medium">{lb.persistence}</p>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold mb-2">Backend Servers</h4>
                    <div className="space-y-2">
                      {lb.backends.map((backend) => (
                        <div key={backend.id} className="flex items-center justify-between p-2 border rounded">
                          <div className="flex items-center gap-2">
                            <div className={`h-2 w-2 rounded-full ${
                              backend.status === 'healthy' ? 'bg-green-500' :
                              backend.status === 'unhealthy' ? 'bg-red-500' : 'bg-yellow-500'
                            }`} />
                            <span className="text-sm font-mono">{backend.ip}:{backend.port}</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant="outline" className="text-xs">
                              Weight: {backend.weight}
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {backend.activeConnections} conn
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {lb.healthCheck.enabled && (
                    <div className="text-sm">
                      <p className="text-muted-foreground">Health Check</p>
                      <p className="font-medium">
                        Every {lb.healthCheck.interval}s • Timeout {lb.healthCheck.timeout}s
                      </p>
                    </div>
                  )}

                  <div className="flex gap-2">
                    {lb.ssl && (
                      <Badge variant="outline">
                        <Lock className="h-3 w-3 mr-1" />
                        SSL
                      </Badge>
                    )}
                    <Badge variant="outline">
                      {lb.backends.filter(b => b.status === 'healthy').length}/{lb.backends.length} healthy
                    </Badge>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="topology" className="space-y-4">
          <Card>
            <CardHeader>
              <div className="flex justify-between items-center">
                <div>
                  <CardTitle>Network Topology</CardTitle>
                  <CardDescription>Visual representation of network architecture</CardDescription>
                </div>
                <Select value={topologyView} onValueChange={setTopologyView}>
                  <SelectTrigger className="w-[150px]">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="logical">Logical View</SelectItem>
                    <SelectItem value="physical">Physical View</SelectItem>
                    <SelectItem value="security">Security Zones</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-[500px] border rounded-lg bg-muted/20 p-4">
                {/* Simplified topology visualization */}
                <div className="flex justify-center items-center h-full">
                  <div className="space-y-8">
                    {/* Internet */}
                    <div className="flex justify-center">
                      <div className="flex items-center gap-2 p-3 bg-background border-2 rounded-lg">
                        <Globe className="h-5 w-5" />
                        <span className="font-medium">Internet</span>
                      </div>
                    </div>
                    
                    {/* Firewall */}
                    <div className="flex justify-center">
                      <div className="w-1 h-8 bg-border" />
                    </div>
                    <div className="flex justify-center">
                      <div className="flex items-center gap-2 p-3 bg-red-50 border-2 border-red-200 rounded-lg">
                        <Shield className="h-5 w-5 text-red-600" />
                        <span className="font-medium">Firewall</span>
                      </div>
                    </div>
                    
                    {/* Load Balancer */}
                    <div className="flex justify-center">
                      <div className="w-1 h-8 bg-border" />
                    </div>
                    <div className="flex justify-center">
                      <div className="flex items-center gap-2 p-3 bg-blue-50 border-2 border-blue-200 rounded-lg">
                        <Layers className="h-5 w-5 text-blue-600" />
                        <span className="font-medium">Load Balancer</span>
                      </div>
                    </div>
                    
                    {/* Networks */}
                    <div className="flex justify-center">
                      <div className="w-1 h-8 bg-border" />
                    </div>
                    <div className="flex justify-center gap-8">
                      {networks.slice(0, 3).map((network) => (
                        <div key={network.id} className="flex flex-col items-center gap-2">
                          <div className="flex items-center gap-2 p-3 bg-green-50 border-2 border-green-200 rounded-lg">
                            <Network className="h-5 w-5 text-green-600" />
                            <span className="font-medium text-sm">{network.name}</span>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            {network.connectedVMs} VMs
                          </Badge>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="flows" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Network Traffic Flows</CardTitle>
              <CardDescription>Real-time traffic analysis and flow monitoring</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex items-center gap-4">
                  <div className="flex items-center gap-2">
                    <Filter className="h-4 w-4 text-muted-foreground" />
                    <Select defaultValue="all">
                      <SelectTrigger className="w-[150px]">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Traffic</SelectItem>
                        <SelectItem value="tcp">TCP Only</SelectItem>
                        <SelectItem value="udp">UDP Only</SelectItem>
                        <SelectItem value="high">High Volume</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="mr-2 h-4 w-4" />
                    Export
                  </Button>
                </div>

                <div className="space-y-2">
                  {networkFlows.map((flow, index) => (
                    <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-4 flex-1">
                        <Activity className="h-4 w-4 text-muted-foreground" />
                        <div className="flex items-center gap-2 flex-1">
                          <span className="font-mono text-sm">{flow.source}</span>
                          <ArrowRight className="h-3 w-3 text-muted-foreground" />
                          <span className="font-mono text-sm">{flow.destination}</span>
                        </div>
                        <Badge variant="outline">{flow.protocol}/{flow.port}</Badge>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        <div>
                          <p className="font-medium">{(flow.bytes / 1024).toFixed(1)} KB</p>
                          <p className="text-xs text-muted-foreground">{flow.packets} packets</p>
                        </div>
                        <span className="text-xs text-muted-foreground">
                          {new Date(flow.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Create Network Dialog */}
      <Dialog open={createNetworkDialog} onOpenChange={setCreateNetworkDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Create Virtual Network</DialogTitle>
            <DialogDescription>
              Configure a new virtual network segment
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="network-name">Network Name</Label>
              <Input
                id="network-name"
                value={newNetwork.name}
                onChange={(e) => setNewNetwork({ ...newNetwork, name: e.target.value })}
                placeholder="production-network"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="cidr">CIDR Block</Label>
                <Input
                  id="cidr"
                  value={newNetwork.cidr}
                  onChange={(e) => setNewNetwork({ ...newNetwork, cidr: e.target.value })}
                  placeholder="10.0.0.0/24"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="vlan">VLAN ID</Label>
                <Input
                  id="vlan"
                  type="number"
                  value={newNetwork.vlan}
                  onChange={(e) => setNewNetwork({ ...newNetwork, vlan: parseInt(e.target.value) })}
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label>Network Type</Label>
              <Select
                value={newNetwork.type}
                onValueChange={(value) => setNewNetwork({ ...newNetwork, type: value as any })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="overlay">Overlay (VXLAN)</SelectItem>
                  <SelectItem value="bridged">Bridged</SelectItem>
                  <SelectItem value="isolated">Isolated</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <Label htmlFor="dhcp">Enable DHCP</Label>
              <Switch
                id="dhcp"
                checked={newNetwork.dhcp}
                onCheckedChange={(checked) => setNewNetwork({ ...newNetwork, dhcp: checked })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateNetworkDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateNetwork}>Create Network</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Firewall Rule Dialog */}
      <Dialog open={createRuleDialog} onOpenChange={setCreateRuleDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Firewall Rule</DialogTitle>
            <DialogDescription>
              Create a new firewall rule to control traffic
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="rule-name">Rule Name</Label>
              <Input
                id="rule-name"
                value={newRule.name}
                onChange={(e) => setNewRule({ ...newRule, name: e.target.value })}
                placeholder="Allow HTTP Traffic"
              />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Direction</Label>
                <Select
                  value={newRule.direction}
                  onValueChange={(value) => setNewRule({ ...newRule, direction: value as any })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="inbound">Inbound</SelectItem>
                    <SelectItem value="outbound">Outbound</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label>Action</Label>
                <Select
                  value={newRule.action}
                  onValueChange={(value) => setNewRule({ ...newRule, action: value as any })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="allow">Allow</SelectItem>
                    <SelectItem value="deny">Deny</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label>Protocol</Label>
                <Select
                  value={newRule.protocol}
                  onValueChange={(value) => setNewRule({ ...newRule, protocol: value as any })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tcp">TCP</SelectItem>
                    <SelectItem value="udp">UDP</SelectItem>
                    <SelectItem value="icmp">ICMP</SelectItem>
                    <SelectItem value="any">Any</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="port">Port</Label>
                <Input
                  id="port"
                  value={newRule.port}
                  onChange={(e) => setNewRule({ ...newRule, port: e.target.value })}
                  placeholder="80, 443, or any"
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="source">Source</Label>
                <Input
                  id="source"
                  value={newRule.source}
                  onChange={(e) => setNewRule({ ...newRule, source: e.target.value })}
                  placeholder="0.0.0.0/0 or any"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="destination">Destination</Label>
                <Input
                  id="destination"
                  value={newRule.destination}
                  onChange={(e) => setNewRule({ ...newRule, destination: e.target.value })}
                  placeholder="10.0.0.0/16 or any"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="priority">Priority (lower = higher priority)</Label>
              <Input
                id="priority"
                type="number"
                value={newRule.priority}
                onChange={(e) => setNewRule({ ...newRule, priority: parseInt(e.target.value) })}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setCreateRuleDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleCreateRule}>Create Rule</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default NetworkConfigurationPanel;