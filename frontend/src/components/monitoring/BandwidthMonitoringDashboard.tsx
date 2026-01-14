import React, { useState, useMemo, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  Activity, TrendingUp, TrendingDown, AlertCircle, Network,
  Globe, Wifi, WifiOff, BarChart3, Gauge, Server, Cloud
} from 'lucide-react';
import { PredictiveChart } from '@/components/visualizations/PredictiveChart';
import { useBandwidthMonitoringWebSocket } from '@/hooks/useWebSocket';
import type { BandwidthMetrics, QoSMetrics, NetworkInterface, TrafficFlow } from '@/lib/api/types';

interface BandwidthMonitoringDashboardProps {
  clusterId?: string;
  timeRange?: '5min' | '1hr' | '24hr' | '7days';
  autoRefresh?: boolean;
}

export const BandwidthMonitoringDashboard: React.FC<BandwidthMonitoringDashboardProps> = ({
  clusterId,
  timeRange: initialTimeRange = '1hr',
  autoRefresh = true,
}) => {
  const [timeRange, setTimeRange] = useState(initialTimeRange);
  const [selectedInterface, setSelectedInterface] = useState<string | null>(null);
  const [showAlerts, setShowAlerts] = useState(true);

  // Real-time bandwidth data from WebSocket
  const { data: bandwidthData, isConnected } = useBandwidthMonitoringWebSocket();

  // Mock data for demonstration - would be replaced with real API data
  const mockBandwidthMetrics: BandwidthMetrics = {
    timestamp: new Date().toISOString(),
    totalBandwidth: 10000, // Mbps
    usedBandwidth: 6800, // Mbps
    availableBandwidth: 3200, // Mbps
    utilizationPercent: 68,
    uploadRate: 3200, // Mbps
    downloadRate: 3600, // Mbps
    peakUsage: 8500, // Mbps
    averageUsage: 5200, // Mbps
  };

  const mockQoSMetrics: QoSMetrics = {
    latency: 12.5, // ms
    jitter: 2.3, // ms
    packetLoss: 0.02, // %
    throughput: 6500, // Mbps
    qosCompliance: 98.5, // %
    priorityQueues: {
      high: { usage: 45, latency: 5.2 },
      medium: { usage: 30, latency: 12.8 },
      low: { usage: 25, latency: 22.4 },
    },
  };

  const mockInterfaces: NetworkInterface[] = [
    {
      id: 'eth0',
      name: 'eth0',
      type: 'ethernet',
      status: 'up',
      speed: 10000, // Mbps
      duplex: 'full',
      mtu: 1500,
      utilization: 72,
      errorRate: 0.001,
      packetsSent: 1234567890,
      packetsReceived: 987654321,
      bytesSent: 9876543210000,
      bytesReceived: 8765432100000,
    },
    {
      id: 'eth1',
      name: 'eth1',
      type: 'ethernet',
      status: 'up',
      speed: 10000,
      duplex: 'full',
      mtu: 9000, // Jumbo frames
      utilization: 54,
      errorRate: 0.0005,
      packetsSent: 987654321,
      packetsReceived: 876543210,
      bytesSent: 7654321000000,
      bytesReceived: 6543210000000,
    },
    {
      id: 'wg0',
      name: 'wg0',
      type: 'vpn',
      status: 'up',
      speed: 1000,
      duplex: 'full',
      mtu: 1420,
      utilization: 35,
      errorRate: 0.002,
      packetsSent: 123456789,
      packetsReceived: 98765432,
      bytesSent: 987654321000,
      bytesReceived: 876543210000,
    },
  ];

  const mockTrafficFlows: TrafficFlow[] = [
    {
      id: '1',
      source: 'cluster-us-east',
      destination: 'cluster-us-west',
      protocol: 'TCP',
      port: 443,
      bandwidth: 1200, // Mbps
      packets: 98765432,
      bytes: 1234567890000,
      startTime: new Date(Date.now() - 3600000).toISOString(),
    },
    {
      id: '2',
      source: 'vm-web-001',
      destination: 'vm-db-001',
      protocol: 'TCP',
      port: 3306,
      bandwidth: 450,
      packets: 45678901,
      bytes: 567890123000,
      startTime: new Date(Date.now() - 1800000).toISOString(),
    },
    {
      id: '3',
      source: 'cluster-eu-central',
      destination: 'cluster-us-east',
      protocol: 'UDP',
      port: 4789, // VXLAN
      bandwidth: 890,
      packets: 67890123,
      bytes: 890123456000,
      startTime: new Date(Date.now() - 900000).toISOString(),
    },
  ];

  // Calculate current metrics (would come from WebSocket in production)
  const currentMetrics = useMemo(() => {
    return bandwidthData || mockBandwidthMetrics;
  }, [bandwidthData]);

  // Generate time-series data for charts
  const timeSeriesData = useMemo(() => {
    const points = 100;
    const now = Date.now();
    const interval = {
      '5min': 3000, // 3 seconds
      '1hr': 36000, // 36 seconds
      '24hr': 864000, // 14.4 minutes
      '7days': 6048000, // 1.68 hours
    }[timeRange];

    return Array.from({ length: points }, (_, i) => ({
      timestamp: new Date(now - (points - i) * interval).toISOString(),
      bandwidth: Math.random() * 8000 + 2000,
      upload: Math.random() * 4000 + 1000,
      download: Math.random() * 4000 + 1000,
      utilization: Math.random() * 30 + 50,
    }));
  }, [timeRange]);

  // Format bytes to human-readable format
  const formatBytes = (bytes: number) => {
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let unitIndex = 0;
    let value = bytes;

    while (value >= 1024 && unitIndex < units.length - 1) {
      value /= 1024;
      unitIndex++;
    }

    return `${value.toFixed(2)} ${units[unitIndex]}`;
  };

  // Format bandwidth to human-readable format
  const formatBandwidth = (mbps: number) => {
    if (mbps >= 1000) {
      return `${(mbps / 1000).toFixed(1)} Gbps`;
    }
    return `${mbps.toFixed(0)} Mbps`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold">Bandwidth Monitoring</h2>
          <p className="text-muted-foreground">
            Real-time network bandwidth analysis and QoS monitoring
          </p>
        </div>
        <div className="flex items-center gap-4">
          <Select value={timeRange} onValueChange={(v: any) => setTimeRange(v)}>
            <SelectTrigger className="w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="5min">5 Minutes</SelectItem>
              <SelectItem value="1hr">1 Hour</SelectItem>
              <SelectItem value="24hr">24 Hours</SelectItem>
              <SelectItem value="7days">7 Days</SelectItem>
            </SelectContent>
          </Select>
          {isConnected ? (
            <Badge variant="outline">
              <Wifi className="h-3 w-3 mr-1 text-green-500" />
              Live
            </Badge>
          ) : (
            <Badge variant="outline">
              <WifiOff className="h-3 w-3 mr-1 text-red-500" />
              Offline
            </Badge>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Bandwidth</CardTitle>
            <Network className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatBandwidth(currentMetrics.totalBandwidth)}
            </div>
            <Progress value={currentMetrics.utilizationPercent} className="mt-2" />
            <p className="text-xs text-muted-foreground mt-2">
              {currentMetrics.utilizationPercent}% utilized
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Upload Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatBandwidth(currentMetrics.uploadRate)}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Peak: {formatBandwidth(currentMetrics.peakUsage * 0.45)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Download Rate</CardTitle>
            <TrendingDown className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatBandwidth(currentMetrics.downloadRate)}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              Peak: {formatBandwidth(currentMetrics.peakUsage * 0.55)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Available</CardTitle>
            <Gauge className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatBandwidth(currentMetrics.availableBandwidth)}
            </div>
            <p className="text-xs text-muted-foreground mt-2">
              {((currentMetrics.availableBandwidth / currentMetrics.totalBandwidth) * 100).toFixed(0)}% free
            </p>
          </CardContent>
        </Card>
      </div>

      {/* QoS Metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Quality of Service (QoS)</CardTitle>
          <CardDescription>Network performance and reliability metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <p className="text-sm font-medium">Latency</p>
              <p className="text-2xl font-bold">{mockQoSMetrics.latency.toFixed(1)}ms</p>
              <Badge variant={mockQoSMetrics.latency < 20 ? 'default' : 'destructive'} className="mt-1">
                {mockQoSMetrics.latency < 20 ? 'Good' : 'High'}
              </Badge>
            </div>
            <div>
              <p className="text-sm font-medium">Jitter</p>
              <p className="text-2xl font-bold">{mockQoSMetrics.jitter.toFixed(1)}ms</p>
              <Badge variant={mockQoSMetrics.jitter < 5 ? 'default' : 'secondary'} className="mt-1">
                {mockQoSMetrics.jitter < 5 ? 'Stable' : 'Variable'}
              </Badge>
            </div>
            <div>
              <p className="text-sm font-medium">Packet Loss</p>
              <p className="text-2xl font-bold">{mockQoSMetrics.packetLoss.toFixed(3)}%</p>
              <Badge variant={mockQoSMetrics.packetLoss < 0.1 ? 'default' : 'destructive'} className="mt-1">
                {mockQoSMetrics.packetLoss < 0.1 ? 'Minimal' : 'High'}
              </Badge>
            </div>
            <div>
              <p className="text-sm font-medium">Throughput</p>
              <p className="text-2xl font-bold">{formatBandwidth(mockQoSMetrics.throughput)}</p>
              <Progress value={(mockQoSMetrics.throughput / currentMetrics.totalBandwidth) * 100} className="mt-2" />
            </div>
            <div>
              <p className="text-sm font-medium">QoS Compliance</p>
              <p className="text-2xl font-bold">{mockQoSMetrics.qosCompliance.toFixed(1)}%</p>
              <Badge variant={mockQoSMetrics.qosCompliance > 95 ? 'default' : 'secondary'} className="mt-1">
                {mockQoSMetrics.qosCompliance > 95 ? 'Compliant' : 'Review'}
              </Badge>
            </div>
          </div>

          {/* Priority Queue Status */}
          <div className="mt-6">
            <h4 className="text-sm font-medium mb-3">Priority Queue Distribution</h4>
            <div className="space-y-2">
              {Object.entries(mockQoSMetrics.priorityQueues).map(([priority, stats]) => (
                <div key={priority} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant={priority === 'high' ? 'default' : priority === 'medium' ? 'secondary' : 'outline'}>
                      {priority.toUpperCase()}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      Latency: {stats.latency.toFixed(1)}ms
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm">{stats.usage}%</span>
                    <Progress value={stats.usage} className="w-24" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Bandwidth Trend Charts */}
      <Tabs defaultValue="bandwidth" className="space-y-4">
        <TabsList>
          <TabsTrigger value="bandwidth">Bandwidth Usage</TabsTrigger>
          <TabsTrigger value="interfaces">Network Interfaces</TabsTrigger>
          <TabsTrigger value="traffic">Traffic Analysis</TabsTrigger>
          <TabsTrigger value="crosscluster">Cross-Cluster</TabsTrigger>
        </TabsList>

        <TabsContent value="bandwidth" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Bandwidth Trends</CardTitle>
              <CardDescription>Historical bandwidth usage and predictions</CardDescription>
            </CardHeader>
            <CardContent>
              <PredictiveChart
                data={timeSeriesData.map(d => ({
                  timestamp: d.timestamp,
                  actual: d.bandwidth,
                  predicted: d.bandwidth * (1 + Math.random() * 0.2 - 0.1),
                  confidence: {
                    lower: d.bandwidth * 0.8,
                    upper: d.bandwidth * 1.2,
                  },
                }))}
                title="Network Bandwidth Usage"
                yAxisLabel="Bandwidth (Mbps)"
                showConfidence={true}
                height={300}
              />
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Upload vs Download</CardTitle>
              </CardHeader>
              <CardContent>
                <PredictiveChart
                  data={timeSeriesData.map(d => ({
                    timestamp: d.timestamp,
                    actual: d.upload,
                    predicted: d.download,
                  }))}
                  title=""
                  yAxisLabel="Rate (Mbps)"
                  actualLabel="Upload"
                  predictedLabel="Download"
                  height={200}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Utilization Rate</CardTitle>
              </CardHeader>
              <CardContent>
                <PredictiveChart
                  data={timeSeriesData.map(d => ({
                    timestamp: d.timestamp,
                    actual: d.utilization,
                    predicted: Math.min(100, d.utilization + Math.random() * 10),
                  }))}
                  title=""
                  yAxisLabel="Utilization (%)"
                  height={200}
                />
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="interfaces" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {mockInterfaces.map((iface) => (
              <Card key={iface.id} className={selectedInterface === iface.id ? 'ring-2 ring-primary' : ''}>
                <CardHeader className="cursor-pointer" onClick={() => setSelectedInterface(iface.id)}>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm">{iface.name}</CardTitle>
                    <Badge variant={iface.status === 'up' ? 'default' : 'destructive'}>
                      {iface.status.toUpperCase()}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Speed</span>
                      <span>{formatBandwidth(iface.speed)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Utilization</span>
                      <span>{iface.utilization}%</span>
                    </div>
                    <Progress value={iface.utilization} className="h-2" />
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Error Rate</span>
                      <span className={iface.errorRate > 0.01 ? 'text-red-500' : ''}>
                        {(iface.errorRate * 100).toFixed(3)}%
                      </span>
                    </div>
                    <div className="pt-2 border-t">
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">TX</span>
                        <span>{formatBytes(iface.bytesSent)}</span>
                      </div>
                      <div className="flex justify-between text-xs">
                        <span className="text-muted-foreground">RX</span>
                        <span>{formatBytes(iface.bytesReceived)}</span>
                      </div>
                    </div>
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>Duplex: {iface.duplex}</span>
                      <span>MTU: {iface.mtu}</span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="traffic" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Top Traffic Flows</CardTitle>
              <CardDescription>Real-time traffic analysis by source and destination</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {mockTrafficFlows.map((flow) => (
                  <div key={flow.id} className="flex items-center justify-between p-4 rounded-lg bg-accent/50">
                    <div className="flex items-center gap-4">
                      <div>
                        <p className="text-sm font-medium">{flow.source}</p>
                        <p className="text-xs text-muted-foreground">Source</p>
                      </div>
                      <Activity className="h-4 w-4 text-muted-foreground" />
                      <div>
                        <p className="text-sm font-medium">{flow.destination}</p>
                        <p className="text-xs text-muted-foreground">Destination</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-6">
                      <div className="text-right">
                        <p className="text-sm font-medium">{formatBandwidth(flow.bandwidth)}</p>
                        <p className="text-xs text-muted-foreground">Bandwidth</p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm">{flow.protocol}:{flow.port}</p>
                        <p className="text-xs text-muted-foreground">Protocol</p>
                      </div>
                      <div className="text-right">
                        <p className="text-sm">{formatBytes(flow.bytes)}</p>
                        <p className="text-xs text-muted-foreground">Total Transfer</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Protocol Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm">TCP</span>
                  <div className="flex items-center gap-2">
                    <Progress value={65} className="w-32" />
                    <span className="text-sm w-12 text-right">65%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">UDP</span>
                  <div className="flex items-center gap-2">
                    <Progress value={25} className="w-32" />
                    <span className="text-sm w-12 text-right">25%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">ICMP</span>
                  <div className="flex items-center gap-2">
                    <Progress value={5} className="w-32" />
                    <span className="text-sm w-12 text-right">5%</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm">Other</span>
                  <div className="flex items-center gap-2">
                    <Progress value={5} className="w-32" />
                    <span className="text-sm w-12 text-right">5%</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="crosscluster" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Cross-Cluster Bandwidth</CardTitle>
              <CardDescription>WAN optimization and inter-cluster traffic</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <h4 className="text-sm font-medium mb-3">Federation Links</h4>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between p-2 rounded bg-accent/50">
                      <div className="flex items-center gap-2">
                        <Cloud className="h-4 w-4" />
                        <span className="text-sm">US-East ↔ US-West</span>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">2.4 Gbps</p>
                        <p className="text-xs text-muted-foreground">45ms latency</p>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-2 rounded bg-accent/50">
                      <div className="flex items-center gap-2">
                        <Cloud className="h-4 w-4" />
                        <span className="text-sm">US-East ↔ EU-Central</span>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">1.8 Gbps</p>
                        <p className="text-xs text-muted-foreground">95ms latency</p>
                      </div>
                    </div>
                    <div className="flex items-center justify-between p-2 rounded bg-accent/50">
                      <div className="flex items-center gap-2">
                        <Cloud className="h-4 w-4" />
                        <span className="text-sm">EU-Central ↔ AP-South</span>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">1.2 Gbps</p>
                        <p className="text-xs text-muted-foreground">140ms latency</p>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium mb-3">WAN Optimization</h4>
                  <div className="space-y-3">
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Compression Ratio</span>
                        <span>2.8:1</span>
                      </div>
                      <Progress value={65} />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Deduplication Rate</span>
                        <span>42%</span>
                      </div>
                      <Progress value={42} />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Cache Hit Rate</span>
                        <span>78%</span>
                      </div>
                      <Progress value={78} />
                    </div>
                    <div>
                      <div className="flex justify-between text-sm mb-1">
                        <span>Transfer Efficiency</span>
                        <span>89%</span>
                      </div>
                      <Progress value={89} />
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Bandwidth Alerts */}
      {showAlerts && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Bandwidth Alerts</CardTitle>
              <Button
                size="sm"
                variant="ghost"
                onClick={() => setShowAlerts(false)}
              >
                Dismiss All
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <Alert>
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>High Bandwidth Utilization</AlertTitle>
                <AlertDescription>
                  Interface eth0 is at 92% capacity. Consider load balancing or capacity upgrade.
                </AlertDescription>
              </Alert>
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>QoS Violation Detected</AlertTitle>
                <AlertDescription>
                  High-priority queue experiencing 28ms latency (threshold: 20ms).
                </AlertDescription>
              </Alert>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default BandwidthMonitoringDashboard;