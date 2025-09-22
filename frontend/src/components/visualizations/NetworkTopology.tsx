import React, { useEffect, useRef, useCallback, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { useDistributedTopologyWebSocket } from '@/hooks/useWebSocket';
import { Activity, Globe, Server, Cloud, Network as NetworkIcon } from 'lucide-react';

// Enhanced types for distributed network data
export interface Node {
  id: string;
  name: string;
  type: 'vm' | 'host' | 'storage' | 'network' | 'service' | 'cluster' | 'federation';
  status: 'healthy' | 'warning' | 'error' | 'unknown' | 'migrating';
  clusterId?: string;
  region?: string;
  metrics?: {
    cpuUsage?: number;
    memoryUsage?: number;
    diskUsage?: number;
    networkIn?: number;
    networkOut?: number;
    [key: string]: number | undefined;
  };
  position?: { x: number; y: number; fixed?: boolean };
}

export interface Edge {
  source: string;
  target: string;
  type: 'network' | 'storage' | 'dependency' | 'cluster' | 'federation' | 'migration';
  metrics?: {
    latency?: number;
    bandwidth?: number;
    packetLoss?: number;
    utilization?: number;
    capacity?: number;
    qos?: 'high' | 'medium' | 'low';
  };
  animated?: boolean;
  bidirectional?: boolean;
}

export interface ClusterInfo {
  id: string;
  name: string;
  region: string;
  nodeCount: number;
  status: 'healthy' | 'degraded' | 'error';
  bounds?: { x: number; y: number; width: number; height: number };
}

export interface NetworkData {
  nodes: Node[];
  edges: Edge[];
  clusters?: ClusterInfo[];
}

interface NetworkTopologyProps {
  data?: NetworkData;
  title?: string;
  description?: string;
  height?: number;
  showDistributed?: boolean;
  showBandwidth?: boolean;
  showPerformanceMetrics?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
  onNodeClick?: (node: Node) => void;
  onEdgeClick?: (edge: Edge) => void;
  onClusterClick?: (cluster: ClusterInfo) => void;
}

export const NetworkTopology: React.FC<NetworkTopologyProps> = ({
  data: initialData,
  title = 'Network Topology',
  description,
  height = 400,
  showDistributed = false,
  showBandwidth = false,
  showPerformanceMetrics = false,
  autoRefresh = true,
  refreshInterval = 5000,
  onNodeClick,
  onEdgeClick,
  onClusterClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationFrameRef = useRef<ReturnType<typeof requestAnimationFrame>>();
  const [layoutType, setLayoutType] = React.useState<string>('force');
  const [highlightedNode, setHighlightedNode] = React.useState<string | null>(null);
  const [selectedCluster, setSelectedCluster] = React.useState<string | null>(null);
  const [showLabels, setShowLabels] = React.useState(true);
  const [detailLevel, setDetailLevel] = React.useState<'low' | 'medium' | 'high'>('medium');
  const [expandedClusters, setExpandedClusters] = React.useState<Set<string>>(new Set());

  // Convert props to internal state for toggling
  const [isDistributed, setIsDistributed] = React.useState(showDistributed);
  const [isBandwidth, setIsBandwidth] = React.useState(showBandwidth);
  const [isMetrics, setIsMetrics] = React.useState(showPerformanceMetrics);

  // Sync when props change
  useEffect(() => setIsDistributed(showDistributed), [showDistributed]);
  useEffect(() => setIsBandwidth(showBandwidth), [showBandwidth]);
  useEffect(() => setIsMetrics(showPerformanceMetrics), [showPerformanceMetrics]);

  // Use WebSocket for real-time updates when distributed mode is enabled
  const { data: wsData, isConnected } = isDistributed ?
    useDistributedTopologyWebSocket() :
    { data: null, isConnected: false };

  // Merge WebSocket data with initial data
  const data = React.useMemo(() => {
    if (wsData && isDistributed) {
      return wsData as NetworkData;
    }
    return initialData || { nodes: [], edges: [], clusters: [] };
  }, [wsData, initialData, isDistributed]);
  
  // Simple force-directed layout simulation
  useEffect(() => {
    if (!canvasRef.current || !data.nodes.length) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions with device pixel ratio
    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(canvas.clientWidth * dpr);
      canvas.height = Math.floor(canvas.clientHeight * dpr);
      canvas.style.width = canvas.clientWidth + 'px';
      canvas.style.height = canvas.clientHeight + 'px';
      ctx.scale(dpr, dpr);
    };

    window.addEventListener('resize', resize);
    resize();
    
    // Node positions and simulation state
    const nodePositions = new Map<string, { x: number; y: number; vx: number; vy: number }>();

    // Initialize node positions based on layout type
    const initializeLayout = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      data.nodes.forEach((node, index) => {
        let x, y;

        switch (layoutType) {
          case 'circular':
            const angle = (index / data.nodes.length) * 2 * Math.PI;
            const radius = Math.min(width, height) * 0.3;
            x = width / 2 + Math.cos(angle) * radius;
            y = height / 2 + Math.sin(angle) * radius;
            break;
          case 'hierarchical':
            const layers = new Map<string, number>();
            let layer = 0;
            if (node.type === 'cluster') layer = 0;
            else if (node.type === 'host') layer = 1;
            else if (node.type === 'vm') layer = 2;
            else layer = 3;

            const nodesInLayer = data.nodes.filter(n => {
              let nLayer = 0;
              if (n.type === 'cluster') nLayer = 0;
              else if (n.type === 'host') nLayer = 1;
              else if (n.type === 'vm') nLayer = 2;
              else nLayer = 3;
              return nLayer === layer;
            });
            const layerIndex = nodesInLayer.findIndex(n => n.id === node.id);

            x = (width / (nodesInLayer.length + 1)) * (layerIndex + 1);
            y = (height / 5) * (layer + 1);
            break;
          case 'geographic':
            // Use region data if available
            const regionCoords = {
              'us-east-1': { x: width * 0.8, y: height * 0.3 },
              'us-west-2': { x: width * 0.2, y: height * 0.4 },
              'eu-west-1': { x: width * 0.6, y: height * 0.2 },
              'ap-southeast-1': { x: width * 0.9, y: height * 0.7 },
            };
            const coords = regionCoords[node.region as keyof typeof regionCoords];
            x = coords?.x || Math.random() * width;
            y = coords?.y || Math.random() * height;
            break;
          case 'clustered':
            if (node.clusterId && data.clusters) {
              const cluster = data.clusters.find(c => c.id === node.clusterId);
              if (cluster?.bounds) {
                x = cluster.bounds.x + Math.random() * cluster.bounds.width;
                y = cluster.bounds.y + Math.random() * cluster.bounds.height;
              } else {
                x = Math.random() * width;
                y = Math.random() * height;
              }
            } else {
              x = Math.random() * width;
              y = Math.random() * height;
            }
            break;
          default: // force
            x = Math.random() * width;
            y = Math.random() * height;
        }

        nodePositions.set(node.id, { x, y, vx: 0, vy: 0 });
      });
    };

    initializeLayout();
    
    // Enhanced node colors for distributed types
    const nodeColors = {
      vm: '#4f46e5', // Indigo
      host: '#0891b2', // Cyan
      storage: '#7c3aed', // Violet
      network: '#2563eb', // Blue
      service: '#059669', // Emerald
      cluster: '#f59e0b', // Amber
      federation: '#ec4899', // Pink
    };
    
    // Node status colors
    const statusColors = {
      healthy: '#10b981', // Green
      warning: '#f59e0b', // Amber
      error: '#ef4444', // Red
      unknown: '#6b7280', // Gray
    };
    
    // Enhanced edge colors with bandwidth visualization
    const edgeColors = {
      network: 'rgba(37, 99, 235, 0.5)', // Blue
      storage: 'rgba(124, 58, 237, 0.5)', // Violet
      dependency: 'rgba(156, 163, 175, 0.5)', // Gray
      cluster: 'rgba(245, 158, 11, 0.5)', // Amber
      federation: 'rgba(236, 72, 153, 0.5)', // Pink
      migration: 'rgba(34, 197, 94, 0.5)', // Green
    };

    // Get edge color based on bandwidth utilization
    const getBandwidthColor = (utilization?: number) => {
      if (!utilization) return edgeColors.network;
      if (utilization > 80) return 'rgba(239, 68, 68, 0.7)'; // Red for high
      if (utilization > 60) return 'rgba(245, 158, 11, 0.7)'; // Amber for medium
      return 'rgba(34, 197, 94, 0.7)'; // Green for low
    };

    // Get edge width based on bandwidth
    const getEdgeWidth = (edge: Edge) => {
      if (!isBandwidth || !edge.metrics?.bandwidth) return 1;
      const width = Math.max(1, Math.min(8, edge.metrics.bandwidth / 10));
      return edge.metrics.utilization ? width * (edge.metrics.utilization / 100) : width;
    };
    
    // Draw cluster boundaries for distributed visualization
    const drawCluster = (cluster: ClusterInfo) => {
      if (!cluster.bounds || !isDistributed) return;

      const { x, y, width, height } = cluster.bounds;

      // Draw cluster boundary
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.3)';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(x, y, width, height);
      ctx.setLineDash([]);

      // Draw cluster label
      ctx.font = '12px sans-serif';
      ctx.fillStyle = '#f59e0b';
      ctx.textAlign = 'left';
      ctx.fillText(`${cluster.name} (${cluster.region})`, x + 5, y - 5);

      // Draw cluster status indicator
      const statusColor = {
        healthy: '#10b981',
        degraded: '#f59e0b',
        error: '#ef4444',
      }[cluster.status];

      ctx.fillStyle = statusColor;
      ctx.beginPath();
      ctx.arc(x + width - 10, y + 10, 5, 0, Math.PI * 2);
      ctx.fill();
    };

    // Enhanced node drawing with performance metrics
    const drawNode = (node: Node, position: { x: number; y: number }) => {
      const radius = node.type === 'host' || node.type === 'cluster' ? 15 : 10;
      
      // Draw node circle
      ctx.beginPath();
      ctx.arc(position.x, position.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = nodeColors[node.type];
      ctx.fill();
      
      // Draw status ring
      ctx.beginPath();
      ctx.arc(position.x, position.y, radius + 3, 0, Math.PI * 2);
      ctx.strokeStyle = statusColors[node.status];
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw node label
      if (showLabels) {
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.name.substring(0, 2), position.x, position.y);

        // Draw node name below
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#374151';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(node.name, position.x, position.y + radius + 5);
      }

      // Draw performance metrics if enabled
      if (isMetrics && node.metrics && detailLevel !== 'low') {
        ctx.font = '9px sans-serif';
        ctx.fillStyle = '#6b7280';
        ctx.textAlign = 'center';

        if (node.metrics.cpuUsage !== undefined) {
          ctx.fillText(`CPU: ${node.metrics.cpuUsage}%`, position.x, position.y + radius + 18);
        }
        if (node.metrics.memoryUsage !== undefined && detailLevel === 'high') {
          ctx.fillText(`Mem: ${node.metrics.memoryUsage}%`, position.x, position.y + radius + 28);
        }
      }

      // Highlight if node is migrating
      if (node.status === 'migrating') {
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.8)';
        ctx.lineWidth = 3;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.arc(position.x, position.y, radius + 6, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    };
    
    // Enhanced edge drawing with bandwidth and QoS visualization
    const drawEdge = (edge: Edge, sourcePos: { x: number; y: number }, targetPos: { x: number; y: number }) => {
      // Calculate edge color based on bandwidth utilization
      const edgeColor = isBandwidth && edge.metrics?.utilization
        ? getBandwidthColor(edge.metrics.utilization)
        : edgeColors[edge.type];

      ctx.beginPath();
      ctx.moveTo(sourcePos.x, sourcePos.y);
      ctx.lineTo(targetPos.x, targetPos.y);
      ctx.strokeStyle = edgeColor;
      ctx.lineWidth = getEdgeWidth(edge);
      
      // Draw dashed line for dependency
      if (edge.type === 'dependency') {
        ctx.setLineDash([5, 3]);
      } else {
        ctx.setLineDash([]);
      }
      
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Draw edge metrics if available
      if (isMetrics && edge.metrics && detailLevel !== 'low') {
        const midX = (sourcePos.x + targetPos.x) / 2;
        const midY = (sourcePos.y + targetPos.y) / 2;

        ctx.font = '9px sans-serif';
        ctx.fillStyle = '#6b7280';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        const metrics = [];
        if (edge.metrics.latency) metrics.push(`${edge.metrics.latency.toFixed(1)}ms`);
        if (isBandwidth && edge.metrics.bandwidth) {
          metrics.push(`${edge.metrics.bandwidth}Mbps`);
        }
        if (edge.metrics.utilization && detailLevel === 'high') {
          metrics.push(`${edge.metrics.utilization}%`);
        }

        if (metrics.length > 0) {
          ctx.fillText(metrics.join(' | '), midX, midY - 8);
        }

        // Draw QoS indicator
        if (edge.metrics.qos) {
          const qosColor = {
            high: '#10b981',
            medium: '#f59e0b',
            low: '#ef4444',
          }[edge.metrics.qos];

          ctx.fillStyle = qosColor;
          ctx.beginPath();
          ctx.arc(midX, midY + 8, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // Animate edge if it's a migration path
      if (edge.animated || edge.type === 'migration') {
        const gradient = ctx.createLinearGradient(
          sourcePos.x, sourcePos.y, targetPos.x, targetPos.y
        );
        gradient.addColorStop(0, 'rgba(34, 197, 94, 0)');
        gradient.addColorStop(0.5, 'rgba(34, 197, 94, 0.8)');
        gradient.addColorStop(1, 'rgba(34, 197, 94, 0)');

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    };
    
    // Simplified Barnes-Hut quadtree for force approximation
    class QuadTree {
      constructor(x: number, y: number, width: number, height: number) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.nodes = [];
        this.centerOfMass = { x: 0, y: 0 };
        this.totalMass = 0;
        this.children = [];
      }

      x: number;
      y: number;
      width: number;
      height: number;
      nodes: any[];
      centerOfMass: { x: number; y: number };
      totalMass: number;
      children: QuadTree[];

      insert(node: any, position: any) {
        if (!this.contains(position.x, position.y)) return false;

        if (this.nodes.length === 0 && this.children.length === 0) {
          this.nodes.push({ node, position });
          this.updateCenterOfMass();
          return true;
        }

        if (this.children.length === 0) {
          this.subdivide();

          // Move existing nodes to children
          for (const existingNode of this.nodes) {
            for (const child of this.children) {
              if (child.insert(existingNode.node, existingNode.position)) {
                break;
              }
            }
          }
          this.nodes = [];
        }

        // Try to insert in children
        for (const child of this.children) {
          if (child.insert(node, position)) {
            this.updateCenterOfMass();
            return true;
          }
        }

        return false;
      }

      contains(x: number, y: number): boolean {
        return x >= this.x && x < this.x + this.width &&
               y >= this.y && y < this.y + this.height;
      }

      subdivide() {
        const halfWidth = this.width / 2;
        const halfHeight = this.height / 2;

        this.children.push(
          new QuadTree(this.x, this.y, halfWidth, halfHeight),
          new QuadTree(this.x + halfWidth, this.y, halfWidth, halfHeight),
          new QuadTree(this.x, this.y + halfHeight, halfWidth, halfHeight),
          new QuadTree(this.x + halfWidth, this.y + halfHeight, halfWidth, halfHeight)
        );
      }

      updateCenterOfMass() {
        let totalX = 0;
        let totalY = 0;
        let mass = 0;

        for (const { position } of this.nodes) {
          totalX += position.x;
          totalY += position.y;
          mass += 1;
        }

        for (const child of this.children) {
          totalX += child.centerOfMass.x * child.totalMass;
          totalY += child.centerOfMass.y * child.totalMass;
          mass += child.totalMass;
        }

        if (mass > 0) {
          this.centerOfMass.x = totalX / mass;
          this.centerOfMass.y = totalY / mass;
        }
        this.totalMass = mass;
      }

      calculateForce(position: any, theta = 0.5): { fx: number; fy: number } {
        if (this.totalMass === 0) return { fx: 0, fy: 0 };

        const dx = this.centerOfMass.x - position.x;
        const dy = this.centerOfMass.y - position.y;
        const distance = Math.sqrt(dx * dx + dy * dy);

        if (distance === 0) return { fx: 0, fy: 0 };

        // If this is a leaf node or the node is far enough, use center of mass
        const s = Math.max(this.width, this.height);
        if (this.children.length === 0 || s / distance < theta) {
          const force = 1000 / (distance * distance);
          const fx = (dx / distance) * force;
          const fy = (dy / distance) * force;
          return { fx, fy };
        }

        // Otherwise, recurse on children
        let fx = 0, fy = 0;
        for (const child of this.children) {
          const childForce = child.calculateForce(position, theta);
          fx += childForce.fx;
          fy += childForce.fy;
        }

        return { fx, fy };
      }
    }

    // Compute static positions for non-force layouts
    const computeCircularPositions = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const radius = Math.min(width, height) * 0.3;
      const result = new Map<string, { x: number; y: number; vx: number; vy: number }>();

      data.nodes.forEach((node, index) => {
        const angle = (index / data.nodes.length) * 2 * Math.PI;
        const x = width / 2 + Math.cos(angle) * radius;
        const y = height / 2 + Math.sin(angle) * radius;
        result.set(node.id, { x, y, vx: 0, vy: 0 });
      });

      return result;
    };

    const computeHierarchicalPositions = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const result = new Map<string, { x: number; y: number; vx: number; vy: number }>();

      // Group nodes by type for hierarchical layout
      const layers = new Map<number, Node[]>();
      data.nodes.forEach(node => {
        let layer = 0;
        if (node.type === 'cluster') layer = 0;
        else if (node.type === 'host') layer = 1;
        else if (node.type === 'vm') layer = 2;
        else layer = 3;

        if (!layers.has(layer)) layers.set(layer, []);
        layers.get(layer)!.push(node);
      });

      Array.from(layers.entries()).forEach(([layer, nodes]) => {
        const y = (height / 5) * (layer + 1);
        nodes.forEach((node, index) => {
          const x = (width / (nodes.length + 1)) * (index + 1);
          result.set(node.id, { x, y, vx: 0, vy: 0 });
        });
      });

      return result;
    };

    const computeGeographicPositions = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const result = new Map<string, { x: number; y: number; vx: number; vy: number }>();

      const regionCoords = {
        'us-east-1': { x: width * 0.8, y: height * 0.3 },
        'us-west-2': { x: width * 0.2, y: height * 0.4 },
        'eu-west-1': { x: width * 0.6, y: height * 0.2 },
        'ap-southeast-1': { x: width * 0.9, y: height * 0.7 },
      };

      data.nodes.forEach(node => {
        const coords = regionCoords[node.region as keyof typeof regionCoords];
        const x = coords?.x || Math.random() * width;
        const y = coords?.y || Math.random() * height;
        result.set(node.id, { x, y, vx: 0, vy: 0 });
      });

      return result;
    };

    const computeClusteredPositions = () => {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      const result = new Map<string, { x: number; y: number; vx: number; vy: number }>();

      data.nodes.forEach(node => {
        let x, y;
        if (node.clusterId && data.clusters) {
          const cluster = data.clusters.find(c => c.id === node.clusterId);
          if (cluster?.bounds) {
            x = cluster.bounds.x + Math.random() * cluster.bounds.width;
            y = cluster.bounds.y + Math.random() * cluster.bounds.height;
          } else {
            x = Math.random() * width;
            y = Math.random() * height;
          }
        } else {
          x = Math.random() * width;
          y = Math.random() * height;
        }
        result.set(node.id, { x, y, vx: 0, vy: 0 });
      });

      return result;
    };

    // Render function for static layouts
    const renderStatic = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw clusters first (background layer)
      if (data.clusters && isDistributed) {
        // Calculate cluster bounds based on contained nodes
        data.clusters.forEach(cluster => {
          const clusterNodes = data.nodes.filter(node => node.clusterId === cluster.id);
          if (clusterNodes.length > 0) {
            const positions = clusterNodes.map(node => nodePositions.get(node.id)!).filter(Boolean);
            if (positions.length > 0) {
              const minX = Math.min(...positions.map(p => p.x)) - 50;
              const maxX = Math.max(...positions.map(p => p.x)) + 50;
              const minY = Math.min(...positions.map(p => p.y)) - 50;
              const maxY = Math.max(...positions.map(p => p.y)) + 50;
              cluster.bounds = {
                x: minX,
                y: minY,
                width: maxX - minX,
                height: maxY - minY
              };
            }
          }
          drawCluster(cluster);
        });
      }

      // Draw edges
      data.edges.forEach(edge => {
        const sourcePos = nodePositions.get(edge.source);
        const targetPos = nodePositions.get(edge.target);

        if (sourcePos && targetPos) {
          drawEdge(edge, sourcePos, targetPos);
        }
      });

      // Draw nodes
      data.nodes.forEach(node => {
        const position = nodePositions.get(node.id);
        if (position) {
          drawNode(node, position);
        }
      });
    };

    // Force-directed layout simulation with quadtree optimization
    const runForceSimulation = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw clusters first (background layer)
      if (data.clusters && isDistributed) {
        // Calculate cluster bounds based on contained nodes
        data.clusters.forEach(cluster => {
          const clusterNodes = data.nodes.filter(node => node.clusterId === cluster.id);
          if (clusterNodes.length > 0) {
            const positions = clusterNodes.map(node => nodePositions.get(node.id)!).filter(Boolean);
            if (positions.length > 0) {
              const minX = Math.min(...positions.map(p => p.x)) - 50;
              const maxX = Math.max(...positions.map(p => p.x)) + 50;
              const minY = Math.min(...positions.map(p => p.y)) - 50;
              const maxY = Math.max(...positions.map(p => p.y)) + 50;
              cluster.bounds = {
                x: minX,
                y: minY,
                width: maxX - minX,
                height: maxY - minY
              };
            }
          }
          drawCluster(cluster);
        });
      }

      // Apply forces - use quadtree for large networks (>100 nodes)
      const iterations = data.nodes.length > 100 ? 3 : 10;
      const useQuadTree = data.nodes.length > 100;

      for (let i = 0; i < iterations; i++) {
        if (useQuadTree) {
          // Build quadtree for this iteration
          const quadTree = new QuadTree(0, 0, canvas.clientWidth, canvas.clientHeight);

          // Insert all nodes
          data.nodes.forEach(node => {
            const position = nodePositions.get(node.id);
            if (position) {
              quadTree.insert(node, position);
            }
          });

          // Apply repulsive forces using quadtree
          data.nodes.forEach(node => {
            const position = nodePositions.get(node.id);
            if (position) {
              const force = quadTree.calculateForce(position);
              position.vx -= force.fx;
              position.vy -= force.fy;
            }
          });
        } else {
          // Use direct O(n²) calculation for smaller networks
          data.nodes.forEach(node1 => {
            data.nodes.forEach(node2 => {
              if (node1.id === node2.id) return;

              const pos1 = nodePositions.get(node1.id)!;
              const pos2 = nodePositions.get(node2.id)!;

              const dx = pos2.x - pos1.x;
              const dy = pos2.y - pos1.y;
              const distance = Math.sqrt(dx * dx + dy * dy);

              if (distance === 0) return;

              // Repulsive force (inverse square law)
              const force = 1000 / (distance * distance);
              const forceX = (dx / distance) * force;
              const forceY = (dy / distance) * force;

              pos1.vx -= forceX;
              pos1.vy -= forceY;
              pos2.vx += forceX;
              pos2.vy += forceY;
            });
          });
        }

        // Attractive forces along edges (always direct calculation)
        data.edges.forEach(edge => {
          const sourcePos = nodePositions.get(edge.source)!;
          const targetPos = nodePositions.get(edge.target)!;

          if (!sourcePos || !targetPos) return;

          const dx = targetPos.x - sourcePos.x;
          const dy = targetPos.y - sourcePos.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance === 0) return;

          // Attractive force (spring-like)
          const force = 0.01 * (distance - 100);
          const forceX = (dx / distance) * force;
          const forceY = (dy / distance) * force;

          sourcePos.vx += forceX;
          sourcePos.vy += forceY;
          targetPos.vx -= forceX;
          targetPos.vy -= forceY;
        });

        // Update positions with velocity and damping
        nodePositions.forEach((pos) => {
          pos.x += pos.vx * 0.1;
          pos.y += pos.vy * 0.1;
          pos.vx *= 0.9; // Damping
          pos.vy *= 0.9; // Damping

          // Constrain to canvas
          pos.x = Math.max(50, Math.min(canvas.clientWidth - 50, pos.x));
          pos.y = Math.max(50, Math.min(canvas.clientHeight - 50, pos.y));
        });
      }

      // Draw edges
      data.edges.forEach(edge => {
        const sourcePos = nodePositions.get(edge.source);
        const targetPos = nodePositions.get(edge.target);

        if (sourcePos && targetPos) {
          drawEdge(edge, sourcePos, targetPos);
        }
      });

      // Draw nodes
      data.nodes.forEach(node => {
        const position = nodePositions.get(node.id);
        if (position) {
          drawNode(node, position);
        }
      });

      // Continue animation for force layout only
      if (layoutType === 'force') {
        animationFrameRef.current = requestAnimationFrame(runForceSimulation);
      }
    };
    
    // Start appropriate rendering based on layout type
    const startRendering = () => {
      // Cancel any existing animation frame
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }

      if (layoutType === 'force') {
        // Use force simulation with continuous animation
        runForceSimulation();
      } else {
        // Compute static positions based on layout type
        let staticPositions: Map<string, { x: number; y: number; vx: number; vy: number }>;

        switch (layoutType) {
          case 'circular':
            staticPositions = computeCircularPositions();
            break;
          case 'hierarchical':
            staticPositions = computeHierarchicalPositions();
            break;
          case 'geographic':
            staticPositions = computeGeographicPositions();
            break;
          case 'clustered':
            staticPositions = computeClusteredPositions();
            break;
          default:
            staticPositions = new Map();
        }

        // Update node positions with computed static positions
        staticPositions.forEach((position, nodeId) => {
          nodePositions.set(nodeId, position);
        });

        // Render once with static positions
        renderStatic();
      }
    };

    startRendering();

    // Auto-refresh data if enabled
    let refreshTimer: ReturnType<typeof setInterval> | null = null;
    if (autoRefresh && refreshInterval > 0) {
      refreshTimer = setInterval(() => {
        // Trigger data refresh (would be handled by WebSocket or API call)
        console.log('Refreshing network topology data...');
      }, refreshInterval);
    }
    
    // Handle canvas click events
    const handleCanvasClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      // Check if a node was clicked
      for (const node of data.nodes) {
        const position = nodePositions.get(node.id);
        if (!position) continue;

        const radius = node.type === 'host' || node.type === 'cluster' ? 15 : 10;
        const distance = Math.sqrt(
          Math.pow(position.x - x, 2) + Math.pow(position.y - y, 2)
        );

        if (distance <= radius + 3) { // Use actual render radius + status ring
          if (onNodeClick) {
            onNodeClick(node);
          }
          return;
        }
      }

      // Check if a cluster was clicked
      if (data.clusters && isDistributed) {
        for (const cluster of data.clusters) {
          if (cluster.bounds) {
            const { x: cx, y: cy, width, height } = cluster.bounds;
            if (x >= cx && x <= cx + width && y >= cy && y <= cy + height) {
              if (onClusterClick) {
                onClusterClick(cluster);
              }
              // Toggle cluster expansion
              const expanded = expandedClusters.has(cluster.id);
              const newExpanded = new Set(expandedClusters);
              if (expanded) {
                newExpanded.delete(cluster.id);
              } else {
                newExpanded.add(cluster.id);
              }
              setExpandedClusters(newExpanded);
              return;
            }
          }
        }
      }
      
      // Check if an edge was clicked
      for (const edge of data.edges) {
        const sourcePos = nodePositions.get(edge.source);
        const targetPos = nodePositions.get(edge.target);
        
        if (!sourcePos || !targetPos) continue;
        
        // Calculate distance from point to line segment
        const A = x - sourcePos.x;
        const B = y - sourcePos.y;
        const C = targetPos.x - sourcePos.x;
        const D = targetPos.y - sourcePos.y;
        
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        let param = -1;
        
        if (lenSq !== 0) {
          param = dot / lenSq;
        }
        
        let xx, yy;
        
        if (param < 0) {
          xx = sourcePos.x;
          yy = sourcePos.y;
        } else if (param > 1) {
          xx = targetPos.x;
          yy = targetPos.y;
        } else {
          xx = sourcePos.x + param * C;
          yy = sourcePos.y + param * D;
        }
        
        const dx = x - xx;
        const dy = y - yy;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        const edgeWidth = getEdgeWidth(edge);
        if (distance <= edgeWidth + 2) { // Use actual edge width + margin
          if (onEdgeClick) {
            onEdgeClick(edge);
          }
          return;
        }
      }
    };
    
    canvas.addEventListener('click', handleCanvasClick);
    
    return () => {
      canvas.removeEventListener('click', handleCanvasClick);
      window.removeEventListener('resize', resize);
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (refreshTimer) {
        clearInterval(refreshTimer);
      }
    };
  }, [data, layoutType, isDistributed, isBandwidth, isMetrics,
      showLabels, detailLevel, autoRefresh, refreshInterval, onNodeClick, onEdgeClick, onClusterClick, expandedClusters]);
  
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div>
          <CardTitle>{title}</CardTitle>
          {description && <p className="text-sm text-muted-foreground">{description}</p>}
        </div>
        <Select value={layoutType} onValueChange={setLayoutType}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Layout Type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="force">Force-Directed</SelectItem>
            <SelectItem value="circular">Circular</SelectItem>
            <SelectItem value="hierarchical">Hierarchical</SelectItem>
            <SelectItem value="geographic">Geographic</SelectItem>
            <SelectItem value="clustered">Clustered</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        {/* Control Panel */}
        <div className="flex flex-wrap gap-2 mb-4">
          <Button
            size="sm"
            variant={isDistributed ? 'default' : 'outline'}
            onClick={() => setIsDistributed(!isDistributed)}
          >
            <Globe className="h-4 w-4 mr-1" />
            Distributed View
          </Button>
          <Button
            size="sm"
            variant={isBandwidth ? 'default' : 'outline'}
            onClick={() => setIsBandwidth(!isBandwidth)}
          >
            <Activity className="h-4 w-4 mr-1" />
            Bandwidth
          </Button>
          <Button
            size="sm"
            variant={isMetrics ? 'default' : 'outline'}
            onClick={() => setIsMetrics(!isMetrics)}
          >
            <Server className="h-4 w-4 mr-1" />
            Metrics
          </Button>
          <Select value={detailLevel} onValueChange={(v) => setDetailLevel(v as any)}>
            <SelectTrigger className="w-[120px] h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="low">Low Detail</SelectItem>
              <SelectItem value="medium">Medium Detail</SelectItem>
              <SelectItem value="high">High Detail</SelectItem>
            </SelectContent>
          </Select>
          {isDistributed && isConnected && (
            <Badge variant="outline" className="ml-auto">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse" />
              Live Updates
            </Badge>
          )}
        </div>
        <div className="relative" style={{ height: `${height}px` }}>
          <canvas
            ref={canvasRef}
            className="w-full h-full"
            style={{ touchAction: 'none' }}
          />
        </div>
        <div className="flex justify-center mt-4 space-x-4">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-[#4f46e5] mr-2"></div>
            <span className="text-xs">VM</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-[#0891b2] mr-2"></div>
            <span className="text-xs">Host</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-[#7c3aed] mr-2"></div>
            <span className="text-xs">Storage</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-[#2563eb] mr-2"></div>
            <span className="text-xs">Network</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-[#059669] mr-2"></div>
            <span className="text-xs">Service</span>
          </div>
          {isDistributed && (
            <>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-[#f59e0b] mr-2"></div>
                <span className="text-xs">Cluster</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 rounded-full bg-[#ec4899] mr-2"></div>
                <span className="text-xs">Federation</span>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  );
};