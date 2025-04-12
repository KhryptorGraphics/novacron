import React, { useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

// Define types for the network data
export interface Node {
  id: string;
  name: string;
  type: 'vm' | 'host' | 'storage' | 'network' | 'service';
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  metrics?: {
    [key: string]: number;
  };
}

export interface Edge {
  source: string;
  target: string;
  type: 'network' | 'storage' | 'dependency';
  metrics?: {
    latency?: number;
    bandwidth?: number;
    packetLoss?: number;
  };
}

export interface NetworkData {
  nodes: Node[];
  edges: Edge[];
}

interface NetworkTopologyProps {
  data: NetworkData;
  title?: string;
  description?: string;
  height?: number;
  onNodeClick?: (node: Node) => void;
  onEdgeClick?: (edge: Edge) => void;
}

export const NetworkTopology: React.FC<NetworkTopologyProps> = ({
  data,
  title = 'Network Topology',
  description,
  height = 400,
  onNodeClick,
  onEdgeClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [layoutType, setLayoutType] = React.useState<string>('force');
  const [highlightedNode, setHighlightedNode] = React.useState<string | null>(null);
  
  // Simple force-directed layout simulation
  useEffect(() => {
    if (!canvasRef.current || !data.nodes.length) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    
    // Node positions and simulation state
    const nodePositions = new Map<string, { x: number; y: number; vx: number; vy: number }>();
    
    // Initialize node positions
    data.nodes.forEach(node => {
      nodePositions.set(node.id, {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        vx: 0,
        vy: 0,
      });
    });
    
    // Node colors based on type
    const nodeColors = {
      vm: '#4f46e5', // Indigo
      host: '#0891b2', // Cyan
      storage: '#7c3aed', // Violet
      network: '#2563eb', // Blue
      service: '#059669', // Emerald
    };
    
    // Node status colors
    const statusColors = {
      healthy: '#10b981', // Green
      warning: '#f59e0b', // Amber
      error: '#ef4444', // Red
      unknown: '#6b7280', // Gray
    };
    
    // Edge colors based on type
    const edgeColors = {
      network: 'rgba(37, 99, 235, 0.5)', // Blue
      storage: 'rgba(124, 58, 237, 0.5)', // Violet
      dependency: 'rgba(156, 163, 175, 0.5)', // Gray
    };
    
    // Function to draw a node
    const drawNode = (node: Node, position: { x: number; y: number }) => {
      const radius = node.type === 'host' ? 15 : 10;
      
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
    };
    
    // Function to draw an edge
    const drawEdge = (edge: Edge, sourcePos: { x: number; y: number }, targetPos: { x: number; y: number }) => {
      ctx.beginPath();
      ctx.moveTo(sourcePos.x, sourcePos.y);
      ctx.lineTo(targetPos.x, targetPos.y);
      ctx.strokeStyle = edgeColors[edge.type];
      ctx.lineWidth = edge.metrics?.bandwidth 
        ? Math.max(1, Math.min(5, edge.metrics.bandwidth / 20)) 
        : 1;
      
      // Draw dashed line for dependency
      if (edge.type === 'dependency') {
        ctx.setLineDash([5, 3]);
      } else {
        ctx.setLineDash([]);
      }
      
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Draw latency if available
      if (edge.metrics?.latency) {
        const midX = (sourcePos.x + targetPos.x) / 2;
        const midY = (sourcePos.y + targetPos.y) / 2;
        
        ctx.font = '9px sans-serif';
        ctx.fillStyle = '#6b7280';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${edge.metrics.latency.toFixed(1)}ms`, midX, midY - 8);
      }
    };
    
    // Force-directed layout simulation
    const runSimulation = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Apply forces
      for (let i = 0; i < 10; i++) { // Run multiple iterations per frame for faster convergence
        // Repulsive forces between nodes
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
        
        // Attractive forces along edges
        data.edges.forEach(edge => {
          const sourcePos = nodePositions.get(edge.source)!;
          const targetPos = nodePositions.get(edge.target)!;
          
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
          pos.x = Math.max(50, Math.min(canvas.width - 50, pos.x));
          pos.y = Math.max(50, Math.min(canvas.height - 50, pos.y));
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
      
      // Continue animation
      requestAnimationFrame(runSimulation);
    };
    
    // Start simulation
    runSimulation();
    
    // Handle canvas click events
    const handleCanvasClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      // Check if a node was clicked
      for (const node of data.nodes) {
        const position = nodePositions.get(node.id);
        if (!position) continue;
        
        const distance = Math.sqrt(
          Math.pow(position.x - x, 2) + Math.pow(position.y - y, 2)
        );
        
        if (distance <= 15) { // Node radius + some margin
          if (onNodeClick) {
            onNodeClick(node);
          }
          return;
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
        
        if (distance <= 5) { // Edge width + some margin
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
    };
  }, [data, layoutType, onNodeClick, onEdgeClick]);
  
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
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
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
        </div>
      </CardContent>
    </Card>
  );
};