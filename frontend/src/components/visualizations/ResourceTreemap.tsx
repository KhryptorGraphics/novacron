import React, { useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

interface TreemapNode {
  id: string;
  name: string;
  value: number;
  color?: string;
  children?: TreemapNode[];
}

interface ResourceTreemapProps {
  data: TreemapNode;
  title?: string;
  description?: string;
  height?: number;
  colorMode?: 'category' | 'value';
  valueFormat?: (value: number) => string;
  onNodeClick?: (node: TreemapNode) => void;
}

export const ResourceTreemap: React.FC<ResourceTreemapProps> = ({
  data,
  title = 'Resource Utilization',
  description,
  height = 400,
  colorMode = 'value',
  valueFormat = (value) => `${value.toFixed(1)}%`,
  onNodeClick,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedMetric, setSelectedMetric] = React.useState<string>('usage');
  
  // Color scales
  const valueColorScale = [
    '#f7fbff', // Very low - almost white
    '#deebf7', // Low
    '#c6dbef', // Low-medium
    '#9ecae1', // Medium
    '#6baed6', // Medium-high
    '#4292c6', // High
    '#2171b5', // Very high
    '#084594', // Extremely high
  ];
  
  // Category colors
  const categoryColors: Record<string, string> = {
    vm: '#4f46e5', // Indigo
    host: '#0891b2', // Cyan
    storage: '#7c3aed', // Violet
    network: '#2563eb', // Blue
    service: '#059669', // Emerald
    default: '#6b7280', // Gray
  };
  
  // Get color based on value
  const getValueColor = (value: number, min: number, max: number) => {
    const normalizedValue = max > min ? (value - min) / (max - min) : 0.5;
    const colorIndex = Math.min(
      Math.floor(normalizedValue * valueColorScale.length),
      valueColorScale.length - 1
    );
    return valueColorScale[colorIndex];
  };
  
  // Get color based on category
  const getCategoryColor = (node: TreemapNode) => {
    // Try to infer category from node id or name
    const nodeText = (node.id + node.name).toLowerCase();
    
    for (const [category, color] of Object.entries(categoryColors)) {
      if (nodeText.includes(category)) {
        return color;
      }
    }
    
    return categoryColors.default;
  };
  
  // Flatten tree to get min/max values
  const flattenTree = (node: TreemapNode): TreemapNode[] => {
    const result: TreemapNode[] = [node];
    if (node.children) {
      node.children.forEach(child => {
        result.push(...flattenTree(child));
      });
    }
    return result;
  };
  
  // Layout algorithm (simple treemap)
  interface LayoutedNode extends TreemapNode {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
    color: string;
    children?: LayoutedNode[];
  }
  
  const layoutTreemap = (
    node: TreemapNode,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    minValue: number,
    maxValue: number
  ): LayoutedNode => {
    // Base case: leaf node
    if (!node.children || node.children.length === 0) {
      return {
        ...node,
        x0,
        y0,
        x1,
        y1,
        color: node.color || (colorMode === 'value'
          ? getValueColor(node.value, minValue, maxValue)
          : getCategoryColor(node)
        ),
        children: undefined
      } as LayoutedNode;
    }
    
    // Calculate total value of children
    const total = node.children.reduce((sum, child) => sum + child.value, 0);
    
    // Sort children by value (descending)
    const sortedChildren = [...node.children].sort((a, b) => b.value - a.value);
    
    // Layout children
    let currentX = x0;
    let currentY = y0;
    const width = x1 - x0;
    const height = y1 - y0;
    
    // Choose direction (horizontal or vertical) based on aspect ratio
    const isHorizontal = width > height;
    
    const layoutChildren: LayoutedNode[] = sortedChildren.map(child => {
      const childSize = (child.value / total) * (isHorizontal ? width : height);
      let childX0, childY0, childX1, childY1;
      
      if (isHorizontal) {
        childX0 = currentX;
        childY0 = y0;
        childX1 = currentX + childSize;
        childY1 = y1;
        currentX += childSize;
      } else {
        childX0 = x0;
        childY0 = currentY;
        childX1 = x1;
        childY1 = currentY + childSize;
        currentY += childSize;
      }
      
      return layoutTreemap(
        child,
        childX0,
        childY0,
        childX1,
        childY1,
        minValue,
        maxValue
      );
    });
    
    return {
      ...node,
      x0,
      y0,
      x1,
      y1,
      children: layoutChildren,
      color: node.color || (colorMode === 'value' 
        ? getValueColor(node.value, minValue, maxValue)
        : getCategoryColor(node)
      ),
    };
  };
  
  // Draw treemap
  useEffect(() => {
    if (!canvasRef.current || !data) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    
    // Get min/max values for color scaling
    const flatNodes = flattenTree(data);
    const values = flatNodes.map(node => node.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    
    // Layout treemap
    const layoutedTree = layoutTreemap(
      data,
      0,
      0,
      canvas.width,
      canvas.height,
      minValue,
      maxValue
    );
    
    // Function to draw a node
    const drawNode = (node: any, depth = 0) => {
      const padding = 2;
      
      // Draw rectangle
      ctx.fillStyle = node.color;
      ctx.fillRect(
        node.x0 + padding,
        node.y0 + padding,
        node.x1 - node.x0 - padding * 2,
        node.y1 - node.y0 - padding * 2
      );
      
      // Draw border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1;
      ctx.strokeRect(
        node.x0 + padding,
        node.y0 + padding,
        node.x1 - node.x0 - padding * 2,
        node.y1 - node.y0 - padding * 2
      );
      
      // Draw text if node is large enough
      const width = node.x1 - node.x0;
      const height = node.y1 - node.y0;
      
      if (width > 60 && height > 30) {
        const fontSize = Math.min(14, Math.max(10, Math.floor(width / 10)));
        ctx.font = `${fontSize}px sans-serif`;
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        
        // Draw name
        const centerX = (node.x0 + node.x1) / 2;
        const centerY = (node.y0 + node.y1) / 2 - fontSize / 2;
        
        ctx.fillText(
          node.name,
          centerX,
          centerY,
          width - padding * 4
        );
        
        // Draw value
        ctx.font = `${fontSize - 2}px sans-serif`;
        ctx.fillText(
          valueFormat(node.value),
          centerX,
          centerY + fontSize,
          width - padding * 4
        );
      }
      
      // Recursively draw children
      if (node.children) {
        node.children.forEach((child: any) => {
          drawNode(child, depth + 1);
        });
      }
    };
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw treemap
    drawNode(layoutedTree);
    
    // Handle canvas click events
    const handleCanvasClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      // Function to find node at position
      const findNodeAtPosition = (node: any): any => {
        if (
          x >= node.x0 &&
          x <= node.x1 &&
          y >= node.y0 &&
          y <= node.y1
        ) {
          // Check if a child contains the point
          if (node.children) {
            for (const child of node.children) {
              const foundInChild = findNodeAtPosition(child);
              if (foundInChild) {
                return foundInChild;
              }
            }
          }
          
          // No child contains the point, so this node contains it
          return node;
        }
        
        return null;
      };
      
      const clickedNode = findNodeAtPosition(layoutedTree);
      
      if (clickedNode && onNodeClick) {
        // Create a clean version of the node without layout properties
        const cleanNode: TreemapNode = {
          id: clickedNode.id,
          name: clickedNode.name,
          value: clickedNode.value,
          children: clickedNode.children?.map((child: any) => ({
            id: child.id,
            name: child.name,
            value: child.value,
          })),
        };
        
        onNodeClick(cleanNode);
      }
    };
    
    canvas.addEventListener('click', handleCanvasClick);
    
    return () => {
      canvas.removeEventListener('click', handleCanvasClick);
    };
  }, [data, colorMode, valueFormat, onNodeClick]);
  
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div>
          <CardTitle>{title}</CardTitle>
          {description && <p className="text-sm text-muted-foreground">{description}</p>}
        </div>
        <Select value={selectedMetric} onValueChange={setSelectedMetric}>
          <SelectTrigger className="w-[180px]">
            <SelectValue placeholder="Metric" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="usage">Usage %</SelectItem>
            <SelectItem value="iops">IOPS</SelectItem>
            <SelectItem value="latency">Latency</SelectItem>
          </SelectContent>
        </Select>
      </CardHeader>
      <CardContent>
        <div className="relative" style={{ height: `${height}px` }}>
          <canvas
            ref={canvasRef}
            className="w-full h-full"
          />
        </div>
        <div className="flex justify-between mt-4">
          <div className="flex items-center">
            <div className="w-full h-2 bg-gradient-to-r from-[#f7fbff] to-[#084594] rounded mr-2" style={{ width: '100px' }}></div>
            <span className="text-xs">Resource Utilization</span>
          </div>
          <div className="text-xs text-muted-foreground">
            Click on a section to view details
          </div>
        </div>
      </CardContent>
    </Card>
  );
};