import React, { useEffect, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { format } from 'date-fns';
import { AlertCircle, AlertTriangle, ArrowRight, ChevronDown, ChevronRight, Clock, Info, Search, X } from 'lucide-react';

export interface Alert {
  id: string;
  name: string;
  description: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  status: 'firing' | 'resolved' | 'acknowledged';
  startTime: string;
  endTime?: string;
  labels: Record<string, string>;
  value: number;
  resource: string;
}

export interface AlertRelation {
  source: string; // Alert ID
  target: string; // Alert ID
  type: 'causes' | 'correlates' | 'follows';
  confidence: number; // 0-1
}

interface AlertCorrelationProps {
  alerts: Alert[];
  relations: AlertRelation[];
  onAlertClick?: (alert: Alert) => void;
  onAcknowledge?: (id: string) => void;
}

export const AlertCorrelation: React.FC<AlertCorrelationProps> = ({
  alerts,
  relations,
  onAlertClick,
  onAcknowledge,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedAlert, setSelectedAlert] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState<string>('');
  const [expandedGroups, setExpandedGroups] = useState<Record<string, boolean>>({});
  
  // Group alerts by resource
  const alertsByResource = alerts.reduce<Record<string, Alert[]>>((acc, alert) => {
    const resource = alert.resource || 'Unknown';
    if (!acc[resource]) {
      acc[resource] = [];
    }
    acc[resource].push(alert);
    return acc;
  }, {});
  
  // Get color for severity
  const getSeverityColor = (severity: Alert['severity']): string => {
    switch (severity) {
      case 'critical': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
      case 'error': return 'text-orange-600 bg-orange-100 dark:bg-orange-900/20 dark:text-orange-400';
      case 'warning': return 'text-amber-600 bg-amber-100 dark:bg-amber-900/20 dark:text-amber-400';
      case 'info': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
    }
  };
  
  // Get icon for severity
  const getSeverityIcon = (severity: Alert['severity']) => {
    switch (severity) {
      case 'critical': return <AlertCircle className="h-4 w-4" />;
      case 'error': return <AlertCircle className="h-4 w-4" />;
      case 'warning': return <AlertTriangle className="h-4 w-4" />;
      case 'info': return <Info className="h-4 w-4" />;
      default: return <Info className="h-4 w-4" />;
    }
  };
  
  // Get color for status
  const getStatusColor = (status: Alert['status']): string => {
    switch (status) {
      case 'firing': return 'text-red-600 bg-red-100 dark:bg-red-900/20 dark:text-red-400';
      case 'acknowledged': return 'text-blue-600 bg-blue-100 dark:bg-blue-900/20 dark:text-blue-400';
      case 'resolved': return 'text-green-600 bg-green-100 dark:bg-green-900/20 dark:text-green-400';
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-800 dark:text-gray-400';
    }
  };
  
  // Get related alerts
  const getRelatedAlerts = (alertId: string): { alert: Alert; relation: AlertRelation }[] => {
    const relatedAlerts: { alert: Alert; relation: AlertRelation }[] = [];
    
    // Find relations where this alert is the source
    relations
      .filter(relation => relation.source === alertId)
      .forEach(relation => {
        const targetAlert = alerts.find(alert => alert.id === relation.target);
        if (targetAlert) {
          relatedAlerts.push({ alert: targetAlert, relation });
        }
      });
    
    // Find relations where this alert is the target
    relations
      .filter(relation => relation.target === alertId)
      .forEach(relation => {
        const sourceAlert = alerts.find(alert => alert.id === relation.source);
        if (sourceAlert) {
          relatedAlerts.push({ alert: sourceAlert, relation: { ...relation, type: reverseRelationType(relation.type) } });
        }
      });
    
    return relatedAlerts;
  };
  
  // Reverse relation type
  const reverseRelationType = (type: AlertRelation['type']): AlertRelation['type'] => {
    switch (type) {
      case 'causes': return 'follows';
      case 'follows': return 'causes';
      default: return type;
    }
  };
  
  // Get relation description
  const getRelationDescription = (relation: AlertRelation): string => {
    switch (relation.type) {
      case 'causes': return 'Likely causes';
      case 'follows': return 'Likely effect of';
      case 'correlates': return 'Correlated with';
      default: return 'Related to';
    }
  };
  
  // Filter alerts by search term
  const filteredAlerts = searchTerm
    ? alerts.filter(alert => 
        alert.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        alert.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        alert.resource.toLowerCase().includes(searchTerm.toLowerCase()) ||
        Object.entries(alert.labels).some(([key, value]) => 
          key.toLowerCase().includes(searchTerm.toLowerCase()) ||
          value.toLowerCase().includes(searchTerm.toLowerCase())
        )
      )
    : alerts;
  
  // Toggle group expansion
  const toggleGroup = (resource: string) => {
    setExpandedGroups(prev => ({
      ...prev,
      [resource]: !prev[resource]
    }));
  };
  
  // Draw alert correlation graph
  useEffect(() => {
    if (!canvasRef.current || !selectedAlert) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Get selected alert
    const alert = alerts.find(a => a.id === selectedAlert);
    if (!alert) return;
    
    // Get related alerts
    const relatedAlerts = getRelatedAlerts(selectedAlert);
    if (relatedAlerts.length === 0) {
      // Draw message when no relations
      ctx.font = '14px sans-serif';
      ctx.fillStyle = '#6b7280';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('No alert correlations found', canvas.width / 2, canvas.height / 2);
      return;
    }
    
    // Node positions
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) * 0.7;
    
    // Draw central node (selected alert)
    const drawCentralNode = () => {
      ctx.beginPath();
      ctx.arc(centerX, centerY, 30, 0, Math.PI * 2);
      
      // Fill based on severity
      const severityColors: Record<Alert['severity'], string> = {
        critical: '#ef4444',
        error: '#f97316',
        warning: '#f59e0b',
        info: '#3b82f6',
      };
      
      ctx.fillStyle = severityColors[alert.severity] || '#6b7280';
      ctx.fill();
      
      // Draw label
      ctx.font = 'bold 12px sans-serif';
      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(alert.name.substring(0, 10), centerX, centerY);
    };
    
    // Draw related nodes
    const drawRelatedNodes = () => {
      const angleStep = (2 * Math.PI) / relatedAlerts.length;
      
      relatedAlerts.forEach((related, index) => {
        const angle = index * angleStep;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        
        // Draw connection line
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(x, y);
        
        // Line style based on relation type
        const relationColors: Record<AlertRelation['type'], string> = {
          causes: '#ef4444', // Red
          follows: '#3b82f6', // Blue
          correlates: '#8b5cf6', // Purple
        };
        
        ctx.strokeStyle = relationColors[related.relation.type] || '#6b7280';
        ctx.lineWidth = related.relation.confidence * 3 + 1;
        ctx.stroke();
        
        // Draw arrow for direction
        const arrowLength = 10;
        const arrowWidth = 6;
        const dx = x - centerX;
        const dy = y - centerY;
        const length = Math.sqrt(dx * dx + dy * dy);
        const unitX = dx / length;
        const unitY = dy / length;
        
        const arrowX = centerX + unitX * (length - 40); // Offset from target node
        const arrowY = centerY + unitY * (length - 40);
        
        const perpX = -unitY;
        const perpY = unitX;
        
        ctx.beginPath();
        ctx.moveTo(arrowX, arrowY);
        ctx.lineTo(arrowX - arrowLength * unitX + arrowWidth * perpX, arrowY - arrowLength * unitY + arrowWidth * perpY);
        ctx.lineTo(arrowX - arrowLength * unitX - arrowWidth * perpX, arrowY - arrowLength * unitY - arrowWidth * perpY);
        ctx.closePath();
        ctx.fillStyle = relationColors[related.relation.type] || '#6b7280';
        ctx.fill();
        
        // Draw relation label
        ctx.font = '10px sans-serif';
        ctx.fillStyle = '#6b7280';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const labelX = centerX + unitX * (length / 2);
        const labelY = centerY + unitY * (length / 2) - 10;
        ctx.fillText(getRelationDescription(related.relation), labelX, labelY);
        
        // Draw confidence
        ctx.font = '9px sans-serif';
        ctx.fillText(`${Math.round(related.relation.confidence * 100)}%`, labelX, labelY + 12);
        
        // Draw node
        ctx.beginPath();
        ctx.arc(x, y, 25, 0, Math.PI * 2);
        
        // Fill based on severity
        const severityColors: Record<Alert['severity'], string> = {
          critical: '#ef4444',
          error: '#f97316',
          warning: '#f59e0b',
          info: '#3b82f6',
        };
        
        ctx.fillStyle = severityColors[related.alert.severity] || '#6b7280';
        ctx.fill();
        
        // Draw label
        ctx.font = 'bold 11px sans-serif';
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(related.alert.name.substring(0, 8), x, y);
      });
    };
    
    // Draw graph
    drawCentralNode();
    drawRelatedNodes();
    
    // Handle canvas click events
    const handleCanvasClick = (event: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      // Check if a related node was clicked
      const angleStep = (2 * Math.PI) / relatedAlerts.length;
      
      for (let i = 0; i < relatedAlerts.length; i++) {
        const angle = i * angleStep;
        const nodeX = centerX + radius * Math.cos(angle);
        const nodeY = centerY + radius * Math.sin(angle);
        
        const distance = Math.sqrt(
          Math.pow(nodeX - x, 2) + Math.pow(nodeY - y, 2)
        );
        
        if (distance <= 25) { // Node radius
          setSelectedAlert(relatedAlerts[i].alert.id);
          break;
        }
      }
    };
    
    canvas.addEventListener('click', handleCanvasClick);
    
    return () => {
      canvas.removeEventListener('click', handleCanvasClick);
    };
  }, [alerts, relations, selectedAlert]);
  
  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle>Alert Correlation</CardTitle>
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search alerts..."
            className="pl-8 h-9 w-[200px] rounded-md border border-input bg-background px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
          />
          {searchTerm && (
            <button
              className="absolute right-2.5 top-2.5 h-4 w-4 text-muted-foreground"
              onClick={() => setSearchTerm('')}
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="grid grid-cols-1 md:grid-cols-2 h-full">
          {/* Alert List */}
          <div className="border-r border-border overflow-y-auto" style={{ maxHeight: '500px' }}>
            {Object.entries(alertsByResource).map(([resource, resourceAlerts]) => {
              const filteredResourceAlerts = resourceAlerts.filter(alert => 
                filteredAlerts.some(a => a.id === alert.id)
              );
              
              if (filteredResourceAlerts.length === 0) return null;
              
              return (
                <div key={resource} className="border-b border-border last:border-b-0">
                  <div 
                    className="flex items-center justify-between p-3 cursor-pointer hover:bg-muted/50"
                    onClick={() => toggleGroup(resource)}
                  >
                    <div className="font-medium flex items-center">
                      {expandedGroups[resource] ? <ChevronDown className="h-4 w-4 mr-1" /> : <ChevronRight className="h-4 w-4 mr-1" />}
                      {resource}
                    </div>
                    <Badge variant="outline">
                      {filteredResourceAlerts.length} alert{filteredResourceAlerts.length !== 1 ? 's' : ''}
                    </Badge>
                  </div>
                  
                  {expandedGroups[resource] && (
                    <div className="pl-4">
                      {filteredResourceAlerts.map(alert => (
                        <div 
                          key={alert.id} 
                          className={`p-2 border-l-2 mb-1 ml-2 cursor-pointer hover:bg-muted/50 ${
                            selectedAlert === alert.id ? 'bg-muted border-primary' : `border-transparent ${getSeverityColor(alert.severity)}`
                          }`}
                          onClick={() => {
                            setSelectedAlert(alert.id);
                            if (onAlertClick) onAlertClick(alert);
                          }}
                        >
                          <div className="flex items-start">
                            <div className="flex-shrink-0 mt-0.5">
                              {getSeverityIcon(alert.severity)}
                            </div>
                            <div className="ml-2 flex-1">
                              <h3 className="text-sm font-medium">{alert.name}</h3>
                              <div className="mt-1 text-xs text-muted-foreground">
                                <p>{alert.description}</p>
                              </div>
                              <div className="mt-1 flex flex-wrap gap-1">
                                <Badge variant="outline" className={getStatusColor(alert.status)}>
                                  {alert.status}
                                </Badge>
                                <span className="text-xs flex items-center text-muted-foreground">
                                  <Clock className="h-3 w-3 inline mr-1" />
                                  {format(new Date(alert.startTime), 'MMM d, HH:mm')}
                                </span>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
            
            {filteredAlerts.length === 0 && (
              <div className="p-4 text-center text-muted-foreground">
                No alerts match your search
              </div>
            )}
          </div>
          
          {/* Alert Details and Correlation */}
          <div className="p-4">
            {selectedAlert ? (
              <>
                {/* Selected Alert Details */}
                {(() => {
                  const alert = alerts.find(a => a.id === selectedAlert);
                  if (!alert) return null;
                  
                  return (
                    <div className="mb-4">
                      <h3 className="text-lg font-medium mb-2">{alert.name}</h3>
                      <p className="text-sm text-muted-foreground mb-2">{alert.description}</p>
                      
                      <div className="flex flex-wrap gap-2 mb-3">
                        <Badge variant="outline" className={getSeverityColor(alert.severity)}>
                          {alert.severity}
                        </Badge>
                        <Badge variant="outline" className={getStatusColor(alert.status)}>
                          {alert.status}
                        </Badge>
                        <span className="text-xs flex items-center text-muted-foreground">
                          <Clock className="h-3 w-3 inline mr-1" />
                          {format(new Date(alert.startTime), 'MMM d, HH:mm:ss')}
                        </span>
                      </div>
                      
                      {alert.status === 'firing' && onAcknowledge && (
                        <Button 
                          size="sm" 
                          variant="outline"
                          onClick={() => onAcknowledge(alert.id)}
                          className="mb-3"
                        >
                          Acknowledge
                        </Button>
                      )}
                      
                      {/* Alert Labels */}
                      {Object.entries(alert.labels).length > 0 && (
                        <div className="mb-3">
                          <h4 className="text-sm font-medium mb-1">Labels</h4>
                          <div className="flex flex-wrap gap-1">
                            {Object.entries(alert.labels).map(([key, value]) => (
                              <Badge key={key} variant="outline" className="text-xs">
                                {key}: {value}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })()}
                
                {/* Correlation Graph */}
                <div>
                  <h4 className="text-sm font-medium mb-2">Alert Correlation</h4>
                  <div className="border rounded-md" style={{ height: '300px' }}>
                    <canvas
                      ref={canvasRef}
                      className="w-full h-full"
                    />
                  </div>
                </div>
              </>
            ) : (
              <div className="h-full flex items-center justify-center text-muted-foreground">
                Select an alert to view details and correlations
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};