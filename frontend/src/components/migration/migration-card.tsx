import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Clock, ArrowRight, AlertTriangle, CheckCircle, RefreshCw, XCircle } from 'lucide-react';
import { formatDistanceToNow, format } from 'date-fns';
import { bytesToSize } from '@/lib/utils';
import { useToast } from '@/components/ui/use-toast';

export interface MigrationStatus {
  migrationId: string;
  vmId: string;
  vmName: string;
  sourceNodeId: string;
  targetNodeId: string;
  state: string;
  progress: number;
  startTime: string;
  completionTime?: string;
  transferRate?: number;
  bytesTransferred?: number;
  totalBytes?: number;
  errorMessage?: string;
}

interface MigrationCardProps {
  migration: MigrationStatus;
  onCancel?: (migrationId: string) => Promise<void>;
  onRetry?: (vmId: string, targetNodeId: string) => Promise<void>;
}

export function MigrationCard({ migration, onCancel, onRetry }: MigrationCardProps) {
  const { toast } = useToast();
  const [progressValue, setProgressValue] = useState(migration.progress * 100);
  
  useEffect(() => {
    setProgressValue(migration.progress * 100);
  }, [migration.progress]);
  
  const isActive = !['completed', 'failed', 'rolledback'].includes(migration.state);
  const isFailed = ['failed', 'rolledback'].includes(migration.state);
  const isRollingBack = migration.state === 'rollingback';
  
  const handleCancel = async () => {
    if (!onCancel) return;
    
    try {
      await onCancel(migration.migrationId);
      toast({
        title: "Migration cancelled",
        description: `Cancellation request for ${migration.vmName} has been submitted.`,
      });
    } catch (error) {
      toast({
        title: "Error cancelling migration",
        description: `Failed to cancel migration: ${error.message}`,
        variant: "destructive",
      });
    }
  };
  
  const handleRetry = async () => {
    if (!onRetry) return;
    
    try {
      await onRetry(migration.vmId, migration.targetNodeId);
      toast({
        title: "Migration retry initiated",
        description: `Started a new migration for ${migration.vmName}.`,
      });
    } catch (error) {
      toast({
        title: "Error retrying migration",
        description: `Failed to retry migration: ${error.message}`,
        variant: "destructive",
      });
    }
  };

  return (
    <Card className={`w-full ${isFailed ? 'border-red-300' : isActive ? 'border-blue-300' : 'border-green-300'}`}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-start">
          <div>
            <CardTitle className="text-lg font-semibold">{migration.vmName}</CardTitle>
            <CardDescription>Migration {migration.migrationId.substring(0, 8)}</CardDescription>
          </div>
          <StatusBadge state={migration.state} />
        </div>
      </CardHeader>
      <CardContent className="pb-2">
        <div className="space-y-4">
          <div className="flex items-center space-x-4 text-sm">
            <div className="font-medium">{migration.sourceNodeId}</div>
            <ArrowRight className="h-4 w-4 text-gray-500" />
            <div className="font-medium">{migration.targetNodeId}</div>
          </div>
          
          {isActive && (
            <div className="space-y-2">
              <div className="flex justify-between text-xs text-gray-500">
                <span>Progress</span>
                <span>{progressValue.toFixed(1)}%</span>
              </div>
              <Progress value={progressValue} className="h-2" />
            </div>
          )}
          
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div>
              <div className="text-gray-500">Started</div>
              <div className="font-medium">{formatDistanceToNow(new Date(migration.startTime), { addSuffix: true })}</div>
              <div className="text-xs text-gray-400">{format(new Date(migration.startTime), 'MMM d, yyyy HH:mm:ss')}</div>
            </div>
            
            {migration.completionTime && (
              <div>
                <div className="text-gray-500">Completed</div>
                <div className="font-medium">{formatDistanceToNow(new Date(migration.completionTime), { addSuffix: true })}</div>
                <div className="text-xs text-gray-400">{format(new Date(migration.completionTime), 'MMM d, yyyy HH:mm:ss')}</div>
              </div>
            )}
            
            {migration.transferRate && (
              <div>
                <div className="text-gray-500">Transfer Rate</div>
                <div className="font-medium">{bytesToSize(migration.transferRate)}/s</div>
              </div>
            )}
            
            {migration.bytesTransferred && migration.totalBytes && (
              <div>
                <div className="text-gray-500">Data Transferred</div>
                <div className="font-medium">{bytesToSize(migration.bytesTransferred)} / {bytesToSize(migration.totalBytes)}</div>
              </div>
            )}
          </div>
          
          {migration.errorMessage && (
            <div className="mt-2 text-sm text-red-600 bg-red-50 p-2 rounded border border-red-200">
              <div className="font-medium">Error:</div>
              <div>{migration.errorMessage}</div>
            </div>
          )}
        </div>
      </CardContent>
      
      <CardFooter className="pt-0">
        {isActive && !isRollingBack && onCancel && (
          <Button variant="outline" size="sm" className="text-red-600" onClick={handleCancel}>
            <XCircle className="h-4 w-4 mr-1" /> Cancel Migration
          </Button>
        )}
        
        {isFailed && onRetry && (
          <Button variant="outline" size="sm" className="text-blue-600" onClick={handleRetry}>
            <RefreshCw className="h-4 w-4 mr-1" /> Retry Migration
          </Button>
        )}
        
        {isRollingBack && (
          <div className="flex items-center text-sm text-amber-600">
            <RefreshCw className="h-4 w-4 mr-1 animate-spin" /> Rolling back...
          </div>
        )}
      </CardFooter>
    </Card>
  );
}

function StatusBadge({ state }: { state: string }) {
  switch (state) {
    case 'pending':
      return <Badge variant="outline" className="bg-gray-100">Pending</Badge>;
    case 'initiating':
      return <Badge variant="outline" className="bg-blue-100 text-blue-800">Initiating</Badge>;
    case 'transferring':
      return <Badge variant="outline" className="bg-blue-100 text-blue-800">Transferring</Badge>;
    case 'activating':
      return <Badge variant="outline" className="bg-amber-100 text-amber-800">Activating</Badge>;
    case 'completed':
      return <Badge variant="outline" className="bg-green-100 text-green-800">Completed</Badge>;
    case 'failed':
      return <Badge variant="outline" className="bg-red-100 text-red-800">Failed</Badge>;
    case 'rollingback':
      return <Badge variant="outline" className="bg-amber-100 text-amber-800">Rolling Back</Badge>;
    case 'rolledback':
      return <Badge variant="outline" className="bg-amber-100 text-amber-800">Rolled Back</Badge>;
    default:
      return <Badge variant="outline">{state}</Badge>;
  }
}
