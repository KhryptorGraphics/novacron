'use client';

import CoreVMsPage from '@/app/core/vms/page';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { WifiOff } from 'lucide-react';

const VMOperationsDashboard: React.FC = () => {
  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <h1 className="text-2xl font-bold">VM Operations</h1>
        <Badge variant="outline" className="bg-yellow-50 text-yellow-900">
          <WifiOff className="mr-2 h-3 w-3" />
          Snapshot Mode
        </Badge>
      </div>

      <Alert>
        <WifiOff className="h-4 w-4" />
        <AlertTitle>Realtime VM streams are unavailable</AlertTitle>
        <AlertDescription>
          The canonical websocket contract does not expose `/api/ws/vms*` channels yet. This
          dashboard uses the live canonical VM REST surface instead of opening speculative
          websocket connections.
        </AlertDescription>
      </Alert>

      <CoreVMsPage />
    </div>
  );
};

export default VMOperationsDashboard;
