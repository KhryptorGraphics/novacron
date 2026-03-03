"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { format } from "date-fns";

interface ExecutionEvent {
  id: string;
  name: string;
  type: 'job' | 'workflow';
  status: 'success' | 'failed' | 'running';
  timestamp: Date;
}

export function ExecutionTimeline() {
  // Mock execution events
  const executionEvents: ExecutionEvent[] = [
    {
      id: "1",
      name: "backup-daily",
      type: "job",
      status: "success",
      timestamp: new Date(Date.now() - 1000 * 60 * 5), // 5 minutes ago
    },
    {
      id: "2",
      name: "data-pipeline",
      type: "workflow",
      status: "success",
      timestamp: new Date(Date.now() - 1000 * 60 * 15), // 15 minutes ago
    },
    {
      id: "3",
      name: "cleanup-temp",
      type: "job",
      status: "failed",
      timestamp: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
    },
    {
      id: "4",
      name: "report-generation",
      type: "workflow",
      status: "running",
      timestamp: new Date(Date.now() - 1000 * 60 * 45), // 45 minutes ago
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Execution Timeline</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {executionEvents.map((event) => (
            <div key={event.id} className="flex items-start">
              <div className="flex flex-col items-center mr-4">
                <div className={`w-3 h-3 rounded-full ${
                  event.status === 'success' ? 'bg-green-500' : 
                  event.status === 'failed' ? 'bg-red-500' : 'bg-blue-500'
                }`}></div>
                {event.id !== executionEvents[executionEvents.length - 1].id && (
                  <div className="w-0.5 h-full bg-gray-200 mt-1"></div>
                )}
              </div>
              <div className="flex-1 pb-4">
                <div className="flex justify-between">
                  <h3 className="font-medium">{event.name}</h3>
                  <Badge 
                    variant={event.status === 'success' ? 'default' : 
                            event.status === 'failed' ? 'destructive' : 'secondary'}
                  >
                    {event.status}
                  </Badge>
                </div>
                <p className="text-sm text-gray-500">
                  {event.type === 'job' ? 'Job' : 'Workflow'} • {format(event.timestamp, "HH:mm")}
                </p>
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}