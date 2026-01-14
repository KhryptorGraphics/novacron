"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { 
  Server, 
  Play, 
  Pause, 
  Square, 
  RotateCw, 
  Settings, 
  Monitor, 
  HardDrive,
  Cpu,
  MemoryStick,
  Network,
  ArrowRight,
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  MoreHorizontal
} from "lucide-react";
import { 
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface VM {
  id: string;
  name: string;
  status: string;
  os: string;
  cpu: { cores: number; usage: number };
  memory: { total: number; used: number; usage: number };
  disk: { total: number; used: number; usage: number };
  network: { rx: number; tx: number };
  uptime: string;
  host: string;
  ip: string;
  created: string;
}

interface VMDetailsCardProps {
  vm: VM;
  onAction: (vmId: string, action: string) => void;
  onMigration: (vmId: string) => void;
  onSelect: (vmId: string) => void;
}

export function VMDetailsCard({ vm, onAction, onMigration, onSelect }: VMDetailsCardProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "running": return "bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400";
      case "stopped": return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
      case "error": return "bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400";
      case "starting": return "bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400";
      default: return "bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-400";
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "running": return <CheckCircle className="h-4 w-4" />;
      case "stopped": return <Square className="h-4 w-4" />;
      case "error": return <AlertCircle className="h-4 w-4" />;
      case "starting": return <Clock className="h-4 w-4" />;
      default: return <AlertCircle className="h-4 w-4" />;
    }
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
              <Server className="h-5 w-5 text-blue-600 dark:text-blue-400" />
            </div>
            <div>
              <CardTitle className="text-lg">{vm.name}</CardTitle>
              <p className="text-sm text-muted-foreground">{vm.os}</p>
            </div>
          </div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="h-8 w-8 p-0">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuLabel>Actions</DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={() => onSelect(vm.id)}>
                <Monitor className="mr-2 h-4 w-4" />
                View Details
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => onMigration(vm.id)}>
                <ArrowRight className="mr-2 h-4 w-4" />
                Migrate VM
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4" />
                Configure
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        <div className="flex items-center space-x-4 text-sm text-muted-foreground">
          <Badge className={getStatusColor(vm.status)}>
            {getStatusIcon(vm.status)}
            <span className="ml-1 capitalize">{vm.status}</span>
          </Badge>
          <span>{vm.host}</span>
          <span>{vm.ip}</span>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Resource Usage */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">CPU</span>
            </div>
            <span className="text-sm text-muted-foreground">
              {vm.cpu.cores} cores • {vm.cpu.usage}%
            </span>
          </div>
          <Progress value={vm.cpu.usage} className="h-2" />
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <MemoryStick className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Memory</span>
            </div>
            <span className="text-sm text-muted-foreground">
              {vm.memory.used.toFixed(1)} / {vm.memory.total} GB
            </span>
          </div>
          <Progress value={vm.memory.usage} className="h-2" />
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <HardDrive className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Storage</span>
            </div>
            <span className="text-sm text-muted-foreground">
              {vm.disk.used} / {vm.disk.total} GB
            </span>
          </div>
          <Progress value={vm.disk.usage} className="h-2" />
          
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Network className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">Network</span>
            </div>
            <span className="text-sm text-muted-foreground">
              ↓ {vm.network.rx} MB/s ↑ {vm.network.tx} MB/s
            </span>
          </div>
        </div>
        
        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-2 border-t">
          <div className="flex items-center space-x-2">
            {vm.status === "running" ? (
              <>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => onAction(vm.id, "stop")}
                >
                  <Square className="h-4 w-4 mr-1" />
                  Stop
                </Button>
                <Button 
                  variant="outline" 
                  size="sm"
                  onClick={() => onAction(vm.id, "restart")}
                >
                  <RotateCw className="h-4 w-4 mr-1" />
                  Restart
                </Button>
              </>
            ) : vm.status === "stopped" ? (
              <Button 
                variant="outline" 
                size="sm"
                onClick={() => onAction(vm.id, "start")}
              >
                <Play className="h-4 w-4 mr-1" />
                Start
              </Button>
            ) : null}
          </div>
          
          <div className="flex items-center text-xs text-muted-foreground">
            <Activity className="h-3 w-3 mr-1" />
            <span>Uptime: {vm.uptime}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}