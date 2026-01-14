"use client";

import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { 
  ArrowRight, 
  Server, 
  Cpu, 
  MemoryStick, 
  HardDrive,
  Network,
  CheckCircle,
  AlertCircle,
  Clock,
  Zap,
  Settings,
  Activity
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Switch } from "@/components/ui/switch";

interface VMMigrationDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  vmId: string | null;
}

// Mock data - replace with API calls
const mockVM = {
  id: "vm-001",
  name: "web-server-01",
  status: "running",
  currentHost: "node-01",
  cpu: { cores: 4, usage: 45 },
  memory: { total: 8, used: 5.2 },
  disk: { total: 100, used: 32 },
  network: { rx: 15.2, tx: 8.7 }
};

const mockHosts = [
  { 
    id: "node-01", 
    name: "node-01", 
    cpu: { total: 16, used: 8 }, 
    memory: { total: 64, used: 28 }, 
    storage: { total: 1000, used: 450 },
    status: "online",
    load: 0.52,
    isCurrentHost: true
  },
  { 
    id: "node-02", 
    name: "node-02", 
    cpu: { total: 24, used: 12 }, 
    memory: { total: 128, used: 48 }, 
    storage: { total: 2000, used: 680 },
    status: "online",
    load: 0.38,
    isCurrentHost: false
  },
  { 
    id: "node-03", 
    name: "node-03", 
    cpu: { total: 8, used: 6 }, 
    memory: { total: 32, used: 24 }, 
    storage: { total: 500, used: 180 },
    status: "online",
    load: 0.78,
    isCurrentHost: false
  },
  { 
    id: "node-04", 
    name: "node-04", 
    cpu: { total: 32, used: 4 }, 
    memory: { total: 256, used: 32 }, 
    storage: { total: 4000, used: 200 },
    status: "maintenance",
    load: 0.15,
    isCurrentHost: false
  }
];

export function VMMigrationDialog({ open, onOpenChange, vmId }: VMMigrationDialogProps) {
  const [selectedHost, setSelectedHost] = useState("");
  const [migrationType, setMigrationType] = useState("live");
  const [migrationStatus, setMigrationStatus] = useState<"idle" | "preparing" | "migrating" | "completing" | "completed" | "error">("idle");
  const [migrationProgress, setMigrationProgress] = useState(0);
  const [estimatedTime, setEstimatedTime] = useState("");
  const [bandwidthLimit, setBandwidthLimit] = useState(false);
  const [compressionEnabled, setCompressionEnabled] = useState(true);
  const [validationOnly, setValidationOnly] = useState(false);

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (open) {
      setSelectedHost("");
      setMigrationStatus("idle");
      setMigrationProgress(0);
      setEstimatedTime("");
    }
  }, [open]);

  // Mock migration process
  const startMigration = () => {
    if (!selectedHost) return;
    
    setMigrationStatus("preparing");
    setMigrationProgress(0);
    
    // Simulate migration phases
    const phases = [
      { status: "preparing", duration: 2000, progress: 10 },
      { status: "migrating", duration: 8000, progress: 90 },
      { status: "completing", duration: 1000, progress: 100 },
      { status: "completed", duration: 500, progress: 100 }
    ];
    
    let currentPhase = 0;
    
    const runPhase = () => {
      const phase = phases[currentPhase];
      setMigrationStatus(phase.status as any);
      
      const startTime = Date.now();
      const interval = setInterval(() => {
        const elapsed = Date.now() - startTime;
        const progress = Math.min((elapsed / phase.duration) * (phase.progress - (currentPhase > 0 ? phases[currentPhase - 1].progress : 0)), phase.progress);
        setMigrationProgress(currentPhase > 0 ? phases[currentPhase - 1].progress + progress : progress);
        
        if (elapsed >= phase.duration) {
          clearInterval(interval);
          currentPhase++;
          if (currentPhase < phases.length) {
            runPhase();
          }
        }
      }, 100);
    };
    
    runPhase();
  };

  const getHostCompatibility = (host: any) => {
    if (host.isCurrentHost) return { score: 100, issues: [] };
    
    const issues = [];
    let score = 100;
    
    // Check CPU compatibility
    if (host.cpu.total - host.cpu.used < mockVM.cpu.cores) {
      issues.push("Insufficient CPU cores");
      score -= 30;
    }
    
    // Check memory compatibility
    if (host.memory.total - host.memory.used < mockVM.memory.total) {
      issues.push("Insufficient memory");
      score -= 30;
    }
    
    // Check load
    if (host.load > 0.8) {
      issues.push("High system load");
      score -= 20;
    }
    
    // Check status
    if (host.status !== "online") {
      issues.push("Host not online");
      score -= 50;
    }
    
    return { score: Math.max(0, score), issues };
  };

  const getCompatibilityColor = (score: number) => {
    if (score >= 80) return "text-green-600 dark:text-green-400";
    if (score >= 60) return "text-yellow-600 dark:text-yellow-400";
    return "text-red-600 dark:text-red-400";
  };

  const getCompatibilityBadge = (score: number) => {
    if (score >= 80) return { variant: "default" as const, text: "Excellent" };
    if (score >= 60) return { variant: "secondary" as const, text: "Good" };
    return { variant: "destructive" as const, text: "Poor" };
  };

  const selectedHostData = mockHosts.find(h => h.id === selectedHost);
  const compatibility = selectedHostData ? getHostCompatibility(selectedHostData) : null;

  const getMigrationTypeDescription = (type: string) => {
    switch (type) {
      case "live":
        return "Minimal downtime migration with memory pre-copy";
      case "cold":
        return "VM stopped during migration for maximum reliability";
      case "warm":
        return "Brief pause for final synchronization";
      default:
        return "";
    }
  };

  const getEstimatedMigrationTime = () => {
    if (!selectedHostData) return "";
    
    const dataSize = mockVM.memory.used + (mockVM.disk.used * 0.1); // Simplified calculation
    const transferRate = bandwidthLimit ? 100 : 1000; // MB/s
    const compressionFactor = compressionEnabled ? 0.7 : 1;
    
    const timeSeconds = (dataSize * compressionFactor) / transferRate;
    const minutes = Math.ceil(timeSeconds / 60);
    
    return `~${minutes} minute${minutes > 1 ? 's' : ''}`;
  };

  useEffect(() => {
    setEstimatedTime(getEstimatedMigrationTime());
  }, [selectedHost, bandwidthLimit, compressionEnabled]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ArrowRight className="h-5 w-5" />
            Migrate Virtual Machine
          </DialogTitle>
          <DialogDescription>
            {vmId && `Migrate ${mockVM.name} to a different host`}
          </DialogDescription>
        </DialogHeader>

        {migrationStatus === "idle" ? (
          <Tabs defaultValue="destination" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="destination">Destination</TabsTrigger>
              <TabsTrigger value="options">Options</TabsTrigger>
              <TabsTrigger value="validation">Validation</TabsTrigger>
            </TabsList>

            <TabsContent value="destination" className="space-y-4">
              <div className="grid gap-4">
                {/* Current VM Info */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm font-medium">Source VM</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center space-x-4">
                      <div className="p-2 bg-blue-100 dark:bg-blue-900/20 rounded-lg">
                        <Server className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                      </div>
                      <div className="flex-1">
                        <div className="font-medium">{mockVM.name}</div>
                        <div className="text-sm text-muted-foreground">
                          Currently on {mockVM.currentHost}
                        </div>
                      </div>
                      <div className="text-right text-sm">
                        <div>CPU: {mockVM.cpu.cores} cores</div>
                        <div>Memory: {mockVM.memory.total} GB</div>
                        <div>Storage: {mockVM.disk.total} GB</div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Host Selection */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm font-medium">Target Host</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <Select value={selectedHost} onValueChange={setSelectedHost}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select target host" />
                      </SelectTrigger>
                      <SelectContent>
                        {mockHosts.filter(host => !host.isCurrentHost).map(host => {
                          const compatibility = getHostCompatibility(host);
                          const badge = getCompatibilityBadge(compatibility.score);
                          
                          return (
                            <SelectItem 
                              key={host.id} 
                              value={host.id}
                              disabled={host.status !== "online"}
                            >
                              <div className="flex items-center justify-between w-full">
                                <div className="flex flex-col">
                                  <span>{host.name}</span>
                                  <span className="text-xs text-muted-foreground">
                                    Load: {Math.round(host.load * 100)}% • {host.status}
                                  </span>
                                </div>
                                <Badge variant={badge.variant} className="ml-2">
                                  {badge.text}
                                </Badge>
                              </div>
                            </SelectItem>
                          );
                        })}
                      </SelectContent>
                    </Select>

                    {selectedHostData && (
                      <div className="space-y-3">
                        <div className="grid grid-cols-3 gap-4 text-sm">
                          <div className="space-y-1">
                            <div className="font-medium flex items-center gap-1">
                              <Cpu className="h-3 w-3" />
                              CPU
                            </div>
                            <div className="text-muted-foreground">
                              {selectedHostData.cpu.used} / {selectedHostData.cpu.total} cores
                            </div>
                            <Progress 
                              value={(selectedHostData.cpu.used / selectedHostData.cpu.total) * 100} 
                              className="h-1"
                            />
                          </div>
                          <div className="space-y-1">
                            <div className="font-medium flex items-center gap-1">
                              <MemoryStick className="h-3 w-3" />
                              Memory
                            </div>
                            <div className="text-muted-foreground">
                              {selectedHostData.memory.used} / {selectedHostData.memory.total} GB
                            </div>
                            <Progress 
                              value={(selectedHostData.memory.used / selectedHostData.memory.total) * 100} 
                              className="h-1"
                            />
                          </div>
                          <div className="space-y-1">
                            <div className="font-medium flex items-center gap-1">
                              <HardDrive className="h-3 w-3" />
                              Storage
                            </div>
                            <div className="text-muted-foreground">
                              {selectedHostData.storage.used} / {selectedHostData.storage.total} GB
                            </div>
                            <Progress 
                              value={(selectedHostData.storage.used / selectedHostData.storage.total) * 100} 
                              className="h-1"
                            />
                          </div>
                        </div>

                        {compatibility && (
                          <Alert>
                            <CheckCircle className="h-4 w-4" />
                            <AlertTitle>
                              Compatibility Score: 
                              <span className={getCompatibilityColor(compatibility.score)}>
                                {compatibility.score}%
                              </span>
                            </AlertTitle>
                            {compatibility.issues.length > 0 && (
                              <AlertDescription>
                                Issues: {compatibility.issues.join(", ")}
                              </AlertDescription>
                            )}
                          </Alert>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="options" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Migration Type</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <Select value={migrationType} onValueChange={setMigrationType}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="live">
                        <div className="flex flex-col">
                          <span>Live Migration</span>
                          <span className="text-xs text-muted-foreground">
                            Minimal downtime (recommended)
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="warm">
                        <div className="flex flex-col">
                          <span>Warm Migration</span>
                          <span className="text-xs text-muted-foreground">
                            Brief pause for synchronization
                          </span>
                        </div>
                      </SelectItem>
                      <SelectItem value="cold">
                        <div className="flex flex-col">
                          <span>Cold Migration</span>
                          <span className="text-xs text-muted-foreground">
                            VM stopped during migration
                          </span>
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-muted-foreground">
                    {getMigrationTypeDescription(migrationType)}
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Advanced Options</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Enable Compression</Label>
                      <div className="text-sm text-muted-foreground">
                        Reduce bandwidth usage during transfer
                      </div>
                    </div>
                    <Switch
                      checked={compressionEnabled}
                      onCheckedChange={setCompressionEnabled}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Bandwidth Limiting</Label>
                      <div className="text-sm text-muted-foreground">
                        Limit network usage to avoid impacting other VMs
                      </div>
                    </div>
                    <Switch
                      checked={bandwidthLimit}
                      onCheckedChange={setBandwidthLimit}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="space-y-0.5">
                      <Label>Validation Only</Label>
                      <div className="text-sm text-muted-foreground">
                        Check compatibility without performing migration
                      </div>
                    </div>
                    <Switch
                      checked={validationOnly}
                      onCheckedChange={setValidationOnly}
                    />
                  </div>
                </CardContent>
              </Card>

              {estimatedTime && (
                <Alert>
                  <Clock className="h-4 w-4" />
                  <AlertTitle>Estimated Migration Time</AlertTitle>
                  <AlertDescription>
                    {estimatedTime} (based on current settings and network conditions)
                  </AlertDescription>
                </Alert>
              )}
            </TabsContent>

            <TabsContent value="validation" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Pre-migration Checks</CardTitle>
                  <CardDescription>
                    Verify migration requirements before proceeding
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Host connectivity</span>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Resource availability</span>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Storage compatibility</span>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">Network configuration</span>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm">VM health status</span>
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        ) : (
          /* Migration Progress */
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span>Migration Progress</span>
                  <Badge variant="outline" className="capitalize">
                    {migrationStatus}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <Progress value={migrationProgress} className="h-2" />
                <div className="flex justify-between text-sm text-muted-foreground">
                  <span>Migrating {mockVM.name}</span>
                  <span>{Math.round(migrationProgress)}% complete</span>
                </div>

                {migrationStatus === "completed" && (
                  <Alert>
                    <CheckCircle className="h-4 w-4" />
                    <AlertTitle>Migration Completed Successfully</AlertTitle>
                    <AlertDescription>
                      {mockVM.name} has been successfully migrated to {selectedHostData?.name}
                    </AlertDescription>
                  </Alert>
                )}
              </CardContent>
            </Card>

            {migrationStatus !== "completed" && (
              <Card>
                <CardHeader>
                  <CardTitle className="text-sm font-medium">Current Activity</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center space-x-3">
                    <Activity className="h-4 w-4 animate-pulse" />
                    <span className="text-sm">
                      {migrationStatus === "preparing" && "Preparing VM for migration..."}
                      {migrationStatus === "migrating" && "Transferring VM data..."}
                      {migrationStatus === "completing" && "Finalizing migration..."}
                    </span>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            {migrationStatus === "completed" ? "Close" : "Cancel"}
          </Button>
          {migrationStatus === "idle" && (
            <Button 
              onClick={startMigration}
              disabled={!selectedHost}
            >
              {validationOnly ? "Validate Migration" : "Start Migration"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}