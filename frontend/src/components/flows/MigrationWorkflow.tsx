'use client';

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Checkbox } from '@/components/ui/checkbox';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { 
  ArrowRight,
  ArrowLeft,
  Server,
  HardDrive,
  Network,
  Clock,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Pause,
  Play,
  RotateCcw,
  Activity,
  Zap,
  Shield,
  Database,
  Wifi,
  Eye,
  Download,
  RefreshCw,
  Settings
} from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';

interface VMInstance {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'suspended' | 'error';
  os: string;
  cpu: number;
  memory: number;
  storage: number;
  network: string;
  location: string;
  tags: string[];
}

interface MigrationCheck {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'passed' | 'failed' | 'warning';
  message: string;
  critical: boolean;
  duration: number;
}

interface MigrationPlan {
  sourceVM: string;
  targetLocation: string;
  migrationMode: 'live' | 'offline' | 'hybrid';
  dataTransferMethod: 'network' | 'storage' | 'hybrid';
  bandwidth: number;
  estimatedDuration: number;
  estimatedDowntime: number;
  rollbackPlan: boolean;
}

const mockVMs: VMInstance[] = [
  {
    id: 'vm-001',
    name: 'web-server-01',
    status: 'running',
    os: 'Ubuntu 22.04',
    cpu: 4,
    memory: 8,
    storage: 100,
    network: 'VLAN-100',
    location: 'DataCenter-East',
    tags: ['production', 'web', 'critical']
  },
  {
    id: 'vm-002', 
    name: 'app-server-02',
    status: 'running',
    os: 'CentOS 8',
    cpu: 8,
    memory: 16,
    storage: 200,
    network: 'VLAN-200',
    location: 'DataCenter-East',
    tags: ['production', 'app', 'high-priority']
  },
  {
    id: 'vm-003',
    name: 'db-server-01',
    status: 'running',
    os: 'Windows Server 2022',
    cpu: 16,
    memory: 64,
    storage: 500,
    network: 'VLAN-300',
    location: 'DataCenter-East',
    tags: ['production', 'database', 'critical']
  }
];

const targetLocations = [
  { id: 'dc-west', name: 'DataCenter-West', region: 'US-West', available: true },
  { id: 'dc-central', name: 'DataCenter-Central', region: 'US-Central', available: true },
  { id: 'cloud-aws', name: 'AWS Cloud', region: 'us-east-1', available: true },
  { id: 'cloud-azure', name: 'Azure Cloud', region: 'eastus', available: false }
];

export function MigrationWorkflow() {
  const [selectedVM, setSelectedVM] = useState<string>('');
  const [migrationPlan, setMigrationPlan] = useState<MigrationPlan>({
    sourceVM: '',
    targetLocation: '',
    migrationMode: 'live',
    dataTransferMethod: 'network',
    bandwidth: 1000,
    estimatedDuration: 0,
    estimatedDowntime: 0,
    rollbackPlan: true
  });
  
  const [currentPhase, setCurrentPhase] = useState<'planning' | 'validation' | 'execution' | 'verification' | 'completed'>('planning');
  const [preChecks, setPreChecks] = useState<MigrationCheck[]>([]);
  const [isRunningChecks, setIsRunningChecks] = useState(false);
  const [migrationProgress, setMigrationProgress] = useState(0);
  const [isMigrating, setIsMigrating] = useState(false);
  const [migrationPaused, setMigrationPaused] = useState(false);
  const [migrationLogs, setMigrationLogs] = useState<string[]>([]);
  const [realTimeMetrics, setRealTimeMetrics] = useState({
    dataTransferred: 0,
    transferRate: 0,
    eta: 0,
    networkUtilization: 0,
    cpuUsage: 0,
    memoryUsage: 0
  });

  const { toast } = useToast();
  const logEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [migrationLogs]);

  // Calculate migration estimates
  useEffect(() => {
    if (selectedVM && migrationPlan.targetLocation) {
      const vm = mockVMs.find(v => v.id === selectedVM);
      if (vm) {
        // Simple estimation based on storage size and bandwidth
        const transferTimeHours = vm.storage / (migrationPlan.bandwidth * 0.125); // Convert Mbps to GB/h
        const baseDowntime = migrationPlan.migrationMode === 'live' ? 5 : 
                            migrationPlan.migrationMode === 'offline' ? transferTimeHours * 60 : 15;
        
        setMigrationPlan(prev => ({
          ...prev,
          sourceVM: selectedVM,
          estimatedDuration: Math.ceil(transferTimeHours * 60), // minutes
          estimatedDowntime: Math.ceil(baseDowntime) // minutes
        }));
      }
    }
  }, [selectedVM, migrationPlan.targetLocation, migrationPlan.bandwidth, migrationPlan.migrationMode]);

  const runPreMigrationChecks = async () => {
    setIsRunningChecks(true);
    setCurrentPhase('validation');
    
    const checks: MigrationCheck[] = [
      { id: 'source-health', name: 'Source VM Health Check', status: 'pending', message: '', critical: true, duration: 0 },
      { id: 'target-capacity', name: 'Target Capacity Validation', status: 'pending', message: '', critical: true, duration: 0 },
      { id: 'network-connectivity', name: 'Network Connectivity Test', status: 'pending', message: '', critical: true, duration: 0 },
      { id: 'storage-compatibility', name: 'Storage Compatibility Check', status: 'pending', message: '', critical: true, duration: 0 },
      { id: 'license-validation', name: 'License Validation', status: 'pending', message: '', critical: false, duration: 0 },
      { id: 'backup-verification', name: 'Backup Verification', status: 'pending', message: '', critical: false, duration: 0 }
    ];
    
    setPreChecks(checks);

    // Simulate running checks
    for (let i = 0; i < checks.length; i++) {
      const check = checks[i];
      
      // Update status to running
      setPreChecks(prev => prev.map((c, idx) => 
        idx === i ? { ...c, status: 'running' } : c
      ));

      // Simulate check duration
      await new Promise(resolve => setTimeout(resolve, 1000 + Math.random() * 2000));
      
      // Randomly determine result (90% pass rate for demo)
      const passed = Math.random() > 0.1;
      const hasWarning = !passed && !check.critical && Math.random() > 0.5;
      
      const status = passed ? 'passed' : hasWarning ? 'warning' : 'failed';
      const message = passed ? 'Check completed successfully' :
                     hasWarning ? 'Minor issues detected but migration can proceed' :
                     'Critical issue detected - migration blocked';

      setPreChecks(prev => prev.map((c, idx) => 
        idx === i ? { ...c, status, message, duration: 1000 + Math.random() * 2000 } : c
      ));
    }
    
    setIsRunningChecks(false);
    
    const failedCritical = checks.some(c => c.status === 'failed' && c.critical);
    if (failedCritical) {
      toast({
        title: "Pre-migration Checks Failed",
        description: "Critical issues detected. Please resolve before proceeding.",
        variant: "destructive"
      });
    } else {
      toast({
        title: "Pre-migration Checks Completed",
        description: "Ready to proceed with migration.",
      });
    }
  };

  const startMigration = async () => {
    setIsMigrating(true);
    setCurrentPhase('execution');
    setMigrationProgress(0);
    setMigrationLogs([]);

    const vm = mockVMs.find(v => v.id === selectedVM);
    const target = targetLocations.find(l => l.id === migrationPlan.targetLocation);
    
    const addLog = (message: string) => {
      setMigrationLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: ${message}`]);
    };

    addLog(`Starting migration of ${vm?.name} to ${target?.name}`);
    addLog(`Migration mode: ${migrationPlan.migrationMode}`);
    addLog(`Data transfer method: ${migrationPlan.dataTransferMethod}`);

    // Simulate migration phases
    const phases = [
      { name: 'Initializing migration', duration: 5000, progress: 5 },
      { name: 'Creating snapshot', duration: 8000, progress: 15 },
      { name: 'Preparing target resources', duration: 6000, progress: 25 },
      { name: 'Starting data transfer', duration: 3000, progress: 30 },
      { name: 'Transferring system files', duration: 15000, progress: 60 },
      { name: 'Transferring application data', duration: 10000, progress: 80 },
      { name: 'Synchronizing final changes', duration: 5000, progress: 90 },
      { name: 'Starting target VM', duration: 4000, progress: 95 },
      { name: 'Finalizing migration', duration: 3000, progress: 100 }
    ];

    for (const phase of phases) {
      if (!isMigrating || migrationPaused) break;
      
      addLog(`Phase: ${phase.name}`);
      
      const startTime = Date.now();
      const progressInterval = setInterval(() => {
        if (migrationPaused) return;
        
        const elapsed = Date.now() - startTime;
        const phaseProgress = Math.min((elapsed / phase.duration) * (phase.progress - migrationProgress), 
                                      phase.progress - migrationProgress);
        
        setMigrationProgress(prev => Math.min(prev + phaseProgress * 0.1, phase.progress));
        
        // Update real-time metrics
        setRealTimeMetrics(prev => ({
          ...prev,
          dataTransferred: (phase.progress / 100) * (vm?.storage || 0),
          transferRate: 800 + Math.random() * 400,
          eta: Math.max(0, (100 - phase.progress) * 2),
          networkUtilization: 60 + Math.random() * 30,
          cpuUsage: 40 + Math.random() * 30,
          memoryUsage: 50 + Math.random() * 20
        }));
      }, 500);

      await new Promise(resolve => setTimeout(resolve, phase.duration));
      clearInterval(progressInterval);
      
      setMigrationProgress(phase.progress);
      addLog(`Completed: ${phase.name}`);
    }

    if (migrationProgress >= 100) {
      addLog('Migration completed successfully!');
      setCurrentPhase('verification');
      toast({
        title: "Migration Completed",
        description: "VM has been successfully migrated. Running post-migration verification.",
      });
      
      // Run verification
      setTimeout(() => {
        addLog('Post-migration verification passed');
        addLog('Target VM is healthy and responding');
        setCurrentPhase('completed');
        setIsMigrating(false);
        
        toast({
          title: "Migration Verified",
          description: "VM is running successfully at the target location.",
        });
      }, 3000);
    }
  };

  const pauseMigration = () => {
    setMigrationPaused(true);
    setMigrationLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: Migration paused`]);
    toast({
      title: "Migration Paused",
      description: "Migration has been paused and can be resumed at any time.",
    });
  };

  const resumeMigration = () => {
    setMigrationPaused(false);
    setMigrationLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: Migration resumed`]);
    toast({
      title: "Migration Resumed",
      description: "Migration has been resumed.",
    });
  };

  const rollbackMigration = () => {
    setIsMigrating(false);
    setMigrationPaused(false);
    setMigrationProgress(0);
    setCurrentPhase('planning');
    setMigrationLogs(prev => [...prev, `${new Date().toLocaleTimeString()}: Migration rolled back`]);
    
    toast({
      title: "Migration Rolled Back",
      description: "Migration has been cancelled and system restored to original state.",
      variant: "destructive"
    });
  };

  const selectedVMData = mockVMs.find(vm => vm.id === selectedVM);
  const targetLocationData = targetLocations.find(loc => loc.id === migrationPlan.targetLocation);

  const canProceedToMigration = preChecks.length > 0 && 
    !preChecks.some(check => check.status === 'failed' && check.critical) &&
    preChecks.every(check => check.status !== 'pending' && check.status !== 'running');

  return (
    <div className="max-w-7xl mx-auto p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold">VM Migration Workflow</h2>
          <p className="text-gray-600 mt-1">Migrate virtual machines between locations with zero-downtime</p>
        </div>
        <Badge variant="outline" className="text-lg px-4 py-2">
          Phase: {currentPhase.charAt(0).toUpperCase() + currentPhase.slice(1)}
        </Badge>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Source & Target Selection */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Migration Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Source VM *</Label>
                <Select value={selectedVM} onValueChange={setSelectedVM}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select VM to migrate" />
                  </SelectTrigger>
                  <SelectContent>
                    {mockVMs.map((vm) => (
                      <SelectItem key={vm.id} value={vm.id}>
                        <div className="flex items-center justify-between w-full">
                          <span>{vm.name}</span>
                          <Badge 
                            variant={vm.status === 'running' ? 'default' : 'secondary'}
                            className="ml-2"
                          >
                            {vm.status}
                          </Badge>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Target Location *</Label>
                <Select 
                  value={migrationPlan.targetLocation} 
                  onValueChange={(value) => setMigrationPlan(prev => ({ ...prev, targetLocation: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select target location" />
                  </SelectTrigger>
                  <SelectContent>
                    {targetLocations.map((location) => (
                      <SelectItem 
                        key={location.id} 
                        value={location.id}
                        disabled={!location.available}
                      >
                        <div className="flex items-center justify-between w-full">
                          <div>
                            <div>{location.name}</div>
                            <div className="text-sm text-gray-500">{location.region}</div>
                          </div>
                          {!location.available && (
                            <Badge variant="secondary">Unavailable</Badge>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Migration Mode</Label>
                <Select
                  value={migrationPlan.migrationMode}
                  onValueChange={(value: 'live' | 'offline' | 'hybrid') => 
                    setMigrationPlan(prev => ({ ...prev, migrationMode: value }))
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="live">
                      <div>
                        <div className="font-medium">Live Migration</div>
                        <div className="text-sm text-gray-500">Zero downtime, slower transfer</div>
                      </div>
                    </SelectItem>
                    <SelectItem value="offline">
                      <div>
                        <div className="font-medium">Offline Migration</div>
                        <div className="text-sm text-gray-500">Faster transfer, planned downtime</div>
                      </div>
                    </SelectItem>
                    <SelectItem value="hybrid">
                      <div>
                        <div className="font-medium">Hybrid Migration</div>
                        <div className="text-sm text-gray-500">Balanced approach, minimal downtime</div>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Bandwidth Allocation (Mbps)</Label>
                <Input
                  type="number"
                  value={migrationPlan.bandwidth}
                  onChange={(e) => setMigrationPlan(prev => ({ 
                    ...prev, 
                    bandwidth: parseInt(e.target.value) || 1000 
                  }))}
                  min={100}
                  max={10000}
                  step={100}
                />
              </div>

              {selectedVMData && migrationPlan.targetLocation && (
                <Card className="bg-blue-50 border-blue-200">
                  <CardContent className="pt-4">
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Estimated Duration:</span>
                        <span className="font-medium">{migrationPlan.estimatedDuration} min</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Estimated Downtime:</span>
                        <span className="font-medium">{migrationPlan.estimatedDowntime} min</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Data to Transfer:</span>
                        <span className="font-medium">{selectedVMData.storage} GB</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Main Content Area */}
        <div className="lg:col-span-2">
          <Tabs value={currentPhase} className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="planning" disabled={currentPhase !== 'planning'}>Planning</TabsTrigger>
              <TabsTrigger value="validation" disabled={currentPhase !== 'validation' && !canProceedToMigration}>Validation</TabsTrigger>
              <TabsTrigger value="execution" disabled={currentPhase !== 'execution'}>Execution</TabsTrigger>
              <TabsTrigger value="verification" disabled={currentPhase !== 'verification'}>Verification</TabsTrigger>
              <TabsTrigger value="completed" disabled={currentPhase !== 'completed'}>Completed</TabsTrigger>
            </TabsList>

            <TabsContent value="planning" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Migration Planning</CardTitle>
                  <CardDescription>
                    Configure your migration settings and review the plan
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  {selectedVMData && targetLocationData ? (
                    <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <div className="p-2 bg-blue-100 rounded">
                          <Server className="h-6 w-6 text-blue-600" />
                        </div>
                        <div>
                          <h4 className="font-medium">{selectedVMData.name}</h4>
                          <p className="text-sm text-gray-600">{selectedVMData.location}</p>
                        </div>
                      </div>
                      
                      <ArrowRight className="h-8 w-8 text-gray-400" />
                      
                      <div className="flex items-center space-x-4">
                        <div className="p-2 bg-green-100 rounded">
                          <Server className="h-6 w-6 text-green-600" />
                        </div>
                        <div>
                          <h4 className="font-medium">{targetLocationData.name}</h4>
                          <p className="text-sm text-gray-600">{targetLocationData.region}</p>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <Alert>
                      <AlertTriangle className="h-4 w-4" />
                      <AlertDescription>
                        Please select both source VM and target location to continue.
                      </AlertDescription>
                    </Alert>
                  )}

                  <div className="flex justify-end">
                    <Button
                      onClick={runPreMigrationChecks}
                      disabled={!selectedVM || !migrationPlan.targetLocation || isRunningChecks}
                    >
                      {isRunningChecks ? (
                        <>
                          <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                          Running Checks...
                        </>
                      ) : (
                        <>
                          <CheckCircle className="mr-2 h-4 w-4" />
                          Run Pre-Migration Checks
                        </>
                      )}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="validation" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Pre-Migration Validation</CardTitle>
                  <CardDescription>
                    Validating migration requirements and compatibility
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {preChecks.map((check) => (
                      <div
                        key={check.id}
                        className={`flex items-center justify-between p-4 rounded-lg border ${
                          check.status === 'passed' ? 'bg-green-50 border-green-200' :
                          check.status === 'failed' ? 'bg-red-50 border-red-200' :
                          check.status === 'warning' ? 'bg-yellow-50 border-yellow-200' :
                          check.status === 'running' ? 'bg-blue-50 border-blue-200' :
                          'bg-gray-50 border-gray-200'
                        }`}
                      >
                        <div className="flex items-center space-x-3">
                          <div>
                            {check.status === 'passed' && <CheckCircle className="h-5 w-5 text-green-600" />}
                            {check.status === 'failed' && <XCircle className="h-5 w-5 text-red-600" />}
                            {check.status === 'warning' && <AlertTriangle className="h-5 w-5 text-yellow-600" />}
                            {check.status === 'running' && <RefreshCw className="h-5 w-5 text-blue-600 animate-spin" />}
                            {check.status === 'pending' && <Clock className="h-5 w-5 text-gray-400" />}
                          </div>
                          <div>
                            <h4 className="font-medium">{check.name}</h4>
                            {check.message && (
                              <p className="text-sm text-gray-600">{check.message}</p>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          {check.critical && (
                            <Badge variant="destructive">Critical</Badge>
                          )}
                          <Badge variant="outline">
                            {check.status === 'running' ? 'Running...' : 
                             check.duration > 0 ? `${(check.duration / 1000).toFixed(1)}s` : 
                             check.status}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>

                  {canProceedToMigration && (
                    <div className="flex justify-end mt-6">
                      <Button onClick={startMigration} className="bg-green-600 hover:bg-green-700">
                        <Play className="mr-2 h-4 w-4" />
                        Start Migration
                      </Button>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="execution" className="mt-6">
              <div className="space-y-6">
                {/* Progress Overview */}
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>Migration Progress</CardTitle>
                      <div className="flex space-x-2">
                        {!migrationPaused ? (
                          <Button variant="outline" size="sm" onClick={pauseMigration}>
                            <Pause className="mr-2 h-4 w-4" />
                            Pause
                          </Button>
                        ) : (
                          <Button variant="outline" size="sm" onClick={resumeMigration}>
                            <Play className="mr-2 h-4 w-4" />
                            Resume
                          </Button>
                        )}
                        <Button variant="destructive" size="sm" onClick={rollbackMigration}>
                          <RotateCcw className="mr-2 h-4 w-4" />
                          Rollback
                        </Button>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">Overall Progress</span>
                        <span className="text-sm font-bold">{Math.round(migrationProgress)}%</span>
                      </div>
                      <Progress value={migrationProgress} className="w-full" />
                      
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-blue-600">
                            {realTimeMetrics.dataTransferred.toFixed(1)} GB
                          </div>
                          <div className="text-sm text-gray-600">Data Transferred</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-green-600">
                            {realTimeMetrics.transferRate.toFixed(0)} Mbps
                          </div>
                          <div className="text-sm text-gray-600">Transfer Rate</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-orange-600">
                            {realTimeMetrics.eta} min
                          </div>
                          <div className="text-sm text-gray-600">ETA</div>
                        </div>
                        <div className="text-center">
                          <div className="text-2xl font-bold text-purple-600">
                            {realTimeMetrics.networkUtilization.toFixed(0)}%
                          </div>
                          <div className="text-sm text-gray-600">Network Usage</div>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Real-time Logs */}
                <Card>
                  <CardHeader>
                    <CardTitle>Migration Logs</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="bg-black text-green-400 p-4 rounded-lg h-64 overflow-y-auto text-sm font-mono">
                      {migrationLogs.map((log, index) => (
                        <div key={index} className="mb-1">{log}</div>
                      ))}
                      <div ref={logEndRef} />
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="verification" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle>Post-Migration Verification</CardTitle>
                  <CardDescription>
                    Verifying migration success and VM functionality
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center space-x-2">
                      <RefreshCw className="h-5 w-5 animate-spin text-blue-600" />
                      <span>Running post-migration health checks...</span>
                    </div>
                    
                    <Progress value={75} className="w-full" />
                    
                    <div className="text-sm text-gray-600">
                      Verifying VM boot sequence, network connectivity, and application status...
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="completed" className="mt-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <CheckCircle className="h-6 w-6 text-green-600" />
                    <span>Migration Completed Successfully</span>
                  </CardTitle>
                  <CardDescription>
                    Your VM has been successfully migrated to the target location
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-3">
                      <h4 className="font-medium">Migration Summary</h4>
                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span>Total Duration:</span>
                          <span className="font-medium">{migrationPlan.estimatedDuration} minutes</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Actual Downtime:</span>
                          <span className="font-medium">{migrationPlan.estimatedDowntime} minutes</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Data Transferred:</span>
                          <span className="font-medium">{selectedVMData?.storage} GB</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Average Speed:</span>
                          <span className="font-medium">850 Mbps</span>
                        </div>
                      </div>
                    </div>
                    
                    <div className="space-y-3">
                      <h4 className="font-medium">Target VM Status</h4>
                      <div className="space-y-2">
                        <div className="flex items-center space-x-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          <span className="text-sm">VM is running and healthy</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          <span className="text-sm">Network connectivity verified</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          <span className="text-sm">All services started successfully</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <CheckCircle className="h-4 w-4 text-green-600" />
                          <span className="text-sm">Performance metrics normal</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="flex justify-end space-x-2">
                    <Button variant="outline">
                      <Download className="mr-2 h-4 w-4" />
                      Download Report
                    </Button>
                    <Button>
                      <Eye className="mr-2 h-4 w-4" />
                      View Target VM
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}