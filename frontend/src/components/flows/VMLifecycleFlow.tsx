'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Checkbox } from '@/components/ui/checkbox';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Slider } from '@/components/ui/slider';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { 
  ChevronLeft, 
  ChevronRight, 
  Server, 
  HardDrive, 
  Network, 
  Cpu, 
  MemoryStick, 
  DollarSign,
  CheckCircle,
  AlertTriangle,
  Info,
  Play,
  Settings,
  Globe,
  Shield,
  Clock,
  Zap
} from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';

interface VMConfiguration {
  name: string;
  description: string;
  template: string;
  resources: {
    cpu: number;
    memory: number;
    storage: number;
    storageTier: string;
  };
  network: {
    vlan: string;
    subnet: string;
    publicIP: boolean;
    firewallRules: string[];
  };
  security: {
    encryption: boolean;
    backupEnabled: boolean;
    monitoringEnabled: boolean;
  };
  scheduling: {
    autoStart: boolean;
    shutdownSchedule: string;
    maintenanceWindow: string;
  };
}

interface Template {
  id: string;
  name: string;
  os: string;
  version: string;
  description: string;
  minCpu: number;
  minMemory: number;
  minStorage: number;
  category: string;
  tags: string[];
  cost: number;
}

interface StorageTier {
  id: string;
  name: string;
  type: string;
  iops: number;
  costPerGB: number;
  description: string;
}

const templates: Template[] = [
  {
    id: 'ubuntu-22-04',
    name: 'Ubuntu Server 22.04 LTS',
    os: 'Ubuntu',
    version: '22.04',
    description: 'Latest Ubuntu LTS with security updates',
    minCpu: 1,
    minMemory: 2,
    minStorage: 20,
    category: 'Linux',
    tags: ['web', 'development', 'production'],
    cost: 0.05
  },
  {
    id: 'windows-2022',
    name: 'Windows Server 2022',
    os: 'Windows',
    version: '2022',
    description: 'Latest Windows Server with full features',
    minCpu: 2,
    minMemory: 4,
    minStorage: 40,
    category: 'Windows',
    tags: ['enterprise', 'ad', 'iis'],
    cost: 0.15
  },
  {
    id: 'centos-8',
    name: 'CentOS Stream 8',
    os: 'CentOS',
    version: '8',
    description: 'Enterprise-grade Linux distribution',
    minCpu: 1,
    minMemory: 2,
    minStorage: 20,
    category: 'Linux',
    tags: ['enterprise', 'rhel', 'production'],
    cost: 0.06
  }
];

const storageTiers: StorageTier[] = [
  {
    id: 'standard',
    name: 'Standard SSD',
    type: 'SSD',
    iops: 3000,
    costPerGB: 0.10,
    description: 'General purpose SSD storage'
  },
  {
    id: 'premium',
    name: 'Premium SSD',
    type: 'NVMe',
    iops: 7000,
    costPerGB: 0.15,
    description: 'High-performance NVMe storage'
  },
  {
    id: 'archive',
    name: 'Archive Storage',
    type: 'HDD',
    iops: 500,
    costPerGB: 0.05,
    description: 'Cost-effective archival storage'
  }
];

const STEPS = [
  { id: 'basic', label: 'Basic Info', icon: Info },
  { id: 'template', label: 'Template', icon: Server },
  { id: 'resources', label: 'Resources', icon: Cpu },
  { id: 'network', label: 'Network', icon: Network },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'review', label: 'Review', icon: CheckCircle }
];

export function VMLifecycleFlow() {
  const [currentStep, setCurrentStep] = useState(0);
  const [configuration, setConfiguration] = useState<VMConfiguration>({
    name: '',
    description: '',
    template: '',
    resources: {
      cpu: 2,
      memory: 4,
      storage: 50,
      storageTier: 'standard'
    },
    network: {
      vlan: '',
      subnet: '',
      publicIP: false,
      firewallRules: []
    },
    security: {
      encryption: true,
      backupEnabled: true,
      monitoringEnabled: true
    },
    scheduling: {
      autoStart: true,
      shutdownSchedule: '',
      maintenanceWindow: ''
    }
  });
  
  const [validationErrors, setValidationErrors] = useState<Record<string, string>>({});
  const [isDeploying, setIsDeploying] = useState(false);
  const [deploymentProgress, setDeploymentProgress] = useState(0);
  const [costEstimate, setCostEstimate] = useState(0);
  const [savedConfigurations, setSavedConfigurations] = useState<VMConfiguration[]>([]);
  const { toast } = useToast();

  // Auto-save configuration
  useEffect(() => {
    const timer = setTimeout(() => {
      const savedConfigs = JSON.parse(localStorage.getItem('vm-drafts') || '[]');
      const existingIndex = savedConfigs.findIndex((config: VMConfiguration) => 
        config.name === configuration.name && configuration.name !== ''
      );
      
      if (configuration.name && configuration.name.trim() !== '') {
        if (existingIndex >= 0) {
          savedConfigs[existingIndex] = configuration;
        } else {
          savedConfigs.push(configuration);
        }
        localStorage.setItem('vm-drafts', JSON.stringify(savedConfigs));
        setSavedConfigurations(savedConfigs);
      }
    }, 2000);

    return () => clearTimeout(timer);
  }, [configuration]);

  // Calculate cost estimate
  useEffect(() => {
    const selectedTemplate = templates.find(t => t.id === configuration.template);
    const selectedTier = storageTiers.find(t => t.id === configuration.resources.storageTier);
    
    if (selectedTemplate && selectedTier) {
      const templateCost = selectedTemplate.cost * configuration.resources.cpu;
      const memoryCost = configuration.resources.memory * 0.02;
      const storageCost = configuration.resources.storage * selectedTier.costPerGB;
      const networkCost = configuration.network.publicIP ? 5 : 0;
      
      setCostEstimate(templateCost + memoryCost + storageCost + networkCost);
    }
  }, [configuration]);

  const validateCurrentStep = (): boolean => {
    const errors: Record<string, string> = {};
    
    switch (currentStep) {
      case 0: // Basic Info
        if (!configuration.name.trim()) {
          errors.name = 'VM name is required';
        } else if (configuration.name.length < 3) {
          errors.name = 'VM name must be at least 3 characters';
        }
        if (!configuration.description.trim()) {
          errors.description = 'Description is required';
        }
        break;
      case 1: // Template
        if (!configuration.template) {
          errors.template = 'Please select a template';
        }
        break;
      case 2: // Resources
        const selectedTemplate = templates.find(t => t.id === configuration.template);
        if (selectedTemplate) {
          if (configuration.resources.cpu < selectedTemplate.minCpu) {
            errors.cpu = `Minimum ${selectedTemplate.minCpu} CPU cores required`;
          }
          if (configuration.resources.memory < selectedTemplate.minMemory) {
            errors.memory = `Minimum ${selectedTemplate.minMemory}GB memory required`;
          }
          if (configuration.resources.storage < selectedTemplate.minStorage) {
            errors.storage = `Minimum ${selectedTemplate.minStorage}GB storage required`;
          }
        }
        break;
      case 3: // Network
        if (!configuration.network.vlan) {
          errors.vlan = 'Please select a VLAN';
        }
        if (!configuration.network.subnet) {
          errors.subnet = 'Please select a subnet';
        }
        break;
    }
    
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const nextStep = () => {
    if (validateCurrentStep()) {
      setCurrentStep(prev => Math.min(prev + 1, STEPS.length - 1));
    } else {
      toast({
        title: "Validation Error",
        description: "Please fix the errors before continuing",
        variant: "destructive"
      });
    }
  };

  const prevStep = () => {
    setCurrentStep(prev => Math.max(prev - 1, 0));
  };

  const deployVM = async () => {
    if (!validateCurrentStep()) return;
    
    setIsDeploying(true);
    setDeploymentProgress(0);
    
    // Simulate deployment progress
    const progressInterval = setInterval(() => {
      setDeploymentProgress(prev => {
        if (prev >= 100) {
          clearInterval(progressInterval);
          setIsDeploying(false);
          toast({
            title: "VM Deployed Successfully",
            description: `${configuration.name} has been created and is starting up`,
          });
          // Clear draft
          const savedConfigs = JSON.parse(localStorage.getItem('vm-drafts') || '[]');
          const filteredConfigs = savedConfigs.filter((config: VMConfiguration) => 
            config.name !== configuration.name
          );
          localStorage.setItem('vm-drafts', JSON.stringify(filteredConfigs));
          return 100;
        }
        return prev + Math.random() * 15;
      });
    }, 500);
  };

  const loadSavedConfiguration = (config: VMConfiguration) => {
    setConfiguration(config);
    setCurrentStep(0);
    toast({
      title: "Configuration Loaded",
      description: `Loaded saved configuration for ${config.name}`,
    });
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0: // Basic Info
        return (
          <div className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="vm-name">VM Name *</Label>
              <Input
                id="vm-name"
                placeholder="Enter VM name (e.g., web-server-01)"
                value={configuration.name}
                onChange={(e) => setConfiguration(prev => ({ ...prev, name: e.target.value }))}
                className={validationErrors.name ? 'border-red-500' : ''}
              />
              {validationErrors.name && (
                <p className="text-sm text-red-500">{validationErrors.name}</p>
              )}
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="vm-description">Description *</Label>
              <Input
                id="vm-description"
                placeholder="Brief description of VM purpose"
                value={configuration.description}
                onChange={(e) => setConfiguration(prev => ({ ...prev, description: e.target.value }))}
                className={validationErrors.description ? 'border-red-500' : ''}
              />
              {validationErrors.description && (
                <p className="text-sm text-red-500">{validationErrors.description}</p>
              )}
            </div>

            {savedConfigurations.length > 0 && (
              <div className="space-y-2">
                <Label>Load Saved Configuration</Label>
                <div className="grid grid-cols-1 gap-2">
                  {savedConfigurations.slice(0, 3).map((config, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => loadSavedConfiguration(config)}
                      className="justify-start"
                    >
                      <Clock className="mr-2 h-4 w-4" />
                      {config.name} - {config.description}
                    </Button>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 1: // Template Selection
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {templates.map((template) => (
                <Card
                  key={template.id}
                  className={`cursor-pointer transition-all ${
                    configuration.template === template.id 
                      ? 'ring-2 ring-blue-500 bg-blue-50' 
                      : 'hover:shadow-md'
                  }`}
                  onClick={() => setConfiguration(prev => ({ ...prev, template: template.id }))}
                >
                  <CardHeader className="pb-2">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-lg">{template.name}</CardTitle>
                      <Badge variant="secondary">{template.category}</Badge>
                    </div>
                    <CardDescription>{template.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-center space-x-4 text-sm text-gray-600">
                      <div className="flex items-center">
                        <Cpu className="mr-1 h-4 w-4" />
                        {template.minCpu}+ cores
                      </div>
                      <div className="flex items-center">
                        <MemoryStick className="mr-1 h-4 w-4" />
                        {template.minMemory}+ GB RAM
                      </div>
                      <div className="flex items-center">
                        <HardDrive className="mr-1 h-4 w-4" />
                        {template.minStorage}+ GB
                      </div>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex gap-1">
                        {template.tags.slice(0, 3).map(tag => (
                          <Badge key={tag} variant="outline" className="text-xs">{tag}</Badge>
                        ))}
                      </div>
                      <div className="flex items-center text-sm font-medium">
                        <DollarSign className="mr-1 h-4 w-4" />
                        {template.cost}/hour per core
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
            {validationErrors.template && (
              <Alert variant="destructive">
                <AlertTriangle className="h-4 w-4" />
                <AlertDescription>{validationErrors.template}</AlertDescription>
              </Alert>
            )}
          </div>
        );

      case 2: // Resources
        const selectedTemplate = templates.find(t => t.id === configuration.template);
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>CPU Cores: {configuration.resources.cpu}</Label>
                  <Slider
                    value={[configuration.resources.cpu]}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      resources: { ...prev.resources, cpu: value[0] }
                    }))}
                    min={selectedTemplate?.minCpu || 1}
                    max={32}
                    step={1}
                    className="w-full"
                  />
                  {validationErrors.cpu && (
                    <p className="text-sm text-red-500">{validationErrors.cpu}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label>Memory: {configuration.resources.memory} GB</Label>
                  <Slider
                    value={[configuration.resources.memory]}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      resources: { ...prev.resources, memory: value[0] }
                    }))}
                    min={selectedTemplate?.minMemory || 1}
                    max={128}
                    step={1}
                    className="w-full"
                  />
                  {validationErrors.memory && (
                    <p className="text-sm text-red-500">{validationErrors.memory}</p>
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>Storage: {configuration.resources.storage} GB</Label>
                  <Slider
                    value={[configuration.resources.storage]}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      resources: { ...prev.resources, storage: value[0] }
                    }))}
                    min={selectedTemplate?.minStorage || 20}
                    max={2000}
                    step={10}
                    className="w-full"
                  />
                  {validationErrors.storage && (
                    <p className="text-sm text-red-500">{validationErrors.storage}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="storage-tier">Storage Tier</Label>
                  <Select
                    value={configuration.resources.storageTier}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      resources: { ...prev.resources, storageTier: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select storage tier" />
                    </SelectTrigger>
                    <SelectContent>
                      {storageTiers.map((tier) => (
                        <SelectItem key={tier.id} value={tier.id}>
                          <div className="flex items-center justify-between w-full">
                            <span>{tier.name}</span>
                            <Badge variant="secondary">{tier.iops} IOPS</Badge>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Resource Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span>Current configuration:</span>
                    <span className="font-medium">
                      {configuration.resources.cpu} cores, {configuration.resources.memory}GB RAM
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Recommended for:</span>
                    <div className="flex gap-1">
                      {configuration.resources.cpu >= 4 && configuration.resources.memory >= 8 && (
                        <Badge variant="outline">Production</Badge>
                      )}
                      {configuration.resources.cpu >= 2 && configuration.resources.memory >= 4 && (
                        <Badge variant="outline">Development</Badge>
                      )}
                      {configuration.resources.cpu < 2 || configuration.resources.memory < 4 ? (
                        <Badge variant="outline">Testing</Badge>
                      ) : null}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        );

      case 3: // Network
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="vlan">VLAN *</Label>
                  <Select
                    value={configuration.network.vlan}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      network: { ...prev.network, vlan: value }
                    }))}
                  >
                    <SelectTrigger className={validationErrors.vlan ? 'border-red-500' : ''}>
                      <SelectValue placeholder="Select VLAN" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="vlan-100">VLAN 100 - Production</SelectItem>
                      <SelectItem value="vlan-200">VLAN 200 - Development</SelectItem>
                      <SelectItem value="vlan-300">VLAN 300 - Testing</SelectItem>
                      <SelectItem value="vlan-400">VLAN 400 - DMZ</SelectItem>
                    </SelectContent>
                  </Select>
                  {validationErrors.vlan && (
                    <p className="text-sm text-red-500">{validationErrors.vlan}</p>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="subnet">Subnet *</Label>
                  <Select
                    value={configuration.network.subnet}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      network: { ...prev.network, subnet: value }
                    }))}
                  >
                    <SelectTrigger className={validationErrors.subnet ? 'border-red-500' : ''}>
                      <SelectValue placeholder="Select subnet" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="subnet-1">10.0.1.0/24 - Web Tier</SelectItem>
                      <SelectItem value="subnet-2">10.0.2.0/24 - App Tier</SelectItem>
                      <SelectItem value="subnet-3">10.0.3.0/24 - DB Tier</SelectItem>
                      <SelectItem value="subnet-4">10.0.4.0/24 - Management</SelectItem>
                    </SelectContent>
                  </Select>
                  {validationErrors.subnet && (
                    <p className="text-sm text-red-500">{validationErrors.subnet}</p>
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="public-ip"
                    checked={configuration.network.publicIP}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      network: { ...prev.network, publicIP: !!checked }
                    }))}
                  />
                  <Label htmlFor="public-ip">Assign Public IP (+$5/month)</Label>
                </div>

                <div className="space-y-2">
                  <Label>Firewall Rules</Label>
                  <div className="space-y-2">
                    {['HTTP (80)', 'HTTPS (443)', 'SSH (22)', 'RDP (3389)'].map((rule) => (
                      <div key={rule} className="flex items-center space-x-2">
                        <Checkbox
                          id={rule}
                          checked={configuration.network.firewallRules.includes(rule)}
                          onCheckedChange={(checked) => {
                            if (checked) {
                              setConfiguration(prev => ({
                                ...prev,
                                network: {
                                  ...prev.network,
                                  firewallRules: [...prev.network.firewallRules, rule]
                                }
                              }));
                            } else {
                              setConfiguration(prev => ({
                                ...prev,
                                network: {
                                  ...prev.network,
                                  firewallRules: prev.network.firewallRules.filter(r => r !== rule)
                                }
                              }));
                            }
                          }}
                        />
                        <Label htmlFor={rule}>{rule}</Label>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 4: // Security
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="encryption"
                    checked={configuration.security.encryption}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      security: { ...prev.security, encryption: !!checked }
                    }))}
                  />
                  <Label htmlFor="encryption">Enable Disk Encryption</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="backup"
                    checked={configuration.security.backupEnabled}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      security: { ...prev.security, backupEnabled: !!checked }
                    }))}
                  />
                  <Label htmlFor="backup">Enable Automated Backups</Label>
                </div>

                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="monitoring"
                    checked={configuration.security.monitoringEnabled}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      security: { ...prev.security, monitoringEnabled: !!checked }
                    }))}
                  />
                  <Label htmlFor="monitoring">Enable Security Monitoring</Label>
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="auto-start"
                    checked={configuration.scheduling.autoStart}
                    onCheckedChange={(checked) => setConfiguration(prev => ({
                      ...prev,
                      scheduling: { ...prev.scheduling, autoStart: !!checked }
                    }))}
                  />
                  <Label htmlFor="auto-start">Auto-start VM on boot</Label>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="shutdown-schedule">Shutdown Schedule</Label>
                  <Select
                    value={configuration.scheduling.shutdownSchedule}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      scheduling: { ...prev.scheduling, shutdownSchedule: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select schedule" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="">No scheduled shutdown</SelectItem>
                      <SelectItem value="daily-6pm">Daily at 6:00 PM</SelectItem>
                      <SelectItem value="weekends">Weekends only</SelectItem>
                      <SelectItem value="custom">Custom schedule</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="maintenance-window">Maintenance Window</Label>
                  <Select
                    value={configuration.scheduling.maintenanceWindow}
                    onValueChange={(value) => setConfiguration(prev => ({
                      ...prev,
                      scheduling: { ...prev.scheduling, maintenanceWindow: value }
                    }))}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select window" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="sunday-2am">Sunday 2:00-4:00 AM</SelectItem>
                      <SelectItem value="saturday-midnight">Saturday Midnight-2:00 AM</SelectItem>
                      <SelectItem value="weekday-3am">Weekdays 3:00-5:00 AM</SelectItem>
                      <SelectItem value="custom">Custom window</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <Alert>
              <Shield className="h-4 w-4" />
              <AlertDescription>
                Security recommendations are automatically applied based on your template selection and network configuration.
              </AlertDescription>
            </Alert>
          </div>
        );

      case 5: // Review
        const selectedTemplateForReview = templates.find(t => t.id === configuration.template);
        const selectedTierForReview = storageTiers.find(t => t.id === configuration.resources.storageTier);
        
        return (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Configuration Summary</CardTitle>
                <CardDescription>Review your VM configuration before deployment</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-3">
                    <div>
                      <Label className="text-sm font-medium">VM Name</Label>
                      <p className="text-sm">{configuration.name}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Template</Label>
                      <p className="text-sm">{selectedTemplateForReview?.name}</p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Resources</Label>
                      <p className="text-sm">
                        {configuration.resources.cpu} cores, {configuration.resources.memory}GB RAM, 
                        {configuration.resources.storage}GB {selectedTierForReview?.name}
                      </p>
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Network</Label>
                      <p className="text-sm">
                        {configuration.network.vlan}, {configuration.network.subnet}
                        {configuration.network.publicIP && ' + Public IP'}
                      </p>
                    </div>
                  </div>
                  
                  <div className="space-y-3">
                    <div>
                      <Label className="text-sm font-medium">Security Features</Label>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {configuration.security.encryption && <Badge variant="outline">Encrypted</Badge>}
                        {configuration.security.backupEnabled && <Badge variant="outline">Backups</Badge>}
                        {configuration.security.monitoringEnabled && <Badge variant="outline">Monitoring</Badge>}
                      </div>
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Firewall Rules</Label>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {configuration.network.firewallRules.map(rule => (
                          <Badge key={rule} variant="secondary">{rule}</Badge>
                        ))}
                      </div>
                    </div>
                    <div>
                      <Label className="text-sm font-medium">Monthly Cost Estimate</Label>
                      <p className="text-lg font-bold text-green-600">
                        ${(costEstimate * 24 * 30).toFixed(2)}
                      </p>
                      <p className="text-xs text-gray-500">${costEstimate.toFixed(3)}/hour</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {isDeploying && (
              <Card>
                <CardContent className="pt-6">
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <Label>Deployment Progress</Label>
                      <span className="text-sm font-medium">{Math.round(deploymentProgress)}%</span>
                    </div>
                    <Progress value={deploymentProgress} className="w-full" />
                    <div className="text-sm text-gray-600">
                      {deploymentProgress < 20 && "Validating configuration..."}
                      {deploymentProgress >= 20 && deploymentProgress < 40 && "Allocating resources..."}
                      {deploymentProgress >= 40 && deploymentProgress < 60 && "Creating VM instance..."}
                      {deploymentProgress >= 60 && deploymentProgress < 80 && "Installing template..."}
                      {deploymentProgress >= 80 && deploymentProgress < 100 && "Configuring network..."}
                      {deploymentProgress >= 100 && "VM deployed successfully!"}
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Progress Indicator */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-bold">Create New VM</h2>
            <Badge variant="outline">
              Step {currentStep + 1} of {STEPS.length}
            </Badge>
          </div>
          
          <div className="flex items-center space-x-2">
            {STEPS.map((step, index) => {
              const Icon = step.icon;
              const isCompleted = index < currentStep;
              const isCurrent = index === currentStep;
              
              return (
                <React.Fragment key={step.id}>
                  <div
                    className={`flex items-center space-x-2 px-3 py-2 rounded-lg transition-all ${
                      isCompleted
                        ? 'bg-green-100 text-green-800'
                        : isCurrent
                        ? 'bg-blue-100 text-blue-800'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                    <span className="text-sm font-medium">{step.label}</span>
                    {isCompleted && <CheckCircle className="h-4 w-4" />}
                  </div>
                  {index < STEPS.length - 1 && (
                    <ChevronRight className="h-4 w-4 text-gray-400" />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </CardContent>
      </Card>

      {/* Step Content */}
      <Card>
        <CardHeader>
          <CardTitle>{STEPS[currentStep].label}</CardTitle>
        </CardHeader>
        <CardContent>
          {renderStepContent()}
        </CardContent>
      </Card>

      {/* Navigation Buttons */}
      <div className="flex justify-between">
        <Button
          variant="outline"
          onClick={prevStep}
          disabled={currentStep === 0}
        >
          <ChevronLeft className="mr-2 h-4 w-4" />
          Previous
        </Button>

        <div className="flex space-x-2">
          {currentStep < STEPS.length - 1 ? (
            <Button onClick={nextStep}>
              Next
              <ChevronRight className="ml-2 h-4 w-4" />
            </Button>
          ) : (
            <Button
              onClick={deployVM}
              disabled={isDeploying}
              className="bg-green-600 hover:bg-green-700"
            >
              {isDeploying ? (
                <>
                  <Settings className="mr-2 h-4 w-4 animate-spin" />
                  Deploying...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Deploy VM
                </>
              )}
            </Button>
          )}
        </div>
      </div>

      {/* Cost Estimate (always visible) */}
      <Card className="bg-gradient-to-r from-green-50 to-blue-50">
        <CardContent className="pt-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-5 w-5 text-green-600" />
              <span className="font-medium">Estimated Monthly Cost:</span>
            </div>
            <div className="text-right">
              <span className="text-2xl font-bold text-green-600">
                ${(costEstimate * 24 * 30).toFixed(2)}
              </span>
              <p className="text-xs text-gray-500">${costEstimate.toFixed(3)}/hour</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}