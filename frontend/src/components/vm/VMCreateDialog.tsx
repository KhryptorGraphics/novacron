"use client";

import { useState } from "react";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  Server, 
  HardDrive, 
  Cpu, 
  MemoryStick, 
  Network, 
  Shield,
  Settings,
  Plus,
  X
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface VMCreateDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function VMCreateDialog({ open, onOpenChange }: VMCreateDialogProps) {
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    template: "",
    os: "",
    cpu: 2,
    memory: 4,
    storage: 50,
    networkType: "bridged",
    host: "",
    autoStart: true,
    enableBackup: true,
    tags: [] as string[],
  });

  const [currentTag, setCurrentTag] = useState("");

  const templates = [
    { id: "ubuntu-22.04", name: "Ubuntu 22.04 LTS", description: "Latest LTS release" },
    { id: "centos-8", name: "CentOS Stream 8", description: "Enterprise Linux" },
    { id: "debian-11", name: "Debian 11", description: "Stable release" },
    { id: "windows-server-2022", name: "Windows Server 2022", description: "Microsoft Windows Server" },
    { id: "custom", name: "Custom ISO", description: "Upload your own ISO" }
  ];

  const hosts = [
    { id: "node-01", name: "node-01", cpu: "16 cores", memory: "64 GB", status: "available" },
    { id: "node-02", name: "node-02", cpu: "24 cores", memory: "128 GB", status: "available" },
    { id: "node-03", name: "node-03", cpu: "8 cores", memory: "32 GB", status: "maintenance" }
  ];

  const handleCreate = () => {
    // Mock VM creation - replace with API call
    console.log("Creating VM with data:", formData);
    onOpenChange(false);
    // Reset form
    setFormData({
      name: "",
      description: "",
      template: "",
      os: "",
      cpu: 2,
      memory: 4,
      storage: 50,
      networkType: "bridged",
      host: "",
      autoStart: true,
      enableBackup: true,
      tags: [],
    });
  };

  const addTag = () => {
    if (currentTag && !formData.tags.includes(currentTag)) {
      setFormData(prev => ({
        ...prev,
        tags: [...prev.tags, currentTag]
      }));
      setCurrentTag("");
    }
  };

  const removeTag = (tag: string) => {
    setFormData(prev => ({
      ...prev,
      tags: prev.tags.filter(t => t !== tag)
    }));
  };

  const getResourceColor = (value: number, max: number) => {
    const percentage = (value / max) * 100;
    if (percentage < 50) return "text-green-600";
    if (percentage < 80) return "text-yellow-600";
    return "text-red-600";
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Server className="h-5 w-5" />
            Create New Virtual Machine
          </DialogTitle>
          <DialogDescription>
            Configure and deploy a new virtual machine on your infrastructure.
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="basic">Basic</TabsTrigger>
            <TabsTrigger value="resources">Resources</TabsTrigger>
            <TabsTrigger value="network">Network</TabsTrigger>
            <TabsTrigger value="advanced">Advanced</TabsTrigger>
          </TabsList>

          <TabsContent value="basic" className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="vm-name">VM Name*</Label>
                <Input
                  id="vm-name"
                  placeholder="e.g., web-server-01"
                  value={formData.name}
                  onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="vm-template">Template*</Label>
                <Select 
                  value={formData.template} 
                  onValueChange={(value) => setFormData(prev => ({ ...prev, template: value }))}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a template" />
                  </SelectTrigger>
                  <SelectContent>
                    {templates.map(template => (
                      <SelectItem key={template.id} value={template.id}>
                        <div className="flex flex-col">
                          <span>{template.name}</span>
                          <span className="text-xs text-muted-foreground">
                            {template.description}
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="vm-description">Description</Label>
              <Textarea
                id="vm-description"
                placeholder="Optional description for this VM..."
                value={formData.description}
                onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="vm-host">Target Host*</Label>
              <Select 
                value={formData.host} 
                onValueChange={(value) => setFormData(prev => ({ ...prev, host: value }))}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Select target host" />
                </SelectTrigger>
                <SelectContent>
                  {hosts.map(host => (
                    <SelectItem 
                      key={host.id} 
                      value={host.id}
                      disabled={host.status === "maintenance"}
                    >
                      <div className="flex items-center justify-between w-full">
                        <div className="flex flex-col">
                          <span>{host.name}</span>
                          <span className="text-xs text-muted-foreground">
                            {host.cpu} • {host.memory}
                          </span>
                        </div>
                        <Badge 
                          variant={host.status === "available" ? "default" : "secondary"}
                          className="ml-2"
                        >
                          {host.status}
                        </Badge>
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </TabsContent>

          <TabsContent value="resources" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="h-4 w-4" />
                  CPU Configuration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>CPU Cores: {formData.cpu}</Label>
                    <Slider
                      value={[formData.cpu]}
                      onValueChange={(value) => setFormData(prev => ({ ...prev, cpu: value[0] }))}
                      max={16}
                      min={1}
                      step={1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>1 core</span>
                      <span className={getResourceColor(formData.cpu, 16)}>
                        {formData.cpu} / 16 cores allocated
                      </span>
                      <span>16 cores</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <MemoryStick className="h-4 w-4" />
                  Memory Configuration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Memory: {formData.memory} GB</Label>
                    <Slider
                      value={[formData.memory]}
                      onValueChange={(value) => setFormData(prev => ({ ...prev, memory: value[0] }))}
                      max={64}
                      min={1}
                      step={1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>1 GB</span>
                      <span className={getResourceColor(formData.memory, 64)}>
                        {formData.memory} / 64 GB allocated
                      </span>
                      <span>64 GB</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <HardDrive className="h-4 w-4" />
                  Storage Configuration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label>Storage: {formData.storage} GB</Label>
                    <Slider
                      value={[formData.storage]}
                      onValueChange={(value) => setFormData(prev => ({ ...prev, storage: value[0] }))}
                      max={1000}
                      min={10}
                      step={10}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>10 GB</span>
                      <span className={getResourceColor(formData.storage, 1000)}>
                        {formData.storage} / 1000 GB allocated
                      </span>
                      <span>1000 GB</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="network" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Network className="h-4 w-4" />
                  Network Configuration
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="network-type">Network Type</Label>
                  <Select 
                    value={formData.networkType} 
                    onValueChange={(value) => setFormData(prev => ({ ...prev, networkType: value }))}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="bridged">Bridged Network</SelectItem>
                      <SelectItem value="nat">NAT</SelectItem>
                      <SelectItem value="host-only">Host-Only</SelectItem>
                      <SelectItem value="isolated">Isolated</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="text-sm text-muted-foreground">
                  {formData.networkType === "bridged" && 
                    "VM will have direct access to the physical network"}
                  {formData.networkType === "nat" && 
                    "VM will share host's IP address with port forwarding"}
                  {formData.networkType === "host-only" && 
                    "VM can only communicate with host and other VMs"}
                  {formData.networkType === "isolated" && 
                    "VM has no network connectivity"}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="advanced" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Advanced Settings
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Auto Start</Label>
                    <div className="text-sm text-muted-foreground">
                      Start VM automatically when host boots
                    </div>
                  </div>
                  <Switch
                    checked={formData.autoStart}
                    onCheckedChange={(checked) => setFormData(prev => ({ ...prev, autoStart: checked }))}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="space-y-0.5">
                    <Label>Enable Backup</Label>
                    <div className="text-sm text-muted-foreground">
                      Include VM in scheduled backups
                    </div>
                  </div>
                  <Switch
                    checked={formData.enableBackup}
                    onCheckedChange={(checked) => setFormData(prev => ({ ...prev, enableBackup: checked }))}
                  />
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label>Tags</Label>
                  <div className="flex gap-2">
                    <Input
                      placeholder="Add tag..."
                      value={currentTag}
                      onChange={(e) => setCurrentTag(e.target.value)}
                      onKeyPress={(e) => e.key === "Enter" && addTag()}
                    />
                    <Button type="button" variant="outline" onClick={addTag}>
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-2">
                    {formData.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="flex items-center gap-1">
                        {tag}
                        <X 
                          className="h-3 w-3 cursor-pointer" 
                          onClick={() => removeTag(tag)}
                        />
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button 
            onClick={handleCreate}
            disabled={!formData.name || !formData.template || !formData.host}
          >
            Create VM
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}