"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import { 
  Settings, 
  User, 
  Bell,
  Shield,
  Globe,
  Database,
  Mail,
  Smartphone,
  Key,
  Save,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Upload,
  Download
} from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    // General Settings
    systemName: "NovaCron Production",
    description: "Main production hypervisor cluster",
    timezone: "UTC",
    language: "en",
    
    // Notifications
    emailNotifications: true,
    pushNotifications: false,
    slackIntegration: false,
    webhookUrl: "",
    
    // Security
    twoFactorAuth: true,
    sessionTimeout: 30,
    passwordPolicy: "strong",
    auditLogging: true,
    
    // Performance
    autoOptimization: true,
    resourceThresholds: {
      cpu: 80,
      memory: 85,
      disk: 75
    },
    
    // Backup & Recovery
    autoBackup: true,
    backupRetention: 30,
    backupLocation: "/backup",
    
    // API Settings
    apiRateLimit: 1000,
    apiTimeout: 30,
    apiLogging: true
  });

  const [saved, setSaved] = useState(false);

  const handleSave = async () => {
    // Simulate save operation
    await new Promise(resolve => setTimeout(resolve, 1000));
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const updateNestedSetting = (parent: string, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [parent]: {
        ...prev[parent as keyof typeof prev],
        [key]: value
      }
    }));
  };

  return (
    <div className="container mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">System Settings</h1>
          <p className="text-muted-foreground">Configure system preferences and behavior</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline">
            <RefreshCw className="h-4 w-4 mr-2" />
            Reset to Defaults
          </Button>
          <Button onClick={handleSave}>
            <Save className="h-4 w-4 mr-2" />
            Save Changes
          </Button>
        </div>
      </div>

      {saved && (
        <Alert>
          <CheckCircle className="h-4 w-4" />
          <AlertTitle>Settings Saved</AlertTitle>
          <AlertDescription>
            Your changes have been saved successfully.
          </AlertDescription>
        </Alert>
      )}

      <Tabs defaultValue="general" className="w-full">
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="general">General</TabsTrigger>
          <TabsTrigger value="notifications">Notifications</TabsTrigger>
          <TabsTrigger value="security">Security</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="backup">Backup</TabsTrigger>
          <TabsTrigger value="api">API</TabsTrigger>
        </TabsList>

        <TabsContent value="general" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                General Settings
              </CardTitle>
              <CardDescription>Basic system configuration and preferences</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="system-name">System Name</Label>
                  <Input
                    id="system-name"
                    value={settings.systemName}
                    onChange={(e) => updateSetting('systemName', e.target.value)}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="timezone">Timezone</Label>
                  <Select value={settings.timezone} onValueChange={(value) => updateSetting('timezone', value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="UTC">UTC</SelectItem>
                      <SelectItem value="America/New_York">Eastern Time</SelectItem>
                      <SelectItem value="America/Los_Angeles">Pacific Time</SelectItem>
                      <SelectItem value="Europe/London">London</SelectItem>
                      <SelectItem value="Asia/Tokyo">Tokyo</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="description">System Description</Label>
                <Textarea
                  id="description"
                  value={settings.description}
                  onChange={(e) => updateSetting('description', e.target.value)}
                  placeholder="Describe this NovaCron instance..."
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="language">Interface Language</Label>
                <Select value={settings.language} onValueChange={(value) => updateSetting('language', value)}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="en">English</SelectItem>
                    <SelectItem value="es">Spanish</SelectItem>
                    <SelectItem value="fr">French</SelectItem>
                    <SelectItem value="de">German</SelectItem>
                    <SelectItem value="ja">Japanese</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="notifications" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Bell className="h-5 w-5" />
                Notification Settings
              </CardTitle>
              <CardDescription>Configure how you receive alerts and updates</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Email Notifications</Label>
                  <div className="text-sm text-muted-foreground">
                    Receive alerts and updates via email
                  </div>
                </div>
                <Switch
                  checked={settings.emailNotifications}
                  onCheckedChange={(checked) => updateSetting('emailNotifications', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Push Notifications</Label>
                  <div className="text-sm text-muted-foreground">
                    Browser push notifications for critical alerts
                  </div>
                </div>
                <Switch
                  checked={settings.pushNotifications}
                  onCheckedChange={(checked) => updateSetting('pushNotifications', checked)}
                />
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Slack Integration</Label>
                  <div className="text-sm text-muted-foreground">
                    Send notifications to Slack channels
                  </div>
                </div>
                <Switch
                  checked={settings.slackIntegration}
                  onCheckedChange={(checked) => updateSetting('slackIntegration', checked)}
                />
              </div>
              
              <Separator />
              
              <div className="space-y-2">
                <Label htmlFor="webhook-url">Webhook URL</Label>
                <Input
                  id="webhook-url"
                  placeholder="https://hooks.slack.com/services/..."
                  value={settings.webhookUrl}
                  onChange={(e) => updateSetting('webhookUrl', e.target.value)}
                />
                <p className="text-xs text-muted-foreground">
                  Optional webhook endpoint for custom notification integrations
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="security" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Shield className="h-5 w-5" />
                Security Settings
              </CardTitle>
              <CardDescription>Configure authentication and access controls</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Two-Factor Authentication</Label>
                  <div className="text-sm text-muted-foreground">
                    Require 2FA for all admin accounts
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Switch
                    checked={settings.twoFactorAuth}
                    onCheckedChange={(checked) => updateSetting('twoFactorAuth', checked)}
                  />
                  <Badge variant={settings.twoFactorAuth ? "default" : "secondary"}>
                    {settings.twoFactorAuth ? "Enabled" : "Disabled"}
                  </Badge>
                </div>
              </div>
              
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Audit Logging</Label>
                  <div className="text-sm text-muted-foreground">
                    Log all user actions and system events
                  </div>
                </div>
                <Switch
                  checked={settings.auditLogging}
                  onCheckedChange={(checked) => updateSetting('auditLogging', checked)}
                />
              </div>
              
              <Separator />
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="session-timeout">Session Timeout (minutes)</Label>
                  <Input
                    id="session-timeout"
                    type="number"
                    value={settings.sessionTimeout}
                    onChange={(e) => updateSetting('sessionTimeout', parseInt(e.target.value))}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="password-policy">Password Policy</Label>
                  <Select value={settings.passwordPolicy} onValueChange={(value) => updateSetting('passwordPolicy', value)}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="weak">Weak (6+ characters)</SelectItem>
                      <SelectItem value="medium">Medium (8+ chars, mixed case)</SelectItem>
                      <SelectItem value="strong">Strong (12+ chars, symbols)</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="performance" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Performance Settings
              </CardTitle>
              <CardDescription>Configure system performance and resource management</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Auto Optimization</Label>
                  <div className="text-sm text-muted-foreground">
                    Automatically optimize resource allocation
                  </div>
                </div>
                <Switch
                  checked={settings.autoOptimization}
                  onCheckedChange={(checked) => updateSetting('autoOptimization', checked)}
                />
              </div>
              
              <Separator />
              
              <div>
                <Label className="text-base font-medium">Resource Thresholds</Label>
                <p className="text-sm text-muted-foreground mb-4">
                  Alert thresholds for system resources
                </p>
                
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="cpu-threshold">CPU Warning (%)</Label>
                    <Input
                      id="cpu-threshold"
                      type="number"
                      value={settings.resourceThresholds.cpu}
                      onChange={(e) => updateNestedSetting('resourceThresholds', 'cpu', parseInt(e.target.value))}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="memory-threshold">Memory Warning (%)</Label>
                    <Input
                      id="memory-threshold"
                      type="number"
                      value={settings.resourceThresholds.memory}
                      onChange={(e) => updateNestedSetting('resourceThresholds', 'memory', parseInt(e.target.value))}
                    />
                  </div>
                  
                  <div className="space-y-2">
                    <Label htmlFor="disk-threshold">Disk Warning (%)</Label>
                    <Input
                      id="disk-threshold"
                      type="number"
                      value={settings.resourceThresholds.disk}
                      onChange={(e) => updateNestedSetting('resourceThresholds', 'disk', parseInt(e.target.value))}
                    />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="backup" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Backup & Recovery
              </CardTitle>
              <CardDescription>Configure automated backups and recovery options</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>Automatic Backup</Label>
                  <div className="text-sm text-muted-foreground">
                    Enable scheduled system backups
                  </div>
                </div>
                <Switch
                  checked={settings.autoBackup}
                  onCheckedChange={(checked) => updateSetting('autoBackup', checked)}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="backup-retention">Retention Period (days)</Label>
                  <Input
                    id="backup-retention"
                    type="number"
                    value={settings.backupRetention}
                    onChange={(e) => updateSetting('backupRetention', parseInt(e.target.value))}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="backup-location">Backup Location</Label>
                  <Input
                    id="backup-location"
                    value={settings.backupLocation}
                    onChange={(e) => updateSetting('backupLocation', e.target.value)}
                  />
                </div>
              </div>
              
              <Separator />
              
              <div className="flex gap-2">
                <Button variant="outline">
                  <Upload className="h-4 w-4 mr-2" />
                  Create Backup Now
                </Button>
                <Button variant="outline">
                  <Download className="h-4 w-4 mr-2" />
                  Download Latest Backup
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="api" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Globe className="h-5 w-5" />
                API Settings
              </CardTitle>
              <CardDescription>Configure API access and rate limiting</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="flex items-center justify-between">
                <div className="space-y-0.5">
                  <Label>API Request Logging</Label>
                  <div className="text-sm text-muted-foreground">
                    Log all API requests for debugging
                  </div>
                </div>
                <Switch
                  checked={settings.apiLogging}
                  onCheckedChange={(checked) => updateSetting('apiLogging', checked)}
                />
              </div>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="api-rate-limit">Rate Limit (requests/hour)</Label>
                  <Input
                    id="api-rate-limit"
                    type="number"
                    value={settings.apiRateLimit}
                    onChange={(e) => updateSetting('apiRateLimit', parseInt(e.target.value))}
                  />
                </div>
                
                <div className="space-y-2">
                  <Label htmlFor="api-timeout">Timeout (seconds)</Label>
                  <Input
                    id="api-timeout"
                    type="number"
                    value={settings.apiTimeout}
                    onChange={(e) => updateSetting('apiTimeout', parseInt(e.target.value))}
                  />
                </div>
              </div>
              
              <Separator />
              
              <div className="p-4 border rounded-lg bg-muted/50">
                <div className="flex items-center gap-2 mb-2">
                  <Key className="h-4 w-4" />
                  <span className="font-medium">API Keys</span>
                </div>
                <p className="text-sm text-muted-foreground mb-3">
                  Manage API keys for external integrations
                </p>
                <Button variant="outline" size="sm">
                  Generate New API Key
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}