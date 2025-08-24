"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { 
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { FadeIn } from "@/lib/animations";
import { 
  Settings, 
  Save, 
  RotateCcw, 
  Shield, 
  Database, 
  Network,
  Mail,
  Bell,
  Clock,
  HardDrive,
  Cpu,
  Memory,
  Globe,
  Lock,
  Key,
  AlertTriangle,
  CheckCircle,
  Server,
  FileText,
  Upload
} from "lucide-react";
import { cn } from "@/lib/utils";

// Configuration sections
interface ConfigSection {
  id: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  settings: ConfigSetting[];
}

interface ConfigSetting {
  key: string;
  label: string;
  description: string;
  type: 'boolean' | 'number' | 'string' | 'select' | 'slider' | 'textarea';
  value: any;
  options?: { value: string; label: string }[];
  min?: number;
  max?: number;
  unit?: string;
  sensitive?: boolean;
}

const configSections: ConfigSection[] = [
  {
    id: 'general',
    title: 'General Settings',
    description: 'Basic system configuration and preferences',
    icon: <Settings className="h-5 w-5" />,
    settings: [
      {
        key: 'system_name',
        label: 'System Name',
        description: 'Display name for the NovaCron system',
        type: 'string',
        value: 'NovaCron Production'
      },
      {
        key: 'maintenance_mode',
        label: 'Maintenance Mode',
        description: 'Enable maintenance mode to restrict access',
        type: 'boolean',
        value: false
      },
      {
        key: 'debug_mode',
        label: 'Debug Mode',
        description: 'Enable debug logging and error details',
        type: 'boolean',
        value: false
      },
      {
        key: 'session_timeout',
        label: 'Session Timeout',
        description: 'User session timeout in minutes',
        type: 'slider',
        value: [30],
        min: 5,
        max: 480,
        unit: 'minutes'
      },
      {
        key: 'timezone',
        label: 'System Timezone',
        description: 'Default timezone for the system',
        type: 'select',
        value: 'UTC',
        options: [
          { value: 'UTC', label: 'UTC' },
          { value: 'America/New_York', label: 'Eastern Time' },
          { value: 'America/Chicago', label: 'Central Time' },
          { value: 'America/Denver', label: 'Mountain Time' },
          { value: 'America/Los_Angeles', label: 'Pacific Time' },
          { value: 'Europe/London', label: 'London' },
          { value: 'Europe/Paris', label: 'Paris' },
          { value: 'Asia/Tokyo', label: 'Tokyo' }
        ]
      }
    ]
  },
  {
    id: 'security',
    title: 'Security Settings',
    description: 'Authentication, authorization, and security policies',
    icon: <Shield className="h-5 w-5" />,
    settings: [
      {
        key: 'require_2fa',
        label: 'Require Two-Factor Authentication',
        description: 'Force all users to enable 2FA',
        type: 'boolean',
        value: true
      },
      {
        key: 'password_min_length',
        label: 'Minimum Password Length',
        description: 'Minimum required password length',
        type: 'slider',
        value: [8],
        min: 6,
        max: 32,
        unit: 'characters'
      },
      {
        key: 'login_attempts',
        label: 'Max Login Attempts',
        description: 'Maximum failed login attempts before lockout',
        type: 'slider',
        value: [5],
        min: 3,
        max: 10,
        unit: 'attempts'
      },
      {
        key: 'jwt_secret',
        label: 'JWT Secret Key',
        description: 'Secret key for JWT token signing',
        type: 'string',
        value: '••••••••••••••••••••••••••••••••',
        sensitive: true
      },
      {
        key: 'api_rate_limit',
        label: 'API Rate Limit',
        description: 'Requests per minute per user',
        type: 'slider',
        value: [1000],
        min: 100,
        max: 10000,
        unit: 'req/min'
      }
    ]
  },
  {
    id: 'database',
    title: 'Database Settings',
    description: 'Database connection and performance configuration',
    icon: <Database className="h-5 w-5" />,
    settings: [
      {
        key: 'db_pool_size',
        label: 'Connection Pool Size',
        description: 'Maximum database connections',
        type: 'slider',
        value: [50],
        min: 10,
        max: 200,
        unit: 'connections'
      },
      {
        key: 'query_timeout',
        label: 'Query Timeout',
        description: 'Maximum query execution time',
        type: 'slider',
        value: [30],
        min: 5,
        max: 300,
        unit: 'seconds'
      },
      {
        key: 'enable_query_logging',
        label: 'Query Logging',
        description: 'Log all database queries for debugging',
        type: 'boolean',
        value: false
      },
      {
        key: 'backup_retention',
        label: 'Backup Retention',
        description: 'Number of days to keep backups',
        type: 'slider',
        value: [30],
        min: 7,
        max: 365,
        unit: 'days'
      }
    ]
  },
  {
    id: 'vm',
    title: 'VM Management',
    description: 'Virtual machine and resource management settings',
    icon: <Server className="h-5 w-5" />,
    settings: [
      {
        key: 'max_concurrent_migrations',
        label: 'Max Concurrent Migrations',
        description: 'Maximum number of simultaneous VM migrations',
        type: 'slider',
        value: [3],
        min: 1,
        max: 10,
        unit: 'migrations'
      },
      {
        key: 'migration_bandwidth_limit',
        label: 'Migration Bandwidth Limit',
        description: 'Bandwidth limit for VM migrations',
        type: 'slider',
        value: [1000],
        min: 100,
        max: 10000,
        unit: 'MB/s'
      },
      {
        key: 'vm_snapshot_retention',
        label: 'Snapshot Retention',
        description: 'Number of VM snapshots to retain',
        type: 'slider',
        value: [10],
        min: 3,
        max: 50,
        unit: 'snapshots'
      },
      {
        key: 'auto_scaling_enabled',
        label: 'Auto Scaling',
        description: 'Automatically scale resources based on demand',
        type: 'boolean',
        value: true
      }
    ]
  },
  {
    id: 'notifications',
    title: 'Notifications',
    description: 'Email and alert notification settings',
    icon: <Bell className="h-5 w-5" />,
    settings: [
      {
        key: 'smtp_server',
        label: 'SMTP Server',
        description: 'Email server hostname',
        type: 'string',
        value: 'smtp.novacron.io'
      },
      {
        key: 'smtp_port',
        label: 'SMTP Port',
        description: 'Email server port',
        type: 'number',
        value: 587
      },
      {
        key: 'email_notifications',
        label: 'Email Notifications',
        description: 'Send email notifications for system events',
        type: 'boolean',
        value: true
      },
      {
        key: 'notification_frequency',
        label: 'Notification Frequency',
        description: 'How often to send digest notifications',
        type: 'select',
        value: 'daily',
        options: [
          { value: 'immediate', label: 'Immediate' },
          { value: 'hourly', label: 'Hourly' },
          { value: 'daily', label: 'Daily' },
          { value: 'weekly', label: 'Weekly' }
        ]
      }
    ]
  }
];

export function SystemConfiguration() {
  const [config, setConfig] = useState<Record<string, any>>(() => {
    const initialConfig: Record<string, any> = {};
    configSections.forEach(section => {
      section.settings.forEach(setting => {
        initialConfig[setting.key] = setting.value;
      });
    });
    return initialConfig;
  });

  const [activeSection, setActiveSection] = useState('general');
  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);

  const updateSetting = (key: string, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = async () => {
    setSaving(true);
    
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    setHasChanges(false);
    setSaving(false);
  };

  const handleReset = () => {
    // Reset to original values
    const resetConfig: Record<string, any> = {};
    configSections.forEach(section => {
      section.settings.forEach(setting => {
        resetConfig[setting.key] = setting.value;
      });
    });
    setConfig(resetConfig);
    setHasChanges(false);
  };

  const renderSetting = (setting: ConfigSetting) => {
    const value = config[setting.key];

    switch (setting.type) {
      case 'boolean':
        return (
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <label className="text-sm font-medium">{setting.label}</label>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                {setting.description}
              </p>
            </div>
            <Switch
              checked={value}
              onCheckedChange={(checked) => updateSetting(setting.key, checked)}
            />
          </div>
        );

      case 'string':
        return (
          <div>
            <label className="text-sm font-medium">{setting.label}</label>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 mb-2">
              {setting.description}
            </p>
            <Input
              type={setting.sensitive ? "password" : "text"}
              value={value}
              onChange={(e) => updateSetting(setting.key, e.target.value)}
              className="w-full"
            />
          </div>
        );

      case 'number':
        return (
          <div>
            <label className="text-sm font-medium">{setting.label}</label>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 mb-2">
              {setting.description}
            </p>
            <Input
              type="number"
              value={value}
              onChange={(e) => updateSetting(setting.key, parseInt(e.target.value))}
              className="w-full"
              min={setting.min}
              max={setting.max}
            />
          </div>
        );

      case 'slider':
        return (
          <div>
            <div className="flex justify-between items-center mb-2">
              <label className="text-sm font-medium">{setting.label}</label>
              <Badge variant="outline">
                {Array.isArray(value) ? value[0] : value} {setting.unit}
              </Badge>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              {setting.description}
            </p>
            <Slider
              value={Array.isArray(value) ? value : [value]}
              onValueChange={(newValue) => updateSetting(setting.key, newValue)}
              min={setting.min}
              max={setting.max}
              step={1}
              className="w-full"
            />
          </div>
        );

      case 'select':
        return (
          <div>
            <label className="text-sm font-medium">{setting.label}</label>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 mb-2">
              {setting.description}
            </p>
            <Select value={value} onValueChange={(newValue) => updateSetting(setting.key, newValue)}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {setting.options?.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        );

      case 'textarea':
        return (
          <div>
            <label className="text-sm font-medium">{setting.label}</label>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 mb-2">
              {setting.description}
            </p>
            <Textarea
              value={value}
              onChange={(e) => updateSetting(setting.key, e.target.value)}
              className="w-full"
              rows={3}
            />
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold flex items-center gap-2">
            <Settings className="h-6 w-6" />
            System Configuration
          </h2>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Manage system settings and preferences
          </p>
        </div>
        
        <div className="flex items-center gap-2">
          {hasChanges && (
            <Badge variant="secondary">
              <AlertTriangle className="h-3 w-3 mr-1" />
              Unsaved Changes
            </Badge>
          )}
          <Button 
            variant="outline" 
            onClick={handleReset}
            disabled={!hasChanges || saving}
          >
            <RotateCcw className="h-4 w-4 mr-2" />
            Reset
          </Button>
          <Button 
            onClick={handleSave}
            disabled={!hasChanges || saving}
          >
            <Save className="h-4 w-4 mr-2" />
            {saving ? 'Saving...' : 'Save Changes'}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Settings Navigation */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Settings Categories</CardTitle>
            </CardHeader>
            <CardContent className="p-0">
              <div className="space-y-1">
                {configSections.map((section) => (
                  <button
                    key={section.id}
                    onClick={() => setActiveSection(section.id)}
                    className={cn(
                      "w-full text-left px-4 py-3 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-3",
                      activeSection === section.id && "bg-blue-50 dark:bg-blue-950 border-r-2 border-blue-500"
                    )}
                  >
                    {section.icon}
                    <div>
                      <div className="font-medium">{section.title}</div>
                      <div className="text-sm text-gray-600 dark:text-gray-400">
                        {section.description}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Settings Content */}
        <div className="lg:col-span-3">
          {configSections
            .filter(section => section.id === activeSection)
            .map((section) => (
              <FadeIn key={section.id}>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {section.icon}
                      {section.title}
                    </CardTitle>
                    <CardDescription>{section.description}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {section.settings.map((setting) => (
                      <div key={setting.key} className="p-4 border rounded-lg">
                        {renderSetting(setting)}
                      </div>
                    ))}
                  </CardContent>
                </Card>
              </FadeIn>
            ))}

          {/* Configuration Export/Import */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Configuration Management
              </CardTitle>
              <CardDescription>
                Export or import system configuration
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-4">
                <Button variant="outline">
                  <Upload className="h-4 w-4 mr-2" />
                  Export Config
                </Button>
                <Button variant="outline">
                  <Upload className="h-4 w-4 mr-2" />
                  Import Config
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}