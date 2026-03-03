'use client';

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { 
  Smartphone,
  Play,
  Pause,
  RotateCcw,
  Bell,
  Shield,
  Moon,
  Sun,
  Fingerprint,
  Download,
  WifiOff,
  Wifi,
  Activity,
  Zap,
  Settings,
  AlertTriangle,
  CheckCircle,
  RefreshCw,
  Volume2,
  Vibrate,
  Eye,
  EyeOff,
  Lock,
  Unlock
} from 'lucide-react';
import { useToast } from '@/components/ui/use-toast';

// Mock React Native-like component structure for demonstration
interface MobileVM {
  id: string;
  name: string;
  status: 'running' | 'stopped' | 'suspended' | 'error';
  cpu: number;
  memory: number;
  lastUpdated: Date;
  environment: 'production' | 'development' | 'testing';
  quickActions: string[];
}

interface MobileNotification {
  id: string;
  title: string;
  body: string;
  type: 'alert' | 'info' | 'warning' | 'success';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  timestamp: Date;
  vmId?: string;
  actionable: boolean;
  read: boolean;
}

interface MobileSettings {
  biometricAuth: boolean;
  pushNotifications: boolean;
  darkMode: boolean;
  offlineMode: boolean;
  autoSync: boolean;
  soundEnabled: boolean;
  vibrationEnabled: boolean;
  dataCompression: boolean;
  securityTimeout: number; // minutes
  gestureControls: boolean;
}

// Mock data
const mockMobileVMs: MobileVM[] = [
  {
    id: 'vm-mobile-001',
    name: 'prod-web-01',
    status: 'running',
    cpu: 45.2,
    memory: 67.8,
    lastUpdated: new Date(),
    environment: 'production',
    quickActions: ['restart', 'stop', 'monitor']
  },
  {
    id: 'vm-mobile-002',
    name: 'dev-app-02',
    status: 'stopped',
    cpu: 0,
    memory: 0,
    lastUpdated: new Date(Date.now() - 15 * 60 * 1000),
    environment: 'development',
    quickActions: ['start', 'delete', 'clone']
  },
  {
    id: 'vm-mobile-003',
    name: 'test-db-01',
    status: 'suspended',
    cpu: 12.5,
    memory: 35.4,
    lastUpdated: new Date(Date.now() - 5 * 60 * 1000),
    environment: 'testing',
    quickActions: ['resume', 'stop', 'snapshot']
  }
];

const mockNotifications: MobileNotification[] = [
  {
    id: 'notif-001',
    title: 'VM Alert',
    body: 'prod-web-01 CPU usage above 90%',
    type: 'alert',
    priority: 'urgent',
    timestamp: new Date(Date.now() - 5 * 60 * 1000),
    vmId: 'vm-mobile-001',
    actionable: true,
    read: false
  },
  {
    id: 'notif-002',
    title: 'Backup Complete',
    body: 'Scheduled backup for test-db-01 completed successfully',
    type: 'success',
    priority: 'low',
    timestamp: new Date(Date.now() - 30 * 60 * 1000),
    vmId: 'vm-mobile-003',
    actionable: false,
    read: true
  },
  {
    id: 'notif-003',
    title: 'Maintenance Window',
    body: 'Scheduled maintenance starts in 2 hours',
    type: 'info',
    priority: 'medium',
    timestamp: new Date(Date.now() - 60 * 60 * 1000),
    actionable: false,
    read: false
  }
];

// React Native-style component
export function MobileAppControls() {
  const [selectedTab, setSelectedTab] = useState('dashboard');
  const [mobileVMs, setMobileVMs] = useState<MobileVM[]>(mockMobileVMs);
  const [notifications, setNotifications] = useState<MobileNotification[]>(mockNotifications);
  const [isOffline, setIsOffline] = useState(false);
  const [lastSync, setLastSync] = useState<Date>(new Date());
  const [biometricAuthenticated, setBiometricAuthenticated] = useState(false);
  const [showBiometricPrompt, setShowBiometricPrompt] = useState(false);
  
  const [mobileSettings, setMobileSettings] = useState<MobileSettings>({
    biometricAuth: true,
    pushNotifications: true,
    darkMode: false,
    offlineMode: true,
    autoSync: true,
    soundEnabled: true,
    vibrationEnabled: true,
    dataCompression: false,
    securityTimeout: 15,
    gestureControls: true
  });

  const { toast } = useToast();

  // Simulate offline/online status
  useEffect(() => {
    const interval = setInterval(() => {
      // Randomly simulate connectivity changes
      if (Math.random() > 0.95) {
        setIsOffline(!isOffline);
      }
      
      // Auto-sync when online
      if (!isOffline && mobileSettings.autoSync) {
        setLastSync(new Date());
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [isOffline, mobileSettings.autoSync]);

  // Biometric authentication simulation
  const authenticateWithBiometric = async () => {
    setShowBiometricPrompt(true);
    
    // Simulate biometric check
    setTimeout(() => {
      const success = Math.random() > 0.1; // 90% success rate
      if (success) {
        setBiometricAuthenticated(true);
        toast({
          title: "Authentication Successful",
          description: "Biometric authentication completed",
        });
      } else {
        toast({
          title: "Authentication Failed",
          description: "Please try again or use alternative method",
          variant: "destructive"
        });
      }
      setShowBiometricPrompt(false);
    }, 2000);
  };

  // Quick VM actions
  const performQuickAction = async (vmId: string, action: string) => {
    if (!biometricAuthenticated && ['stop', 'restart', 'delete'].includes(action)) {
      await authenticateWithBiometric();
      if (!biometricAuthenticated) return;
    }

    const vm = mobileVMs.find(v => v.id === vmId);
    if (!vm) return;

    // Simulate action with haptic feedback
    if (mobileSettings.vibrationEnabled) {
      // In real React Native, this would be Haptics.impactAsync()
      console.log('Vibration feedback');
    }

    let newStatus = vm.status;
    let message = '';

    switch (action) {
      case 'start':
        newStatus = 'running';
        message = `${vm.name} is starting...`;
        break;
      case 'stop':
        newStatus = 'stopped';
        message = `${vm.name} is stopping...`;
        break;
      case 'restart':
        newStatus = 'running';
        message = `${vm.name} is restarting...`;
        break;
      case 'suspend':
        newStatus = 'suspended';
        message = `${vm.name} is suspending...`;
        break;
      case 'resume':
        newStatus = 'running';
        message = `${vm.name} is resuming...`;
        break;
    }

    // Update VM status
    setMobileVMs(prev => prev.map(v => 
      v.id === vmId ? { ...v, status: newStatus, lastUpdated: new Date() } : v
    ));

    // Show notification
    toast({
      title: "Action Initiated",
      description: message,
    });

    // Add to notifications
    const notification: MobileNotification = {
      id: `notif-${Date.now()}`,
      title: 'VM Action',
      body: message,
      type: 'info',
      priority: 'medium',
      timestamp: new Date(),
      vmId: vmId,
      actionable: false,
      read: false
    };

    setNotifications(prev => [notification, ...prev]);
  };

  // Gesture handling (simulated)
  const handleGesture = (gesture: 'swipeLeft' | 'swipeRight' | 'pinch' | 'longPress', vmId?: string) => {
    if (!mobileSettings.gestureControls) return;

    switch (gesture) {
      case 'swipeLeft':
        if (vmId) performQuickAction(vmId, 'stop');
        break;
      case 'swipeRight':
        if (vmId) performQuickAction(vmId, 'start');
        break;
      case 'pinch':
        // Zoom/focus on VM details
        break;
      case 'longPress':
        // Show context menu
        break;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running': return 'text-green-600 bg-green-50';
      case 'stopped': return 'text-red-600 bg-red-50';
      case 'suspended': return 'text-yellow-600 bg-yellow-50';
      case 'error': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'alert': return AlertTriangle;
      case 'success': return CheckCircle;
      case 'warning': return AlertTriangle;
      case 'info': return Activity;
      default: return Activity;
    }
  };

  const unreadNotifications = notifications.filter(n => !n.read).length;

  return (
    <div className="max-w-md mx-auto bg-white min-h-screen relative">
      {/* Mobile Status Bar */}
      <div className="flex items-center justify-between p-2 bg-gray-900 text-white text-sm">
        <div className="flex items-center space-x-2">
          <span>9:41</span>
          {isOffline ? <WifiOff className="h-4 w-4" /> : <Wifi className="h-4 w-4" />}
        </div>
        <div className="flex items-center space-x-1">
          <span>100%</span>
          <div className="w-6 h-3 border border-white rounded-sm">
            <div className="w-full h-full bg-green-500 rounded-sm"></div>
          </div>
        </div>
      </div>

      {/* App Header */}
      <div className="p-4 bg-blue-600 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">NovaCron Mobile</h1>
            <p className="text-blue-100 text-sm">
              {isOffline ? 'Offline Mode' : 'Connected'}
              {!isOffline && ` • Last sync: ${lastSync.toLocaleTimeString()}`}
            </p>
          </div>
          <div className="flex items-center space-x-2">
            {unreadNotifications > 0 && (
              <div className="relative">
                <Bell className="h-6 w-6" />
                <span className="absolute -top-2 -right-2 bg-red-500 text-xs rounded-full h-5 w-5 flex items-center justify-center">
                  {unreadNotifications}
                </span>
              </div>
            )}
            <Settings className="h-6 w-6" />
          </div>
        </div>
      </div>

      {/* Offline Banner */}
      {isOffline && (
        <div className="p-2 bg-yellow-100 border-b border-yellow-200">
          <div className="flex items-center space-x-2 text-yellow-800 text-sm">
            <WifiOff className="h-4 w-4" />
            <span>Working offline - some features limited</span>
          </div>
        </div>
      )}

      {/* Navigation Tabs */}
      <div className="flex border-b border-gray-200">
        {[
          { id: 'dashboard', label: 'Dashboard', icon: Activity },
          { id: 'vms', label: 'VMs', icon: Smartphone },
          { id: 'notifications', label: 'Alerts', icon: Bell },
          { id: 'settings', label: 'Settings', icon: Settings }
        ].map(tab => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id)}
              className={`flex-1 p-3 text-center border-b-2 ${
                selectedTab === tab.id 
                  ? 'border-blue-500 text-blue-600' 
                  : 'border-transparent text-gray-600'
              }`}
            >
              <Icon className="h-5 w-5 mx-auto mb-1" />
              <span className="text-xs">{tab.label}</span>
              {tab.id === 'notifications' && unreadNotifications > 0 && (
                <span className="absolute top-1 right-1 bg-red-500 text-white text-xs rounded-full h-4 w-4 flex items-center justify-center">
                  {unreadNotifications}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Content Area */}
      <div className="flex-1 overflow-y-auto pb-20">
        {selectedTab === 'dashboard' && (
          <div className="p-4 space-y-4">
            {/* Quick Stats */}
            <div className="grid grid-cols-3 gap-2">
              <Card className="p-3">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{mobileVMs.filter(vm => vm.status === 'running').length}</div>
                  <div className="text-xs text-gray-600">Running</div>
                </div>
              </Card>
              <Card className="p-3">
                <div className="text-center">
                  <div className="text-2xl font-bold text-red-600">{mobileVMs.filter(vm => vm.status === 'stopped').length}</div>
                  <div className="text-xs text-gray-600">Stopped</div>
                </div>
              </Card>
              <Card className="p-3">
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-600">{unreadNotifications}</div>
                  <div className="text-xs text-gray-600">Alerts</div>
                </div>
              </Card>
            </div>

            {/* Quick Actions */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Quick Actions</CardTitle>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-2">
                <Button variant="outline" size="sm" className="h-auto p-3">
                  <div className="text-center">
                    <Play className="h-6 w-6 mx-auto mb-1" />
                    <span className="text-xs">Start All</span>
                  </div>
                </Button>
                <Button variant="outline" size="sm" className="h-auto p-3">
                  <div className="text-center">
                    <Pause className="h-6 w-6 mx-auto mb-1" />
                    <span className="text-xs">Stop All</span>
                  </div>
                </Button>
                <Button variant="outline" size="sm" className="h-auto p-3">
                  <div className="text-center">
                    <RotateCcw className="h-6 w-6 mx-auto mb-1" />
                    <span className="text-xs">Restart All</span>
                  </div>
                </Button>
                <Button variant="outline" size="sm" className="h-auto p-3">
                  <div className="text-center">
                    <RefreshCw className="h-6 w-6 mx-auto mb-1" />
                    <span className="text-xs">Refresh</span>
                  </div>
                </Button>
              </CardContent>
            </Card>

            {/* Recent Notifications */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">Recent Alerts</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                {notifications.slice(0, 3).map(notification => {
                  const Icon = getNotificationIcon(notification.type);
                  return (
                    <div key={notification.id} className="flex items-center space-x-3 p-2 rounded-lg border">
                      <Icon className={`h-4 w-4 ${
                        notification.type === 'alert' ? 'text-red-500' :
                        notification.type === 'success' ? 'text-green-500' :
                        notification.type === 'warning' ? 'text-yellow-500' : 'text-blue-500'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{notification.title}</p>
                        <p className="text-xs text-gray-600 truncate">{notification.body}</p>
                      </div>
                      <span className="text-xs text-gray-400">
                        {Math.round((Date.now() - notification.timestamp.getTime()) / 60000)}m
                      </span>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          </div>
        )}

        {selectedTab === 'vms' && (
          <div className="p-4 space-y-3">
            {mobileVMs.map(vm => (
              <Card key={vm.id} className="relative">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <h4 className="font-semibold">{vm.name}</h4>
                      <p className="text-xs text-gray-600 capitalize">{vm.environment}</p>
                    </div>
                    <Badge className={getStatusColor(vm.status)}>
                      {vm.status}
                    </Badge>
                  </div>
                  
                  {vm.status === 'running' && (
                    <div className="space-y-2 mb-3">
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span>CPU</span>
                          <span>{vm.cpu}%</span>
                        </div>
                        <Progress value={vm.cpu} className="h-2" />
                      </div>
                      <div>
                        <div className="flex justify-between text-xs mb-1">
                          <span>Memory</span>
                          <span>{vm.memory}%</span>
                        </div>
                        <Progress value={vm.memory} className="h-2" />
                      </div>
                    </div>
                  )}
                  
                  <div className="flex gap-2">
                    {vm.quickActions.map(action => (
                      <Button
                        key={action}
                        variant="outline"
                        size="sm"
                        className="flex-1 text-xs"
                        onClick={() => performQuickAction(vm.id, action)}
                        onTouchStart={() => handleGesture('longPress', vm.id)}
                      >
                        {action === 'start' && <Play className="h-3 w-3 mr-1" />}
                        {action === 'stop' && <Pause className="h-3 w-3 mr-1" />}
                        {action === 'restart' && <RotateCcw className="h-3 w-3 mr-1" />}
                        {action === 'resume' && <Play className="h-3 w-3 mr-1" />}
                        <span className="capitalize">{action}</span>
                      </Button>
                    ))}
                  </div>
                  
                  <div className="text-xs text-gray-500 mt-2">
                    Updated {vm.lastUpdated.toLocaleTimeString()}
                  </div>
                </CardContent>
                
                {/* Gesture indicators */}
                {mobileSettings.gestureControls && (
                  <div className="absolute top-2 right-2 text-xs text-gray-400">
                    <span>← stop | start →</span>
                  </div>
                )}
              </Card>
            ))}
          </div>
        )}

        {selectedTab === 'notifications' && (
          <div className="p-4 space-y-3">
            {notifications.map(notification => {
              const Icon = getNotificationIcon(notification.type);
              return (
                <Card key={notification.id} className={`${!notification.read ? 'ring-2 ring-blue-500' : ''}`}>
                  <CardContent className="p-4">
                    <div className="flex items-start space-x-3">
                      <Icon className={`h-5 w-5 mt-0.5 ${
                        notification.type === 'alert' ? 'text-red-500' :
                        notification.type === 'success' ? 'text-green-500' :
                        notification.type === 'warning' ? 'text-yellow-500' : 'text-blue-500'
                      }`} />
                      <div className="flex-1">
                        <div className="flex items-start justify-between">
                          <h4 className="font-semibold">{notification.title}</h4>
                          <div className="text-right">
                            <Badge variant={notification.priority === 'urgent' ? 'destructive' : 'secondary'} className="text-xs">
                              {notification.priority}
                            </Badge>
                            <p className="text-xs text-gray-500 mt-1">
                              {notification.timestamp.toLocaleTimeString()}
                            </p>
                          </div>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">{notification.body}</p>
                        {notification.actionable && (
                          <div className="flex gap-2 mt-3">
                            <Button size="sm" variant="outline">View</Button>
                            <Button size="sm">Take Action</Button>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        )}

        {selectedTab === 'settings' && (
          <div className="p-4 space-y-4">
            {/* Security Settings */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center">
                  <Shield className="h-5 w-5 mr-2" />
                  Security
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Fingerprint className="h-4 w-4" />
                    <span className="text-sm">Biometric Authentication</span>
                  </div>
                  <Switch
                    checked={mobileSettings.biometricAuth}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, biometricAuth: checked }))}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Lock className="h-4 w-4" />
                    <span className="text-sm">Auto-lock timeout</span>
                  </div>
                  <select 
                    value={mobileSettings.securityTimeout}
                    onChange={(e) => setMobileSettings(prev => ({ ...prev, securityTimeout: parseInt(e.target.value) }))}
                    className="text-sm border rounded px-2 py-1"
                  >
                    <option value={5}>5 min</option>
                    <option value={15}>15 min</option>
                    <option value={30}>30 min</option>
                    <option value={60}>1 hour</option>
                  </select>
                </div>
                
                {!biometricAuthenticated && mobileSettings.biometricAuth && (
                  <Button 
                    variant="outline" 
                    size="sm" 
                    onClick={authenticateWithBiometric}
                    disabled={showBiometricPrompt}
                  >
                    {showBiometricPrompt ? (
                      <>
                        <RefreshCw className="h-4 w-4 mr-2 animate-spin" />
                        Authenticating...
                      </>
                    ) : (
                      <>
                        <Fingerprint className="h-4 w-4 mr-2" />
                        Authenticate
                      </>
                    )}
                  </Button>
                )}
              </CardContent>
            </Card>

            {/* Notification Settings */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center">
                  <Bell className="h-5 w-5 mr-2" />
                  Notifications
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Bell className="h-4 w-4" />
                    <span className="text-sm">Push Notifications</span>
                  </div>
                  <Switch
                    checked={mobileSettings.pushNotifications}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, pushNotifications: checked }))}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Volume2 className="h-4 w-4" />
                    <span className="text-sm">Sound</span>
                  </div>
                  <Switch
                    checked={mobileSettings.soundEnabled}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, soundEnabled: checked }))}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Vibrate className="h-4 w-4" />
                    <span className="text-sm">Vibration</span>
                  </div>
                  <Switch
                    checked={mobileSettings.vibrationEnabled}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, vibrationEnabled: checked }))}
                  />
                </div>
              </CardContent>
            </Card>

            {/* App Settings */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-lg">App Settings</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    {mobileSettings.darkMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
                    <span className="text-sm">Dark Mode</span>
                  </div>
                  <Switch
                    checked={mobileSettings.darkMode}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, darkMode: checked }))}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <WifiOff className="h-4 w-4" />
                    <span className="text-sm">Offline Mode</span>
                  </div>
                  <Switch
                    checked={mobileSettings.offlineMode}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, offlineMode: checked }))}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <RefreshCw className="h-4 w-4" />
                    <span className="text-sm">Auto Sync</span>
                  </div>
                  <Switch
                    checked={mobileSettings.autoSync}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, autoSync: checked }))}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Zap className="h-4 w-4" />
                    <span className="text-sm">Gesture Controls</span>
                  </div>
                  <Switch
                    checked={mobileSettings.gestureControls}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, gestureControls: checked }))}
                  />
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Download className="h-4 w-4" />
                    <span className="text-sm">Data Compression</span>
                  </div>
                  <Switch
                    checked={mobileSettings.dataCompression}
                    onCheckedChange={(checked) => setMobileSettings(prev => ({ ...prev, dataCompression: checked }))}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Biometric Prompt Overlay */}
      {showBiometricPrompt && (
        <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <Card className="mx-4 w-full max-w-sm">
            <CardContent className="pt-6 text-center">
              <Fingerprint className="h-16 w-16 mx-auto mb-4 text-blue-600 animate-pulse" />
              <h3 className="text-lg font-semibold mb-2">Biometric Authentication</h3>
              <p className="text-gray-600 mb-4">Touch the fingerprint sensor to authenticate</p>
              <Button variant="outline" onClick={() => setShowBiometricPrompt(false)}>
                Cancel
              </Button>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Gesture Hints */}
      {mobileSettings.gestureControls && selectedTab === 'vms' && (
        <div className="fixed bottom-4 left-4 right-4 bg-black bg-opacity-75 text-white text-xs p-2 rounded-lg">
          <p>💡 Swipe left to stop, right to start, long press for menu</p>
        </div>
      )}
    </div>
  );
}