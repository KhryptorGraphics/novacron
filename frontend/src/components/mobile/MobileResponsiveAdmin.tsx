import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Menu, Home, Server, Activity, HardDrive, Network, Shield,
  Bell, Settings, User, Search, ChevronRight, Cpu, MemoryStick,
  Gauge, TrendingUp, AlertCircle, CheckCircle, Plus, Play,
  Pause, Square, RefreshCw, Wifi, WifiOff, Battery, Signal
} from 'lucide-react';
import { useMonitoringWebSocket } from '@/hooks/useWebSocket';

// Types for mobile-optimized data structures
interface MobileMetrics {
  totalVMs: number;
  runningVMs: number;
  stoppedVMs: number;
  cpuUsage: number;
  memoryUsage: number;
  storageUsage: number;
  networkStatus: 'online' | 'degraded' | 'offline';
  alerts: number;
  criticalAlerts: number;
}

interface QuickAction {
  id: string;
  icon: React.ElementType;
  label: string;
  description: string;
  action: () => void;
  disabled?: boolean;
}

interface MobileAlert {
  id: string;
  type: 'info' | 'warning' | 'error' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
}

export const MobileResponsiveAdmin: React.FC = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [activeView, setActiveView] = useState('dashboard');
  const [isOnline, setIsOnline] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [notifications, setNotifications] = useState<MobileAlert[]>([]);
  const [offlineQueue, setOfflineQueue] = useState<Array<{ action: string; data: any; timestamp: number }>>([]);

  // WebSocket for real-time updates
  const { data: wsData, isConnected } = useMonitoringWebSocket();

  // Mock data for mobile interface, update with WebSocket data
  const [mobileMetrics, setMobileMetrics] = useState<MobileMetrics>({
    totalVMs: 47,
    runningVMs: 42,
    stoppedVMs: 5,
    cpuUsage: 68,
    memoryUsage: 72,
    storageUsage: 54,
    networkStatus: 'online',
    alerts: 3,
    criticalAlerts: 0,
  });

  // Update metrics with WebSocket data
  useEffect(() => {
    if (wsData && isConnected) {
      setMobileMetrics(prev => ({
        ...prev,
        ...wsData.metrics,
        networkStatus: 'online',
      }));

      // Add new notifications from WebSocket
      if (wsData.alerts && wsData.alerts.length > 0) {
        setNotifications(prev => [
          ...wsData.alerts.map((alert: any) => ({
            id: alert.id || Math.random().toString(),
            type: alert.severity || 'info',
            title: alert.title || 'System Alert',
            message: alert.message || 'New system event',
            timestamp: alert.timestamp || new Date().toISOString(),
            read: false,
          })),
          ...prev,
        ]);
      }
    }
  }, [wsData, isConnected]);

  // Mock alerts/notifications
  useEffect(() => {
    setNotifications([
      {
        id: '1',
        type: 'warning',
        title: 'High CPU Usage',
        message: 'CPU usage is at 85% on cluster-01',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        read: false,
      },
      {
        id: '2',
        type: 'info',
        title: 'VM Migration Complete',
        message: 'VM-WEB-03 successfully migrated to node-04',
        timestamp: new Date(Date.now() - 600000).toISOString(),
        read: false,
      },
      {
        id: '3',
        type: 'success',
        title: 'Backup Complete',
        message: 'All VMs backed up successfully',
        timestamp: new Date(Date.now() - 1800000).toISOString(),
        read: true,
      },
    ]);
  }, []);

  // Queue action for offline handling
  const queueAction = (action: string, data: any) => {
    if (!isOnline) {
      setOfflineQueue(prev => [
        ...prev,
        { action, data, timestamp: Date.now() }
      ]);
      return true; // Indicate action was queued
    }
    return false; // Indicate action should proceed normally
  };

  // Quick actions for mobile
  const quickActions: QuickAction[] = [
    {
      id: 'create-vm',
      icon: Plus,
      label: 'Create VM',
      description: 'Deploy new virtual machine',
      action: () => {
        if (queueAction('create-vm', {})) {
          // Show queued message
        } else {
          console.log('Create VM');
        }
      },
    },
    {
      id: 'backup',
      icon: HardDrive,
      label: 'Backup',
      description: 'Start system backup',
      action: () => {
        if (queueAction('backup', {})) {
          // Show queued message
        } else {
          console.log('Start backup');
        }
      },
    },
    {
      id: 'health-check',
      icon: Activity,
      label: 'Health Check',
      description: 'Run system diagnostics',
      action: () => {
        if (queueAction('health-check', {})) {
          // Show queued message
        } else {
          console.log('Health check');
        }
      },
    },
    {
      id: 'security-scan',
      icon: Shield,
      label: 'Security',
      description: 'Security scan',
      action: () => {
        if (queueAction('security-scan', {})) {
          // Show queued message
        } else {
          console.log('Security scan');
        }
      },
    },
  ];

  // Navigation items
  const navigationItems = [
    { id: 'dashboard', icon: Home, label: 'Dashboard' },
    { id: 'vms', icon: Server, label: 'Virtual Machines' },
    { id: 'monitoring', icon: Activity, label: 'Monitoring' },
    { id: 'storage', icon: HardDrive, label: 'Storage' },
    { id: 'network', icon: Network, label: 'Network' },
    { id: 'security', icon: Shield, label: 'Security' },
  ];

  // Handle refresh action
  const handleRefresh = async () => {
    setRefreshing(true);
    // Simulate refresh delay
    setTimeout(() => setRefreshing(false), 1000);
  };

  // Handle network status changes and offline queue
  useEffect(() => {
    const handleOnline = () => {
      setIsOnline(true);
      // Process offline queue when coming back online
      if (offlineQueue.length > 0) {
        console.log('Processing offline queue:', offlineQueue);
        // Here you would process the queued actions
        setOfflineQueue([]);
      }
    };
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, [offlineQueue]);

  // Service worker for PWA functionality (only register if sw.js exists)
  useEffect(() => {
    if ('serviceWorker' in navigator && process.env.NODE_ENV === 'production') {
      fetch('/sw.js')
        .then(response => {
          if (response.ok) {
            navigator.serviceWorker.register('/sw.js')
              .then((registration) => {
                console.log('SW registered: ', registration);
              })
              .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
              });
          }
        })
        .catch(() => {
          // sw.js not available, skip registration
        });
    }
  }, []);

  // Format timestamp for mobile display
  const formatMobileTime = (timestamp: string) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diff = now.getTime() - time.getTime();

    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
    return time.toLocaleDateString();
  };

  // Get alert icon
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error': return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      case 'success': return <CheckCircle className="h-4 w-4 text-green-500" />;
      default: return <AlertCircle className="h-4 w-4 text-blue-500" />;
    }
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Mobile Header */}
      <div className="sticky top-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
        <div className="flex items-center justify-between p-4">
          <Sheet open={isMenuOpen} onOpenChange={setIsMenuOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-[280px]">
              <SheetHeader>
                <SheetTitle>NovaCron Mobile</SheetTitle>
              </SheetHeader>
              <ScrollArea className="h-full mt-6">
                <div className="space-y-2">
                  {navigationItems.map((item) => {
                    const Icon = item.icon;
                    return (
                      <Button
                        key={item.id}
                        variant={activeView === item.id ? 'default' : 'ghost'}
                        className="w-full justify-start"
                        onClick={() => {
                          setActiveView(item.id);
                          setIsMenuOpen(false);
                        }}
                      >
                        <Icon className="h-4 w-4 mr-3" />
                        {item.label}
                      </Button>
                    );
                  })}
                </div>

                {/* Mobile User Section */}
                <div className="mt-8 p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="h-10 w-10 rounded-full bg-primary flex items-center justify-center">
                      <User className="h-5 w-5 text-primary-foreground" />
                    </div>
                    <div>
                      <p className="font-medium">Admin User</p>
                      <p className="text-sm text-muted-foreground">admin@novacron.dev</p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm" className="w-full mt-3">
                    <Settings className="h-4 w-4 mr-2" />
                    Settings
                  </Button>
                </div>
              </ScrollArea>
            </SheetContent>
          </Sheet>

          <div className="flex items-center gap-2">
            <h1 className="text-lg font-semibold">NovaCron</h1>
            <Badge variant="outline" className="text-xs">
              Mobile
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            {/* Network Status */}
            {isOnline ? (
              <div className="flex items-center gap-1">
                <Wifi className="h-4 w-4 text-green-500" />
                {isConnected && <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />}
              </div>
            ) : (
              <WifiOff className="h-4 w-4 text-red-500" />
            )}

            {/* Refresh Button */}
            <Button
              variant="ghost"
              size="icon"
              onClick={handleRefresh}
              disabled={refreshing}
            >
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
            </Button>

            {/* Notifications */}
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" className="relative">
                  <Bell className="h-4 w-4" />
                  {notifications.filter(n => !n.read).length > 0 && (
                    <div className="absolute -top-1 -right-1 h-4 w-4 bg-red-500 rounded-full text-xs text-white flex items-center justify-center">
                      {notifications.filter(n => !n.read).length}
                    </div>
                  )}
                </Button>
              </SheetTrigger>
              <SheetContent>
                <SheetHeader>
                  <SheetTitle>Notifications</SheetTitle>
                </SheetHeader>
                <ScrollArea className="h-full mt-6">
                  <div className="space-y-4">
                    {notifications.map((alert) => (
                      <div
                        key={alert.id}
                        className={`p-3 rounded-lg border ${alert.read ? 'opacity-60' : 'bg-accent/50'}`}
                      >
                        <div className="flex items-start gap-3">
                          {getAlertIcon(alert.type)}
                          <div className="flex-1">
                            <h4 className="font-medium text-sm">{alert.title}</h4>
                            <p className="text-sm text-muted-foreground mt-1">{alert.message}</p>
                            <p className="text-xs text-muted-foreground mt-2">
                              {formatMobileTime(alert.timestamp)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </SheetContent>
            </Sheet>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="p-4 space-y-4">
        {activeView === 'dashboard' && (
          <>
            {/* System Status Cards */}
            <div className="grid grid-cols-2 gap-4">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">VMs</p>
                      <p className="text-2xl font-bold">{mobileMetrics.totalVMs}</p>
                    </div>
                    <Server className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <div className="flex gap-1 mt-2">
                    <Badge variant="default" className="text-xs">
                      {mobileMetrics.runningVMs} On
                    </Badge>
                    <Badge variant="outline" className="text-xs">
                      {mobileMetrics.stoppedVMs} Off
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">CPU</p>
                      <p className="text-2xl font-bold">{mobileMetrics.cpuUsage}%</p>
                    </div>
                    <Cpu className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <Progress value={mobileMetrics.cpuUsage} className="mt-2 h-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">Memory</p>
                      <p className="text-2xl font-bold">{mobileMetrics.memoryUsage}%</p>
                    </div>
                    <MemoryStick className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <Progress value={mobileMetrics.memoryUsage} className="mt-2 h-2" />
                </CardContent>
              </Card>

              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">Storage</p>
                      <p className="text-2xl font-bold">{mobileMetrics.storageUsage}%</p>
                    </div>
                    <HardDrive className="h-8 w-8 text-muted-foreground" />
                  </div>
                  <Progress value={mobileMetrics.storageUsage} className="mt-2 h-2" />
                </CardContent>
              </Card>
            </div>

            {/* Quick Actions */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Quick Actions</CardTitle>
                <CardDescription className="text-sm">Common operations</CardDescription>
              </CardHeader>
              <CardContent className="p-4 pt-0">
                <div className="grid grid-cols-2 gap-3">
                  {quickActions.map((action) => {
                    const Icon = action.icon;
                    return (
                      <Button
                        key={action.id}
                        variant="outline"
                        className="h-auto p-4 flex flex-col gap-2"
                        onClick={action.action}
                        disabled={action.disabled}
                      >
                        <Icon className="h-6 w-6" />
                        <div className="text-center">
                          <div className="text-sm font-medium">{action.label}</div>
                          <div className="text-xs text-muted-foreground">
                            {action.description}
                          </div>
                        </div>
                      </Button>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* System Health */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">System Health</CardTitle>
              </CardHeader>
              <CardContent className="p-4 pt-0">
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      <span className="text-sm">All Systems</span>
                    </div>
                    <Badge variant="default">Operational</Badge>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Network className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Network</span>
                    </div>
                    <Badge variant="default">Online</Badge>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Shield className="h-4 w-4 text-green-500" />
                      <span className="text-sm">Security</span>
                    </div>
                    <Badge variant="default">Protected</Badge>
                  </div>

                  {mobileMetrics.alerts > 0 && (
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <AlertCircle className="h-4 w-4 text-yellow-500" />
                        <span className="text-sm">Alerts</span>
                      </div>
                      <Badge variant="secondary">{mobileMetrics.alerts} Active</Badge>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Recent Activity */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Recent Activity</CardTitle>
              </CardHeader>
              <CardContent className="p-4 pt-0">
                <div className="space-y-3">
                  <div className="flex items-center gap-3 p-2 rounded bg-accent/50">
                    <CheckCircle className="h-4 w-4 text-green-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">VM Migration Complete</p>
                      <p className="text-xs text-muted-foreground">5 minutes ago</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 p-2 rounded bg-accent/50">
                    <HardDrive className="h-4 w-4 text-blue-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">Backup Completed</p>
                      <p className="text-xs text-muted-foreground">2 hours ago</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-3 p-2 rounded bg-accent/50">
                    <Shield className="h-4 w-4 text-purple-500 flex-shrink-0" />
                    <div className="flex-1">
                      <p className="text-sm font-medium">Security Scan</p>
                      <p className="text-xs text-muted-foreground">4 hours ago</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}

        {/* Other Views - Simplified for Mobile */}
        {activeView === 'vms' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Virtual Machines</CardTitle>
              <CardDescription>Manage your VM infrastructure</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <Server className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">VM management interface</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Optimized for mobile interactions
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {activeView === 'monitoring' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">System Monitoring</CardTitle>
              <CardDescription>Real-time performance metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <Activity className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Monitoring dashboard</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Touch-optimized charts and metrics
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {activeView === 'storage' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Storage Management</CardTitle>
              <CardDescription>Manage storage resources</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <HardDrive className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Storage management</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Mobile-friendly storage controls
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {activeView === 'network' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Network Configuration</CardTitle>
              <CardDescription>Network settings and monitoring</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <Network className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Network management</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Simplified network configuration
                </p>
              </div>
            </CardContent>
          </Card>
        )}

        {activeView === 'security' && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Security & Compliance</CardTitle>
              <CardDescription>Security monitoring and policies</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <Shield className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                <p className="text-muted-foreground">Security dashboard</p>
                <p className="text-sm text-muted-foreground mt-2">
                  Mobile security management
                </p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Mobile Bottom Navigation (Alternative) */}
      <div className="sm:hidden fixed bottom-0 left-0 right-0 bg-background border-t">
        <div className="grid grid-cols-4 gap-1 p-2">
          {navigationItems.slice(0, 4).map((item) => {
            const Icon = item.icon;
            return (
              <Button
                key={item.id}
                variant={activeView === item.id ? 'default' : 'ghost'}
                size="sm"
                className="flex flex-col gap-1 h-auto py-2"
                onClick={() => setActiveView(item.id)}
              >
                <Icon className="h-4 w-4" />
                <span className="text-xs">{item.label.split(' ')[0]}</span>
              </Button>
            );
          })}
        </div>
      </div>

      {/* Offline Indicator */}
      {!isOnline && (
        <div className="fixed bottom-20 left-4 right-4 bg-destructive text-destructive-foreground p-3 rounded-lg text-center text-sm">
          <WifiOff className="h-4 w-4 inline mr-2" />
          You're offline. {offlineQueue.length > 0 && `${offlineQueue.length} actions queued.`}
        </div>
      )}

      {/* PWA Install Banner (Hidden by default) */}
      <div className="hidden fixed bottom-20 left-4 right-4 bg-primary text-primary-foreground p-3 rounded-lg">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium">Install NovaCron</p>
            <p className="text-xs opacity-90">Add to home screen for quick access</p>
          </div>
          <Button size="sm" variant="secondary">
            Install
          </Button>
        </div>
      </div>
    </div>
  );
};

export default MobileResponsiveAdmin;