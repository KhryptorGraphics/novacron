"use client";

import { useState, useEffect } from "react";
import { ErrorBoundary } from "@/components/error-boundary";
import { SkipToMain } from "@/components/accessibility/a11y-components";
import { ThemeToggle } from "@/components/theme/theme-toggle";
import { MobileNavigation, DesktopSidebar } from "@/components/ui/mobile-navigation";
import { MetricsCard } from "@/components/monitoring/MetricsCard";
import { VMStatusGrid } from "@/components/monitoring/VMStatusGrid";
import { Container } from "@/components/ui/layout";
import { LazyTabs } from "@/components/ui/progressive-disclosure";
import { 
  DashboardSkeleton, 
  RefreshIndicator,
  LoadingStates,
  ErrorState
} from "@/components/ui/loading-states";
import { 
  AnimatedDiv,
  StaggeredList,
  LoadingSpinner
} from "@/lib/animations";
import { 
  Activity, 
  Server, 
  Database, 
  Network,
  Shield,
  BarChart3,
  Settings,
  Bell
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useWebSocket } from "@/hooks/useAPI";

// Mock data for demonstration
const mockVMs = [
  { id: "1", name: "web-server-01", status: "running" as const, cpu: 45, memory: 62, disk: 30, network: { in: 125, out: 89 }, uptime: "5d 12h", host: "node-01" },
  { id: "2", name: "database-01", status: "running" as const, cpu: 78, memory: 85, disk: 65, network: { in: 245, out: 189 }, uptime: "12d 4h", host: "node-02" },
  { id: "3", name: "api-server-01", status: "stopped" as const, cpu: 0, memory: 0, disk: 45, network: { in: 0, out: 0 }, uptime: "-", host: "node-01" },
  { id: "4", name: "cache-server", status: "paused" as const, cpu: 12, memory: 35, disk: 20, network: { in: 45, out: 32 }, uptime: "2d 8h", host: "node-03" },
  { id: "5", name: "worker-01", status: "migrating" as const, cpu: 56, memory: 48, disk: 38, network: { in: 78, out: 65 }, uptime: "8d 16h", host: "node-02" },
];

export default function UnifiedDashboard() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notifications, setNotifications] = useState(3);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  const { connected, lastMessage } = useWebSocket();
  
  // Loading states
  const [loadingState, setLoadingState] = useState({
    isLoading: true,
    isError: false,
    isSuccess: false,
    message: ""
  });
  
  const user = {
    name: "Admin User",
    email: "admin@novacron.io",
  };
  
  // Simulate data loading
  useEffect(() => {
    const timer = setTimeout(() => {
      setLoadingState({
        isLoading: false,
        isError: false,
        isSuccess: true,
        message: ""
      });
    }, 1500);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Handle refresh
  const handleRefresh = async () => {
    setRefreshing(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    setLastUpdated(new Date());
    setRefreshing(false);
  };
  
  // Handle VM actions
  const handleVMAction = (vmId: string, action: string) => {
    console.log(`VM ${vmId}: ${action}`);
  };
  
  // Check if user has admin privileges
  const isAdmin = user.email?.includes('admin') || user.name?.toLowerCase().includes('admin');
  
  // Tab content
  const dashboardTabs = [
    {
      id: "overview",
      label: "Overview",
      icon: <Activity className="h-4 w-4" />,
      content: (
        <div className="space-y-6">
          {/* Metrics Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <FadeIn delay={0.1}>
              <MetricsCard
                title="Total VMs"
                value={<AnimatedCounter value={156} />}
                change={12}
                changeLabel="vs last week"
                status="success"
                icon={<Server className="h-5 w-5" />}
              />
            </FadeIn>
            <FadeIn delay={0.2}>
              <MetricsCard
                title="CPU Usage"
                value={<AnimatedCounter value={68} suffix="%" />}
                change={-5}
                changeLabel="vs last hour"
                status="warning"
                icon={<Activity className="h-5 w-5" />}
              />
            </FadeIn>
            <FadeIn delay={0.3}>
              <MetricsCard
                title="Memory Usage"
                value={<AnimatedCounter value={82} suffix="%" />}
                change={3}
                changeLabel="vs last hour"
                status="error"
                icon={<Database className="h-5 w-5" />}
              />
            </FadeIn>
            <FadeIn delay={0.4}>
              <MetricsCard
                title="Network I/O"
                value="2.4"
                unit="GB/s"
                change={15}
                changeLabel="vs last hour"
                status="success"
                icon={<Network className="h-5 w-5" />}
              />
            </FadeIn>
          </div>
          
          {/* VM Status Grid */}
          <FadeIn delay={0.5}>
            <VMStatusGrid
              vms={mockVMs}
              onVMAction={handleVMAction}
              loading={refreshing}
            />
          </FadeIn>
        </div>
      )
    },
    {
      id: "monitoring",
      label: "Monitoring",
      icon: <BarChart3 className="h-4 w-4" />,
      badge: notifications,
      content: async () => {
        // Simulate lazy loading
        await new Promise(resolve => setTimeout(resolve, 1000));
        return (
          <div className="p-8 text-center">
            <h3 className="text-lg font-semibold mb-2">Monitoring Dashboard</h3>
            <p className="text-gray-600 dark:text-gray-400">
              Real-time monitoring and analytics content would load here
            </p>
          </div>
        );
      }
    },
    {
      id: "security",
      label: "Security",
      icon: <Shield className="h-4 w-4" />,
      content: (
        <div className="p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">Security Center</h3>
          <p className="text-gray-600 dark:text-gray-400">
            Security policies and compliance monitoring
          </p>
        </div>
      )
    },
    // Admin tab - only visible to admin users
    ...(isAdmin ? [{
      id: "admin",
      label: "Administration",
      icon: <Shield className="h-4 w-4 text-red-600" />,
      badge: 5, // Admin notifications
      content: (
        <div className="p-8 text-center">
          <h3 className="text-lg font-semibold mb-4 text-red-600">Admin Panel</h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Elevated admin access - use with caution
          </p>
          <div className="flex justify-center">
            <Button 
              onClick={() => window.open('/admin', '_blank')}
              className="bg-red-600 hover:bg-red-700"
            >
              Open Admin Dashboard
            </Button>
          </div>
        </div>
      )
    }] : []),
    {
      id: "settings",
      label: "Settings",
      icon: <Settings className="h-4 w-4" />,
      content: (
        <div className="p-8 text-center">
          <h3 className="text-lg font-semibold mb-2">System Settings</h3>
          <p className="text-gray-600 dark:text-gray-400">
            Configure system preferences and options
          </p>
        </div>
      )
    }
  ];
  
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <SkipToMain />
        
        {/* Mobile Navigation */}
        <MobileNavigation user={user} onLogout={() => console.log("Logout")} />
        
        {/* Desktop Sidebar */}
        <DesktopSidebar 
          user={user} 
          collapsed={sidebarCollapsed}
          onCollapse={setSidebarCollapsed}
        />
        
        {/* Main Content */}
        <main
          id="main-content"
          className={cn(
            "transition-all duration-300 min-h-screen",
            "lg:ml-64",
            sidebarCollapsed && "lg:ml-16",
            "pb-16 sm:pb-0 flex flex-col" // Account for mobile bottom nav
          )}
        >
          {/* Enhanced Header with Breadcrumb */}
          <header className="sticky top-0 z-30 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 border-b">
            <Container size="full" className="py-0">
              <div className="flex flex-col gap-4 py-4 md:py-0">
                {/* Main Header Row */}
                <div className="flex items-center justify-between min-h-16">
                  <div className="flex items-center gap-4 min-w-0 flex-1">
                    <div className="space-y-1 min-w-0 flex-1">
                      <div className="flex items-center gap-3">
                        <h1 className="text-2xl md:text-3xl font-bold tracking-tight">
                          Dashboard
                        </h1>
                        {connected && (
                          <Badge variant="success" className="animate-pulse shrink-0">
                            <span className="flex items-center gap-1.5">
                              <span className="h-2 w-2 rounded-full bg-success" />
                              Live
                            </span>
                          </Badge>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground hidden sm:block">
                        Monitor and manage your VM infrastructure
                      </p>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-2 shrink-0">
                    <RefreshIndicator
                      isRefreshing={refreshing}
                      onRefresh={handleRefresh}
                      lastUpdated={lastUpdated}
                      className="hidden sm:flex"
                    />
                    
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="relative hover:bg-accent"
                      aria-label={`Notifications (${notifications} unread)`}
                    >
                      <Bell className="h-5 w-5" />
                      {notifications > 0 && (
                        <span className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-destructive text-destructive-foreground text-xs flex items-center justify-center animate-pulse">
                          {notifications}
                        </span>
                      )}
                    </Button>
                    
                    <ThemeToggle />
                  </div>
                </div>
                
                {/* Mobile Refresh Indicator */}
                <div className="sm:hidden">
                  <RefreshIndicator
                    isRefreshing={refreshing}
                    onRefresh={handleRefresh}
                    lastUpdated={lastUpdated}
                    variant="compact"
                  />
                </div>
              </div>
            </Container>
          </header>
          
          {/* Page Content with improved layout */}
          <div className="flex-1 flex flex-col">
            <Container size="full" className="flex-1 py-6 md:py-8">
              <LoadingStates
                state={loadingState}
                loadingComponent={<DashboardSkeleton />}
                errorComponent={
                  <ErrorState 
                    error="Failed to load dashboard data"
                    onRetry={() => window.location.reload()}
                  />
                }
              >
                <div className="space-y-6">
                  <LazyTabs
                    tabs={dashboardTabs}
                    defaultTab="overview"
                    className="w-full"
                    variant="enhanced"
                  />
                </div>
              </LoadingStates>
            </Container>
          </div>
        </main>
      </div>
    </ErrorBoundary>
  );
}

// Re-export cn utility
import { cn } from "@/lib/utils";