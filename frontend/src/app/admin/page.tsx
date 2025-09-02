"use client";

// Disable static generation for this page
export const dynamic = 'force-dynamic';

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useToast } from "@/components/ui/use-toast";
import { ErrorBoundary } from "@/components/error-boundary";
import { SkipToMain } from "@/components/accessibility/a11y-components";
import { ThemeToggle } from "@/components/theme/theme-toggle";
import { MobileNavigation, DesktopSidebar } from "@/components/ui/mobile-navigation";
import { LazyTabs } from "@/components/ui/progressive-disclosure";
import { 
  DashboardSkeleton, 
  RefreshIndicator,
  LoadingStates 
} from "@/components/ui/loading-states";
import { FadeIn } from "@/lib/animations";
import { 
  Settings,
  Users,
  Database,
  Shield,
  BarChart3,
  Server,
  Bell,
  UserCheck,
  Key,
  FileText,
  AlertTriangle,
  Activity
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

// Admin Components
import { DatabaseEditor } from "@/components/admin/DatabaseEditor";
import { UserManagement } from "@/components/admin/UserManagement";
import { SystemConfiguration } from "@/components/admin/SystemConfiguration";
import { SecurityDashboard } from "@/components/admin/SecurityDashboard";
import { AdminMetrics } from "@/components/admin/AdminMetrics";
import { AuditLogs } from "@/components/admin/AuditLogs";
import { RolePermissionManager } from "@/components/admin/RolePermissionManager";
import { RealTimeDashboard } from "@/components/admin/RealTimeDashboard";

export default function AdminDashboard() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [notifications, setNotifications] = useState(5);
  const [refreshing, setRefreshing] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(new Date());
  
  // Loading states
  const [loadingState, setLoadingState] = useState({
    isLoading: true,
    isError: false,
    isSuccess: false,
    message: ""
  });
  
  const user = {
    name: "System Administrator",
    email: "admin@novacron.io",
    role: "admin"
  };

  const router = useRouter();
  const { toast } = useToast();
  const handleLogout = () => {
    try { localStorage.removeItem("authToken"); } catch {}
    toast({ title: "Logged out" });
    router.push("/auth/login");
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
    }, 1000);
    
    return () => clearTimeout(timer);
  }, []);
  
  // Handle refresh
  const handleRefresh = async () => {
    setRefreshing(true);
    await new Promise(resolve => setTimeout(resolve, 2000));
    setLastUpdated(new Date());
    setRefreshing(false);
  };
  
  // Admin tab configuration
  const adminTabs = [
    {
      id: "overview",
      label: "Overview",
      icon: <BarChart3 className="h-4 w-4" />,
      content: (
        <FadeIn>
          <AdminMetrics />
        </FadeIn>
      )
    },
    {
      id: "realtime",
      label: "Real-time Monitor",
      icon: <Activity className="h-4 w-4" />,
      content: (
        <FadeIn delay={0.1}>
          <RealTimeDashboard />
        </FadeIn>
      )
    },
    {
      id: "users",
      label: "User Management",
      icon: <Users className="h-4 w-4" />,
      badge: 3, // Pending user requests
      content: (
        <FadeIn delay={0.1}>
          <UserManagement />
        </FadeIn>
      )
    },
    {
      id: "database",
      label: "Database Editor",
      icon: <Database className="h-4 w-4" />,
      content: (
        <FadeIn delay={0.2}>
          <DatabaseEditor />
        </FadeIn>
      )
    },
    {
      id: "security",
      label: "Security",
      icon: <Shield className="h-4 w-4" />,
      badge: 2, // Security alerts
      content: (
        <FadeIn delay={0.3}>
          <SecurityDashboard />
        </FadeIn>
      )
    },
    {
      id: "permissions",
      label: "Roles & Permissions",
      icon: <UserCheck className="h-4 w-4" />,
      content: (
        <FadeIn delay={0.4}>
          <RolePermissionManager />
        </FadeIn>
      )
    },
    {
      id: "audit",
      label: "Audit Logs",
      icon: <FileText className="h-4 w-4" />,
      content: (
        <FadeIn delay={0.5}>
          <AuditLogs />
        </FadeIn>
      )
    },
    {
      id: "settings",
      label: "System Config",
      icon: <Settings className="h-4 w-4" />,
      content: (
        <FadeIn delay={0.6}>
          <SystemConfiguration />
        </FadeIn>
      )
    }
  ];
  
  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
        <SkipToMain />
        
        {/* Mobile Navigation */}
        <MobileNavigation user={user} onLogout={handleLogout} />

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
            "transition-all duration-300",
            "lg:ml-64",
            sidebarCollapsed && "lg:ml-16",
            "pb-16 sm:pb-0"
          )}
        >
          {/* Header */}
          <header className="sticky top-0 z-30 bg-white dark:bg-gray-900 border-b dark:border-gray-800">
            <div className="px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between h-16">
                <div className="flex items-center gap-4">
                  <h1 className="text-2xl font-bold text-red-600 dark:text-red-400">
                    Admin Dashboard
                  </h1>
                  <Badge variant="destructive" className="animate-pulse">
                    <AlertTriangle className="h-3 w-3 mr-1" />
                    Elevated Access
                  </Badge>
                </div>
                
                <div className="flex items-center gap-2">
                  <RefreshIndicator
                    isRefreshing={refreshing}
                    onRefresh={handleRefresh}
                    lastUpdated={lastUpdated}
                  />
                  
                  <Button variant="ghost" size="icon" className="relative">
                    <Bell className="h-5 w-5" />
                    {notifications > 0 && (
                      <span className="absolute -top-1 -right-1 h-5 w-5 rounded-full bg-red-500 text-white text-xs flex items-center justify-center">
                        {notifications}
                      </span>
                    )}
                  </Button>
                  
                  <ThemeToggle />
                </div>
              </div>
            </div>
          </header>
          
          {/* Page Content */}
          <div className="px-4 sm:px-6 lg:px-8 py-8">
            <LoadingStates
              state={loadingState}
              loadingComponent={<DashboardSkeleton />}
            >
              <LazyTabs
                tabs={adminTabs}
                defaultTab="realtime"
                className="w-full"
              />
            </LoadingStates>
          </div>
        </main>
      </div>
    </ErrorBoundary>
  );
}