"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { cn } from "@/lib/utils";
import {
  Menu,
  X,
  Home,
  Server,
  Activity,
  Settings,
  Database,
  Network,
  Shield,
  BarChart3,
  Users,
  LogOut,
  ChevronRight,
  Moon,
  Sun
} from "lucide-react";
import { Button } from "./button";
import { motion, AnimatePresence } from "framer-motion";

interface MobileNavigationProps {
  user?: {
    name: string;
    email: string;
    avatar?: string;
  };
  onLogout?: () => void;
}

const navigationItems = [
  {
    name: "Dashboard",
    href: "/dashboard",
    icon: Home,
    description: "Overview and metrics"
  },
  {
    name: "VMs",
    href: "/vms",
    icon: Server,
    description: "Virtual machine management"
  },
  {
    name: "Core VMs",
    href: "/core/vms",
    icon: Server,
    description: "Core-mode VM list"
  },
  {
    name: "Monitoring",
    href: "/monitoring",
    icon: Activity,
    description: "Real-time system monitoring"
  },
  {
    name: "Storage",
    href: "/storage",
    icon: Database,
    description: "Storage management"
  },
  {
    name: "Network",
    href: "/network",
    icon: Network,
    description: "Network configuration"
  },
  {
    name: "Security",
    href: "/security",
    icon: Shield,
    description: "Security settings"
  },
  {
    name: "Analytics",
    href: "/analytics",
    icon: BarChart3,
    description: "Performance analytics"
  },
  {
    name: "Users",
    href: "/users",
    icon: Users,
    description: "User management"
  },
  {
    name: "Settings",
    href: "/settings",
    icon: Settings,
    description: "System settings"
  },
  {
    name: "Admin",
    href: "/admin",
    icon: Shield,
    description: "Admin panel"
  },
];


export function MobileNavigation({ user, onLogout }: MobileNavigationProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const pathname = usePathname();

  // Close menu on route change
  useEffect(() => {
    setIsOpen(false);
  }, [pathname]);

  // Prevent body scroll when menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }
    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    // In a real app, this would update the theme context
    document.documentElement.setAttribute(
      "data-theme",
      isDarkMode ? "light" : "dark"
    );
  };

  return (
    <>
      {/* Mobile Header Bar */}
      <div className="sticky top-0 z-40 flex h-16 items-center gap-4 border-b bg-white px-4 dark:bg-gray-900 dark:border-gray-800 lg:hidden">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setIsOpen(!isOpen)}
          className="lg:hidden"
          aria-label="Toggle menu"
        >
          {isOpen ? (
            <X className="h-6 w-6" />
          ) : (
            <Menu className="h-6 w-6" />
          )}
        </Button>

        <div className="flex flex-1 items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center">
              <span className="text-white font-bold text-sm">N</span>
            </div>
            <span className="font-semibold text-lg">NovaCron</span>
          </Link>

          <Button
            variant="ghost"
            size="icon"
            onClick={toggleDarkMode}
            aria-label="Toggle dark mode"
          >
            {isDarkMode ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>

      {/* Mobile Slide-out Menu */}
      <AnimatePresence>
        {isOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-40 bg-black/50 lg:hidden"
              onClick={() => setIsOpen(false)}
            />

            {/* Slide-out Panel */}
            <motion.div
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ type: "spring", damping: 25, stiffness: 300 }}
              className="fixed inset-y-0 left-0 z-50 w-72 bg-white dark:bg-gray-900 shadow-xl lg:hidden"
            >
              <div className="flex h-full flex-col">
                {/* Header */}
                <div className="flex h-16 items-center gap-4 border-b px-4 dark:border-gray-800">
                  <Link href="/" className="flex items-center gap-2">
                    <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center">
                      <span className="text-white font-bold text-sm">N</span>
                    </div>
                    <span className="font-semibold text-lg">NovaCron</span>
                  </Link>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={() => setIsOpen(false)}
                    className="ml-auto"
                    aria-label="Close menu"
                  >
                    <X className="h-5 w-5" />
                  </Button>
                </div>

                {/* User Info */}
                {user && (
                  <div className="border-b px-4 py-4 dark:border-gray-800">
                    <div className="flex items-center gap-3">
                      <div className="h-10 w-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-semibold">
                        {user.name.charAt(0).toUpperCase()}
                      </div>
                      <div className="flex-1">
                        <p className="text-sm font-medium">{user.name}</p>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          {user.email}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Navigation Items */}
                <nav className="flex-1 overflow-y-auto px-2 py-4">
                  <div className="space-y-1">
                    {navigationItems.map((item) => {
                      const Icon = item.icon;
                      const isActive = pathname === item.href;

                      return (
                        <Link
                          key={item.name}
                          href={item.href}
                          className={cn(
                            "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all",
                            "hover:bg-gray-100 dark:hover:bg-gray-800",
                            isActive
                              ? "bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
                              : "text-gray-700 dark:text-gray-300"
                          )}
                        >
                          <Icon className="h-5 w-5 flex-shrink-0" />
                          <div className="flex-1">
                            <p className="font-medium">{item.name}</p>
                            <p className="text-xs text-gray-500 dark:text-gray-400">
                              {item.description}
                            </p>
                          </div>
                          {isActive && (
                            <ChevronRight className="h-4 w-4 flex-shrink-0" />
                          )}
                        </Link>
                      );
                    })}
                  </div>
                </nav>

                {/* Footer Actions */}
                <div className="border-t px-4 py-4 dark:border-gray-800">
                  <div className="space-y-2">
                    <Button
                      variant="outline"
                      className="w-full justify-start"
                      onClick={toggleDarkMode}
                    >
                      {isDarkMode ? (
                        <>
                          <Sun className="mr-2 h-4 w-4" />
                          Light Mode
                        </>
                      ) : (
                        <>
                          <Moon className="mr-2 h-4 w-4" />
                          Dark Mode
                        </>
                      )}
                    </Button>

                    {onLogout && (
                      <Button
                        variant="ghost"
                        className="w-full justify-start text-red-600 hover:text-red-700 hover:bg-red-50 dark:hover:bg-red-900/20"
                        onClick={onLogout}
                      >
                        <LogOut className="mr-2 h-4 w-4" />
                        Sign Out
                      </Button>
                    )}
                  </div>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Bottom Tab Bar for Mobile */}
      <div className="fixed bottom-0 left-0 right-0 z-30 flex h-16 items-center justify-around border-t bg-white px-2 dark:bg-gray-900 dark:border-gray-800 sm:hidden">
        {navigationItems.slice(0, 5).map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.href;

          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                "flex flex-col items-center justify-center gap-1 p-2 rounded-lg transition-colors",
                "hover:bg-gray-100 dark:hover:bg-gray-800",
                isActive
                  ? "text-blue-600 dark:text-blue-400"
                  : "text-gray-500 dark:text-gray-400"
              )}
            >
              <Icon className="h-5 w-5" />
              <span className="text-xs font-medium">{item.name}</span>
            </Link>
          );
        })}
      </div>
    </>
  );
}

// Responsive Sidebar for Desktop
interface DesktopSidebarProps {
  user?: {
    name: string;
    email: string;
    avatar?: string;
  };
  collapsed?: boolean;
  onCollapse?: (collapsed: boolean) => void;
}

export function DesktopSidebar({
  user,
  collapsed = false,
  onCollapse
}: DesktopSidebarProps) {
  const pathname = usePathname();

  return (
    <aside className={cn(
      "hidden lg:flex lg:flex-col",
      "fixed inset-y-0 left-0 z-30",
      "border-r bg-white dark:bg-gray-900 dark:border-gray-800",
      "transition-all duration-300",
      collapsed ? "w-16" : "w-64"
    )}>
      {/* Logo */}
      <div className="flex h-16 items-center gap-4 border-b px-4 dark:border-gray-800">
        <Link href="/" className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
            <span className="text-white font-bold text-sm">N</span>
          </div>
          {!collapsed && (
            <span className="font-semibold text-lg">NovaCron</span>
          )}
        </Link>

        <Button
          variant="ghost"
          size="icon"
          onClick={() => onCollapse?.(!collapsed)}
          className="ml-auto"
          aria-label="Toggle sidebar"
        >
          <Menu className="h-4 w-4" />
        </Button>
      </div>

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-2 py-4">
        <div className="space-y-1">
          {navigationItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-all",
                  "hover:bg-gray-100 dark:hover:bg-gray-800",
                  isActive
                    ? "bg-blue-50 text-blue-600 dark:bg-blue-900/20 dark:text-blue-400"
                    : "text-gray-700 dark:text-gray-300",
                  collapsed && "justify-center px-2"
                )}
                title={collapsed ? item.name : undefined}
              >
                <Icon className="h-5 w-5 flex-shrink-0" />
                {!collapsed && (
                  <>
                    <span className="flex-1 font-medium">{item.name}</span>
                    {isActive && (
                      <ChevronRight className="h-4 w-4 flex-shrink-0" />
                    )}
                  </>
                )}
              </Link>
            );
          })}
        </div>
      </nav>

      {/* User section */}
      {user && !collapsed && (
        <div className="border-t px-4 py-4 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-semibold text-sm">
              {user.name.charAt(0).toUpperCase()}
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium">{user.name}</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {user.email}
              </p>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}