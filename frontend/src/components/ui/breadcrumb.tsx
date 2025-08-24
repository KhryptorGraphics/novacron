"use client";

import * as React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { ChevronRight, Home } from "lucide-react";
import { motion } from "framer-motion";

interface BreadcrumbItem {
  label: string;
  href?: string;
  icon?: React.ComponentType<{ className?: string }>;
}

interface BreadcrumbProps {
  items?: BreadcrumbItem[];
  className?: string;
  showHome?: boolean;
  separator?: React.ReactNode;
}

// Default route mapping for automatic breadcrumbs
const routeLabels: Record<string, string> = {
  "/": "Dashboard",
  "/dashboard": "Dashboard", 
  "/vms": "Virtual Machines",
  "/monitoring": "Monitoring",
  "/storage": "Storage",
  "/network": "Network",
  "/security": "Security", 
  "/analytics": "Analytics",
  "/users": "Users",
  "/settings": "Settings",
  "/auth": "Authentication",
  "/auth/login": "Sign In",
  "/auth/register": "Sign Up",
};

function generateBreadcrumbs(pathname: string): BreadcrumbItem[] {
  const segments = pathname.split('/').filter(Boolean);
  const breadcrumbs: BreadcrumbItem[] = [];
  
  // Add home if not root
  if (segments.length > 0) {
    breadcrumbs.push({
      label: "Home", 
      href: "/",
      icon: Home
    });
  }
  
  // Build breadcrumbs from path segments
  let currentPath = "";
  segments.forEach((segment, index) => {
    currentPath += `/${segment}`;
    const isLast = index === segments.length - 1;
    
    breadcrumbs.push({
      label: routeLabels[currentPath] || segment.charAt(0).toUpperCase() + segment.slice(1),
      href: isLast ? undefined : currentPath,
    });
  });
  
  return breadcrumbs;
}

export function Breadcrumb({ 
  items,
  className,
  showHome = true,
  separator,
  ...props 
}: BreadcrumbProps & React.HTMLAttributes<HTMLElement>) {
  const pathname = usePathname();
  
  // Use provided items or generate from pathname
  const breadcrumbItems = items || generateBreadcrumbs(pathname);
  
  // Don't show breadcrumbs for root or single level paths
  if (breadcrumbItems.length <= 1 && !showHome) {
    return null;
  }
  
  const defaultSeparator = separator || <ChevronRight className="h-4 w-4 text-muted-foreground" />;
  
  return (
    <nav
      role="navigation"
      aria-label="Breadcrumb"
      className={cn("flex items-center space-x-1 text-sm", className)}
      {...props}
    >
      <ol className="flex items-center space-x-1">
        {breadcrumbItems.map((item, index) => {
          const isLast = index === breadcrumbItems.length - 1;
          const IconComponent = item.icon;
          
          return (
            <motion.li
              key={`${item.href}-${index}`}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1, duration: 0.2 }}
              className="flex items-center space-x-1"
            >
              {index > 0 && (
                <span className="mx-2" role="presentation">
                  {defaultSeparator}
                </span>
              )}
              
              {item.href && !isLast ? (
                <Link
                  href={item.href}
                  className={cn(
                    "flex items-center gap-1.5 px-2 py-1 rounded-md",
                    "text-muted-foreground hover:text-foreground",
                    "hover:bg-accent hover:text-accent-foreground",
                    "transition-colors duration-200",
                    "focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
                  )}
                >
                  {IconComponent && (
                    <IconComponent className="h-4 w-4" />
                  )}
                  <span className="font-medium">{item.label}</span>
                </Link>
              ) : (
                <span
                  className={cn(
                    "flex items-center gap-1.5 px-2 py-1 rounded-md",
                    isLast
                      ? "text-foreground font-semibold"
                      : "text-muted-foreground"
                  )}
                  aria-current={isLast ? "page" : undefined}
                >
                  {IconComponent && (
                    <IconComponent className="h-4 w-4" />
                  )}
                  <span>{item.label}</span>
                </span>
              )}
            </motion.li>
          );
        })}
      </ol>
    </nav>
  );
}

// Compact version for mobile
export function CompactBreadcrumb({
  items,
  className,
  ...props
}: BreadcrumbProps & React.HTMLAttributes<HTMLElement>) {
  const pathname = usePathname();
  const breadcrumbItems = items || generateBreadcrumbs(pathname);
  
  if (breadcrumbItems.length <= 1) {
    return null;
  }
  
  // Show only the last item and home on mobile
  const firstItem = breadcrumbItems[0];
  const lastItem = breadcrumbItems[breadcrumbItems.length - 1];
  const hasMoreItems = breadcrumbItems.length > 2;
  
  return (
    <nav
      role="navigation"
      aria-label="Breadcrumb"
      className={cn("flex items-center space-x-1 text-sm", className)}
      {...props}
    >
      <ol className="flex items-center space-x-1">
        {/* Home */}
        {firstItem.href && (
          <li>
            <Link
              href={firstItem.href}
              className="flex items-center gap-1.5 px-1 py-1 rounded text-muted-foreground hover:text-foreground transition-colors"
            >
              {firstItem.icon && <firstItem.icon className="h-4 w-4" />}
            </Link>
          </li>
        )}
        
        {/* Separator and ellipsis if there are middle items */}
        {hasMoreItems && (
          <>
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
            <span className="text-muted-foreground">...</span>
            <ChevronRight className="h-3 w-3 text-muted-foreground" />
          </>
        )}
        
        {/* Current page */}
        <li>
          <span className="font-medium text-foreground px-1 py-1">
            {lastItem.label}
          </span>
        </li>
      </ol>
    </nav>
  );
}

// Breadcrumb with actions
interface BreadcrumbWithActionsProps extends BreadcrumbProps {
  actions?: React.ReactNode;
  title?: string;
  description?: string;
}

export function BreadcrumbWithActions({
  actions,
  title,
  description,
  className,
  ...breadcrumbProps
}: BreadcrumbWithActionsProps) {
  return (
    <div className={cn("space-y-3", className)}>
      <Breadcrumb {...breadcrumbProps} />
      
      {(title || description || actions) && (
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0 flex-1">
            {title && (
              <h1 className="text-2xl font-bold leading-tight tracking-tight text-foreground">
                {title}
              </h1>
            )}
            {description && (
              <p className="mt-1 text-muted-foreground text-balance">
                {description}
              </p>
            )}
          </div>
          
          {actions && (
            <div className="flex-shrink-0">
              {actions}
            </div>
          )}
        </div>
      )}
    </div>
  );
}