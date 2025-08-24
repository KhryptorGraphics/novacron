"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { AlertCircle, Info, CheckCircle, XCircle } from "lucide-react";

// Skip to Main Content Link
export function SkipToMain() {
  return (
    <a
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 focus:px-4 focus:py-2 focus:bg-blue-600 focus:text-white focus:rounded-md focus:outline-none focus:ring-2 focus:ring-blue-600 focus:ring-offset-2"
    >
      Skip to main content
    </a>
  );
}

// Visually Hidden (Screen Reader Only)
interface VisuallyHiddenProps {
  children: React.ReactNode;
  as?: React.ElementType;
}

export function VisuallyHidden({ 
  children, 
  as: Component = "span" 
}: VisuallyHiddenProps) {
  return (
    <Component className="sr-only">
      {children}
    </Component>
  );
}

// Live Region for Dynamic Updates
interface LiveRegionProps {
  children: React.ReactNode;
  mode?: "polite" | "assertive" | "off";
  relevant?: "additions" | "removals" | "text" | "all";
  atomic?: boolean;
  className?: string;
}

export function LiveRegion({
  children,
  mode = "polite",
  relevant = "additions",
  atomic = false,
  className
}: LiveRegionProps) {
  return (
    <div
      aria-live={mode}
      aria-relevant={relevant}
      aria-atomic={atomic}
      className={cn("sr-only", className)}
    >
      {children}
    </div>
  );
}

// Accessible Alert Component
interface AccessibleAlertProps {
  type?: "info" | "success" | "warning" | "error";
  title?: string;
  children: React.ReactNode;
  onClose?: () => void;
  className?: string;
}

export function AccessibleAlert({
  type = "info",
  title,
  children,
  onClose,
  className
}: AccessibleAlertProps) {
  const icons = {
    info: <Info className="h-5 w-5" aria-hidden="true" />,
    success: <CheckCircle className="h-5 w-5" aria-hidden="true" />,
    warning: <AlertCircle className="h-5 w-5" aria-hidden="true" />,
    error: <XCircle className="h-5 w-5" aria-hidden="true" />
  };
  
  const colors = {
    info: "bg-blue-50 text-blue-800 border-blue-200 dark:bg-blue-900/20 dark:text-blue-400 dark:border-blue-800",
    success: "bg-green-50 text-green-800 border-green-200 dark:bg-green-900/20 dark:text-green-400 dark:border-green-800",
    warning: "bg-yellow-50 text-yellow-800 border-yellow-200 dark:bg-yellow-900/20 dark:text-yellow-400 dark:border-yellow-800",
    error: "bg-red-50 text-red-800 border-red-200 dark:bg-red-900/20 dark:text-red-400 dark:border-red-800"
  };
  
  return (
    <div
      role="alert"
      aria-live={type === "error" ? "assertive" : "polite"}
      className={cn(
        "flex gap-3 p-4 rounded-lg border",
        colors[type],
        className
      )}
    >
      <div className="flex-shrink-0">
        {icons[type]}
      </div>
      <div className="flex-1">
        {title && (
          <h3 className="font-semibold mb-1">{title}</h3>
        )}
        <div className="text-sm">{children}</div>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          className="flex-shrink-0 p-1 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors"
          aria-label="Close alert"
        >
          <XCircle className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}

// Focus Trap Hook
export function useFocusTrap(ref: React.RefObject<HTMLElement>) {
  React.useEffect(() => {
    const element = ref.current;
    if (!element) return;
    
    const focusableElements = element.querySelectorAll(
      'a[href], button, textarea, input[type="text"], input[type="radio"], input[type="checkbox"], select, [tabindex]:not([tabindex="-1"])'
    );
    
    const firstFocusable = focusableElements[0] as HTMLElement;
    const lastFocusable = focusableElements[focusableElements.length - 1] as HTMLElement;
    
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Tab") return;
      
      if (e.shiftKey) {
        if (document.activeElement === firstFocusable) {
          e.preventDefault();
          lastFocusable?.focus();
        }
      } else {
        if (document.activeElement === lastFocusable) {
          e.preventDefault();
          firstFocusable?.focus();
        }
      }
    };
    
    element.addEventListener("keydown", handleKeyDown);
    firstFocusable?.focus();
    
    return () => {
      element.removeEventListener("keydown", handleKeyDown);
    };
  }, [ref]);
}

// Keyboard Navigation Hook
export function useKeyboardNavigation(
  items: any[],
  onSelect: (item: any, index: number) => void
) {
  const [focusedIndex, setFocusedIndex] = React.useState(-1);
  
  const handleKeyDown = React.useCallback((e: KeyboardEvent) => {
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setFocusedIndex(prev => 
          prev < items.length - 1 ? prev + 1 : 0
        );
        break;
      case "ArrowUp":
        e.preventDefault();
        setFocusedIndex(prev => 
          prev > 0 ? prev - 1 : items.length - 1
        );
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        if (focusedIndex >= 0 && focusedIndex < items.length) {
          onSelect(items[focusedIndex], focusedIndex);
        }
        break;
      case "Home":
        e.preventDefault();
        setFocusedIndex(0);
        break;
      case "End":
        e.preventDefault();
        setFocusedIndex(items.length - 1);
        break;
    }
  }, [items, focusedIndex, onSelect]);
  
  React.useEffect(() => {
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);
  
  return { focusedIndex, setFocusedIndex };
}

// Accessible Form Field
interface AccessibleFieldProps {
  label: string;
  error?: string;
  description?: string;
  required?: boolean;
  children: React.ReactElement;
}

export function AccessibleField({
  label,
  error,
  description,
  required,
  children
}: AccessibleFieldProps) {
  const id = React.useId();
  const errorId = `${id}-error`;
  const descriptionId = `${id}-description`;
  
  const childWithProps = React.cloneElement(children, {
    id,
    "aria-invalid": !!error,
    "aria-describedby": [
      error && errorId,
      description && descriptionId
    ].filter(Boolean).join(" ") || undefined,
    "aria-required": required
  });
  
  return (
    <div className="space-y-2">
      <label htmlFor={id} className="block text-sm font-medium">
        {label}
        {required && (
          <span className="text-red-500 ml-1" aria-label="required">
            *
          </span>
        )}
      </label>
      
      {description && (
        <p id={descriptionId} className="text-sm text-gray-600 dark:text-gray-400">
          {description}
        </p>
      )}
      
      {childWithProps}
      
      {error && (
        <p id={errorId} role="alert" className="text-sm text-red-600 dark:text-red-400">
          {error}
        </p>
      )}
    </div>
  );
}

// Announce Component for Screen Readers
interface AnnounceProps {
  message: string;
  priority?: "polite" | "assertive";
}

export function Announce({ message, priority = "polite" }: AnnounceProps) {
  const [announcement, setAnnouncement] = React.useState("");
  
  React.useEffect(() => {
    setAnnouncement(message);
    const timer = setTimeout(() => setAnnouncement(""), 100);
    return () => clearTimeout(timer);
  }, [message]);
  
  return (
    <div
      role="status"
      aria-live={priority}
      aria-atomic="true"
      className="sr-only"
    >
      {announcement}
    </div>
  );
}

// Focus Visible Indicator
export function FocusRing({ 
  children,
  className 
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn(
      "focus-within:ring-2 focus-within:ring-blue-500 focus-within:ring-offset-2 rounded-md",
      className
    )}>
      {children}
    </div>
  );
}

// ARIA Descriptions Helper
export function useAriaDescriptions() {
  const [descriptions, setDescriptions] = React.useState<Record<string, string>>({});
  
  const addDescription = React.useCallback((key: string, text: string) => {
    setDescriptions(prev => ({ ...prev, [key]: text }));
  }, []);
  
  const removeDescription = React.useCallback((key: string) => {
    setDescriptions(prev => {
      const next = { ...prev };
      delete next[key];
      return next;
    });
  }, []);
  
  const getDescriptionIds = React.useCallback((keys: string[]) => {
    return keys
      .filter(key => descriptions[key])
      .map(key => `desc-${key}`)
      .join(" ");
  }, [descriptions]);
  
  return { descriptions, addDescription, removeDescription, getDescriptionIds };
}

// High Contrast Mode Detection
export function useHighContrast() {
  const [isHighContrast, setIsHighContrast] = React.useState(false);
  
  React.useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-contrast: high)");
    
    const handleChange = (e: MediaQueryListEvent) => {
      setIsHighContrast(e.matches);
    };
    
    setIsHighContrast(mediaQuery.matches);
    mediaQuery.addEventListener("change", handleChange);
    
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);
  
  return isHighContrast;
}

// Reduced Motion Detection
export function useReducedMotion() {
  const [prefersReducedMotion, setPrefersReducedMotion] = React.useState(false);
  
  React.useEffect(() => {
    const mediaQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
    
    const handleChange = (e: MediaQueryListEvent) => {
      setPrefersReducedMotion(e.matches);
    };
    
    setPrefersReducedMotion(mediaQuery.matches);
    mediaQuery.addEventListener("change", handleChange);
    
    return () => mediaQuery.removeEventListener("change", handleChange);
  }, []);
  
  return prefersReducedMotion;
}