"use client";

import * as React from "react";
import { useState, useEffect } from "react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { 
  ChevronDown, 
  ChevronRight,
  Plus,
  Minus,
  Eye,
  EyeOff,
  Info,
  Filter,
  Settings,
  ChevronUp
} from "lucide-react";
import { Button } from "./button";
import { Badge } from "./badge";

// Collapsible Section Component
interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
  badge?: string | number;
  icon?: React.ReactNode;
  className?: string;
  headerClassName?: string;
  contentClassName?: string;
  onToggle?: (isOpen: boolean) => void;
}

export function CollapsibleSection({
  title,
  children,
  defaultOpen = false,
  badge,
  icon,
  className,
  headerClassName,
  contentClassName,
  onToggle
}: CollapsibleSectionProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  
  const handleToggle = () => {
    const newState = !isOpen;
    setIsOpen(newState);
    onToggle?.(newState);
  };
  
  return (
    <div className={cn("border rounded-lg", className)}>
      <button
        onClick={handleToggle}
        className={cn(
          "w-full px-4 py-3 flex items-center justify-between",
          "hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors",
          "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-inset",
          headerClassName
        )}
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-3">
          {isOpen ? (
            <ChevronDown className="h-5 w-5 text-gray-500" />
          ) : (
            <ChevronRight className="h-5 w-5 text-gray-500" />
          )}
          {icon && <div className="text-gray-600 dark:text-gray-400">{icon}</div>}
          <h3 className="font-medium text-left">{title}</h3>
          {badge && (
            <Badge variant="secondary" className="ml-2">
              {badge}
            </Badge>
          )}
        </div>
      </button>
      
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className={cn("px-4 py-3 border-t", contentClassName)}>
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Accordion Component
interface AccordionItem {
  id: string;
  title: string;
  content: React.ReactNode;
  icon?: React.ReactNode;
  badge?: string | number;
}

interface AccordionProps {
  items: AccordionItem[];
  allowMultiple?: boolean;
  defaultOpenItems?: string[];
  className?: string;
}

export function Accordion({
  items,
  allowMultiple = false,
  defaultOpenItems = [],
  className
}: AccordionProps) {
  const [openItems, setOpenItems] = useState<string[]>(defaultOpenItems);
  
  const toggleItem = (itemId: string) => {
    if (allowMultiple) {
      setOpenItems(prev =>
        prev.includes(itemId)
          ? prev.filter(id => id !== itemId)
          : [...prev, itemId]
      );
    } else {
      setOpenItems(prev =>
        prev.includes(itemId) ? [] : [itemId]
      );
    }
  };
  
  return (
    <div className={cn("space-y-2", className)}>
      {items.map((item) => (
        <CollapsibleSection
          key={item.id}
          title={item.title}
          defaultOpen={openItems.includes(item.id)}
          badge={item.badge}
          icon={item.icon}
          onToggle={() => toggleItem(item.id)}
        >
          {item.content}
        </CollapsibleSection>
      ))}
    </div>
  );
}

// Show More/Less Component
interface ShowMoreProps {
  children: React.ReactNode;
  maxHeight?: number;
  maxLines?: number;
  showLabel?: string;
  hideLabel?: string;
  className?: string;
  gradientColor?: string;
}

export function ShowMore({
  children,
  maxHeight = 200,
  maxLines,
  showLabel = "Show more",
  hideLabel = "Show less",
  className,
  gradientColor = "white"
}: ShowMoreProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [needsExpansion, setNeedsExpansion] = useState(false);
  const contentRef = React.useRef<HTMLDivElement>(null);
  
  useEffect(() => {
    if (contentRef.current) {
      const height = contentRef.current.scrollHeight;
      setNeedsExpansion(height > maxHeight);
    }
  }, [maxHeight, children]);
  
  const style = !isExpanded && needsExpansion
    ? {
        maxHeight: `${maxHeight}px`,
        overflow: "hidden",
        position: "relative" as const
      }
    : {};
  
  return (
    <div className={className}>
      <div ref={contentRef} style={style}>
        {children}
        {!isExpanded && needsExpansion && (
          <div
            className="absolute bottom-0 left-0 right-0 h-20 pointer-events-none"
            style={{
              background: `linear-gradient(transparent, ${gradientColor})`
            }}
          />
        )}
      </div>
      
      {needsExpansion && (
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="mt-2"
        >
          {isExpanded ? (
            <>
              <ChevronUp className="mr-2 h-4 w-4" />
              {hideLabel}
            </>
          ) : (
            <>
              <ChevronDown className="mr-2 h-4 w-4" />
              {showLabel}
            </>
          )}
        </Button>
      )}
    </div>
  );
}

// Progressive Data Table
interface ProgressiveTableProps<T> {
  data: T[];
  columns: {
    key: keyof T | string;
    header: string;
    render?: (value: any, item: T) => React.ReactNode;
    priority?: "always" | "default" | "optional";
  }[];
  initialRows?: number;
  incrementBy?: number;
  className?: string;
}

export function ProgressiveTable<T>({
  data,
  columns,
  initialRows = 10,
  incrementBy = 10,
  className
}: ProgressiveTableProps<T>) {
  const [visibleRows, setVisibleRows] = useState(initialRows);
  const [showOptionalColumns, setShowOptionalColumns] = useState(false);
  
  const visibleData = data.slice(0, visibleRows);
  const hasMore = visibleRows < data.length;
  
  const visibleColumns = columns.filter(col =>
    col.priority === "always" ||
    col.priority === "default" ||
    (col.priority === "optional" && showOptionalColumns)
  );
  
  const optionalColumnsCount = columns.filter(
    col => col.priority === "optional"
  ).length;
  
  return (
    <div className={className}>
      {/* Column visibility toggle */}
      {optionalColumnsCount > 0 && (
        <div className="mb-4 flex justify-end">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowOptionalColumns(!showOptionalColumns)}
          >
            {showOptionalColumns ? (
              <>
                <EyeOff className="mr-2 h-4 w-4" />
                Hide details
              </>
            ) : (
              <>
                <Eye className="mr-2 h-4 w-4" />
                Show details ({optionalColumnsCount})
              </>
            )}
          </Button>
        </div>
      )}
      
      {/* Table */}
      <div className="overflow-x-auto rounded-lg border">
        <table className="w-full">
          <thead className="bg-gray-50 dark:bg-gray-800">
            <tr>
              {visibleColumns.map((column, index) => (
                <th
                  key={index}
                  className="px-4 py-3 text-left text-sm font-medium text-gray-700 dark:text-gray-300"
                >
                  {column.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y">
            {visibleData.map((item, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-gray-50 dark:hover:bg-gray-800">
                {visibleColumns.map((column, colIndex) => (
                  <td key={colIndex} className="px-4 py-3 text-sm">
                    {column.render
                      ? column.render((item as any)[column.key], item)
                      : (item as any)[column.key]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {/* Load more button */}
      {hasMore && (
        <div className="mt-4 text-center">
          <Button
            variant="outline"
            onClick={() => setVisibleRows(prev => prev + incrementBy)}
          >
            Load more ({data.length - visibleRows} remaining)
          </Button>
        </div>
      )}
    </div>
  );
}

// Detail Summary Component
interface DetailSummaryProps {
  summary: React.ReactNode;
  details: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}

export function DetailSummary({
  summary,
  details,
  icon,
  className
}: DetailSummaryProps) {
  const [isOpen, setIsOpen] = useState(false);
  
  return (
    <div className={cn("group", className)}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full text-left focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-lg p-2 -m-2"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-2">
          {icon || <Info className="h-4 w-4 text-gray-400" />}
          <span className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors">
            {summary}
          </span>
          <ChevronDown 
            className={cn(
              "h-4 w-4 text-gray-400 transition-transform",
              isOpen && "rotate-180"
            )}
          />
        </div>
      </button>
      
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="pt-2 pl-6">
              {details}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// Tabs with Lazy Loading
interface LazyTab {
  id: string;
  label: string;
  content: React.ReactNode | (() => Promise<React.ReactNode>);
  icon?: React.ReactNode;
  badge?: string | number;
}

interface LazyTabsProps {
  tabs: LazyTab[];
  defaultTab?: string;
  className?: string;
  onTabChange?: (tabId: string) => void;
}

export function LazyTabs({
  tabs,
  defaultTab,
  className,
  onTabChange
}: LazyTabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);
  const [loadedTabs, setLoadedTabs] = useState<Set<string>>(new Set([activeTab]));
  const [tabContent, setTabContent] = useState<Record<string, React.ReactNode>>({});
  
  const handleTabChange = async (tabId: string) => {
    setActiveTab(tabId);
    onTabChange?.(tabId);
    
    if (!loadedTabs.has(tabId)) {
      const tab = tabs.find(t => t.id === tabId);
      if (tab && typeof tab.content === "function") {
        const content = await tab.content();
        setTabContent(prev => ({ ...prev, [tabId]: content }));
      }
      setLoadedTabs(prev => new Set([...prev, tabId]));
    }
  };
  
  const activeTabData = tabs.find(t => t.id === activeTab);
  
  return (
    <div className={className}>
      {/* Tab headers */}
      <div className="flex gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => handleTabChange(tab.id)}
            className={cn(
              "flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md transition-colors",
              "focus:outline-none focus:ring-2 focus:ring-blue-500",
              activeTab === tab.id
                ? "bg-white dark:bg-gray-700 shadow-sm"
                : "hover:bg-gray-200 dark:hover:bg-gray-700"
            )}
          >
            {tab.icon}
            <span className="font-medium">{tab.label}</span>
            {tab.badge && (
              <Badge variant="secondary" className="ml-1">
                {tab.badge}
              </Badge>
            )}
          </button>
        ))}
      </div>
      
      {/* Tab content */}
      <div className="mt-4">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
          >
            {activeTabData && (
              typeof activeTabData.content === "function"
                ? tabContent[activeTab] || <div>Loading...</div>
                : activeTabData.content
            )}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}