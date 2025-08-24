"use client";

import * as React from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, ChevronRight } from "lucide-react";

interface ResponsiveTableProps {
  children: React.ReactNode;
  className?: string;
  containerClassName?: string;
}

export function ResponsiveTable({ 
  children, 
  className,
  containerClassName 
}: ResponsiveTableProps) {
  return (
    <div className={cn(
      "w-full overflow-auto rounded-lg border border-gray-200 dark:border-gray-700",
      containerClassName
    )}>
      <table className={cn(
        "w-full caption-bottom text-sm",
        className
      )}>
        {children}
      </table>
    </div>
  );
}

interface ResponsiveTableHeaderProps {
  children: React.ReactNode;
  className?: string;
  sticky?: boolean;
}

export function ResponsiveTableHeader({ 
  children, 
  className,
  sticky = true 
}: ResponsiveTableHeaderProps) {
  return (
    <thead className={cn(
      "border-b bg-gray-50 dark:bg-gray-800",
      sticky && "sticky top-0 z-10",
      className
    )}>
      {children}
    </thead>
  );
}

interface ResponsiveTableBodyProps {
  children: React.ReactNode;
  className?: string;
}

export function ResponsiveTableBody({ 
  children, 
  className 
}: ResponsiveTableBodyProps) {
  return (
    <tbody className={cn(
      "[&_tr:last-child]:border-0",
      className
    )}>
      {children}
    </tbody>
  );
}

interface ResponsiveTableRowProps {
  children: React.ReactNode;
  className?: string;
  onClick?: () => void;
  expandable?: boolean;
  expanded?: boolean;
}

export function ResponsiveTableRow({ 
  children, 
  className,
  onClick,
  expandable = false,
  expanded = false
}: ResponsiveTableRowProps) {
  return (
    <tr 
      className={cn(
        "border-b transition-colors",
        "hover:bg-gray-50 dark:hover:bg-gray-800/50",
        onClick && "cursor-pointer",
        expandable && "group",
        className
      )}
      onClick={onClick}
    >
      {expandable && (
        <td className="w-8 px-2">
          <div className="flex items-center justify-center">
            {expanded ? (
              <ChevronDown className="h-4 w-4 text-gray-500" />
            ) : (
              <ChevronRight className="h-4 w-4 text-gray-500" />
            )}
          </div>
        </td>
      )}
      {children}
    </tr>
  );
}

interface ResponsiveTableCellProps {
  children: React.ReactNode;
  className?: string;
  mobileLabel?: string;
  priority?: "high" | "medium" | "low";
  align?: "left" | "center" | "right";
}

export function ResponsiveTableCell({ 
  children, 
  className,
  mobileLabel,
  priority = "medium",
  align = "left"
}: ResponsiveTableCellProps) {
  const alignClass = {
    left: "text-left",
    center: "text-center",
    right: "text-right"
  }[align];

  const priorityClass = {
    high: "",
    medium: "hidden sm:table-cell",
    low: "hidden lg:table-cell"
  }[priority];

  return (
    <td className={cn(
      "p-3",
      alignClass,
      priorityClass,
      className
    )}>
      {mobileLabel && (
        <span className="font-medium text-gray-600 dark:text-gray-400 sm:hidden">
          {mobileLabel}:{" "}
        </span>
      )}
      {children}
    </td>
  );
}

interface ResponsiveTableHeadProps {
  children: React.ReactNode;
  className?: string;
  priority?: "high" | "medium" | "low";
  align?: "left" | "center" | "right";
  sortable?: boolean;
  sorted?: "asc" | "desc" | null;
  onSort?: () => void;
}

export function ResponsiveTableHead({ 
  children, 
  className,
  priority = "medium",
  align = "left",
  sortable = false,
  sorted = null,
  onSort
}: ResponsiveTableHeadProps) {
  const alignClass = {
    left: "text-left",
    center: "text-center",
    right: "text-right"
  }[align];

  const priorityClass = {
    high: "",
    medium: "hidden sm:table-cell",
    low: "hidden lg:table-cell"
  }[priority];

  return (
    <th className={cn(
      "h-10 px-3 font-medium text-gray-700 dark:text-gray-300",
      alignClass,
      priorityClass,
      sortable && "cursor-pointer select-none hover:bg-gray-100 dark:hover:bg-gray-700",
      className
    )}
    onClick={sortable ? onSort : undefined}
    >
      <div className="flex items-center gap-2">
        {children}
        {sortable && (
          <div className="flex flex-col">
            <svg 
              className={cn(
                "h-3 w-3",
                sorted === "asc" ? "text-primary-600" : "text-gray-400"
              )} 
              fill="currentColor" 
              viewBox="0 0 20 20"
            >
              <path d="M7 7l3-3 3 3" />
            </svg>
            <svg 
              className={cn(
                "h-3 w-3 -mt-1",
                sorted === "desc" ? "text-primary-600" : "text-gray-400"
              )} 
              fill="currentColor" 
              viewBox="0 0 20 20"
            >
              <path d="M7 13l3 3 3-3" />
            </svg>
          </div>
        )}
      </div>
    </th>
  );
}

// Mobile Card View Component
interface MobileCardViewProps {
  items: any[];
  renderCard: (item: any, index: number) => React.ReactNode;
  className?: string;
}

export function MobileCardView({ 
  items, 
  renderCard, 
  className 
}: MobileCardViewProps) {
  return (
    <div className={cn(
      "space-y-3 sm:hidden",
      className
    )}>
      {items.map((item, index) => (
        <div
          key={index}
          className="rounded-lg border bg-white p-4 shadow-sm dark:bg-gray-800 dark:border-gray-700"
        >
          {renderCard(item, index)}
        </div>
      ))}
    </div>
  );
}

// Responsive Data Table Component
interface ResponsiveDataTableProps<T> {
  data: T[];
  columns: {
    key: keyof T | string;
    header: string;
    priority?: "high" | "medium" | "low";
    align?: "left" | "center" | "right";
    sortable?: boolean;
    render?: (value: any, item: T) => React.ReactNode;
  }[];
  mobileCard?: (item: T, index: number) => React.ReactNode;
  onRowClick?: (item: T) => void;
  className?: string;
}

export function ResponsiveDataTable<T>({ 
  data, 
  columns,
  mobileCard,
  onRowClick,
  className 
}: ResponsiveDataTableProps<T>) {
  const [sortConfig, setSortConfig] = React.useState<{
    key: string;
    direction: "asc" | "desc";
  } | null>(null);

  const sortedData = React.useMemo(() => {
    if (!sortConfig) return data;

    return [...data].sort((a, b) => {
      const aValue = (a as any)[sortConfig.key];
      const bValue = (b as any)[sortConfig.key];

      if (aValue < bValue) {
        return sortConfig.direction === "asc" ? -1 : 1;
      }
      if (aValue > bValue) {
        return sortConfig.direction === "asc" ? 1 : -1;
      }
      return 0;
    });
  }, [data, sortConfig]);

  const handleSort = (key: string) => {
    setSortConfig(current => {
      if (current?.key === key) {
        if (current.direction === "asc") {
          return { key, direction: "desc" };
        }
        return null;
      }
      return { key, direction: "asc" };
    });
  };

  // Mobile card view
  if (mobileCard) {
    return (
      <>
        <MobileCardView 
          items={sortedData} 
          renderCard={mobileCard}
          className="sm:hidden"
        />
        <div className="hidden sm:block">
          <ResponsiveTable className={className}>
            <ResponsiveTableHeader>
              <ResponsiveTableRow>
                {columns.map((column) => (
                  <ResponsiveTableHead
                    key={column.key as string}
                    priority={column.priority}
                    align={column.align}
                    sortable={column.sortable}
                    sorted={
                      sortConfig?.key === column.key
                        ? sortConfig.direction
                        : null
                    }
                    onSort={() => column.sortable && handleSort(column.key as string)}
                  >
                    {column.header}
                  </ResponsiveTableHead>
                ))}
              </ResponsiveTableRow>
            </ResponsiveTableHeader>
            <ResponsiveTableBody>
              {sortedData.map((item, index) => (
                <ResponsiveTableRow
                  key={index}
                  onClick={() => onRowClick?.(item)}
                >
                  {columns.map((column) => (
                    <ResponsiveTableCell
                      key={column.key as string}
                      priority={column.priority}
                      align={column.align}
                    >
                      {column.render
                        ? column.render((item as any)[column.key], item)
                        : (item as any)[column.key]}
                    </ResponsiveTableCell>
                  ))}
                </ResponsiveTableRow>
              ))}
            </ResponsiveTableBody>
          </ResponsiveTable>
        </div>
      </>
    );
  }

  // Desktop table view only
  return (
    <ResponsiveTable className={className}>
      <ResponsiveTableHeader>
        <ResponsiveTableRow>
          {columns.map((column) => (
            <ResponsiveTableHead
              key={column.key as string}
              priority={column.priority}
              align={column.align}
              sortable={column.sortable}
              sorted={
                sortConfig?.key === column.key
                  ? sortConfig.direction
                  : null
              }
              onSort={() => column.sortable && handleSort(column.key as string)}
            >
              {column.header}
            </ResponsiveTableHead>
          ))}
        </ResponsiveTableRow>
      </ResponsiveTableHeader>
      <ResponsiveTableBody>
        {sortedData.map((item, index) => (
          <ResponsiveTableRow
            key={index}
            onClick={() => onRowClick?.(item)}
          >
            {columns.map((column) => (
              <ResponsiveTableCell
                key={column.key as string}
                priority={column.priority}
                align={column.align}
                mobileLabel={column.header}
              >
                {column.render
                  ? column.render((item as any)[column.key], item)
                  : (item as any)[column.key]}
              </ResponsiveTableCell>
            ))}
          </ResponsiveTableRow>
        ))}
      </ResponsiveTableBody>
    </ResponsiveTable>
  );
}