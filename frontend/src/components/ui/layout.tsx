"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Container component with consistent padding and max-width
const containerVariants = cva(
  "w-full mx-auto",
  {
    variants: {
      size: {
        sm: "max-w-3xl px-4 sm:px-6",
        md: "max-w-5xl px-4 sm:px-6 lg:px-8", 
        lg: "max-w-7xl px-4 sm:px-6 lg:px-8",
        xl: "max-w-screen-2xl px-4 sm:px-6 lg:px-8 xl:px-12",
        full: "max-w-none px-4 sm:px-6 lg:px-8",
      },
    },
    defaultVariants: {
      size: "lg",
    },
  }
);

export interface ContainerProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof containerVariants> {}

const Container = React.forwardRef<HTMLDivElement, ContainerProps>(
  ({ className, size, ...props }, ref) => {
    return (
      <div
        className={cn(containerVariants({ size, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Container.displayName = "Container";

// Stack component for vertical spacing
const stackVariants = cva(
  "flex flex-col",
  {
    variants: {
      spacing: {
        none: "gap-0",
        xs: "gap-1",
        sm: "gap-2", 
        md: "gap-4",
        lg: "gap-6",
        xl: "gap-8",
        "2xl": "gap-12",
      },
      align: {
        start: "items-start",
        center: "items-center", 
        end: "items-end",
        stretch: "items-stretch",
      },
    },
    defaultVariants: {
      spacing: "md",
      align: "stretch",
    },
  }
);

export interface StackProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof stackVariants> {}

const Stack = React.forwardRef<HTMLDivElement, StackProps>(
  ({ className, spacing, align, ...props }, ref) => {
    return (
      <div
        className={cn(stackVariants({ spacing, align, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Stack.displayName = "Stack";

// Inline component for horizontal spacing
const inlineVariants = cva(
  "flex flex-row",
  {
    variants: {
      spacing: {
        none: "gap-0",
        xs: "gap-1",
        sm: "gap-2",
        md: "gap-4", 
        lg: "gap-6",
        xl: "gap-8",
      },
      align: {
        start: "items-start",
        center: "items-center",
        end: "items-end",
        baseline: "items-baseline",
        stretch: "items-stretch",
      },
      justify: {
        start: "justify-start",
        center: "justify-center",
        end: "justify-end", 
        between: "justify-between",
        around: "justify-around",
        evenly: "justify-evenly",
      },
      wrap: {
        true: "flex-wrap",
        false: "flex-nowrap",
      },
    },
    defaultVariants: {
      spacing: "md",
      align: "center",
      justify: "start",
      wrap: false,
    },
  }
);

export interface InlineProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof inlineVariants> {
  wrap?: boolean;
}

const Inline = React.forwardRef<HTMLDivElement, InlineProps>(
  ({ className, spacing, align, justify, wrap, ...props }, ref) => {
    return (
      <div
        className={cn(inlineVariants({ spacing, align, justify, wrap: wrap ? "true" : "false", className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Inline.displayName = "Inline";

// Grid component for consistent grid layouts
const gridVariants = cva(
  "grid",
  {
    variants: {
      cols: {
        1: "grid-cols-1",
        2: "grid-cols-1 sm:grid-cols-2",
        3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3", 
        4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
        dashboard: "grid-cols-dashboard",
        monitoring: "grid-cols-monitoring",
        auto: "grid-cols-[repeat(auto-fit,minmax(280px,1fr))]",
      },
      gap: {
        none: "gap-0",
        xs: "gap-1",
        sm: "gap-2",
        md: "gap-4",
        lg: "gap-6", 
        xl: "gap-8",
      },
    },
    defaultVariants: {
      cols: "auto",
      gap: "md",
    },
  }
);

export interface GridProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof gridVariants> {}

const Grid = React.forwardRef<HTMLDivElement, GridProps>(
  ({ className, cols, gap, ...props }, ref) => {
    return (
      <div
        className={cn(gridVariants({ cols, gap, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);
Grid.displayName = "Grid";

// Page layout components
export interface PageHeaderProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
  actions?: React.ReactNode;
}

const PageHeader = React.forwardRef<HTMLDivElement, PageHeaderProps>(
  ({ className, title, description, actions, children, ...props }, ref) => {
    return (
      <div
        className={cn("flex flex-col gap-4 pb-6 border-b", className)}
        ref={ref}
        {...props}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="space-y-1 min-w-0 flex-1">
            {title && (
              <h1 className="text-2xl font-bold leading-tight tracking-tight truncate">
                {title}
              </h1>
            )}
            {description && (
              <p className="text-muted-foreground text-balance">
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
        {children}
      </div>
    );
  }
);
PageHeader.displayName = "PageHeader";

export interface PageContentProps extends React.HTMLAttributes<HTMLDivElement> {}

const PageContent = React.forwardRef<HTMLDivElement, PageContentProps>(
  ({ className, ...props }, ref) => {
    return (
      <div
        className={cn("flex-1 py-6", className)}
        ref={ref}
        {...props}
      />
    );
  }
);
PageContent.displayName = "PageContent";

// Section component for consistent content sections
export interface SectionProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  description?: string;
  actions?: React.ReactNode;
}

const Section = React.forwardRef<HTMLDivElement, SectionProps>(
  ({ className, title, description, actions, children, ...props }, ref) => {
    return (
      <section
        className={cn("space-y-4", className)}
        ref={ref}
        {...props}
      >
        {(title || description || actions) && (
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1 min-w-0 flex-1">
              {title && (
                <h2 className="text-lg font-semibold leading-tight">
                  {title}
                </h2>
              )}
              {description && (
                <p className="text-sm text-muted-foreground">
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
        {children}
      </section>
    );
  }
);
Section.displayName = "Section";

export {
  Container,
  Stack,
  Inline, 
  Grid,
  PageHeader,
  PageContent,
  Section,
  containerVariants,
  stackVariants,
  inlineVariants,
  gridVariants,
};