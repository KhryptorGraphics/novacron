"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

// Typography variants for consistent text styling
const typographyVariants = cva("", {
  variants: {
    variant: {
      h1: "scroll-m-20 text-4xl font-bold tracking-tight lg:text-5xl",
      h2: "scroll-m-20 text-3xl font-semibold tracking-tight first:mt-0",
      h3: "scroll-m-20 text-2xl font-semibold tracking-tight",
      h4: "scroll-m-20 text-xl font-semibold tracking-tight",
      h5: "scroll-m-20 text-lg font-semibold tracking-tight",
      h6: "scroll-m-20 text-base font-semibold tracking-tight",
      p: "leading-7 [&:not(:first-child)]:mt-6",
      blockquote: "mt-6 border-l-2 pl-6 italic border-border",
      list: "my-6 ml-6 list-disc [&>li]:mt-2",
      code: "relative rounded bg-muted px-[0.3rem] py-[0.2rem] font-mono text-sm font-semibold",
      lead: "text-xl text-muted-foreground",
      large: "text-lg font-semibold",
      small: "text-sm font-medium leading-none",
      muted: "text-sm text-muted-foreground",
      subtle: "text-xs text-muted-foreground",
    },
    color: {
      default: "text-foreground",
      primary: "text-primary",
      secondary: "text-secondary-foreground",
      muted: "text-muted-foreground",
      success: "text-success",
      warning: "text-warning",
      error: "text-destructive",
      info: "text-info",
    },
  },
  defaultVariants: {
    variant: "p",
    color: "default",
  },
});

export interface TypographyProps
  extends React.HTMLAttributes<HTMLElement>,
    VariantProps<typeof typographyVariants> {
  as?: React.ElementType;
}

const Typography = React.forwardRef<HTMLElement, TypographyProps>(
  ({ className, variant, color, as, ...props }, ref) => {
    const Comp = as || getDefaultElement(variant);
    return (
      <Comp
        className={cn(typographyVariants({ variant, color, className }))}
        ref={ref}
        {...props}
      />
    );
  }
);

function getDefaultElement(variant: TypographyProps["variant"]) {
  switch (variant) {
    case "h1":
      return "h1";
    case "h2":
      return "h2";
    case "h3":
      return "h3";
    case "h4":
      return "h4";
    case "h5":
      return "h5";
    case "h6":
      return "h6";
    case "blockquote":
      return "blockquote";
    case "list":
      return "ul";
    case "code":
      return "code";
    default:
      return "p";
  }
}

Typography.displayName = "Typography";

// Pre-built component shortcuts for common use cases
export const Heading1 = React.forwardRef<HTMLHeadingElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="h1" ref={ref} className={className} {...props} />
  )
);
Heading1.displayName = "Heading1";

export const Heading2 = React.forwardRef<HTMLHeadingElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="h2" ref={ref} className={className} {...props} />
  )
);
Heading2.displayName = "Heading2";

export const Heading3 = React.forwardRef<HTMLHeadingElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="h3" ref={ref} className={className} {...props} />
  )
);
Heading3.displayName = "Heading3";

export const Body = React.forwardRef<HTMLParagraphElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="p" ref={ref} className={className} {...props} />
  )
);
Body.displayName = "Body";

export const Lead = React.forwardRef<HTMLParagraphElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="lead" ref={ref} className={className} {...props} />
  )
);
Lead.displayName = "Lead";

export const Caption = React.forwardRef<HTMLParagraphElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="small" ref={ref} className={className} {...props} />
  )
);
Caption.displayName = "Caption";

export const Muted = React.forwardRef<HTMLParagraphElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="muted" ref={ref} className={className} {...props} />
  )
);
Muted.displayName = "Muted";

export const Code = React.forwardRef<HTMLElement, Omit<TypographyProps, "variant">>(
  ({ className, ...props }, ref) => (
    <Typography variant="code" ref={ref} className={className} {...props} />
  )
);
Code.displayName = "Code";

// Status text components
interface StatusTextProps extends Omit<TypographyProps, "color"> {
  status: "success" | "warning" | "error" | "info";
}

export const StatusText = React.forwardRef<HTMLElement, StatusTextProps>(
  ({ status, className, ...props }, ref) => (
    <Typography 
      color={status === "error" ? "error" : status} 
      ref={ref} 
      className={cn("font-medium", className)} 
      {...props} 
    />
  )
);
StatusText.displayName = "StatusText";

export { Typography, typographyVariants };