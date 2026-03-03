"use client";

import React from "react";
import { motion } from "framer-motion";
import { useState, useEffect } from "react";

// FadeIn animation component
export function FadeIn({
  children,
  delay = 0,
  duration = 0.5,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  duration?: number;
  className?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration }}
      className={className}
    >
      {children}
    </motion.div>
  );
}

// AnimatedCounter component
export function AnimatedCounter({
  value,
  duration = 1000,
  suffix = "",
}: {
  value: number;
  duration?: number;
  suffix?: string;
}) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (typeof value !== 'number' || isNaN(value)) {
      setCount(0);
      return;
    }
    
    const start = 0;
    const end = value;
    const increment = end / (duration / 16);
    
    let current = start;
    const timer = setInterval(() => {
      current += increment;
      if (current >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(Math.floor(current));
      }
    }, 16);

    return () => clearInterval(timer);
  }, [value, duration]);

  return <span>{count}{suffix}</span>;
}

// StaggeredList animation component
export function StaggeredList({
  children,
  delay = 0.1,
  className = "",
}: {
  children: React.ReactNode;
  delay?: number;
  className?: string;
}) {
  const childrenArray = React.Children.toArray(children);
  
  return (
    <div className={className}>
      {childrenArray.map((child, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: delay * index, duration: 0.5 }}
        >
          {child}
        </motion.div>
      ))}
    </div>
  );
}

// LoadingSpinner component
export function LoadingSpinner({
  size = "md",
  className = "",
}: {
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  const sizeClasses = {
    sm: "h-4 w-4",
    md: "h-8 w-8", 
    lg: "h-12 w-12"
  };
  
  return (
    <div className={`animate-spin rounded-full border-4 border-blue-500 border-t-transparent ${sizeClasses[size]} ${className}`}></div>
  );
}

// AnimatedDiv wrapper
export function AnimatedDiv({
  children,
  ...props
}: {
  children: React.ReactNode;
  [key: string]: any;
}) {
  return (
    <motion.div {...props}>
      {children}
    </motion.div>
  );
}