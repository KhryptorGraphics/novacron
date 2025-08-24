"use client";

import { motion, Variants, HTMLMotionProps } from "framer-motion";
import * as React from "react";

// Animation presets for consistency
export const animations = {
  // Page transitions
  pageEnter: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
    transition: { duration: 0.3, ease: "easeOut" },
  },
  
  // Modal/overlay animations
  modal: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 },
    transition: { duration: 0.2, ease: "easeOut" },
  },
  
  // Slide animations
  slideUp: {
    initial: { opacity: 0, y: 30 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 30 },
    transition: { duration: 0.3, ease: "easeOut" },
  },
  
  slideDown: {
    initial: { opacity: 0, y: -30 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -30 },
    transition: { duration: 0.3, ease: "easeOut" },
  },
  
  slideLeft: {
    initial: { opacity: 0, x: 30 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 30 },
    transition: { duration: 0.3, ease: "easeOut" },
  },
  
  slideRight: {
    initial: { opacity: 0, x: -30 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: -30 },
    transition: { duration: 0.3, ease: "easeOut" },
  },
  
  // Fade animations
  fadeIn: {
    initial: { opacity: 0 },
    animate: { opacity: 1 },
    exit: { opacity: 0 },
    transition: { duration: 0.2 },
  },
  
  // Scale animations
  scaleIn: {
    initial: { opacity: 0, scale: 0.8 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.8 },
    transition: { duration: 0.2, ease: "easeOut" },
  },
  
  // Stagger animations for lists
  staggerContainer: {
    initial: {},
    animate: {
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.1,
      },
    },
    exit: {
      transition: {
        staggerChildren: 0.05,
        staggerDirection: -1,
      },
    },
  },
  
  staggerItem: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: 20 },
    transition: { duration: 0.3, ease: "easeOut" },
  },
};

// Spring configurations
export const springs = {
  gentle: { type: "spring", stiffness: 120, damping: 14 },
  bouncy: { type: "spring", stiffness: 150, damping: 10 },
  swift: { type: "spring", stiffness: 200, damping: 20 },
  snappy: { type: "spring", stiffness: 300, damping: 25 },
} as const;

// Easing functions
export const easings = {
  easeOut: [0, 0, 0.2, 1],
  easeIn: [0.4, 0, 1, 1],
  easeInOut: [0.4, 0, 0.2, 1],
  bounceOut: [0.34, 1.56, 0.64, 1],
} as const;

// Reusable animated components
interface AnimatedDivProps extends HTMLMotionProps<"div"> {
  preset?: keyof typeof animations;
  delay?: number;
}

export const AnimatedDiv = React.forwardRef<HTMLDivElement, AnimatedDivProps>(
  ({ preset = "fadeIn", delay = 0, children, ...props }, ref) => {
    const animation = animations[preset];
    
    return (
      <motion.div
        ref={ref}
        initial={animation.initial}
        animate={animation.animate}
        exit={animation.exit}
        transition={{
          ...animation.transition,
          delay,
        }}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);
AnimatedDiv.displayName = "AnimatedDiv";

// Staggered list container
interface StaggeredListProps extends HTMLMotionProps<"div"> {
  staggerDelay?: number;
}

export const StaggeredList = React.forwardRef<HTMLDivElement, StaggeredListProps>(
  ({ staggerDelay = 0.1, children, ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        variants={{
          initial: {},
          animate: {
            transition: {
              staggerChildren: staggerDelay,
              delayChildren: staggerDelay,
            },
          },
          exit: {
            transition: {
              staggerChildren: staggerDelay * 0.5,
              staggerDirection: -1,
            },
          },
        }}
        initial="initial"
        animate="animate"
        exit="exit"
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);
StaggeredList.displayName = "StaggeredList";

// Staggered list item
export const StaggeredItem = React.forwardRef<HTMLDivElement, HTMLMotionProps<"div">>(
  ({ children, ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        variants={animations.staggerItem}
        {...props}
      >
        {children}
      </motion.div>
    );
  }
);
StaggeredItem.displayName = "StaggeredItem";

// Loading animations
export const LoadingDots = () => (
  <div className="flex gap-1">
    {[0, 1, 2].map((i) => (
      <motion.div
        key={i}
        className="w-1.5 h-1.5 bg-current rounded-full"
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.7, 1, 0.7],
        }}
        transition={{
          duration: 1.4,
          repeat: Infinity,
          delay: i * 0.2,
        }}
      />
    ))}
  </div>
);

export const LoadingSpinner = ({ className }: { className?: string }) => (
  <motion.div
    className={`border-2 border-current border-t-transparent rounded-full ${className || "w-4 h-4"}`}
    animate={{ rotate: 360 }}
    transition={{
      duration: 1,
      repeat: Infinity,
      ease: "linear",
    }}
  />
);

// Pulse animation for status indicators
export const PulseIndicator = ({ 
  className, 
  children 
}: { 
  className?: string; 
  children?: React.ReactNode; 
}) => (
  <motion.div
    className={className}
    animate={{
      scale: [1, 1.1, 1],
      opacity: [0.8, 1, 0.8],
    }}
    transition={{
      duration: 2,
      repeat: Infinity,
      ease: "easeInOut",
    }}
  >
    {children}
  </motion.div>
);

// Hover animations
export const hoverScale = {
  whileHover: { scale: 1.05 },
  whileTap: { scale: 0.95 },
  transition: { type: "spring", stiffness: 400, damping: 17 },
};

export const hoverLift = {
  whileHover: { y: -2, boxShadow: "0 4px 12px rgba(0, 0, 0, 0.15)" },
  whileTap: { y: 0 },
  transition: { type: "spring", stiffness: 400, damping: 17 },
};

export const hoverGlow = {
  whileHover: { 
    boxShadow: "0 0 20px rgba(59, 130, 246, 0.3)",
    borderColor: "rgba(59, 130, 246, 0.5)",
  },
  transition: { duration: 0.2 },
};

// Card animations
export const cardVariants: Variants = {
  initial: { opacity: 0, y: 20, scale: 0.95 },
  animate: { 
    opacity: 1, 
    y: 0, 
    scale: 1,
    transition: {
      duration: 0.3,
      ease: "easeOut",
    },
  },
  exit: { 
    opacity: 0, 
    y: -20, 
    scale: 0.95,
    transition: {
      duration: 0.2,
    },
  },
  hover: {
    y: -4,
    scale: 1.02,
    boxShadow: "0 8px 25px rgba(0, 0, 0, 0.12)",
    transition: {
      type: "spring",
      stiffness: 400,
      damping: 17,
    },
  },
};

// Progress animations
export const progressAnimation = {
  initial: { scaleX: 0 },
  animate: { scaleX: 1 },
  transition: { duration: 0.8, ease: "easeOut" },
};

export const countUpAnimation = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5, ease: "easeOut" },
};