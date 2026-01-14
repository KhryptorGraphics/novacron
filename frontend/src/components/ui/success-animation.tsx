"use client";

import { motion } from "framer-motion";
import { Icons } from "@/components/ui/icons";
import { cn } from "@/lib/utils";

interface SuccessAnimationProps {
  title: string;
  description?: string;
  className?: string;
  onComplete?: () => void;
}

export function SuccessAnimation({ 
  title, 
  description, 
  className,
  onComplete 
}: SuccessAnimationProps) {
  return (
    <motion.div
      className={cn(
        "flex flex-col items-center justify-center text-center space-y-4 py-8",
        className
      )}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, ease: "easeOut" }}
      onAnimationComplete={onComplete}
    >
      {/* Animated checkmark */}
      <motion.div
        className="relative"
        initial={{ scale: 0 }}
        animate={{ scale: 1 }}
        transition={{ 
          delay: 0.2,
          type: "spring",
          stiffness: 200,
          damping: 10
        }}
      >
        <motion.div
          className="h-16 w-16 rounded-full bg-green-500 flex items-center justify-center"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.1, duration: 0.3 }}
        >
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ delay: 0.3, duration: 0.4 }}
          >
            <Icons.check className="h-8 w-8 text-white" />
          </motion.div>
        </motion.div>
        
        {/* Pulse ring animation */}
        <motion.div
          className="absolute inset-0 h-16 w-16 rounded-full border-4 border-green-500"
          initial={{ scale: 1, opacity: 1 }}
          animate={{ scale: 1.5, opacity: 0 }}
          transition={{ 
            delay: 0.5,
            duration: 1,
            ease: "easeOut"
          }}
        />
      </motion.div>

      {/* Title with slide-up animation */}
      <motion.h2
        className="text-2xl font-bold text-green-700 dark:text-green-400"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.4 }}
      >
        {title}
      </motion.h2>

      {/* Description with stagger animation */}
      {description && (
        <motion.p
          className="text-gray-600 dark:text-gray-400 max-w-md"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.4 }}
        >
          {description}
        </motion.p>
      )}

      {/* Sparkles animation */}
      <motion.div
        className="absolute -inset-4 pointer-events-none"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1, duration: 0.5 }}
      >
        {[...Array(6)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute"
            style={{
              left: `${20 + (i * 12)}%`,
              top: `${30 + (i % 2 ? 10 : -10)}%`,
            }}
            initial={{ scale: 0, rotate: 0 }}
            animate={{ 
              scale: [0, 1, 0],
              rotate: [0, 180, 360],
            }}
            transition={{
              delay: 1 + (i * 0.1),
              duration: 1.5,
              repeat: 1,
              ease: "easeInOut"
            }}
          >
            <Icons.sparkles className="h-4 w-4 text-yellow-400" />
          </motion.div>
        ))}
      </motion.div>
    </motion.div>
  );
}