"use client";

import { useEffect, useState } from "react";
import { validatePassword, PasswordStrength } from "@/lib/validation";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, AlertCircle, Info, Shield, Eye, Lock } from "lucide-react";

interface PasswordStrengthIndicatorProps {
  password: string;
  showSuggestions?: boolean;
  className?: string;
}

export function PasswordStrengthIndicator({
  password,
  showSuggestions = true,
  className
}: PasswordStrengthIndicatorProps) {
  const [strength, setStrength] = useState<PasswordStrength>({
    score: 0,
    feedback: "",
    suggestions: []
  });
  
  useEffect(() => {
    if (password) {
      const result = validatePassword(password);
      setStrength(result);
    } else {
      setStrength({
        score: 0,
        feedback: "",
        suggestions: []
      });
    }
  }, [password]);
  
  const getStrengthConfig = (score: number) => {
    switch (score) {
      case 0: return {
        color: "bg-red-500",
        textColor: "text-red-600 dark:text-red-400",
        label: "Very Weak",
        icon: AlertCircle,
        bgColor: "bg-red-50 dark:bg-red-900/20"
      };
      case 1: return {
        color: "bg-orange-500",
        textColor: "text-orange-600 dark:text-orange-400",
        label: "Weak",
        icon: AlertCircle,
        bgColor: "bg-orange-50 dark:bg-orange-900/20"
      };
      case 2: return {
        color: "bg-yellow-500",
        textColor: "text-yellow-600 dark:text-yellow-400",
        label: "Fair",
        icon: Info,
        bgColor: "bg-yellow-50 dark:bg-yellow-900/20"
      };
      case 3: return {
        color: "bg-blue-500",
        textColor: "text-blue-600 dark:text-blue-400",
        label: "Good",
        icon: Shield,
        bgColor: "bg-blue-50 dark:bg-blue-900/20"
      };
      case 4: return {
        color: "bg-green-500",
        textColor: "text-green-600 dark:text-green-400",
        label: "Strong",
        icon: CheckCircle2,
        bgColor: "bg-green-50 dark:bg-green-900/20"
      };
      default: return {
        color: "bg-gray-200",
        textColor: "text-gray-500",
        label: "",
        icon: Lock,
        bgColor: "bg-gray-50 dark:bg-gray-900/20"
      };
    }
  };
  
  const config = getStrengthConfig(strength.score);
  const IconComponent = config.icon;
  
  if (!password) return null;
  
  return (
    <motion.div 
      className={cn("space-y-3", className)}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2 }}
    >
      {/* Strength meter with animation */}
      <div className="space-y-2">
        <div className="flex gap-1">
          {[0, 1, 2, 3, 4].map((level) => (
            <motion.div
              key={level}
              className={cn(
                "h-2 flex-1 rounded-full transition-colors duration-500",
                level <= strength.score
                  ? config.color
                  : "bg-muted"
              )}
              initial={{ scaleX: 0 }}
              animate={{ 
                scaleX: level <= strength.score ? 1 : 1,
                opacity: level <= strength.score ? 1 : 0.3
              }}
              transition={{ 
                delay: level * 0.1,
                duration: 0.3,
                ease: "easeOut"
              }}
              style={{ transformOrigin: "left" }}
            />
          ))}
        </div>
        
        {/* Strength text with icon */}
        <AnimatePresence mode="wait">
          <motion.div
            key={strength.score}
            className="flex items-center justify-between"
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 10 }}
            transition={{ duration: 0.2 }}
          >
            <div className="flex items-center gap-2">
              <IconComponent className={cn("h-4 w-4", config.textColor)} />
              <span className={cn("text-sm font-medium", config.textColor)}>
                {config.label}
              </span>
            </div>
            
            {/* Checkmark for strong passwords */}
            <AnimatePresence>
              {strength.score >= 3 && (
                <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  exit={{ scale: 0, rotate: 180 }}
                  transition={{ 
                    type: "spring",
                    stiffness: 200,
                    damping: 10
                  }}
                >
                  <CheckCircle2 className="h-4 w-4 text-success" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </AnimatePresence>
      </div>
      
      {/* Enhanced suggestions */}
      <AnimatePresence>
        {showSuggestions && strength.suggestions.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="space-y-1.5 p-3 bg-muted/50 rounded-lg border">
              <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Info className="h-3 w-3" />
                Suggestions to improve your password:
              </div>
              <ul className="space-y-1">
                {strength.suggestions.map((suggestion, index) => (
                  <motion.li
                    key={suggestion}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1, duration: 0.2 }}
                    className="text-xs text-muted-foreground flex items-start gap-2"
                  >
                    <div className="h-1 w-1 rounded-full bg-muted-foreground mt-2 flex-shrink-0" />
                    <span>{suggestion}</span>
                  </motion.li>
                ))}
              </ul>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}