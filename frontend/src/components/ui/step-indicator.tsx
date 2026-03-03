"use client";

import { motion } from "framer-motion";
import { Icons } from "@/components/ui/icons";
import { cn } from "@/lib/utils";

interface Step {
  number: number;
  title: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
}

interface StepIndicatorProps {
  steps: Step[];
  currentStep: number;
  className?: string;
  variant?: "horizontal" | "vertical";
}

export function StepIndicator({ 
  steps, 
  currentStep, 
  className,
  variant = "horizontal" 
}: StepIndicatorProps) {
  return (
    <div className={cn(
      "flex",
      variant === "horizontal" ? "items-center justify-between" : "flex-col space-y-4",
      className
    )}>
      {steps.map((step, index) => {
        const isCompleted = currentStep > step.number;
        const isCurrent = currentStep === step.number;
        const isUpcoming = currentStep < step.number;

        return (
          <div
            key={step.number}
            className={cn(
              "flex items-center",
              variant === "horizontal" ? "flex-col text-center" : "flex-row space-x-4"
            )}
          >
            {/* Step circle */}
            <motion.div
              className={cn(
                "relative flex items-center justify-center rounded-full border-2 transition-all duration-300",
                variant === "horizontal" ? "h-10 w-10 mb-2" : "h-8 w-8 flex-shrink-0",
                isCompleted && "border-green-500 bg-green-500 text-white",
                isCurrent && "border-blue-500 bg-blue-500 text-white",
                isUpcoming && "border-gray-300 bg-white text-gray-400 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-500"
              )}
              initial={false}
              animate={{
                scale: isCurrent ? 1.1 : 1,
                borderColor: isCompleted ? "#10b981" : isCurrent ? "#3b82f6" : "#d1d5db"
              }}
              transition={{ duration: 0.2 }}
            >
              {isCompleted ? (
                <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <Icons.check className={cn(variant === "horizontal" ? "h-5 w-5" : "h-4 w-4")} />
                </motion.div>
              ) : step.icon ? (
                <step.icon className={cn(variant === "horizontal" ? "h-5 w-5" : "h-4 w-4")} />
              ) : (
                <span className={cn(
                  "font-semibold",
                  variant === "horizontal" ? "text-sm" : "text-xs"
                )}>
                  {step.number}
                </span>
              )}

              {/* Pulse animation for current step */}
              {isCurrent && (
                <motion.div
                  className="absolute inset-0 rounded-full border-2 border-blue-500"
                  initial={{ scale: 1, opacity: 1 }}
                  animate={{ scale: 1.5, opacity: 0 }}
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: "easeOut"
                  }}
                />
              )}
            </motion.div>

            {/* Step content */}
            <div className={cn(
              variant === "horizontal" ? "text-center" : "flex-1"
            )}>
              <motion.h3
                className={cn(
                  "font-medium transition-colors duration-300",
                  variant === "horizontal" ? "text-sm" : "text-sm",
                  isCompleted && "text-green-700 dark:text-green-400",
                  isCurrent && "text-blue-700 dark:text-blue-400",
                  isUpcoming && "text-gray-500 dark:text-gray-400"
                )}
                initial={false}
                animate={{
                  color: isCompleted ? "#047857" : isCurrent ? "#1d4ed8" : "#6b7280"
                }}
              >
                {step.title}
              </motion.h3>
              
              {step.description && variant !== "horizontal" && (
                <motion.p
                  className={cn(
                    "text-xs text-gray-500 dark:text-gray-400 mt-1",
                    isCurrent && "text-gray-600 dark:text-gray-300"
                  )}
                  initial={false}
                  animate={{
                    opacity: isCurrent ? 1 : 0.7
                  }}
                >
                  {step.description}
                </motion.p>
              )}
            </div>

            {/* Connector line (horizontal variant) */}
            {variant === "horizontal" && index < steps.length - 1 && (
              <motion.div
                className="flex-1 h-px bg-gray-300 dark:bg-gray-600 mx-4"
                initial={false}
                animate={{
                  backgroundColor: currentStep > step.number ? "#10b981" : "#d1d5db"
                }}
                transition={{ duration: 0.3 }}
              />
            )}

            {/* Connector line (vertical variant) */}
            {variant === "vertical" && index < steps.length - 1 && (
              <motion.div
                className="absolute left-4 top-8 w-px h-8 bg-gray-300 dark:bg-gray-600"
                style={{ transform: "translateX(-50%)" }}
                initial={false}
                animate={{
                  backgroundColor: currentStep > step.number ? "#10b981" : "#d1d5db"
                }}
                transition={{ duration: 0.3 }}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}