import { useState, useEffect } from "react"
import { toast as hotToast } from "react-hot-toast"

export interface Toast {
  id: string
  title?: string
  description?: string
  action?: React.ReactNode
  variant?: "default" | "destructive"
}

interface ToastOptions {
  title?: string
  description?: string
  variant?: "default" | "destructive"
  duration?: number
}

export function useToast() {
  const toast = (options: ToastOptions) => {
    const { title, description, variant = "default", duration = 4000 } = options
    
    const message = (
      <div className="grid gap-1">
        {title && <div className="text-sm font-semibold">{title}</div>}
        {description && <div className="text-sm opacity-90">{description}</div>}
      </div>
    )
    
    if (variant === "destructive") {
      hotToast.error(message, { duration })
    } else {
      hotToast.success(message, { duration })
    }
  }
  
  return { toast }
}