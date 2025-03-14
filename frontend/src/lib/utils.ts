import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * Combines class names with tailwind's merge
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Format bytes to a human-readable string
 * @param bytes Number of bytes
 * @param decimals Number of decimal places to show
 * @returns Formatted string like "1.5 KB" or "2.3 GB"
 */
export function bytesToSize(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return "0 Bytes";

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

/**
 * Format a duration in milliseconds to a human-readable string
 * @param ms Duration in milliseconds
 * @returns Formatted string like "2h 30m" or "45s"
 */
export function formatDuration(ms: number): string {
  if (ms < 0) return "0s";
  
  const seconds = Math.floor((ms / 1000) % 60);
  const minutes = Math.floor((ms / (1000 * 60)) % 60);
  const hours = Math.floor((ms / (1000 * 60 * 60)) % 24);
  const days = Math.floor(ms / (1000 * 60 * 60 * 24));
  
  const parts = [];
  if (days > 0) parts.push(`${days}d`);
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (seconds > 0 || parts.length === 0) parts.push(`${seconds}s`);
  
  return parts.join(" ");
}

/**
 * Generate a random ID
 * @param length Length of the ID to generate
 * @returns Random ID string
 */
export function generateId(length: number = 8): string {
  return Math.random().toString(36).substring(2, 2 + length);
}

/**
 * Parse error objects to extract readable messages
 * @param error The error object
 * @returns A readable error message
 */
export function parseError(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  
  try {
    return JSON.stringify(error);
  } catch {
    return "An unknown error occurred";
  }
}
