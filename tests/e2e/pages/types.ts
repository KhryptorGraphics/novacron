/**
 * Common types and interfaces for Page Object Models
 */

/**
 * Page navigation options
 */
export interface NavigationOptions {
  waitUntil?: 'load' | 'domcontentloaded' | 'networkidle' | 'commit';
  timeout?: number;
}

/**
 * Element wait options
 */
export interface WaitOptions {
  state?: 'attached' | 'detached' | 'visible' | 'hidden';
  timeout?: number;
}

/**
 * Filter base interface
 */
export interface BaseFilter {
  search?: string;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}

/**
 * Pagination info
 */
export interface PaginationInfo {
  currentPage: number;
  totalPages: number;
  pageSize: number;
  totalItems: number;
}

/**
 * Action confirmation options
 */
export interface ConfirmationOptions {
  confirm?: boolean;
  note?: string;
}

/**
 * Resource status
 */
export type ResourceStatus = 'active' | 'inactive' | 'error' | 'pending';

/**
 * Health status
 */
export type HealthStatus = 'healthy' | 'warning' | 'critical' | 'unknown';

/**
 * Priority level
 */
export type Priority = 'low' | 'normal' | 'high' | 'critical';

/**
 * Time range for metrics and charts
 */
export interface TimeRangeConfig {
  start: string;
  end: string;
  granularity?: 'second' | 'minute' | 'hour' | 'day';
}

/**
 * Chart data point
 */
export interface ChartDataPoint {
  timestamp: number;
  value: number;
  label?: string;
}

/**
 * Resource metrics
 */
export interface ResourceMetrics {
  cpu?: number;
  memory?: number;
  disk?: number;
  network?: number;
}

/**
 * User preferences
 */
export interface UserPreferences {
  theme?: 'light' | 'dark' | 'auto';
  language?: string;
  timezone?: string;
  notifications?: boolean;
}

/**
 * Notification configuration
 */
export interface NotificationConfig {
  type: 'email' | 'sms' | 'webhook' | 'slack';
  endpoint: string;
  enabled: boolean;
}

/**
 * Bulk operation result
 */
export interface BulkOperationResult {
  successful: number;
  failed: number;
  total: number;
  errors?: Array<{ id: string; error: string }>;
}

/**
 * API response wrapper
 */
export interface APIResponse<T = any> {
  data?: T;
  error?: string;
  statusCode: number;
  message?: string;
}

/**
 * Form validation error
 */
export interface ValidationError {
  field: string;
  message: string;
  code?: string;
}

/**
 * Modal dialog options
 */
export interface ModalOptions {
  title?: string;
  message?: string;
  confirmText?: string;
  cancelText?: string;
  variant?: 'info' | 'warning' | 'error' | 'success';
}

/**
 * Table column configuration
 */
export interface TableColumn {
  key: string;
  label: string;
  sortable?: boolean;
  width?: string;
}

/**
 * Dashboard widget configuration
 */
export interface WidgetConfig {
  id: string;
  type: string;
  title: string;
  position: { x: number; y: number };
  size: { width: number; height: number };
  config?: Record<string, any>;
}

/**
 * Search result item
 */
export interface SearchResult {
  id: string;
  title: string;
  description?: string;
  type: string;
  url?: string;
}

/**
 * Activity log entry
 */
export interface ActivityLogEntry {
  id: string;
  timestamp: string;
  user: string;
  action: string;
  resource: string;
  details?: string;
}

/**
 * Resource tag
 */
export interface ResourceTag {
  key: string;
  value: string;
  color?: string;
}

/**
 * Export options
 */
export interface ExportOptions {
  format: 'csv' | 'json' | 'pdf' | 'xlsx';
  filename?: string;
  fields?: string[];
}

/**
 * Import result
 */
export interface ImportResult {
  imported: number;
  failed: number;
  total: number;
  errors?: Array<{ line: number; error: string }>;
}

/**
 * Webhook configuration
 */
export interface WebhookConfig {
  url: string;
  method: 'GET' | 'POST' | 'PUT';
  headers?: Record<string, string>;
  payload?: Record<string, any>;
}

/**
 * Schedule configuration
 */
export interface ScheduleConfig {
  enabled: boolean;
  type: 'once' | 'recurring';
  datetime?: string;
  cron?: string;
  timezone?: string;
}

/**
 * Backup configuration
 */
export interface BackupConfig {
  name: string;
  type: 'full' | 'incremental' | 'differential';
  retention: number;
  schedule?: ScheduleConfig;
  destination: string;
}

/**
 * Quota information
 */
export interface QuotaInfo {
  used: number;
  limit: number;
  unit: string;
  percentage: number;
}

/**
 * Network configuration
 */
export interface NetworkConfig {
  name: string;
  type: 'bridged' | 'nat' | 'host-only' | 'internal';
  vlan?: number;
  subnet?: string;
  gateway?: string;
  dns?: string[];
}

/**
 * Storage configuration
 */
export interface StorageConfig {
  name: string;
  type: 'local' | 'nfs' | 'iscsi' | 'ceph';
  path?: string;
  capacity: number;
  allocated: number;
  available: number;
}

/**
 * Permission set
 */
export interface Permission {
  resource: string;
  actions: Array<'read' | 'write' | 'delete' | 'execute'>;
}

/**
 * Role configuration
 */
export interface RoleConfig {
  name: string;
  description?: string;
  permissions: Permission[];
  users?: string[];
}

/**
 * Audit log entry
 */
export interface AuditLogEntry extends ActivityLogEntry {
  ipAddress?: string;
  userAgent?: string;
  result: 'success' | 'failure';
}
