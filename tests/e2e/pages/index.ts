/**
 * Page Object Models Index
 * Centralized export of all page objects for NovaCron E2E tests
 */

// Base Page
export { BasePage } from './base-page';
export type { RequestInterceptionOptions, ScreenshotOptions } from './base-page';

// Authentication Pages
export { LoginPage } from './auth/login-page';
export type { LoginCredentials } from './auth/login-page';

export { RegisterPage } from './auth/register-page';
export type { RegistrationData, RegistrationStep } from './auth/register-page';

export { PasswordResetPage } from './auth/password-reset-page';
export type { PasswordResetFlow } from './auth/password-reset-page';

// VM Management Pages
export { VMListPage } from './vms/vm-list-page';
export type { VMFilterOptions, VMSortField, SortDirection } from './vms/vm-list-page';

export { VMCreatePage } from './vms/vm-create-page';
export type { VMConfiguration, VMCreationStep } from './vms/vm-create-page';

export { VMDetailsPage } from './vms/vm-details-page';
export type { VMOperation, VMTab } from './vms/vm-details-page';

export { VMConsolePage } from './vms/vm-console-page';
export type { ConsoleType, ConsoleQuality } from './vms/vm-console-page';

// Cluster Management Pages
export { ClusterOverviewPage } from './cluster/cluster-overview-page';
export type { ClusterHealth } from './cluster/cluster-overview-page';

export { NodeManagementPage } from './cluster/node-management-page';
export type { NodeStatus, NodeOperation } from './cluster/node-management-page';

export { FederationPage } from './cluster/federation-page';
export type { FederationConfig, FederationStatus } from './cluster/federation-page';

// Monitoring Pages
export { MonitoringDashboardPage } from './monitoring/dashboard-page';
export type { TimeRange } from './monitoring/dashboard-page';

export { MetricsPage } from './monitoring/metrics-page';
export type { MetricType, ChartType } from './monitoring/metrics-page';

export { AlertsPage } from './monitoring/alerts-page';
export type { AlertSeverity, AlertStatus, AlertRuleConfig } from './monitoring/alerts-page';

// Migration Pages
export { MigrationWizardPage } from './migration/migration-wizard-page';
export type { MigrationConfig, MigrationStep } from './migration/migration-wizard-page';

export { MigrationStatusPage } from './migration/migration-status-page';
export type { MigrationStatus, MigrationFilterOptions } from './migration/migration-status-page';
