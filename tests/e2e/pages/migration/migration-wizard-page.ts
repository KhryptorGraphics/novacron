import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Migration configuration interface
 */
export interface MigrationConfig {
  sourceVM: string;
  targetCluster: string;
  targetNode?: string;
  migrationType: 'live' | 'offline';
  priority?: 'low' | 'normal' | 'high';
  bandwidthLimit?: number;
  schedule?: string;
}

/**
 * Migration step type
 */
export type MigrationStep = 'source' | 'destination' | 'options' | 'validation' | 'confirmation';

/**
 * Migration Wizard Page Object Model
 * Handles VM migration wizard workflow
 */
export class MigrationWizardPage extends BasePage {
  // Wizard navigation
  private readonly stepIndicator: Locator;
  private readonly nextButton: Locator;
  private readonly backButton: Locator;
  private readonly cancelButton: Locator;
  private readonly startMigrationButton: Locator;

  // Step 1: Source Selection
  private readonly sourceVMSelect: Locator;
  private readonly sourceClusterInfo: Locator;
  private readonly sourceNodeInfo: Locator;
  private readonly vmDetailsPanel: Locator;

  // Step 2: Destination Selection
  private readonly targetClusterSelect: Locator;
  private readonly targetNodeSelect: Locator;
  private readonly autoSelectNodeCheckbox: Locator;
  private readonly clusterCapacityInfo: Locator;

  // Step 3: Migration Options
  private readonly migrationTypeSelect: Locator;
  private readonly prioritySelect: Locator;
  private readonly bandwidthLimitInput: Locator;
  private readonly enableCompressionCheckbox: Locator;
  private readonly verifyDataCheckbox: Locator;
  private readonly scheduleCheckbox: Locator;
  private readonly scheduleDatetimeInput: Locator;

  // Step 4: Validation
  private readonly validationStatus: Locator;
  private readonly validationChecks: Locator;
  private readonly validationErrors: Locator;
  private readonly validationWarnings: Locator;

  // Step 5: Confirmation
  private readonly migrationSummary: Locator;
  private readonly estimatedDuration: Locator;
  private readonly estimatedDowntime: Locator;

  // Progress
  private readonly progressStatus: Locator;

  constructor(page: Page) {
    super(page);

    // Wizard navigation
    this.stepIndicator = this.page.getByTestId('migration-wizard-step-indicator');
    this.nextButton = this.page.getByTestId('migration-wizard-next-button');
    this.backButton = this.page.getByTestId('migration-wizard-back-button');
    this.cancelButton = this.page.getByTestId('migration-wizard-cancel-button');
    this.startMigrationButton = this.page.getByTestId('migration-wizard-start-button');

    // Source Selection
    this.sourceVMSelect = this.page.getByTestId('migration-source-vm-select');
    this.sourceClusterInfo = this.page.getByTestId('migration-source-cluster-info');
    this.sourceNodeInfo = this.page.getByTestId('migration-source-node-info');
    this.vmDetailsPanel = this.page.getByTestId('migration-vm-details-panel');

    // Destination Selection
    this.targetClusterSelect = this.page.getByTestId('migration-target-cluster-select');
    this.targetNodeSelect = this.page.getByTestId('migration-target-node-select');
    this.autoSelectNodeCheckbox = this.page.getByTestId('migration-auto-select-node-checkbox');
    this.clusterCapacityInfo = this.page.getByTestId('migration-cluster-capacity-info');

    // Migration Options
    this.migrationTypeSelect = this.page.getByTestId('migration-type-select');
    this.prioritySelect = this.page.getByTestId('migration-priority-select');
    this.bandwidthLimitInput = this.page.getByTestId('migration-bandwidth-limit-input');
    this.enableCompressionCheckbox = this.page.getByTestId('migration-compression-checkbox');
    this.verifyDataCheckbox = this.page.getByTestId('migration-verify-data-checkbox');
    this.scheduleCheckbox = this.page.getByTestId('migration-schedule-checkbox');
    this.scheduleDatetimeInput = this.page.getByTestId('migration-schedule-datetime-input');

    // Validation
    this.validationStatus = this.page.getByTestId('migration-validation-status');
    this.validationChecks = this.page.getByTestId('migration-validation-checks');
    this.validationErrors = this.page.getByTestId('migration-validation-errors');
    this.validationWarnings = this.page.getByTestId('migration-validation-warnings');

    // Confirmation
    this.migrationSummary = this.page.getByTestId('migration-summary');
    this.estimatedDuration = this.page.getByTestId('migration-estimated-duration');
    this.estimatedDowntime = this.page.getByTestId('migration-estimated-downtime');

    // Progress
    this.progressStatus = this.page.getByTestId('migration-progress-status');
  }

  /**
   * Navigate to migration wizard
   * @returns MigrationWizardPage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/migration/new');
    await this.waitForLoad();
    return this;
  }

  /**
   * Select source VM (Step 1)
   * @param vmName - VM name
   * @returns MigrationWizardPage instance for chaining
   */
  async selectSourceVM(vmName: string): Promise<this> {
    await this.sourceVMSelect.selectOption(vmName);
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Select destination cluster (Step 2)
   * @param clusterName - Target cluster name
   * @param autoSelectNode - Whether to auto-select target node
   * @returns MigrationWizardPage instance for chaining
   */
  async selectDestination(
    clusterName: string,
    autoSelectNode: boolean = true
  ): Promise<this> {
    await this.targetClusterSelect.selectOption(clusterName);

    if (autoSelectNode) {
      await this.autoSelectNodeCheckbox.check();
    } else {
      await this.autoSelectNodeCheckbox.uncheck();
    }

    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Select specific target node
   * @param nodeName - Target node name
   * @returns MigrationWizardPage instance for chaining
   */
  async selectTargetNode(nodeName: string): Promise<this> {
    await this.autoSelectNodeCheckbox.uncheck();
    await this.targetNodeSelect.selectOption(nodeName);
    return this;
  }

  /**
   * Configure migration options (Step 3)
   * @param type - Migration type
   * @param priority - Migration priority
   * @param bandwidthLimit - Bandwidth limit in Mbps (optional)
   * @returns MigrationWizardPage instance for chaining
   */
  async configureMigration(
    type: 'live' | 'offline',
    priority: 'low' | 'normal' | 'high' = 'normal',
    bandwidthLimit?: number
  ): Promise<this> {
    await this.migrationTypeSelect.selectOption(type);
    await this.prioritySelect.selectOption(priority);

    if (bandwidthLimit) {
      await this.bandwidthLimitInput.fill(bandwidthLimit.toString());
    }

    return this;
  }

  /**
   * Enable compression
   * @param enable - Whether to enable compression
   * @returns MigrationWizardPage instance for chaining
   */
  async setCompression(enable: boolean = true): Promise<this> {
    if (enable) {
      await this.enableCompressionCheckbox.check();
    } else {
      await this.enableCompressionCheckbox.uncheck();
    }
    return this;
  }

  /**
   * Enable data verification
   * @param enable - Whether to enable verification
   * @returns MigrationWizardPage instance for chaining
   */
  async setDataVerification(enable: boolean = true): Promise<this> {
    if (enable) {
      await this.verifyDataCheckbox.check();
    } else {
      await this.verifyDataCheckbox.uncheck();
    }
    return this;
  }

  /**
   * Schedule migration
   * @param datetime - Schedule datetime (ISO format)
   * @returns MigrationWizardPage instance for chaining
   */
  async scheduleMigration(datetime: string): Promise<this> {
    await this.scheduleCheckbox.check();
    await this.scheduleDatetimeInput.fill(datetime);
    return this;
  }

  /**
   * Click next button
   * @returns MigrationWizardPage instance for chaining
   */
  async clickNext(): Promise<this> {
    await this.nextButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click back button
   * @returns MigrationWizardPage instance for chaining
   */
  async clickBack(): Promise<this> {
    await this.backButton.click();
    return this;
  }

  /**
   * Cancel migration wizard
   * @param confirm - Whether to confirm cancellation
   * @returns MigrationWizardPage instance for chaining
   */
  async cancel(confirm: boolean = true): Promise<this> {
    await this.cancelButton.click();

    if (confirm) {
      await this.page.getByTestId('confirm-cancel-migration').click();
      await this.page.waitForURL('**/migration');
    } else {
      await this.page.getByTestId('dismiss-cancel-migration').click();
    }

    return this;
  }

  /**
   * Start migration
   * @returns MigrationWizardPage instance for chaining
   */
  async startMigration(): Promise<this> {
    await this.startMigrationButton.click();
    return this;
  }

  /**
   * Complete migration wizard
   * @param config - Migration configuration
   * @returns MigrationWizardPage instance for chaining
   */
  async completeMigration(config: MigrationConfig): Promise<this> {
    // Step 1: Select source
    await this.selectSourceVM(config.sourceVM);
    await this.clickNext();

    // Step 2: Select destination
    await this.selectDestination(config.targetCluster, !config.targetNode);
    if (config.targetNode) {
      await this.selectTargetNode(config.targetNode);
    }
    await this.clickNext();

    // Step 3: Configure options
    await this.configureMigration(
      config.migrationType,
      config.priority,
      config.bandwidthLimit
    );

    if (config.schedule) {
      await this.scheduleMigration(config.schedule);
    }

    await this.clickNext();

    // Step 4: Validation
    await this.waitForValidation();
    await this.clickNext();

    // Step 5: Confirmation and start
    await this.startMigration();

    return this;
  }

  /**
   * Wait for validation to complete
   * @param timeout - Maximum time to wait
   * @returns MigrationWizardPage instance for chaining
   */
  async waitForValidation(timeout: number = 30000): Promise<this> {
    await this.page.waitForFunction(
      () => {
        const status = document.querySelector(
          '[data-testid="migration-validation-status"]'
        );
        return (
          status?.getAttribute('data-status') === 'complete' ||
          status?.getAttribute('data-status') === 'failed'
        );
      },
      {},
      { timeout }
    );
    return this;
  }

  /**
   * Get validation status
   * @returns Validation status
   */
  async getValidationStatus(): Promise<string> {
    return (await this.validationStatus.getAttribute('data-status')) || '';
  }

  /**
   * Get validation errors
   * @returns Array of error messages
   */
  async getValidationErrors(): Promise<string[]> {
    const errors = await this.validationErrors.locator('[data-testid="error-item"]').all();
    const errorTexts: string[] = [];

    for (const error of errors) {
      const text = await error.textContent();
      if (text) errorTexts.push(text);
    }

    return errorTexts;
  }

  /**
   * Get validation warnings
   * @returns Array of warning messages
   */
  async getValidationWarnings(): Promise<string[]> {
    const warnings = await this.validationWarnings
      .locator('[data-testid="warning-item"]')
      .all();
    const warningTexts: string[] = [];

    for (const warning of warnings) {
      const text = await warning.textContent();
      if (text) warningTexts.push(text);
    }

    return warningTexts;
  }

  /**
   * Get current wizard step
   * @returns Current step
   */
  async getCurrentStep(): Promise<MigrationStep> {
    const stepText = await this.stepIndicator.getAttribute('data-step');
    return (stepText as MigrationStep) || 'source';
  }

  /**
   * Get migration summary
   * @returns Summary text
   */
  async getMigrationSummary(): Promise<string> {
    return (await this.migrationSummary.textContent()) || '';
  }

  /**
   * Get estimated duration
   * @returns Duration text
   */
  async getEstimatedDuration(): Promise<string> {
    return (await this.estimatedDuration.textContent()) || '';
  }

  /**
   * Get estimated downtime
   * @returns Downtime text
   */
  async getEstimatedDowntime(): Promise<string> {
    return (await this.estimatedDowntime.textContent()) || '';
  }

  /**
   * Check if next button is disabled
   * @returns True if disabled
   */
  async isNextButtonDisabled(): Promise<boolean> {
    return await this.nextButton.isDisabled();
  }

  /**
   * Check if start migration button is disabled
   * @returns True if disabled
   */
  async isStartButtonDisabled(): Promise<boolean> {
    return await this.startMigrationButton.isDisabled();
  }
}
