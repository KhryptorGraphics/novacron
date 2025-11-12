import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * VM configuration interface
 */
export interface VMConfiguration {
  name: string;
  template?: string;
  cluster: string;
  cpu: number;
  memory: number;
  disk: number;
  networkInterface?: string;
  storageProfile?: string;
  tags?: string[];
  autoStart?: boolean;
  highAvailability?: boolean;
}

/**
 * VM creation step
 */
export type VMCreationStep = 'basic' | 'resources' | 'network' | 'storage' | 'advanced' | 'review';

/**
 * VM Create Page Object Model
 * Handles VM creation wizard and configuration
 */
export class VMCreatePage extends BasePage {
  // Wizard navigation
  private readonly stepIndicator: Locator;
  private readonly nextButton: Locator;
  private readonly backButton: Locator;
  private readonly cancelButton: Locator;
  private readonly createButton: Locator;

  // Basic Information (Step 1)
  private readonly nameInput: Locator;
  private readonly descriptionInput: Locator;
  private readonly templateSelect: Locator;
  private readonly osTypeSelect: Locator;
  private readonly clusterSelect: Locator;

  // Resource Configuration (Step 2)
  private readonly cpuInput: Locator;
  private readonly cpuSlider: Locator;
  private readonly memoryInput: Locator;
  private readonly memorySlider: Locator;
  private readonly memoryUnitSelect: Locator;

  // Storage Configuration (Step 3)
  private readonly diskSizeInput: Locator;
  private readonly diskSizeSlider: Locator;
  private readonly diskUnitSelect: Locator;
  private readonly storageProfileSelect: Locator;
  private readonly addDiskButton: Locator;

  // Network Configuration (Step 4)
  private readonly networkInterfaceSelect: Locator;
  private readonly ipAddressInput: Locator;
  private readonly dhcpCheckbox: Locator;
  private readonly addNetworkButton: Locator;

  // Advanced Settings (Step 5)
  private readonly autoStartCheckbox: Locator;
  private readonly highAvailabilityCheckbox: Locator;
  private readonly bootOrderSelect: Locator;
  private readonly cloudInitCheckbox: Locator;
  private readonly cloudInitEditor: Locator;

  // Tags
  private readonly tagInput: Locator;
  private readonly addTagButton: Locator;

  // Review (Step 6)
  private readonly configurationSummary: Locator;
  private readonly estimatedCost: Locator;

  // Progress
  private readonly creationProgress: Locator;
  private readonly progressStatus: Locator;

  constructor(page: Page) {
    super(page);

    // Wizard navigation
    this.stepIndicator = this.page.getByTestId('vm-create-step-indicator');
    this.nextButton = this.page.getByTestId('vm-create-next-button');
    this.backButton = this.page.getByTestId('vm-create-back-button');
    this.cancelButton = this.page.getByTestId('vm-create-cancel-button');
    this.createButton = this.page.getByTestId('vm-create-submit-button');

    // Basic Information
    this.nameInput = this.page.getByTestId('vm-create-name-input');
    this.descriptionInput = this.page.getByTestId('vm-create-description-input');
    this.templateSelect = this.page.getByTestId('vm-create-template-select');
    this.osTypeSelect = this.page.getByTestId('vm-create-os-type-select');
    this.clusterSelect = this.page.getByTestId('vm-create-cluster-select');

    // Resource Configuration
    this.cpuInput = this.page.getByTestId('vm-create-cpu-input');
    this.cpuSlider = this.page.getByTestId('vm-create-cpu-slider');
    this.memoryInput = this.page.getByTestId('vm-create-memory-input');
    this.memorySlider = this.page.getByTestId('vm-create-memory-slider');
    this.memoryUnitSelect = this.page.getByTestId('vm-create-memory-unit-select');

    // Storage Configuration
    this.diskSizeInput = this.page.getByTestId('vm-create-disk-size-input');
    this.diskSizeSlider = this.page.getByTestId('vm-create-disk-size-slider');
    this.diskUnitSelect = this.page.getByTestId('vm-create-disk-unit-select');
    this.storageProfileSelect = this.page.getByTestId('vm-create-storage-profile-select');
    this.addDiskButton = this.page.getByTestId('vm-create-add-disk-button');

    // Network Configuration
    this.networkInterfaceSelect = this.page.getByTestId('vm-create-network-interface-select');
    this.ipAddressInput = this.page.getByTestId('vm-create-ip-address-input');
    this.dhcpCheckbox = this.page.getByTestId('vm-create-dhcp-checkbox');
    this.addNetworkButton = this.page.getByTestId('vm-create-add-network-button');

    // Advanced Settings
    this.autoStartCheckbox = this.page.getByTestId('vm-create-auto-start-checkbox');
    this.highAvailabilityCheckbox = this.page.getByTestId('vm-create-ha-checkbox');
    this.bootOrderSelect = this.page.getByTestId('vm-create-boot-order-select');
    this.cloudInitCheckbox = this.page.getByTestId('vm-create-cloud-init-checkbox');
    this.cloudInitEditor = this.page.getByTestId('vm-create-cloud-init-editor');

    // Tags
    this.tagInput = this.page.getByTestId('vm-create-tag-input');
    this.addTagButton = this.page.getByTestId('vm-create-add-tag-button');

    // Review
    this.configurationSummary = this.page.getByTestId('vm-create-summary');
    this.estimatedCost = this.page.getByTestId('vm-create-estimated-cost');

    // Progress
    this.creationProgress = this.page.getByTestId('vm-create-progress');
    this.progressStatus = this.page.getByTestId('vm-create-progress-status');
  }

  /**
   * Navigate to VM create page
   * @returns VMCreatePage instance for chaining
   */
  async navigate(): Promise<this> {
    await this.goto('/vms/create');
    await this.waitForLoad();
    return this;
  }

  /**
   * Fill basic information (Step 1)
   * @param name - VM name
   * @param cluster - Cluster name
   * @param template - Template name (optional)
   * @param description - VM description (optional)
   * @returns VMCreatePage instance for chaining
   */
  async fillBasicInfo(
    name: string,
    cluster: string,
    template?: string,
    description?: string
  ): Promise<this> {
    await this.nameInput.fill(name);
    await this.clusterSelect.selectOption(cluster);

    if (template) {
      await this.templateSelect.selectOption(template);
    }

    if (description) {
      await this.descriptionInput.fill(description);
    }

    return this;
  }

  /**
   * Configure resources (Step 2)
   * @param cpu - Number of CPUs
   * @param memory - Memory size
   * @param memoryUnit - Memory unit (MB, GB)
   * @returns VMCreatePage instance for chaining
   */
  async configureResources(
    cpu: number,
    memory: number,
    memoryUnit: 'MB' | 'GB' = 'GB'
  ): Promise<this> {
    await this.cpuInput.fill(cpu.toString());
    await this.memoryInput.fill(memory.toString());
    await this.memoryUnitSelect.selectOption(memoryUnit);
    return this;
  }

  /**
   * Configure storage (Step 3)
   * @param diskSize - Disk size
   * @param diskUnit - Disk unit (GB, TB)
   * @param storageProfile - Storage profile (optional)
   * @returns VMCreatePage instance for chaining
   */
  async configureStorage(
    diskSize: number,
    diskUnit: 'GB' | 'TB' = 'GB',
    storageProfile?: string
  ): Promise<this> {
    await this.diskSizeInput.fill(diskSize.toString());
    await this.diskUnitSelect.selectOption(diskUnit);

    if (storageProfile) {
      await this.storageProfileSelect.selectOption(storageProfile);
    }

    return this;
  }

  /**
   * Add additional disk
   * @param diskSize - Disk size
   * @param diskUnit - Disk unit
   * @returns VMCreatePage instance for chaining
   */
  async addDisk(diskSize: number, diskUnit: 'GB' | 'TB' = 'GB'): Promise<this> {
    await this.addDiskButton.click();
    const diskIndex = await this.page.getByTestId('vm-disk-item').count();
    await this.page
      .getByTestId(`vm-disk-size-${diskIndex}`)
      .fill(diskSize.toString());
    await this.page.getByTestId(`vm-disk-unit-${diskIndex}`).selectOption(diskUnit);
    return this;
  }

  /**
   * Configure network (Step 4)
   * @param networkInterface - Network interface name
   * @param useDHCP - Whether to use DHCP
   * @param ipAddress - Static IP address (if not using DHCP)
   * @returns VMCreatePage instance for chaining
   */
  async configureNetwork(
    networkInterface: string,
    useDHCP: boolean = true,
    ipAddress?: string
  ): Promise<this> {
    await this.networkInterfaceSelect.selectOption(networkInterface);

    if (useDHCP) {
      await this.dhcpCheckbox.check();
    } else {
      await this.dhcpCheckbox.uncheck();
      if (ipAddress) {
        await this.ipAddressInput.fill(ipAddress);
      }
    }

    return this;
  }

  /**
   * Configure advanced settings (Step 5)
   * @param autoStart - Enable auto-start
   * @param highAvailability - Enable high availability
   * @returns VMCreatePage instance for chaining
   */
  async configureAdvanced(
    autoStart: boolean = false,
    highAvailability: boolean = false
  ): Promise<this> {
    if (autoStart) {
      await this.autoStartCheckbox.check();
    }

    if (highAvailability) {
      await this.highAvailabilityCheckbox.check();
    }

    return this;
  }

  /**
   * Add tag to VM
   * @param tag - Tag name
   * @returns VMCreatePage instance for chaining
   */
  async addTag(tag: string): Promise<this> {
    await this.tagInput.fill(tag);
    await this.addTagButton.click();
    return this;
  }

  /**
   * Add multiple tags
   * @param tags - Array of tag names
   * @returns VMCreatePage instance for chaining
   */
  async addTags(tags: string[]): Promise<this> {
    for (const tag of tags) {
      await this.addTag(tag);
    }
    return this;
  }

  /**
   * Click next button to proceed to next step
   * @returns VMCreatePage instance for chaining
   */
  async clickNext(): Promise<this> {
    await this.nextButton.click();
    await this.waitForLoadingComplete();
    return this;
  }

  /**
   * Click back button to go to previous step
   * @returns VMCreatePage instance for chaining
   */
  async clickBack(): Promise<this> {
    await this.backButton.click();
    return this;
  }

  /**
   * Click cancel button
   * @param confirm - Whether to confirm cancellation
   * @returns VMCreatePage instance for chaining
   */
  async clickCancel(confirm: boolean = true): Promise<this> {
    await this.cancelButton.click();

    if (confirm) {
      await this.page.getByTestId('confirm-cancel-button').click();
      await this.page.waitForURL('**/vms');
    } else {
      await this.page.getByTestId('dismiss-cancel-button').click();
    }

    return this;
  }

  /**
   * Submit VM creation
   * @returns VMCreatePage instance for chaining
   */
  async submit(): Promise<this> {
    await this.createButton.click();
    return this;
  }

  /**
   * Create VM with full configuration
   * @param config - VM configuration
   * @returns VMCreatePage instance for chaining
   */
  async createVM(config: VMConfiguration): Promise<this> {
    // Step 1: Basic Info
    await this.fillBasicInfo(config.name, config.cluster, config.template);
    await this.clickNext();

    // Step 2: Resources
    await this.configureResources(config.cpu, config.memory);
    await this.clickNext();

    // Step 3: Storage
    await this.configureStorage(config.disk, 'GB', config.storageProfile);
    await this.clickNext();

    // Step 4: Network
    if (config.networkInterface) {
      await this.configureNetwork(config.networkInterface);
    }
    await this.clickNext();

    // Step 5: Advanced
    await this.configureAdvanced(config.autoStart, config.highAvailability);

    if (config.tags && config.tags.length > 0) {
      await this.addTags(config.tags);
    }

    await this.clickNext();

    // Step 6: Review and Submit
    await this.submit();

    return this;
  }

  /**
   * Create VM and wait for completion
   * @param config - VM configuration
   * @param timeout - Maximum time to wait
   * @returns VMCreatePage instance for chaining
   */
  async createVMAndWaitForCompletion(
    config: VMConfiguration,
    timeout: number = 60000
  ): Promise<this> {
    await this.createVM(config);
    await this.waitForCreationComplete(timeout);
    return this;
  }

  /**
   * Wait for VM creation to complete
   * @param timeout - Maximum time to wait
   * @returns VMCreatePage instance for chaining
   */
  async waitForCreationComplete(timeout: number = 60000): Promise<this> {
    await this.page.waitForURL('**/vms/**', { timeout });
    await this.waitForToast('VM created successfully');
    return this;
  }

  /**
   * Get current wizard step
   * @returns Current step name
   */
  async getCurrentStep(): Promise<VMCreationStep> {
    const stepText = await this.stepIndicator.getAttribute('data-step');
    return (stepText as VMCreationStep) || 'basic';
  }

  /**
   * Get configuration summary text
   * @returns Configuration summary
   */
  async getConfigurationSummary(): Promise<string> {
    return (await this.configurationSummary.textContent()) || '';
  }

  /**
   * Get estimated cost
   * @returns Estimated cost text
   */
  async getEstimatedCost(): Promise<string> {
    return (await this.estimatedCost.textContent()) || '';
  }

  /**
   * Get creation progress percentage
   * @returns Progress percentage (0-100)
   */
  async getCreationProgress(): Promise<number> {
    const progressText = await this.creationProgress.getAttribute('aria-valuenow');
    return progressText ? parseInt(progressText) : 0;
  }

  /**
   * Get creation status message
   * @returns Status message
   */
  async getCreationStatus(): Promise<string> {
    return (await this.progressStatus.textContent()) || '';
  }

  /**
   * Check if next button is disabled
   * @returns True if next button is disabled
   */
  async isNextButtonDisabled(): Promise<boolean> {
    return await this.nextButton.isDisabled();
  }

  /**
   * Check if create button is disabled
   * @returns True if create button is disabled
   */
  async isCreateButtonDisabled(): Promise<boolean> {
    return await this.createButton.isDisabled();
  }

  /**
   * Get field validation error
   * @param fieldName - Field name
   * @returns Validation error text
   */
  async getFieldValidationError(fieldName: string): Promise<string> {
    const errorLocator = this.page.getByTestId(`vm-create-${fieldName}-error`);
    return (await errorLocator.textContent()) || '';
  }
}
