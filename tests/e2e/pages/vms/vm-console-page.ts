import { Page, Locator } from '@playwright/test';
import { BasePage } from '../base-page';

/**
 * Console connection type
 */
export type ConsoleType = 'vnc' | 'spice' | 'serial';

/**
 * Console quality setting
 */
export type ConsoleQuality = 'low' | 'medium' | 'high' | 'auto';

/**
 * VM Console Page Object Model
 * Handles VM console access and interaction
 */
export class VMConsolePage extends BasePage {
  // Console container
  private readonly consoleContainer: Locator;
  private readonly consoleCanvas: Locator;
  private readonly consoleStatus: Locator;
  private readonly connectionIndicator: Locator;

  // Controls
  private readonly connectButton: Locator;
  private readonly disconnectButton: Locator;
  private readonly fullscreenButton: Locator;
  private readonly screenshotButton: Locator;
  private readonly sendCtrlAltDelButton: Locator;
  private readonly consoleTypeSelect: Locator;
  private readonly qualitySelect: Locator;

  // Keyboard
  private readonly keyboardToggle: Locator;
  private readonly virtualKeyboard: Locator;

  // Clipboard
  private readonly clipboardButton: Locator;
  private readonly clipboardDialog: Locator;
  private readonly clipboardInput: Locator;
  private readonly sendClipboardButton: Locator;

  // Settings
  private readonly settingsButton: Locator;
  private readonly settingsPanel: Locator;
  private readonly scaleToFitCheckbox: Locator;
  private readonly autoResizeCheckbox: Locator;

  // Status indicators
  private readonly latencyIndicator: Locator;
  private readonly bitrateIndicator: Locator;

  constructor(page: Page) {
    super(page);

    // Console container
    this.consoleContainer = this.page.getByTestId('vm-console-container');
    this.consoleCanvas = this.page.getByTestId('vm-console-canvas');
    this.consoleStatus = this.page.getByTestId('vm-console-status');
    this.connectionIndicator = this.page.getByTestId('vm-console-connection');

    // Controls
    this.connectButton = this.page.getByTestId('vm-console-connect-button');
    this.disconnectButton = this.page.getByTestId('vm-console-disconnect-button');
    this.fullscreenButton = this.page.getByTestId('vm-console-fullscreen-button');
    this.screenshotButton = this.page.getByTestId('vm-console-screenshot-button');
    this.sendCtrlAltDelButton = this.page.getByTestId('vm-console-ctrl-alt-del-button');
    this.consoleTypeSelect = this.page.getByTestId('vm-console-type-select');
    this.qualitySelect = this.page.getByTestId('vm-console-quality-select');

    // Keyboard
    this.keyboardToggle = this.page.getByTestId('vm-console-keyboard-toggle');
    this.virtualKeyboard = this.page.getByTestId('vm-console-virtual-keyboard');

    // Clipboard
    this.clipboardButton = this.page.getByTestId('vm-console-clipboard-button');
    this.clipboardDialog = this.page.getByTestId('vm-console-clipboard-dialog');
    this.clipboardInput = this.page.getByTestId('vm-console-clipboard-input');
    this.sendClipboardButton = this.page.getByTestId('vm-console-send-clipboard-button');

    // Settings
    this.settingsButton = this.page.getByTestId('vm-console-settings-button');
    this.settingsPanel = this.page.getByTestId('vm-console-settings-panel');
    this.scaleToFitCheckbox = this.page.getByTestId('vm-console-scale-to-fit-checkbox');
    this.autoResizeCheckbox = this.page.getByTestId('vm-console-auto-resize-checkbox');

    // Status
    this.latencyIndicator = this.page.getByTestId('vm-console-latency');
    this.bitrateIndicator = this.page.getByTestId('vm-console-bitrate');
  }

  /**
   * Navigate to VM console page
   * @param vmId - VM ID or name
   * @returns VMConsolePage instance for chaining
   */
  async navigate(vmId: string): Promise<this> {
    await this.goto(`/vms/${vmId}/console`);
    await this.waitForLoad();
    return this;
  }

  /**
   * Connect to VM console
   * @param consoleType - Console type (vnc, spice, serial)
   * @returns VMConsolePage instance for chaining
   */
  async connect(consoleType: ConsoleType = 'vnc'): Promise<this> {
    await this.consoleTypeSelect.selectOption(consoleType);
    await this.connectButton.click();
    await this.waitForConnection();
    return this;
  }

  /**
   * Disconnect from VM console
   * @returns VMConsolePage instance for chaining
   */
  async disconnect(): Promise<this> {
    await this.disconnectButton.click();
    await this.waitForDisconnection();
    return this;
  }

  /**
   * Wait for console connection to establish
   * @param timeout - Maximum time to wait
   * @returns VMConsolePage instance for chaining
   */
  async waitForConnection(timeout: number = 30000): Promise<this> {
    await this.page.waitForFunction(
      () => {
        const indicator = document.querySelector('[data-testid="vm-console-connection"]');
        return indicator?.getAttribute('data-status') === 'connected';
      },
      {},
      { timeout }
    );
    return this;
  }

  /**
   * Wait for console disconnection
   * @param timeout - Maximum time to wait
   * @returns VMConsolePage instance for chaining
   */
  async waitForDisconnection(timeout: number = 10000): Promise<this> {
    await this.page.waitForFunction(
      () => {
        const indicator = document.querySelector('[data-testid="vm-console-connection"]');
        return indicator?.getAttribute('data-status') === 'disconnected';
      },
      {},
      { timeout }
    );
    return this;
  }

  /**
   * Check if console is connected
   * @returns True if connected
   */
  async isConnected(): Promise<boolean> {
    const status = await this.connectionIndicator.getAttribute('data-status');
    return status === 'connected';
  }

  /**
   * Get connection status
   * @returns Connection status
   */
  async getConnectionStatus(): Promise<string> {
    return (await this.consoleStatus.textContent()) || '';
  }

  /**
   * Enter fullscreen mode
   * @returns VMConsolePage instance for chaining
   */
  async enterFullscreen(): Promise<this> {
    await this.fullscreenButton.click();
    return this;
  }

  /**
   * Exit fullscreen mode
   * @returns VMConsolePage instance for chaining
   */
  async exitFullscreen(): Promise<this> {
    await this.pressKey('Escape');
    return this;
  }

  /**
   * Take console screenshot
   * @returns VMConsolePage instance for chaining
   */
  async takeConsoleScreenshot(): Promise<this> {
    await this.screenshotButton.click();
    await this.waitForToast('Screenshot saved');
    return this;
  }

  /**
   * Send Ctrl+Alt+Del to VM
   * @returns VMConsolePage instance for chaining
   */
  async sendCtrlAltDel(): Promise<this> {
    await this.sendCtrlAltDelButton.click();
    return this;
  }

  /**
   * Type text in console
   * @param text - Text to type
   * @returns VMConsolePage instance for chaining
   */
  async typeInConsole(text: string): Promise<this> {
    await this.consoleCanvas.click();
    await this.page.keyboard.type(text);
    return this;
  }

  /**
   * Press key combination in console
   * @param keys - Key combination (e.g., 'Control+C')
   * @returns VMConsolePage instance for chaining
   */
  async pressKeyCombo(keys: string): Promise<this> {
    await this.consoleCanvas.click();
    await this.page.keyboard.press(keys);
    return this;
  }

  /**
   * Toggle virtual keyboard
   * @param show - Whether to show or hide keyboard
   * @returns VMConsolePage instance for chaining
   */
  async toggleVirtualKeyboard(show: boolean = true): Promise<this> {
    const isVisible = await this.virtualKeyboard.isVisible();

    if (show && !isVisible) {
      await this.keyboardToggle.click();
    } else if (!show && isVisible) {
      await this.keyboardToggle.click();
    }

    return this;
  }

  /**
   * Send text via clipboard
   * @param text - Text to send
   * @returns VMConsolePage instance for chaining
   */
  async sendClipboardText(text: string): Promise<this> {
    await this.clipboardButton.click();
    await this.clipboardInput.fill(text);
    await this.sendClipboardButton.click();
    await this.waitForToast('Clipboard text sent');
    return this;
  }

  /**
   * Set console quality
   * @param quality - Quality setting
   * @returns VMConsolePage instance for chaining
   */
  async setQuality(quality: ConsoleQuality): Promise<this> {
    await this.qualitySelect.selectOption(quality);
    return this;
  }

  /**
   * Open settings panel
   * @returns VMConsolePage instance for chaining
   */
  async openSettings(): Promise<this> {
    await this.settingsButton.click();
    await this.settingsPanel.waitFor({ state: 'visible' });
    return this;
  }

  /**
   * Close settings panel
   * @returns VMConsolePage instance for chaining
   */
  async closeSettings(): Promise<this> {
    await this.pressKey('Escape');
    await this.settingsPanel.waitFor({ state: 'hidden' });
    return this;
  }

  /**
   * Enable scale to fit
   * @param enable - Whether to enable
   * @returns VMConsolePage instance for chaining
   */
  async setScaleToFit(enable: boolean = true): Promise<this> {
    await this.openSettings();

    if (enable) {
      await this.scaleToFitCheckbox.check();
    } else {
      await this.scaleToFitCheckbox.uncheck();
    }

    await this.closeSettings();
    return this;
  }

  /**
   * Enable auto resize
   * @param enable - Whether to enable
   * @returns VMConsolePage instance for chaining
   */
  async setAutoResize(enable: boolean = true): Promise<this> {
    await this.openSettings();

    if (enable) {
      await this.autoResizeCheckbox.check();
    } else {
      await this.autoResizeCheckbox.uncheck();
    }

    await this.closeSettings();
    return this;
  }

  /**
   * Get console latency
   * @returns Latency in milliseconds
   */
  async getLatency(): Promise<number> {
    const latencyText = await this.latencyIndicator.textContent();
    const match = latencyText?.match(/(\d+)/);
    return match ? parseInt(match[1]) : 0;
  }

  /**
   * Get console bitrate
   * @returns Bitrate text
   */
  async getBitrate(): Promise<string> {
    return (await this.bitrateIndicator.textContent()) || '';
  }

  /**
   * Click at specific coordinates in console
   * @param x - X coordinate
   * @param y - Y coordinate
   * @returns VMConsolePage instance for chaining
   */
  async clickAtPosition(x: number, y: number): Promise<this> {
    const box = await this.consoleCanvas.boundingBox();
    if (box) {
      await this.page.mouse.click(box.x + x, box.y + y);
    }
    return this;
  }

  /**
   * Double click at specific coordinates in console
   * @param x - X coordinate
   * @param y - Y coordinate
   * @returns VMConsolePage instance for chaining
   */
  async doubleClickAtPosition(x: number, y: number): Promise<this> {
    const box = await this.consoleCanvas.boundingBox();
    if (box) {
      await this.page.mouse.dblclick(box.x + x, box.y + y);
    }
    return this;
  }

  /**
   * Check if console canvas is visible
   * @returns True if canvas is visible
   */
  async isConsoleVisible(): Promise<boolean> {
    return await this.consoleCanvas.isVisible();
  }

  /**
   * Wait for console to be ready
   * @param timeout - Maximum time to wait
   * @returns VMConsolePage instance for chaining
   */
  async waitForConsoleReady(timeout: number = 30000): Promise<this> {
    await this.consoleCanvas.waitFor({ state: 'visible', timeout });
    await this.waitForConnection(timeout);
    return this;
  }
}
