import { Page, Locator, expect } from '@playwright/test';
import { promises as fs } from 'fs';
import path from 'path';

/**
 * Network request interception options
 */
export interface RequestInterceptionOptions {
  urlPattern?: string | RegExp;
  method?: string;
  postData?: any;
  headers?: Record<string, string>;
}

/**
 * Screenshot options
 */
export interface ScreenshotOptions {
  fullPage?: boolean;
  clip?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
}

/**
 * Base Page Object Model
 * Provides common functionality for all page objects
 */
export class BasePage {
  readonly page: Page;
  protected readonly baseUrl: string;

  constructor(page: Page) {
    this.page = page;
    this.baseUrl = process.env.BASE_URL || 'http://localhost:3000';
  }

  /**
   * Navigate to a specific path
   * @param path - Path to navigate to (relative to baseUrl)
   * @returns BasePage instance for chaining
   */
  async goto(path: string = ''): Promise<this> {
    await this.page.goto(`${this.baseUrl}${path}`);
    return this;
  }

  /**
   * Wait for page to be fully loaded
   * @param timeout - Maximum time to wait in milliseconds
   * @returns BasePage instance for chaining
   */
  async waitForLoad(timeout: number = 30000): Promise<this> {
    await this.page.waitForLoadState('domcontentloaded', { timeout });
    await this.page.waitForLoadState('networkidle', { timeout });
    return this;
  }

  /**
   * Wait for a specific selector to be visible
   * @param selector - CSS selector or test ID
   * @param timeout - Maximum time to wait in milliseconds
   * @returns Locator for the element
   */
  async waitForSelector(
    selector: string,
    timeout: number = 10000
  ): Promise<Locator> {
    const locator = this.page.locator(selector);
    await locator.waitFor({ state: 'visible', timeout });
    return locator;
  }

  /**
   * Wait for element by test ID
   * @param testId - Test ID attribute value
   * @param timeout - Maximum time to wait in milliseconds
   * @returns Locator for the element
   */
  async waitForTestId(testId: string, timeout: number = 10000): Promise<Locator> {
    const locator = this.page.getByTestId(testId);
    await locator.waitFor({ state: 'visible', timeout });
    return locator;
  }

  /**
   * Wait for text to appear on the page
   * @param text - Text to wait for
   * @param timeout - Maximum time to wait in milliseconds
   * @returns Locator for the element containing the text
   */
  async waitForText(text: string, timeout: number = 10000): Promise<Locator> {
    const locator = this.page.getByText(text);
    await locator.waitFor({ state: 'visible', timeout });
    return locator;
  }

  /**
   * Click an element by test ID
   * @param testId - Test ID attribute value
   * @returns BasePage instance for chaining
   */
  async clickByTestId(testId: string): Promise<this> {
    await this.page.getByTestId(testId).click();
    return this;
  }

  /**
   * Fill input by test ID
   * @param testId - Test ID attribute value
   * @param value - Value to fill
   * @returns BasePage instance for chaining
   */
  async fillByTestId(testId: string, value: string): Promise<this> {
    await this.page.getByTestId(testId).fill(value);
    return this;
  }

  /**
   * Select option by test ID
   * @param testId - Test ID attribute value
   * @param value - Value to select
   * @returns BasePage instance for chaining
   */
  async selectByTestId(testId: string, value: string): Promise<this> {
    await this.page.getByTestId(testId).selectOption(value);
    return this;
  }

  /**
   * Check if element is visible
   * @param selector - CSS selector or test ID
   * @returns True if element is visible
   */
  async isVisible(selector: string): Promise<boolean> {
    try {
      return await this.page.locator(selector).isVisible();
    } catch {
      return false;
    }
  }

  /**
   * Check if element by test ID is visible
   * @param testId - Test ID attribute value
   * @returns True if element is visible
   */
  async isTestIdVisible(testId: string): Promise<boolean> {
    try {
      return await this.page.getByTestId(testId).isVisible();
    } catch {
      return false;
    }
  }

  /**
   * Get text content of an element
   * @param selector - CSS selector or test ID
   * @returns Text content
   */
  async getText(selector: string): Promise<string> {
    return (await this.page.locator(selector).textContent()) || '';
  }

  /**
   * Get text content by test ID
   * @param testId - Test ID attribute value
   * @returns Text content
   */
  async getTextByTestId(testId: string): Promise<string> {
    return (await this.page.getByTestId(testId).textContent()) || '';
  }

  /**
   * Wait for API response
   * @param urlPattern - URL pattern to match
   * @param timeout - Maximum time to wait in milliseconds
   * @returns Response object
   */
  async waitForApiResponse(
    urlPattern: string | RegExp,
    timeout: number = 30000
  ) {
    return await this.page.waitForResponse(urlPattern, { timeout });
  }

  /**
   * Intercept network request
   * @param options - Request interception options
   * @param callback - Callback to execute when request is intercepted
   */
  async interceptRequest(
    options: RequestInterceptionOptions,
    callback: (route: any) => void
  ): Promise<void> {
    await this.page.route(options.urlPattern || '**/*', (route) => {
      const request = route.request();

      // Check if request matches criteria
      if (options.method && request.method() !== options.method) {
        return route.continue();
      }

      callback(route);
    });
  }

  /**
   * Mock API response
   * @param urlPattern - URL pattern to match
   * @param response - Response to return
   * @param status - HTTP status code
   */
  async mockApiResponse(
    urlPattern: string | RegExp,
    response: any,
    status: number = 200
  ): Promise<void> {
    await this.page.route(urlPattern, (route) => {
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(response),
      });
    });
  }

  /**
   * Take a screenshot
   * @param filename - Filename for the screenshot
   * @param options - Screenshot options
   */
  async takeScreenshot(
    filename: string,
    options: ScreenshotOptions = { fullPage: true }
  ): Promise<void> {
    const screenshotDir = path.join(process.cwd(), 'test-results', 'screenshots');
    await fs.mkdir(screenshotDir, { recursive: true });

    const filepath = path.join(screenshotDir, filename);
    await this.page.screenshot({ path: filepath, ...options });
  }

  /**
   * Wait for loading indicator to disappear
   * @param testId - Test ID of loading indicator
   * @param timeout - Maximum time to wait
   * @returns BasePage instance for chaining
   */
  async waitForLoadingComplete(
    testId: string = 'loading-spinner',
    timeout: number = 30000
  ): Promise<this> {
    try {
      await this.page.getByTestId(testId).waitFor({ state: 'hidden', timeout });
    } catch {
      // Loading indicator might not exist, continue
    }
    return this;
  }

  /**
   * Wait for toast/notification to appear
   * @param message - Expected message text
   * @param timeout - Maximum time to wait
   * @returns Locator for the toast
   */
  async waitForToast(message?: string, timeout: number = 10000): Promise<Locator> {
    const toastLocator = message
      ? this.page.getByRole('alert').filter({ hasText: message })
      : this.page.getByRole('alert');

    await toastLocator.waitFor({ state: 'visible', timeout });
    return toastLocator;
  }

  /**
   * Assert that success toast appears
   * @param message - Expected success message
   */
  async assertSuccessToast(message: string): Promise<void> {
    const toast = await this.waitForToast(message);
    await expect(toast).toBeVisible();
  }

  /**
   * Assert that error toast appears
   * @param message - Expected error message
   */
  async assertErrorToast(message: string): Promise<void> {
    const toast = await this.waitForToast(message);
    await expect(toast).toBeVisible();
  }

  /**
   * Scroll element into view
   * @param selector - CSS selector
   * @returns BasePage instance for chaining
   */
  async scrollIntoView(selector: string): Promise<this> {
    await this.page.locator(selector).scrollIntoViewIfNeeded();
    return this;
  }

  /**
   * Scroll to element by test ID
   * @param testId - Test ID attribute value
   * @returns BasePage instance for chaining
   */
  async scrollToTestId(testId: string): Promise<this> {
    await this.page.getByTestId(testId).scrollIntoViewIfNeeded();
    return this;
  }

  /**
   * Hover over element
   * @param selector - CSS selector
   * @returns BasePage instance for chaining
   */
  async hover(selector: string): Promise<this> {
    await this.page.locator(selector).hover();
    return this;
  }

  /**
   * Hover over element by test ID
   * @param testId - Test ID attribute value
   * @returns BasePage instance for chaining
   */
  async hoverTestId(testId: string): Promise<this> {
    await this.page.getByTestId(testId).hover();
    return this;
  }

  /**
   * Press keyboard key
   * @param key - Key to press
   * @returns BasePage instance for chaining
   */
  async pressKey(key: string): Promise<this> {
    await this.page.keyboard.press(key);
    return this;
  }

  /**
   * Clear input field
   * @param selector - CSS selector
   * @returns BasePage instance for chaining
   */
  async clearInput(selector: string): Promise<this> {
    await this.page.locator(selector).clear();
    return this;
  }

  /**
   * Clear input field by test ID
   * @param testId - Test ID attribute value
   * @returns BasePage instance for chaining
   */
  async clearInputByTestId(testId: string): Promise<this> {
    await this.page.getByTestId(testId).clear();
    return this;
  }

  /**
   * Get current URL
   * @returns Current page URL
   */
  getCurrentUrl(): string {
    return this.page.url();
  }

  /**
   * Get page title
   * @returns Page title
   */
  async getTitle(): Promise<string> {
    return await this.page.title();
  }

  /**
   * Reload the page
   * @returns BasePage instance for chaining
   */
  async reload(): Promise<this> {
    await this.page.reload();
    return this;
  }

  /**
   * Go back in browser history
   * @returns BasePage instance for chaining
   */
  async goBack(): Promise<this> {
    await this.page.goBack();
    return this;
  }

  /**
   * Go forward in browser history
   * @returns BasePage instance for chaining
   */
  async goForward(): Promise<this> {
    await this.page.goForward();
    return this;
  }

  /**
   * Execute JavaScript in the page context
   * @param script - JavaScript code to execute
   * @param args - Arguments to pass to the script
   * @returns Result of script execution
   */
  async executeScript<T = any>(script: string | Function, ...args: any[]): Promise<T> {
    return await this.page.evaluate(script, ...args);
  }

  /**
   * Wait for specific amount of time
   * @param milliseconds - Time to wait
   * @returns BasePage instance for chaining
   */
  async wait(milliseconds: number): Promise<this> {
    await this.page.waitForTimeout(milliseconds);
    return this;
  }

  /**
   * Close the current page
   */
  async close(): Promise<void> {
    await this.page.close();
  }
}
