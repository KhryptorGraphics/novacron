import { Page, Locator, expect } from '@playwright/test';

/**
 * Test Helper Utilities
 *
 * Custom assertions, wait functions, and test utilities
 */

/**
 * Custom wait for condition with timeout
 */
export async function waitForCondition(
  condition: () => Promise<boolean>,
  options?: { timeout?: number; interval?: number; message?: string }
): Promise<void> {
  const timeout = options?.timeout || 5000;
  const interval = options?.interval || 100;
  const message = options?.message || 'Condition was not met';

  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error(`${message} (timeout: ${timeout}ms)`);
}

/**
 * Wait for element to have specific text
 */
export async function waitForText(
  locator: Locator,
  text: string | RegExp,
  timeout = 5000
): Promise<void> {
  await expect(locator).toHaveText(text, { timeout });
}

/**
 * Wait for element to contain text
 */
export async function waitForTextContains(
  locator: Locator,
  text: string,
  timeout = 5000
): Promise<void> {
  await expect(locator).toContainText(text, { timeout });
}

/**
 * Assert element has attribute with value
 */
export async function assertAttribute(
  locator: Locator,
  attribute: string,
  value: string | RegExp
): Promise<void> {
  await expect(locator).toHaveAttribute(attribute, value);
}

/**
 * Assert element has CSS property
 */
export async function assertCssProperty(
  locator: Locator,
  property: string,
  value: string | RegExp
): Promise<void> {
  await expect(locator).toHaveCSS(property, value);
}

/**
 * Assert element count matches
 */
export async function assertCount(
  locator: Locator,
  count: number
): Promise<void> {
  await expect(locator).toHaveCount(count);
}

/**
 * Assert element is enabled
 */
export async function assertEnabled(locator: Locator): Promise<void> {
  await expect(locator).toBeEnabled();
}

/**
 * Assert element is disabled
 */
export async function assertDisabled(locator: Locator): Promise<void> {
  await expect(locator).toBeDisabled();
}

/**
 * Assert element is checked
 */
export async function assertChecked(locator: Locator): Promise<void> {
  await expect(locator).toBeChecked();
}

/**
 * Assert element is not checked
 */
export async function assertNotChecked(locator: Locator): Promise<void> {
  await expect(locator).not.toBeChecked();
}

/**
 * Assert element is focused
 */
export async function assertFocused(locator: Locator): Promise<void> {
  await expect(locator).toBeFocused();
}

/**
 * Assert URL contains string
 */
export async function assertUrlContains(
  page: Page,
  urlPart: string
): Promise<void> {
  await expect(page).toHaveURL(new RegExp(urlPart));
}

/**
 * Assert URL matches exactly
 */
export async function assertUrlEquals(
  page: Page,
  url: string | RegExp
): Promise<void> {
  await expect(page).toHaveURL(url);
}

/**
 * Assert page title
 */
export async function assertTitle(
  page: Page,
  title: string | RegExp
): Promise<void> {
  await expect(page).toHaveTitle(title);
}

/**
 * Assert console message appeared
 */
export async function waitForConsoleMessage(
  page: Page,
  messagePattern: string | RegExp,
  type?: 'log' | 'error' | 'warning' | 'info'
): Promise<void> {
  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error(`Console message not found: ${messagePattern}`));
    }, 5000);

    page.on('console', (msg) => {
      if (type && msg.type() !== type) return;

      const text = msg.text();
      const matches = typeof messagePattern === 'string'
        ? text.includes(messagePattern)
        : messagePattern.test(text);

      if (matches) {
        clearTimeout(timeout);
        resolve();
      }
    });
  });
}

/**
 * Assert no console errors
 */
export async function assertNoConsoleErrors(page: Page): Promise<void> {
  const errors: string[] = [];

  page.on('console', (msg) => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });

  // Wait a bit to collect any errors
  await new Promise((resolve) => setTimeout(resolve, 1000));

  if (errors.length > 0) {
    throw new Error(`Console errors found:\n${errors.join('\n')}`);
  }
}

/**
 * Wait for animation to complete
 */
export async function waitForAnimation(
  locator: Locator,
  duration = 1000
): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, duration));
  // Also wait for element to be stable
  const box1 = await locator.boundingBox();
  await new Promise((resolve) => setTimeout(resolve, 100));
  const box2 = await locator.boundingBox();

  if (box1 && box2) {
    if (box1.x !== box2.x || box1.y !== box2.y) {
      // Element is still moving, wait more
      await waitForAnimation(locator, duration);
    }
  }
}

/**
 * Retry action with exponential backoff
 */
export async function retryWithBackoff<T>(
  action: () => Promise<T>,
  options?: {
    maxRetries?: number;
    initialDelay?: number;
    maxDelay?: number;
    backoffMultiplier?: number;
  }
): Promise<T> {
  const maxRetries = options?.maxRetries || 3;
  const initialDelay = options?.initialDelay || 1000;
  const maxDelay = options?.maxDelay || 10000;
  const backoffMultiplier = options?.backoffMultiplier || 2;

  let delay = initialDelay;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await action();
    } catch (error) {
      if (i === maxRetries - 1) throw error;

      await new Promise((resolve) => setTimeout(resolve, delay));
      delay = Math.min(delay * backoffMultiplier, maxDelay);
    }
  }

  throw new Error('Retry failed');
}

/**
 * Measure action performance
 */
export async function measurePerformance<T>(
  action: () => Promise<T>,
  label?: string
): Promise<{ result: T; duration: number }> {
  const startTime = Date.now();
  const result = await action();
  const duration = Date.now() - startTime;

  if (label) {
    console.log(`⏱️  ${label}: ${duration}ms`);
  }

  return { result, duration };
}

/**
 * Assert performance threshold
 */
export async function assertPerformance<T>(
  action: () => Promise<T>,
  maxDuration: number,
  label?: string
): Promise<T> {
  const { result, duration } = await measurePerformance(action, label);

  if (duration > maxDuration) {
    throw new Error(
      `Performance assertion failed: ${label || 'Action'} took ${duration}ms (max: ${maxDuration}ms)`
    );
  }

  return result;
}

/**
 * Poll until condition is true
 */
export async function pollUntil(
  condition: () => Promise<boolean>,
  options?: {
    timeout?: number;
    interval?: number;
    message?: string;
  }
): Promise<void> {
  const timeout = options?.timeout || 10000;
  const interval = options?.interval || 500;
  const message = options?.message || 'Polling condition not met';

  const startTime = Date.now();

  while (Date.now() - startTime < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error(`${message} (timeout: ${timeout}ms)`);
}

/**
 * Wait for multiple elements to be visible
 */
export async function waitForAll(
  locators: Locator[],
  timeout = 5000
): Promise<void> {
  await Promise.all(
    locators.map((locator) => expect(locator).toBeVisible({ timeout }))
  );
}

/**
 * Wait for any element to be visible
 */
export async function waitForAny(
  locators: Locator[],
  timeout = 5000
): Promise<number> {
  const promises = locators.map((locator, index) =>
    expect(locator).toBeVisible({ timeout }).then(() => index)
  );

  return await Promise.race(promises);
}

/**
 * Assert elements are in order
 */
export async function assertElementsInOrder(
  page: Page,
  selector: string,
  expectedOrder: string[]
): Promise<void> {
  const elements = await page.locator(selector).all();
  const actualOrder = await Promise.all(
    elements.map((el) => el.textContent())
  );

  expect(actualOrder.map((t) => t?.trim())).toEqual(expectedOrder);
}

/**
 * Assert array contains
 */
export function assertArrayContains<T>(
  actual: T[],
  expected: T[]
): void {
  expected.forEach((item) => {
    expect(actual).toContain(item);
  });
}

/**
 * Assert object partial match
 */
export function assertObjectContains(
  actual: Record<string, any>,
  expected: Record<string, any>
): void {
  Object.keys(expected).forEach((key) => {
    expect(actual[key]).toEqual(expected[key]);
  });
}

/**
 * Sleep/delay utility
 */
export async function sleep(ms: number): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Random number generator
 */
export function randomInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

/**
 * Random string generator
 */
export function randomString(length: number): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let result = '';
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}
