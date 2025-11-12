import { Page, Locator, expect } from '@playwright/test';

/**
 * Playwright Helper Utilities
 *
 * Common utilities and helpers for Playwright tests
 */

/**
 * Wait for network to be idle
 */
export async function waitForNetworkIdle(page: Page, timeout = 5000): Promise<void> {
  await page.waitForLoadState('networkidle', { timeout });
}

/**
 * Wait for element to be visible and stable
 */
export async function waitForElementStable(locator: Locator, timeout = 5000): Promise<void> {
  await locator.waitFor({ state: 'visible', timeout });
  await expect(locator).toBeVisible({ timeout });
}

/**
 * Scroll element into view
 */
export async function scrollIntoView(locator: Locator): Promise<void> {
  await locator.scrollIntoViewIfNeeded();
}

/**
 * Click element with retry logic
 */
export async function clickWithRetry(
  locator: Locator,
  options?: { maxRetries?: number; delay?: number }
): Promise<void> {
  const maxRetries = options?.maxRetries || 3;
  const delay = options?.delay || 1000;

  for (let i = 0; i < maxRetries; i++) {
    try {
      await locator.click({ timeout: 5000 });
      return;
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
}

/**
 * Fill input with clear and verification
 */
export async function fillInput(
  locator: Locator,
  value: string,
  options?: { verify?: boolean }
): Promise<void> {
  await locator.clear();
  await locator.fill(value);

  if (options?.verify !== false) {
    await expect(locator).toHaveValue(value);
  }
}

/**
 * Select dropdown option by text
 */
export async function selectOption(
  locator: Locator,
  option: string | { label?: string; value?: string; index?: number }
): Promise<void> {
  if (typeof option === 'string') {
    await locator.selectOption({ label: option });
  } else if (option.label) {
    await locator.selectOption({ label: option.label });
  } else if (option.value) {
    await locator.selectOption({ value: option.value });
  } else if (option.index !== undefined) {
    await locator.selectOption({ index: option.index });
  }
}

/**
 * Upload file to input
 */
export async function uploadFile(
  locator: Locator,
  filePath: string | string[]
): Promise<void> {
  await locator.setInputFiles(filePath);
}

/**
 * Wait for API response
 */
export async function waitForResponse(
  page: Page,
  urlPattern: string | RegExp,
  options?: { status?: number; timeout?: number }
): Promise<void> {
  await page.waitForResponse(
    (response) => {
      const urlMatch = typeof urlPattern === 'string'
        ? response.url().includes(urlPattern)
        : urlPattern.test(response.url());
      const statusMatch = options?.status ? response.status() === options.status : true;
      return urlMatch && statusMatch;
    },
    { timeout: options?.timeout || 10000 }
  );
}

/**
 * Intercept and mock API response
 */
export async function mockApiResponse(
  page: Page,
  urlPattern: string | RegExp,
  response: any,
  options?: { status?: number; contentType?: string }
): Promise<void> {
  await page.route(urlPattern, (route) => {
    route.fulfill({
      status: options?.status || 200,
      contentType: options?.contentType || 'application/json',
      body: JSON.stringify(response),
    });
  });
}

/**
 * Get all text content from elements
 */
export async function getAllTextContent(locator: Locator): Promise<string[]> {
  const count = await locator.count();
  const texts: string[] = [];

  for (let i = 0; i < count; i++) {
    const text = await locator.nth(i).textContent();
    if (text) texts.push(text.trim());
  }

  return texts;
}

/**
 * Wait for element count
 */
export async function waitForElementCount(
  locator: Locator,
  count: number,
  timeout = 5000
): Promise<void> {
  await expect(locator).toHaveCount(count, { timeout });
}

/**
 * Take screenshot with custom name
 */
export async function takeScreenshot(
  page: Page,
  name: string,
  options?: { fullPage?: boolean }
): Promise<void> {
  await page.screenshot({
    path: `tests/e2e/reports/screenshots/${name}.png`,
    fullPage: options?.fullPage || false,
  });
}

/**
 * Get element bounding box
 */
export async function getBoundingBox(locator: Locator): Promise<{ x: number; y: number; width: number; height: number } | null> {
  return await locator.boundingBox();
}

/**
 * Hover over element
 */
export async function hover(locator: Locator): Promise<void> {
  await locator.hover();
}

/**
 * Double click element
 */
export async function doubleClick(locator: Locator): Promise<void> {
  await locator.dblclick();
}

/**
 * Right click element
 */
export async function rightClick(locator: Locator): Promise<void> {
  await locator.click({ button: 'right' });
}

/**
 * Drag and drop
 */
export async function dragAndDrop(
  source: Locator,
  target: Locator
): Promise<void> {
  await source.dragTo(target);
}

/**
 * Check if element is visible
 */
export async function isVisible(locator: Locator): Promise<boolean> {
  try {
    await expect(locator).toBeVisible({ timeout: 1000 });
    return true;
  } catch {
    return false;
  }
}

/**
 * Check if element is hidden
 */
export async function isHidden(locator: Locator): Promise<boolean> {
  try {
    await expect(locator).toBeHidden({ timeout: 1000 });
    return true;
  } catch {
    return false;
  }
}

/**
 * Wait for URL to contain string
 */
export async function waitForUrlContains(
  page: Page,
  urlPart: string,
  timeout = 5000
): Promise<void> {
  await page.waitForURL((url) => url.toString().includes(urlPart), { timeout });
}

/**
 * Navigate with wait for load
 */
export async function navigateAndWait(
  page: Page,
  url: string,
  options?: { waitUntil?: 'load' | 'domcontentloaded' | 'networkidle' }
): Promise<void> {
  await page.goto(url, { waitUntil: options?.waitUntil || 'networkidle' });
}

/**
 * Execute JavaScript in browser context
 */
export async function evaluateScript<T>(
  page: Page,
  script: string | ((args?: any) => T),
  args?: any
): Promise<T> {
  return await page.evaluate(script, args);
}

/**
 * Get local storage value
 */
export async function getLocalStorage(
  page: Page,
  key: string
): Promise<string | null> {
  return await page.evaluate((k) => localStorage.getItem(k), key);
}

/**
 * Set local storage value
 */
export async function setLocalStorage(
  page: Page,
  key: string,
  value: string
): Promise<void> {
  await page.evaluate(
    ({ k, v }) => localStorage.setItem(k, v),
    { k: key, v: value }
  );
}

/**
 * Clear local storage
 */
export async function clearLocalStorage(page: Page): Promise<void> {
  await page.evaluate(() => localStorage.clear());
}

/**
 * Get cookie value
 */
export async function getCookie(
  page: Page,
  name: string
): Promise<string | undefined> {
  const cookies = await page.context().cookies();
  return cookies.find((c) => c.name === name)?.value;
}

/**
 * Set cookie
 */
export async function setCookie(
  page: Page,
  name: string,
  value: string,
  options?: { domain?: string; path?: string }
): Promise<void> {
  await page.context().addCookies([
    {
      name,
      value,
      domain: options?.domain || 'localhost',
      path: options?.path || '/',
    },
  ]);
}

/**
 * Clear all cookies
 */
export async function clearCookies(page: Page): Promise<void> {
  await page.context().clearCookies();
}

/**
 * Reload page and wait
 */
export async function reloadPage(page: Page): Promise<void> {
  await page.reload({ waitUntil: 'networkidle' });
}

/**
 * Go back in history
 */
export async function goBack(page: Page): Promise<void> {
  await page.goBack({ waitUntil: 'networkidle' });
}

/**
 * Go forward in history
 */
export async function goForward(page: Page): Promise<void> {
  await page.goForward({ waitUntil: 'networkidle' });
}
