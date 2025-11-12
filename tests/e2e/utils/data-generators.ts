/**
 * Test Data Generators
 *
 * Utilities for generating test data
 */

import { randomInt, randomString } from './test-helpers';

/**
 * Generate random email
 */
export function generateEmail(prefix?: string): string {
  const randomPart = randomString(8).toLowerCase();
  const domain = 'novacron-test.com';
  return prefix ? `${prefix}-${randomPart}@${domain}` : `${randomPart}@${domain}`;
}

/**
 * Generate random username
 */
export function generateUsername(prefix?: string): string {
  const randomPart = randomString(6).toLowerCase();
  return prefix ? `${prefix}_${randomPart}` : `user_${randomPart}`;
}

/**
 * Generate random password
 */
export function generatePassword(length = 12): string {
  const lowercase = 'abcdefghijklmnopqrstuvwxyz';
  const uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
  const numbers = '0123456789';
  const special = '!@#$%^&*()_+-=[]{}|;:,.<>?';
  const allChars = lowercase + uppercase + numbers + special;

  let password = '';

  // Ensure at least one of each type
  password += lowercase[randomInt(0, lowercase.length - 1)];
  password += uppercase[randomInt(0, uppercase.length - 1)];
  password += numbers[randomInt(0, numbers.length - 1)];
  password += special[randomInt(0, special.length - 1)];

  // Fill the rest randomly
  for (let i = 4; i < length; i++) {
    password += allChars[randomInt(0, allChars.length - 1)];
  }

  // Shuffle the password
  return password.split('').sort(() => Math.random() - 0.5).join('');
}

/**
 * Generate random phone number
 */
export function generatePhoneNumber(countryCode = '+1'): string {
  const areaCode = randomInt(200, 999);
  const prefix = randomInt(200, 999);
  const lineNumber = randomInt(1000, 9999);
  return `${countryCode}-${areaCode}-${prefix}-${lineNumber}`;
}

/**
 * Generate user data
 */
export interface UserData {
  username: string;
  email: string;
  password: string;
  firstName: string;
  lastName: string;
  phone: string;
  role: 'admin' | 'user' | 'viewer';
}

export function generateUser(overrides?: Partial<UserData>): UserData {
  const firstName = generateFirstName();
  const lastName = generateLastName();

  return {
    username: generateUsername(),
    email: generateEmail(),
    password: generatePassword(),
    firstName,
    lastName,
    phone: generatePhoneNumber(),
    role: 'user',
    ...overrides,
  };
}

/**
 * Generate VM data
 */
export interface VMData {
  name: string;
  description: string;
  cpu: number;
  memory: number; // in GB
  disk: number; // in GB
  os: 'ubuntu' | 'centos' | 'debian' | 'windows';
  status: 'running' | 'stopped' | 'pending' | 'error';
}

export function generateVM(overrides?: Partial<VMData>): VMData {
  const id = randomString(8).toLowerCase();

  return {
    name: `test-vm-${id}`,
    description: `Test VM created for E2E testing - ${id}`,
    cpu: randomInt(1, 8),
    memory: [2, 4, 8, 16][randomInt(0, 3)],
    disk: [20, 50, 100, 200][randomInt(0, 3)],
    os: ['ubuntu', 'centos', 'debian', 'windows'][randomInt(0, 3)] as any,
    status: 'stopped',
    ...overrides,
  };
}

/**
 * Generate project data
 */
export interface ProjectData {
  name: string;
  description: string;
  repository: string;
  branch: string;
  framework: string;
}

export function generateProject(overrides?: Partial<ProjectData>): ProjectData {
  const id = randomString(6).toLowerCase();

  return {
    name: `test-project-${id}`,
    description: `Test project for E2E testing - ${id}`,
    repository: `https://github.com/novacron/test-${id}`,
    branch: 'main',
    framework: ['react', 'vue', 'angular', 'nodejs'][randomInt(0, 3)],
    ...overrides,
  };
}

/**
 * Generate API key
 */
export function generateApiKey(prefix = 'nvc'): string {
  const key = randomString(32);
  return `${prefix}_${key}`;
}

/**
 * Generate token
 */
export function generateToken(type: 'jwt' | 'bearer' = 'bearer'): string {
  if (type === 'jwt') {
    const header = Buffer.from(JSON.stringify({ alg: 'HS256', typ: 'JWT' })).toString('base64');
    const payload = Buffer.from(JSON.stringify({ sub: '1234567890', name: 'Test User', iat: Date.now() })).toString('base64');
    const signature = randomString(32);
    return `${header}.${payload}.${signature}`;
  }
  return randomString(64);
}

/**
 * Generate date range
 */
export interface DateRange {
  startDate: Date;
  endDate: Date;
}

export function generateDateRange(daysAgo = 30): DateRange {
  const endDate = new Date();
  const startDate = new Date();
  startDate.setDate(startDate.getDate() - daysAgo);

  return { startDate, endDate };
}

/**
 * Generate first name
 */
function generateFirstName(): string {
  const firstNames = [
    'John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily',
    'Robert', 'Lisa', 'James', 'Mary', 'William', 'Jennifer',
    'Richard', 'Linda', 'Thomas', 'Patricia', 'Charles', 'Elizabeth',
  ];
  return firstNames[randomInt(0, firstNames.length - 1)];
}

/**
 * Generate last name
 */
function generateLastName(): string {
  const lastNames = [
    'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia',
    'Miller', 'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez',
    'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson',
  ];
  return lastNames[randomInt(0, lastNames.length - 1)];
}

/**
 * Generate address
 */
export interface Address {
  street: string;
  city: string;
  state: string;
  zipCode: string;
  country: string;
}

export function generateAddress(): Address {
  const streets = ['Main St', 'Oak Ave', 'Pine Rd', 'Maple Dr', 'Cedar Ln'];
  const cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'];
  const states = ['NY', 'CA', 'IL', 'TX', 'AZ'];

  return {
    street: `${randomInt(100, 9999)} ${streets[randomInt(0, streets.length - 1)]}`,
    city: cities[randomInt(0, cities.length - 1)],
    state: states[randomInt(0, states.length - 1)],
    zipCode: randomInt(10000, 99999).toString(),
    country: 'USA',
  };
}

/**
 * Generate credit card (test data only)
 */
export interface CreditCard {
  number: string;
  expiry: string;
  cvv: string;
  name: string;
}

export function generateCreditCard(): CreditCard {
  // Generate valid test credit card numbers (Luhn algorithm)
  const testCardNumbers = [
    '4111111111111111', // Visa
    '5555555555554444', // Mastercard
    '378282246310005',  // Amex
  ];

  const month = randomInt(1, 12).toString().padStart(2, '0');
  const year = (new Date().getFullYear() + randomInt(1, 5)).toString().slice(-2);

  return {
    number: testCardNumbers[randomInt(0, testCardNumbers.length - 1)],
    expiry: `${month}/${year}`,
    cvv: randomInt(100, 999).toString(),
    name: `${generateFirstName()} ${generateLastName()}`,
  };
}

/**
 * Generate lorem ipsum text
 */
export function generateLoremIpsum(sentences = 3): string {
  const lorem = [
    'Lorem ipsum dolor sit amet, consectetur adipiscing elit.',
    'Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
    'Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.',
    'Duis aute irure dolor in reprehenderit in voluptate velit esse.',
    'Excepteur sint occaecat cupidatat non proident, sunt in culpa.',
    'Qui officia deserunt mollit anim id est laborum.',
  ];

  let result = '';
  for (let i = 0; i < sentences; i++) {
    result += lorem[randomInt(0, lorem.length - 1)] + ' ';
  }

  return result.trim();
}

/**
 * Generate URL
 */
export function generateUrl(protocol: 'http' | 'https' = 'https'): string {
  const domains = ['example', 'test', 'demo', 'sample'];
  const tlds = ['com', 'org', 'net', 'io'];

  const domain = domains[randomInt(0, domains.length - 1)];
  const tld = tlds[randomInt(0, tlds.length - 1)];
  const path = randomString(8).toLowerCase();

  return `${protocol}://${domain}.${tld}/${path}`;
}

/**
 * Generate bulk test data
 */
export function generateBulkUsers(count: number): UserData[] {
  return Array.from({ length: count }, () => generateUser());
}

export function generateBulkVMs(count: number): VMData[] {
  return Array.from({ length: count }, () => generateVM());
}

export function generateBulkProjects(count: number): ProjectData[] {
  return Array.from({ length: count }, () => generateProject());
}
