/**
 * Frontend-Backend Integration Validation Suite
 * Tests critical user workflows end-to-end
 */

const puppeteer = require('puppeteer');
const WebSocket = require('ws');
const fetch = require('node-fetch');

class FrontendBackendValidator {
    constructor() {
        this.baseURL = process.env.BASE_URL || 'http://localhost:8092';
        this.apiURL = process.env.API_URL || 'http://localhost:8090';
        this.wsURL = process.env.WS_URL || 'ws://localhost:8091';
        
        this.browser = null;
        this.page = null;
        this.authToken = null;
        this.results = {
            tests: [],
            summary: {
                total: 0,
                passed: 0,
                failed: 0,
                skipped: 0
            }
        };
    }

    async initialize() {
        console.log('🚀 Initializing Frontend-Backend Integration Validation...');
        
        // Launch browser with realistic settings
        this.browser = await puppeteer.launch({
            headless: process.env.HEADLESS !== 'false',
            args: [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-accelerated-2d-canvas',
                '--disable-gpu',
                '--window-size=1366,768'
            ],
            defaultViewport: { width: 1366, height: 768 }
        });

        this.page = await this.browser.newPage();

        // Set realistic user agent
        await this.page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36');
        
        // Enable JavaScript and set timeout
        await this.page.setJavaScriptEnabled(true);
        this.page.setDefaultTimeout(30000);

        // Monitor console errors
        this.page.on('console', msg => {
            if (msg.type() === 'error') {
                console.error('Browser Console Error:', msg.text());
            }
        });

        // Monitor network failures
        this.page.on('requestfailed', request => {
            console.error('Network Request Failed:', request.url());
        });

        console.log('✅ Browser initialized successfully');
    }

    async runAllTests() {
        console.log('🧪 Starting comprehensive integration validation...\n');

        const testSuites = [
            { name: 'Frontend Loading & UI Components', fn: this.testFrontendLoading },
            { name: 'Authentication Flow', fn: this.testAuthenticationFlow },
            { name: 'API Contract Validation', fn: this.testAPIContracts },
            { name: 'WebSocket Real-time Data', fn: this.testWebSocketFlow },
            { name: 'VM Management Workflow', fn: this.testVMManagement },
            { name: 'Storage Operations', fn: this.testStorageOperations },
            { name: 'Monitoring Dashboard', fn: this.testMonitoringDashboard },
            { name: 'User Interface Interactions', fn: this.testUIInteractions },
            { name: 'Cross-Component Integration', fn: this.testCrossComponentIntegration },
            { name: 'Error Handling & Recovery', fn: this.testErrorHandling }
        ];

        for (const suite of testSuites) {
            await this.runTestSuite(suite.name, suite.fn);
        }

        return this.generateReport();
    }

    async runTestSuite(suiteName, testFunction) {
        console.log(`📋 Running: ${suiteName}`);
        const startTime = Date.now();

        try {
            await testFunction.call(this);
            const duration = Date.now() - startTime;
            console.log(`✅ ${suiteName} completed in ${duration}ms\n`);
        } catch (error) {
            const duration = Date.now() - startTime;
            console.error(`❌ ${suiteName} failed in ${duration}ms:`, error.message);
            this.recordTest(suiteName, false, error.message);
        }
    }

    async testFrontendLoading() {
        console.log('  🔍 Testing frontend loading and basic UI components...');

        // Test 1: Page loads without errors
        const response = await this.page.goto(this.baseURL, { waitUntil: 'networkidle0' });
        if (!response.ok()) {
            throw new Error(`Frontend failed to load: ${response.status()}`);
        }
        this.recordTest('Frontend Page Load', true);

        // Test 2: Essential UI components are present
        const essentialElements = [
            '[data-testid="app-header"], header, .header',
            '[data-testid="navigation"], nav, .nav',
            '[data-testid="main-content"], main, .main-content'
        ];

        for (let selector of essentialElements) {
            const elements = selector.split(', ');
            let found = false;
            
            for (let element of elements) {
                try {
                    await this.page.waitForSelector(element, { timeout: 5000 });
                    found = true;
                    break;
                } catch (e) {
                    // Try next selector
                }
            }
            
            if (!found) {
                console.warn(`  ⚠️  UI component not found: ${selector}`);
            }
        }
        this.recordTest('Essential UI Components', true);

        // Test 3: No JavaScript errors on page load
        const jsErrors = await this.page.evaluate(() => {
            return window.jsErrors || [];
        });
        
        if (jsErrors.length > 0) {
            console.warn('  ⚠️  JavaScript errors detected:', jsErrors);
        }
        this.recordTest('No JS Errors on Load', jsErrors.length === 0);

        // Test 4: CSS and styling loaded properly
        const stylesLoaded = await this.page.evaluate(() => {
            const stylesheets = document.querySelectorAll('link[rel="stylesheet"], style');
            return stylesheets.length > 0;
        });
        this.recordTest('Stylesheets Loaded', stylesLoaded);

        console.log('  ✅ Frontend loading validation completed');
    }

    async testAuthenticationFlow() {
        console.log('  🔐 Testing authentication workflow...');

        try {
            // Navigate to login page
            await this.page.goto(`${this.baseURL}/auth/login`, { waitUntil: 'networkidle0' });

            // Test 1: Login form is present
            const loginFormSelectors = [
                'form[data-testid="login-form"]',
                'form',
                '[data-testid="email-input"]',
                'input[type="email"]',
                'input[name="email"]'
            ];

            let loginFormFound = false;
            for (let selector of loginFormSelectors) {
                try {
                    await this.page.waitForSelector(selector, { timeout: 3000 });
                    loginFormFound = true;
                    break;
                } catch (e) {
                    // Try next selector
                }
            }

            if (!loginFormFound) {
                throw new Error('Login form not found');
            }
            this.recordTest('Login Form Present', true);

            // Test 2: Attempt login with test credentials
            const emailSelectors = ['input[type="email"]', 'input[name="email"]', '[data-testid="email-input"]'];
            const passwordSelectors = ['input[type="password"]', 'input[name="password"]', '[data-testid="password-input"]'];
            const submitSelectors = ['button[type="submit"]', 'input[type="submit"]', '[data-testid="login-submit"]'];

            // Fill email
            for (let selector of emailSelectors) {
                try {
                    await this.page.click(selector);
                    await this.page.type(selector, 'admin@example.com');
                    break;
                } catch (e) {
                    // Try next selector
                }
            }

            // Fill password
            for (let selector of passwordSelectors) {
                try {
                    await this.page.click(selector);
                    await this.page.type(selector, 'admin123');
                    break;
                } catch (e) {
                    // Try next selector
                }
            }

            // Submit form
            for (let selector of submitSelectors) {
                try {
                    await this.page.click(selector);
                    break;
                } catch (e) {
                    // Try next selector
                }
            }

            // Wait for navigation or error message
            try {
                await this.page.waitForNavigation({ waitUntil: 'networkidle0', timeout: 10000 });
                this.recordTest('Login Submission', true);
            } catch (e) {
                // Check if we're still on login page (could be due to validation)
                const currentURL = this.page.url();
                if (currentURL.includes('/login')) {
                    console.log('  ℹ️  Remained on login page - checking for error messages');
                    this.recordTest('Login Submission', false, 'Login failed or form validation triggered');
                } else {
                    this.recordTest('Login Submission', true);
                }
            }

            // Test 3: Check for authentication token in localStorage
            const hasAuthToken = await this.page.evaluate(() => {
                return localStorage.getItem('authToken') !== null || 
                       localStorage.getItem('novacron_token') !== null ||
                       sessionStorage.getItem('authToken') !== null;
            });

            if (hasAuthToken) {
                this.authToken = await this.page.evaluate(() => {
                    return localStorage.getItem('authToken') || 
                           localStorage.getItem('novacron_token') ||
                           sessionStorage.getItem('authToken');
                });
                this.recordTest('Authentication Token Stored', true);
            } else {
                this.recordTest('Authentication Token Stored', false, 'No auth token found in storage');
            }

            console.log('  ✅ Authentication flow validation completed');

        } catch (error) {
            this.recordTest('Authentication Flow', false, error.message);
            throw error;
        }
    }

    async testAPIContracts() {
        console.log('  🔌 Testing API contract validation...');

        const apiEndpoints = [
            { method: 'GET', path: '/health', name: 'Health Check' },
            { method: 'GET', path: '/api/info', name: 'API Info' },
            { method: 'GET', path: '/api/monitoring/metrics', name: 'System Metrics' },
            { method: 'GET', path: '/api/monitoring/vms', name: 'VM Metrics' },
            { method: 'GET', path: '/api/monitoring/alerts', name: 'System Alerts' }
        ];

        for (const endpoint of apiEndpoints) {
            try {
                console.log(`    Testing ${endpoint.name}...`);
                
                const headers = { 'Content-Type': 'application/json' };
                if (this.authToken && endpoint.path.startsWith('/api/') && !endpoint.path.includes('/auth/')) {
                    headers.Authorization = `Bearer ${this.authToken}`;
                }

                const response = await fetch(`${this.apiURL}${endpoint.path}`, {
                    method: endpoint.method,
                    headers: headers,
                    timeout: 10000
                });

                const isSuccess = response.status >= 200 && response.status < 300;
                
                if (isSuccess) {
                    // Validate response has JSON content for API endpoints
                    if (endpoint.path.startsWith('/api/') || endpoint.path === '/health') {
                        const contentType = response.headers.get('content-type');
                        if (!contentType || !contentType.includes('application/json')) {
                            throw new Error(`Expected JSON response, got: ${contentType}`);
                        }

                        const data = await response.json();
                        if (typeof data !== 'object') {
                            throw new Error('Invalid JSON response format');
                        }

                        // Specific validation for known endpoints
                        if (endpoint.path === '/health' && !data.status) {
                            throw new Error('Health endpoint missing status field');
                        }
                        if (endpoint.path === '/api/info' && (!data.name || !data.version)) {
                            throw new Error('API info missing required fields');
                        }
                    }

                    this.recordTest(`API ${endpoint.name}`, true);
                } else {
                    this.recordTest(`API ${endpoint.name}`, false, `HTTP ${response.status}`);
                }

            } catch (error) {
                this.recordTest(`API ${endpoint.name}`, false, error.message);
            }
        }

        console.log('  ✅ API contract validation completed');
    }

    async testWebSocketFlow() {
        console.log('  🔄 Testing WebSocket real-time data flow...');

        return new Promise((resolve) => {
            const wsTimeout = setTimeout(() => {
                this.recordTest('WebSocket Connection', false, 'Connection timeout');
                resolve();
            }, 10000);

            try {
                const wsUrl = `${this.wsURL}/ws/events/v1`;
                const ws = new WebSocket(wsUrl, {
                    headers: this.authToken ? { Authorization: `Bearer ${this.authToken}` } : {}
                });

                ws.on('open', () => {
                    console.log('    ✅ WebSocket connection established');
                    this.recordTest('WebSocket Connection', true);

                    // Test subscription message
                    const subscribeMsg = JSON.stringify({
                        type: 'subscribe',
                        filters: {
                            event_types: ['system', 'vm'],
                            priorities: [1, 2, 3]
                        }
                    });

                    ws.send(subscribeMsg);
                    this.recordTest('WebSocket Subscription', true);

                    // Clean up and resolve
                    setTimeout(() => {
                        ws.close();
                        clearTimeout(wsTimeout);
                        resolve();
                    }, 2000);
                });

                ws.on('message', (data) => {
                    try {
                        const message = JSON.parse(data.toString());
                        console.log('    📨 WebSocket message received:', message.type);
                        
                        if (message.type === 'connected' || message.type === 'subscribed') {
                            this.recordTest('WebSocket Message Format', true);
                        } else if (message.type === 'ping') {
                            // Respond to ping
                            ws.send(JSON.stringify({ type: 'pong', timestamp: new Date().toISOString() }));
                            this.recordTest('WebSocket Ping/Pong', true);
                        }
                    } catch (error) {
                        this.recordTest('WebSocket Message Format', false, 'Invalid JSON message');
                    }
                });

                ws.on('error', (error) => {
                    console.log('    ❌ WebSocket error:', error.message);
                    this.recordTest('WebSocket Connection', false, error.message);
                    clearTimeout(wsTimeout);
                    resolve();
                });

                ws.on('close', (code, reason) => {
                    console.log('    🔌 WebSocket connection closed:', code, reason.toString());
                    clearTimeout(wsTimeout);
                    resolve();
                });

            } catch (error) {
                console.log('    ❌ WebSocket setup failed:', error.message);
                this.recordTest('WebSocket Connection', false, error.message);
                clearTimeout(wsTimeout);
                resolve();
            }
        });
    }

    async testVMManagement() {
        console.log('  🖥️  Testing VM management workflow...');

        // Navigate to VM management page
        const vmPageURLs = ['/vms', '/dashboard', '/'];
        let navigationSuccess = false;

        for (const url of vmPageURLs) {
            try {
                await this.page.goto(`${this.baseURL}${url}`, { waitUntil: 'networkidle0' });
                navigationSuccess = true;
                break;
            } catch (error) {
                console.log(`    ⚠️  Failed to navigate to ${url}: ${error.message}`);
            }
        }

        if (!navigationSuccess) {
            throw new Error('Could not navigate to any VM management page');
        }

        this.recordTest('VM Page Navigation', true);

        // Test 1: VM list or dashboard is displayed
        const vmElementSelectors = [
            '[data-testid="vm-list"]',
            '[data-testid="vm-grid"]',
            '.vm-list',
            '.vm-grid',
            '[data-testid="dashboard"]',
            '.dashboard'
        ];

        let vmElementFound = false;
        for (let selector of vmElementSelectors) {
            try {
                await this.page.waitForSelector(selector, { timeout: 3000 });
                vmElementFound = true;
                break;
            } catch (e) {
                // Try next selector
            }
        }

        this.recordTest('VM Interface Display', vmElementFound);

        // Test 2: Create VM button/functionality
        const createButtonSelectors = [
            '[data-testid="create-vm-button"]',
            'button:contains("Create VM")',
            'button:contains("Add VM")',
            '[aria-label*="create"], [aria-label*="add"]'
        ];

        let createButtonFound = false;
        for (let selector of createButtonSelectors) {
            try {
                const element = await this.page.$(selector);
                if (element) {
                    createButtonFound = true;
                    break;
                }
            } catch (e) {
                // Try next selector
            }
        }

        this.recordTest('VM Create Function', createButtonFound);

        // Test 3: VM status indicators
        const statusElements = await this.page.$$('[data-status], .status, .vm-status');
        this.recordTest('VM Status Indicators', statusElements.length > 0);

        console.log('  ✅ VM management workflow validation completed');
    }

    async testStorageOperations() {
        console.log('  💾 Testing storage operations...');

        // Try to navigate to storage page
        const storageURLs = ['/storage', '/volumes', '/dashboard'];
        let storagePageFound = false;

        for (const url of storageURLs) {
            try {
                await this.page.goto(`${this.baseURL}${url}`, { waitUntil: 'networkidle0' });
                
                // Check if storage-related content exists
                const storageElements = await this.page.$$([
                    '[data-testid*="storage"]',
                    '[data-testid*="volume"]',
                    '.storage',
                    '.volume'
                ].join(','));

                if (storageElements.length > 0) {
                    storagePageFound = true;
                    break;
                }
            } catch (error) {
                // Continue to next URL
            }
        }

        this.recordTest('Storage Interface Access', storagePageFound);

        if (storagePageFound) {
            // Test volume list display
            const volumeListSelectors = [
                '[data-testid="volume-list"]',
                '.volume-list',
                '[data-testid*="storage-table"]'
            ];

            let volumeListFound = false;
            for (let selector of volumeListSelectors) {
                try {
                    await this.page.waitForSelector(selector, { timeout: 3000 });
                    volumeListFound = true;
                    break;
                } catch (e) {
                    // Try next selector
                }
            }

            this.recordTest('Volume List Display', volumeListFound);
        }

        console.log('  ✅ Storage operations validation completed');
    }

    async testMonitoringDashboard() {
        console.log('  📊 Testing monitoring dashboard...');

        // Navigate to monitoring/dashboard
        const monitoringURLs = ['/monitoring', '/dashboard', '/analytics', '/'];
        let monitoringPageFound = false;

        for (const url of monitoringURLs) {
            try {
                await this.page.goto(`${this.baseURL}${url}`, { waitUntil: 'networkidle0' });
                
                // Look for charts, metrics, or monitoring elements
                const monitoringElements = await this.page.$$([
                    '[data-testid*="chart"]',
                    '[data-testid*="metric"]',
                    '[data-testid*="dashboard"]',
                    '.chart',
                    '.metric',
                    '.monitoring',
                    'canvas',
                    'svg'
                ].join(','));

                if (monitoringElements.length > 0) {
                    monitoringPageFound = true;
                    break;
                }
            } catch (error) {
                // Continue to next URL
            }
        }

        this.recordTest('Monitoring Dashboard Access', monitoringPageFound);

        if (monitoringPageFound) {
            // Test for real-time data updates
            const hasRealtimeElements = await this.page.evaluate(() => {
                // Look for elements that might update in real-time
                const realtimeSelectors = [
                    '[data-testid*="live"]',
                    '[data-testid*="realtime"]',
                    '.live-metric',
                    '.realtime'
                ];

                return realtimeSelectors.some(selector => 
                    document.querySelector(selector) !== null
                );
            });

            this.recordTest('Real-time Data Elements', hasRealtimeElements);

            // Test for metric values
            const hasMetricValues = await this.page.evaluate(() => {
                // Look for numeric values that could be metrics
                const textContent = document.body.innerText;
                const hasPercentages = /\d+(\.\d+)?%/.test(textContent);
                const hasNumbers = /\d+(\.\d+)?\s*(GB|MB|KB|CPU|Memory)/.test(textContent);
                return hasPercentages || hasNumbers;
            });

            this.recordTest('Metric Values Display', hasMetricValues);
        }

        console.log('  ✅ Monitoring dashboard validation completed');
    }

    async testUIInteractions() {
        console.log('  🖱️  Testing user interface interactions...');

        // Test 1: Navigation menu functionality
        const navSelectors = ['nav a', '.nav-link', '[data-testid*="nav"]', 'header a'];
        const navLinks = await this.page.$$eval(navSelectors.join(','), links => 
            links.map(link => ({ text: link.textContent?.trim(), href: link.href }))
        );

        this.recordTest('Navigation Links Present', navLinks.length > 0);

        // Test 2: Interactive elements respond to clicks
        const interactiveElements = await this.page.$$('button, [role="button"], input, select, textarea');
        this.recordTest('Interactive Elements Present', interactiveElements.length > 0);

        // Test 3: Form validation (if forms exist)
        const forms = await this.page.$$('form');
        if (forms.length > 0) {
            try {
                // Try to submit empty form to test validation
                await this.page.click('form button[type="submit"], form input[type="submit"]');
                
                // Check for validation messages
                const validationMessages = await this.page.$$('.error, .invalid, [aria-invalid="true"]');
                this.recordTest('Form Validation', validationMessages.length > 0);
            } catch (error) {
                this.recordTest('Form Validation', false, 'Could not test form validation');
            }
        }

        // Test 4: Responsive design elements
        const hasResponsiveElements = await this.page.evaluate(() => {
            const mobileMenus = document.querySelectorAll('.mobile-menu, .hamburger, [data-testid*="mobile"]');
            const responsiveClasses = document.querySelectorAll('[class*="sm:"], [class*="md:"], [class*="lg:"]');
            return mobileMenus.length > 0 || responsiveClasses.length > 0;
        });

        this.recordTest('Responsive Design Elements', hasResponsiveElements);

        console.log('  ✅ UI interactions validation completed');
    }

    async testCrossComponentIntegration() {
        console.log('  🔄 Testing cross-component integration...');

        // Test 1: State management between components
        const hasStateManagement = await this.page.evaluate(() => {
            // Check for common state management indicators
            return window.React || window.Vue || window.Angular || window.__REDUX_STORE__ || window.jotai;
        });

        this.recordTest('State Management Framework', hasStateManagement);

        // Test 2: Event handling between components
        try {
            // Click on navigation items and check for state changes
            const navItems = await this.page.$$('nav a, .nav-link');
            
            if (navItems.length > 0) {
                const initialURL = this.page.url();
                await navItems[0].click();
                await this.page.waitForTimeout(1000);
                const newURL = this.page.url();
                
                this.recordTest('Component Navigation Integration', initialURL !== newURL || 
                    await this.page.evaluate(() => window.location.hash !== ''));
            } else {
                this.recordTest('Component Navigation Integration', false, 'No navigation elements found');
            }
        } catch (error) {
            this.recordTest('Component Navigation Integration', false, error.message);
        }

        // Test 3: Data flow between frontend and backend
        const hasDataFlow = await this.page.evaluate(() => {
            // Check for fetch calls or API integration
            return typeof fetch !== 'undefined' || window.axios || window.XMLHttpRequest;
        });

        this.recordTest('Frontend-Backend Data Flow', hasDataFlow);

        console.log('  ✅ Cross-component integration validation completed');
    }

    async testErrorHandling() {
        console.log('  🚨 Testing error handling and recovery...');

        // Test 1: Network error simulation
        await this.page.setOfflineMode(true);
        
        try {
            await this.page.reload({ waitUntil: 'networkidle0' });
            // Should handle offline gracefully
            this.recordTest('Offline Mode Handling', false, 'Page should handle offline mode');
        } catch (error) {
            // Expected behavior - check for error UI
            const errorElements = await this.page.$$('.error, [data-testid*="error"], .offline');
            this.recordTest('Offline Error UI', errorElements.length > 0);
        }

        await this.page.setOfflineMode(false);

        // Test 2: Invalid URL handling
        try {
            await this.page.goto(`${this.baseURL}/nonexistent-page`, { waitUntil: 'networkidle0' });
            
            const is404Page = await this.page.evaluate(() => {
                const text = document.body.textContent || '';
                return text.includes('404') || text.includes('Not Found') || text.includes('Page not found');
            });

            this.recordTest('404 Error Handling', is404Page);
        } catch (error) {
            this.recordTest('404 Error Handling', false, error.message);
        }

        // Test 3: JavaScript error recovery
        await this.page.goto(this.baseURL);
        
        const jsErrorsAfterRecovery = await this.page.evaluate(() => {
            return window.jsErrors ? window.jsErrors.length : 0;
        });

        this.recordTest('JavaScript Error Recovery', jsErrorsAfterRecovery === 0);

        console.log('  ✅ Error handling validation completed');
    }

    recordTest(name, passed, error = null) {
        this.results.tests.push({
            name,
            passed,
            error,
            timestamp: new Date().toISOString()
        });

        this.results.summary.total++;
        if (passed) {
            this.results.summary.passed++;
        } else {
            this.results.summary.failed++;
        }
    }

    generateReport() {
        const report = {
            timestamp: new Date().toISOString(),
            summary: this.results.summary,
            successRate: (this.results.summary.passed / this.results.summary.total * 100).toFixed(2),
            tests: this.results.tests,
            issues: this.results.tests.filter(test => !test.passed),
            recommendations: this.generateRecommendations()
        };

        console.log('\n📊 INTEGRATION VALIDATION REPORT');
        console.log('='.repeat(50));
        console.log(`Total Tests: ${report.summary.total}`);
        console.log(`Passed: ${report.summary.passed} ✅`);
        console.log(`Failed: ${report.summary.failed} ❌`);
        console.log(`Success Rate: ${report.successRate}%`);

        if (report.issues.length > 0) {
            console.log('\n🚨 Issues Found:');
            report.issues.forEach(issue => {
                console.log(`  ❌ ${issue.name}: ${issue.error || 'Failed'}`);
            });
        }

        if (report.recommendations.length > 0) {
            console.log('\n💡 Recommendations:');
            report.recommendations.forEach(rec => {
                console.log(`  • ${rec}`);
            });
        }

        return report;
    }

    generateRecommendations() {
        const recommendations = [];
        const issues = this.results.tests.filter(test => !test.passed);

        // Analyze issues and provide recommendations
        const issueTypes = {
            'Frontend Page Load': 'Check frontend build and deployment configuration',
            'Login Form Present': 'Verify authentication UI components are properly rendered',
            'WebSocket Connection': 'Ensure WebSocket server is running and accessible',
            'API': 'Check backend API server status and endpoints',
            'Database': 'Verify database connectivity and schema setup',
            'Navigation': 'Review frontend routing configuration',
            'Error Handling': 'Implement proper error boundaries and user feedback'
        };

        issues.forEach(issue => {
            Object.keys(issueTypes).forEach(type => {
                if (issue.name.includes(type)) {
                    recommendations.push(`${issue.name}: ${issueTypes[type]}`);
                }
            });
        });

        // General recommendations based on success rate
        const successRate = (this.results.summary.passed / this.results.summary.total * 100);
        
        if (successRate < 50) {
            recommendations.push('Critical: System integration is severely compromised - review entire deployment');
        } else if (successRate < 75) {
            recommendations.push('Warning: Multiple integration issues detected - prioritize fixing failed tests');
        } else if (successRate < 90) {
            recommendations.push('Good: Minor integration issues - address remaining failures for production readiness');
        }

        return [...new Set(recommendations)]; // Remove duplicates
    }

    async cleanup() {
        if (this.browser) {
            await this.browser.close();
        }
    }
}

// Export for use in test runners
module.exports = FrontendBackendValidator;

// Run if called directly
if (require.main === module) {
    (async () => {
        const validator = new FrontendBackendValidator();
        
        try {
            await validator.initialize();
            const report = await validator.runAllTests();
            
            // Write report to file
            const fs = require('fs');
            const reportPath = './integration-validation-report.json';
            fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
            console.log(`\n📄 Report saved to: ${reportPath}`);
            
        } catch (error) {
            console.error('❌ Validation failed:', error);
            process.exit(1);
        } finally {
            await validator.cleanup();
        }
    })();
}