# E2E Testing Architecture - Visual Diagrams

This document contains all architectural diagrams for the NovaCron Playwright E2E testing framework in text/ASCII/mermaid format for easy viewing in any environment.

---

## System Architecture Overview

```mermaid
graph TB
    subgraph "Test Layer"
        T1[Smoke Tests]
        T2[Regression Tests]
        T3[Integration Tests]
        T4[Performance Tests]
        T5[Accessibility Tests]
    end

    subgraph "Page Object Layer"
        PO1[Auth Pages]
        PO2[VM Pages]
        PO3[Cluster Pages]
        PO4[Monitoring Pages]
        PO5[Component Pages]
    end

    subgraph "Infrastructure Layer"
        F1[Fixtures]
        F2[Test Data Factories]
        F3[API Clients]
        F4[Assertions]
        F5[Wait Helpers]
    end

    subgraph "Application Under Test"
        UI[Frontend UI]
        API[Backend API]
        WS[WebSocket]
        DB[(Database)]
    end

    T1 --> PO1
    T2 --> PO2
    T3 --> PO3
    T4 --> PO4
    T5 --> PO5

    PO1 --> F1
    PO2 --> F2
    PO3 --> F3
    PO4 --> F4
    PO5 --> F5

    F1 --> UI
    F2 --> API
    F3 --> API
    F4 --> UI
    F5 --> WS

    API --> DB
```

---

## Test Execution Flow

```mermaid
sequenceDiagram
    participant Dev as Developer
    participant CI as CI/CD Pipeline
    participant PW as Playwright
    participant PO as Page Objects
    participant FX as Fixtures
    participant APP as Application
    participant API as Backend API

    Dev->>CI: git push
    CI->>PW: Execute tests (4 workers)

    loop For each test
        PW->>FX: Request fixtures
        FX->>API: Create test data
        API-->>FX: Return test entities
        FX-->>PW: Provide fixtures

        PW->>PO: Execute test steps
        PO->>APP: Interact with UI
        APP-->>PO: UI updates
        PO->>API: Verify backend state
        API-->>PO: Return state
        PO-->>PW: Test results

        PW->>FX: Request cleanup
        FX->>API: Delete test data
    end

    PW->>CI: Generate reports
    CI->>Dev: Test results + artifacts
```

---

## Page Object Model Hierarchy

```
┌─────────────────────────────────────────┐
│         Test Specifications             │
│  (Business logic and assertions)        │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│          Feature Pages                  │
│  (LoginPage, VMCreationPage, etc.)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│          Base Page Class                │
│  (Common methods and utilities)         │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│          Reusable Components            │
│  (Navigation, Forms, Modals, etc.)      │
└─────────────────────────────────────────┘
                  ↓
┌─────────────────────────────────────────┐
│       Playwright Locators               │
│  (DOM element selectors)                │
└─────────────────────────────────────────┘
```

---

## Test Data Management Flow

```mermaid
graph LR
    A[Test Spec] --> B[Fixture Request]
    B --> C{Data Needed?}

    C -->|Yes| D[Factory]
    C -->|No| E[Use Existing]

    D --> F[Generate Data]
    F --> G[API Call]
    G --> H[Backend]
    H --> I[Store in DB]
    I --> J[Return Entity]
    J --> K[Track for Cleanup]
    K --> L[Provide to Test]

    E --> L

    L --> M[Test Execution]
    M --> N[Test Complete]
    N --> O[Fixture Cleanup]
    O --> P[Delete via API]
    P --> H
```

---

## Directory Structure Tree

```
tests/e2e/
├── docs/                           # Documentation
│   ├── ARCHITECTURE.md
│   ├── IMPLEMENTATION_GUIDE.md
│   ├── BEST_PRACTICES.md
│   ├── QUICK_REFERENCE.md
│   └── MIGRATION_PLAN.md
│
├── tests/                          # Test Specifications
│   ├── smoke/
│   │   └── critical-paths.spec.ts
│   ├── auth/
│   │   ├── login.spec.ts
│   │   ├── registration.spec.ts
│   │   └── 2fa.spec.ts
│   ├── vm-management/
│   │   ├── vm-creation.spec.ts
│   │   ├── vm-lifecycle.spec.ts
│   │   ├── vm-migration.spec.ts
│   │   └── vm-snapshots.spec.ts
│   ├── cluster/
│   ├── dwcp/
│   ├── monitoring/
│   ├── edge/
│   ├── multicloud/
│   ├── performance/
│   ├── accessibility/
│   └── integration/
│
├── page-objects/                   # Page Object Model
│   ├── base/
│   │   ├── BasePage.ts
│   │   ├── BaseComponent.ts
│   │   └── BaseModal.ts
│   ├── auth/
│   │   ├── LoginPage.ts
│   │   ├── RegistrationPage.ts
│   │   └── PasswordResetPage.ts
│   ├── vm/
│   │   ├── VMListPage.ts
│   │   ├── VMDetailsPage.ts
│   │   ├── VMCreationPage.ts
│   │   └── VMMigrationPage.ts
│   ├── cluster/
│   ├── monitoring/
│   └── components/
│       ├── NavigationComponent.ts
│       ├── NotificationComponent.ts
│       └── FormComponent.ts
│
├── fixtures/                       # Test Fixtures
│   ├── auth.fixture.ts
│   ├── vm.fixture.ts
│   ├── cluster.fixture.ts
│   └── index.ts
│
├── helpers/                        # Utilities
│   ├── test-data/
│   │   ├── VMFactory.ts
│   │   ├── UserFactory.ts
│   │   └── ClusterFactory.ts
│   ├── api/
│   │   ├── APIClient.ts
│   │   ├── WebSocketClient.ts
│   │   └── GRPCClient.ts
│   ├── assertions/
│   │   ├── CustomMatchers.ts
│   │   └── VisualAssertions.ts
│   └── utilities/
│       ├── WaitHelpers.ts
│       ├── DataCleanup.ts
│       └── ScreenshotHelpers.ts
│
├── config/                         # Configuration
│   ├── playwright.config.ts
│   ├── environments/
│   │   ├── local.config.ts
│   │   ├── ci.config.ts
│   │   ├── staging.config.ts
│   │   └── production.config.ts
│   └── test-categories.ts
│
├── mocks/                          # API Mocking
│   ├── api/
│   │   ├── vm-api.mock.ts
│   │   └── auth-api.mock.ts
│   ├── recordings/
│   │   └── *.har
│   └── handlers/
│       ├── rest-handlers.ts
│       └── ws-handlers.ts
│
└── reports/                        # Test Results
    ├── html/
    ├── traces/
    ├── videos/
    └── screenshots/
```

---

## Test Categorization

```mermaid
graph TD
    A[All Tests] --> B[Smoke Tests]
    A --> C[Regression Tests]
    A --> D[Integration Tests]
    A --> E[Performance Tests]
    A --> F[Accessibility Tests]

    B --> B1[Critical Paths]
    B --> B2[Authentication]
    B --> B3[Core Features]

    C --> C1[VM Management]
    C --> C2[Cluster Ops]
    C --> C3[Monitoring]
    C --> C4[Admin Panel]

    D --> D1[Frontend-Backend]
    D --> D2[WebSocket]
    D --> D3[Cross-Feature]

    E --> E1[Load Tests]
    E --> E2[Stress Tests]
    E --> E3[Scalability]

    F --> F1[WCAG Compliance]
    F --> F2[Keyboard Nav]
    F --> F3[Screen Reader]

    style B fill:#90EE90
    style C fill:#FFD700
    style D fill:#87CEEB
    style E fill:#FFB6C1
    style F fill:#DDA0DD
```

---

## Browser & Viewport Coverage

```
Desktop Browsers          Mobile Browsers
┌────────────────┐        ┌────────────────┐
│   Chrome       │        │  Mobile Chrome │
│   Latest, L-1  │        │   Pixel 5      │
├────────────────┤        ├────────────────┤
│   Firefox      │        │  Mobile Safari │
│   Latest, L-1  │        │   iPhone 12    │
├────────────────┤        └────────────────┘
│   Safari       │
│   Latest, L-1  │        Viewports
├────────────────┤        ┌────────────────┐
│   Edge         │        │ 1920x1080      │
│   Latest       │        │ 1366x768       │
└────────────────┘        │ 1280x720       │
                          │ 1024x768 (tab) │
Priority Matrix           │ 768x1024 (tab) │
High:   Chrome, Firefox   │ 375x667 (mob)  │
Medium: Safari, Edge      │ 320x568 (mob)  │
                          └────────────────┘
```

---

## Parallel Execution Architecture

```
┌─────────────────────────────────────────────────┐
│              Test Runner (Main Thread)          │
└─────────────────────────────────────────────────┘
                    │
        ┌───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼
    ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
    │Worker1│   │Worker2│   │Worker3│   │Worker4│
    └───────┘   └───────┘   └───────┘   └───────┘
        │           │           │           │
    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
    │Browser│   │Browser│   │Browser│   │Browser│
    │Context│   │Context│   │Context│   │Context│
    └───────┘   └───────┘   └───────┘   └───────┘
        │           │           │           │
    ┌───┴───┐   ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
    │ Tests │   │ Tests │   │ Tests │   │ Tests │
    │ 1-25  │   │ 26-50 │   │ 51-75 │   │ 76-100│
    └───────┘   └───────┘   └───────┘   └───────┘

Each worker:
- Independent browser context
- Isolated test data
- Separate storage state
- Concurrent execution
```

---

## Test Sharding for CI/CD

```
┌────────────────────────────────────────────────┐
│           CI/CD Pipeline (4 Machines)          │
└────────────────────────────────────────────────┘
            │
    ┌───────┼───────┬───────┬───────┐
    ▼       ▼       ▼       ▼       ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Machine1│ │Machine2│ │Machine3│ │Machine4│
│Shard1/4│ │Shard2/4│ │Shard3/4│ │Shard4/4│
└────────┘ └────────┘ └────────┘ └────────┘
    │          │          │          │
┌───┴────┐ ┌───┴────┐ ┌───┴────┐ ┌───┴────┐
│Tests   │ │Tests   │ │Tests   │ │Tests   │
│1-50    │ │51-100  │ │101-150 │ │151-200 │
└────────┘ └────────┘ └────────┘ └────────┘
    │          │          │          │
    └──────────┴──────────┴──────────┘
                  │
        ┌─────────▼─────────┐
        │  Merge Results     │
        │  - HTML Report     │
        │  - Test Metrics    │
        │  - Trace Files     │
        │  - Screenshots     │
        └────────────────────┘
```

---

## API Mocking Strategy

```mermaid
graph TD
    A[Test Execution] --> B{Environment}

    B -->|Local Dev| C[MSW Mocking]
    B -->|CI/CD| D[Hybrid Mode]
    B -->|Staging| E[Real API]
    B -->|Production| F[Real API Only]

    C --> G[Fast Tests]
    C --> H[Deterministic]
    C --> I[No Dependencies]

    D --> J[Critical: Real API]
    D --> K[Non-Critical: Mock]
    D --> L[Slow Endpoints: HAR]

    E --> M[Full Integration]
    E --> N[Selective Mock]

    F --> O[Smoke Tests Only]
    F --> P[Health Checks]
```

---

## Fixture Lifecycle

```
Test Start
    │
    ├─► Fixture Setup
    │       ├─► Initialize API Client
    │       ├─► Create Test Data (VM Factory)
    │       ├─► Setup Authentication
    │       └─► Prepare Browser Context
    │
    ├─► Test Execution
    │       ├─► Page Object Actions
    │       ├─► UI Interactions
    │       └─► Assertions
    │
    ├─► Fixture Cleanup (always runs)
    │       ├─► Delete Test Data
    │       ├─► Clear Auth Tokens
    │       ├─► Close Browser Context
    │       └─► Generate Artifacts (if failed)
    │               ├─► Screenshot
    │               ├─► Video
    │               └─► Trace
    │
Test End
```

---

## Page Object Interaction Pattern

```
┌──────────────────────────────────────────────┐
│              Test Specification              │
│                                              │
│  test('create VM', async ({ page }) => {    │
│    const vmPage = new VMCreationPage(page); │
│    await vmPage.createVM(data);             │
│  });                                         │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│           VMCreationPage (POM)               │
│                                              │
│  async createVM(data) {                      │
│    await this.fillForm(data);               │
│    await this.submit();                     │
│  }                                           │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│            BasePage (Common)                 │
│                                              │
│  protected fillInput(locator, value) {...}  │
│  protected clickElement(locator) {...}      │
└──────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────┐
│         Playwright Locators                  │
│                                              │
│  page.locator('[data-testid="vm-name"]')    │
│  page.getByRole('button', {name: 'Submit'}) │
└──────────────────────────────────────────────┘
```

---

## CI/CD Pipeline Flow

```mermaid
graph LR
    A[Git Push] --> B[GitHub Actions]
    B --> C[Install Dependencies]
    C --> D[Install Playwright]
    D --> E{Run Tests}

    E --> F[Shard 1/4]
    E --> G[Shard 2/4]
    E --> H[Shard 3/4]
    E --> I[Shard 4/4]

    F --> J[Merge Results]
    G --> J
    H --> J
    I --> J

    J --> K{Tests Pass?}

    K -->|Yes| L[Deploy to Staging]
    K -->|No| M[Upload Artifacts]

    M --> N[HTML Report]
    M --> O[Trace Files]
    M --> P[Videos]
    M --> Q[Screenshots]

    N --> R[Notify Team]
    O --> R
    P --> R
    Q --> R
```

---

## Authentication Flow

```
┌─────────────────────────────────────────────┐
│          Authentication Strategy            │
└─────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│  UI Login     │       │  API Login    │
│  (Slow)       │       │  (Fast)       │
└───────────────┘       └───────────────┘
        │                       │
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Use for:      │       │ Use for:      │
│ - Auth tests  │       │ - Setup       │
│ - UI flows    │       │ - Fixtures    │
│ - E2E tests   │       │ - Data prep   │
└───────────────┘       └───────────────┘
        │                       │
        └───────────┬───────────┘
                    ▼
        ┌───────────────────────┐
        │  Storage State Cache  │
        │  (Reuse auth session) │
        └───────────────────────┘
```

---

## Test Report Structure

```
reports/
│
├── html/                      # HTML Report
│   ├── index.html             # Main report
│   ├── data/
│   │   ├── test-results.json
│   │   └── screenshots/
│   └── assets/
│       ├── styles.css
│       └── scripts.js
│
├── traces/                    # Debug Traces
│   ├── test-1-retry-1.zip
│   ├── test-2-retry-1.zip
│   └── ...
│
├── videos/                    # Test Videos
│   ├── test-1-chromium.webm
│   ├── test-2-firefox.webm
│   └── ...
│
├── screenshots/               # Screenshots
│   ├── test-1-error.png
│   ├── test-2-final.png
│   └── ...
│
├── test-results.json          # JSON Report
└── junit.xml                  # JUnit Report
```

---

## Error Handling Flow

```mermaid
graph TD
    A[Test Execution] --> B{Test Passes?}

    B -->|Yes| C[Mark as Passed]
    B -->|No| D{Retry Enabled?}

    D -->|No| E[Mark as Failed]
    D -->|Yes| F{Retry Count < Max?}

    F -->|No| E
    F -->|Yes| G[Capture Debug Info]

    G --> H[Screenshot]
    G --> I[Video]
    G --> J[Trace]
    G --> K[Logs]

    H --> L[Retry Test]
    I --> L
    J --> L
    K --> L

    L --> A

    E --> M[Upload Artifacts]
    M --> N[Notify Team]

    C --> O[Continue to Next Test]
```

---

## Visual Regression Testing

```
┌────────────────────────────────────────────┐
│         Visual Regression Testing          │
└────────────────────────────────────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
    ┌──────────────┐ ┌──────────────┐
    │ Baseline     │ │  Current     │
    │ Screenshot   │ │ Screenshot   │
    └──────────────┘ └──────────────┘
            │               │
            └───────┬───────┘
                    ▼
        ┌───────────────────────┐
        │  Pixel Comparison     │
        │  (max diff: 100px)    │
        └───────────────────────┘
                    │
            ┌───────┴───────┐
            ▼               ▼
    ┌──────────────┐ ┌──────────────┐
    │   Match      │ │  Mismatch    │
    │   ✓ Pass     │ │   ✗ Fail     │
    └──────────────┘ └──────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │  Generate Diff Image  │
                │  Save to reports/     │
                └───────────────────────┘
```

---

## Migration Timeline

```
Week 1: Infrastructure
├── Install Playwright
├── Create base structure
├── Implement BasePage
└── First smoke test
    │
Week 2: Foundation
├── Create page objects
├── Implement factories
├── Setup fixtures
└── Test data management
    │
Week 3: Smoke Tests
├── Auth flows (5)
├── VM lifecycle (8)
├── Dashboard (3)
└── Validation
    │
Week 4: Regression Part 1
├── VM management (30)
├── Cluster ops (20)
└── Cross-browser tests
    │
Week 5: Regression Part 2
├── Monitoring (25)
├── Performance (15)
├── Admin panel (20)
└── Visual regression
    │
Week 6: Integration
├── Integration tests (30)
├── Mobile testing
└── Accessibility
    │
Week 7: Completion
├── Puppeteer deprecation
├── Final validation
├── Documentation
└── Team training

Total: 190+ tests migrated
```

---

All diagrams are designed to be viewed in markdown-compatible environments and provide clear visual representation of the E2E testing architecture.

