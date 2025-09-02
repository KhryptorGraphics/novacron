# NovaCron Development Workflow Guide

## Development Environment Setup

This guide provides comprehensive instructions for setting up a development environment, following development workflows, and maintaining code quality standards for the NovaCron platform.

### 1. Prerequisites & Environment Setup

#### System Requirements
```bash
# Minimum development system requirements
OS: Ubuntu 22.04 LTS / macOS 12+ / Windows 11 with WSL2
CPU: 8 cores (Intel i7/AMD Ryzen 7 or better)
RAM: 16GB (32GB recommended)
Storage: 500GB SSD
Network: Stable broadband connection
```

#### Required Software Installation

**Development Tools:**
```bash
# Go development environment
wget https://golang.org/dl/go1.23.0.linux-amd64.tar.gz
sudo tar -C /usr/local -xzf go1.23.0.linux-amd64.tar.gz
echo 'export PATH=$PATH:/usr/local/go/bin' >> ~/.bashrc

# Node.js and npm
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# Development databases
sudo apt install -y postgresql-15 postgresql-contrib redis-server

# Container tools
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER

# Version control
sudo apt install -y git git-lfs

# Code editors and tools
sudo snap install code --classic
sudo snap install postman
```

**KVM/Virtualization Setup (for VM development):**
```bash
# Install KVM and virtualization tools
sudo apt install -y qemu-kvm libvirt-daemon-system libvirt-clients bridge-utils virtinst virt-manager

# Add user to virtualization groups
sudo usermod -aG libvirt $USER
sudo usermod -aG kvm $USER

# Verify KVM installation
sudo virt-host-validate
```

#### IDE Configuration

**VS Code Extensions:**
```json
{
  "recommendations": [
    "golang.go",
    "ms-vscode.vscode-typescript-next",
    "bradlc.vscode-tailwindcss",
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-kubernetes-tools.vscode-kubernetes-tools",
    "ms-vscode-remote.remote-containers",
    "github.copilot",
    "sonarsource.sonarlint-vscode",
    "ms-vscode.test-adapter-converter"
  ]
}
```

**VS Code Settings:**
```json
{
  "go.useLanguageServer": true,
  "go.formatTool": "goimports",
  "go.lintTool": "golangci-lint",
  "go.testFlags": ["-v", "-race"],
  "go.coverOnSave": true,
  "go.coverOnSingleTest": true,
  "typescript.preferences.importModuleSpecifier": "relative",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true,
    "source.fixAll": true
  },
  "files.associations": {
    "*.yaml": "yaml",
    "Dockerfile*": "dockerfile"
  }
}
```

### 2. Repository Setup & Structure

#### Repository Cloning and Setup
```bash
# Clone the repository
git clone https://github.com/novacron/novacron.git
cd novacron

# Set up Git hooks
cp scripts/git-hooks/* .git/hooks/
chmod +x .git/hooks/*

# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Set up development environment
make dev-setup
```

#### Project Structure Overview
```
novacron/
├── backend/                    # Go backend services
│   ├── api/                   # API handlers and routes
│   │   ├── rest/             # REST API implementation
│   │   ├── graphql/          # GraphQL API implementation
│   │   ├── websocket/        # WebSocket handlers
│   │   └── middleware/       # HTTP middleware
│   ├── cmd/                  # Application entry points
│   │   ├── api-server/       # Main API server
│   │   └── migration/        # Database migration tool
│   ├── core/                 # Core business logic
│   │   ├── vm/              # VM management
│   │   ├── auth/            # Authentication
│   │   ├── monitoring/      # System monitoring
│   │   └── backup/          # Backup systems
│   ├── pkg/                 # Shared packages
│   │   ├── config/          # Configuration management
│   │   ├── logger/          # Logging utilities
│   │   └── middleware/      # Shared middleware
│   └── tests/               # Test files
├── frontend/                 # Next.js frontend
│   ├── src/
│   │   ├── app/             # App Router pages
│   │   ├── components/      # React components
│   │   ├── lib/            # Utility libraries
│   │   └── types/          # TypeScript definitions
│   ├── public/             # Static assets
│   └── tests/              # Frontend tests
├── docs/                    # Documentation
├── scripts/                 # Development scripts
├── deployment/              # Deployment configurations
│   ├── docker/             # Docker configurations
│   ├── kubernetes/         # Kubernetes manifests
│   └── terraform/          # Infrastructure as code
└── .github/                # GitHub workflows and templates
```

### 3. Development Workflow

#### Git Workflow & Branching Strategy

**Branching Model:**
```
main                    # Production-ready code
├── develop            # Integration branch for features
│   ├── feature/auth   # Feature development
│   ├── feature/vm     # Feature development
│   └── hotfix/bug     # Critical bug fixes
├── release/v1.2.0     # Release preparation
└── hotfix/critical    # Emergency fixes
```

**Branch Naming Convention:**
```bash
# Feature branches
feature/VM-123-add-migration-support
feature/AUTH-456-implement-sso

# Bug fix branches
bugfix/API-789-fix-timeout-issue
bugfix/UI-012-resolve-loading-state

# Hotfix branches
hotfix/CRITICAL-fix-security-vulnerability

# Release branches
release/v1.2.0
release/v2.0.0-beta
```

#### Commit Message Standards

**Conventional Commits Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Examples:**
```bash
# Feature commits
feat(vm): add live migration support for KVM instances
feat(auth): implement OAuth2 integration with Azure AD

# Bug fixes
fix(api): resolve timeout issues in VM creation endpoint
fix(ui): fix loading spinner not appearing during VM operations

# Documentation
docs(api): add authentication examples to API documentation
docs(deployment): update Kubernetes deployment guide

# Refactoring
refactor(core): extract common VM operations into shared package
refactor(frontend): migrate from axios to fetch API

# Testing
test(vm): add integration tests for VM lifecycle operations
test(auth): increase test coverage for JWT validation
```

#### Pull Request Workflow

**Pull Request Template:**
```markdown
## Description
Brief description of changes and their purpose.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] Performance improvement

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance testing (if applicable)

## Checklist
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to help explain your changes.

## Additional Notes
Any additional information that reviewers should know.
```

**Code Review Guidelines:**

1. **Reviewer Responsibilities:**
   - Review code for correctness, performance, and maintainability
   - Check for security vulnerabilities
   - Verify test coverage and quality
   - Ensure documentation is updated
   - Validate adherence to coding standards

2. **Review Checklist:**
```markdown
## Code Quality
- [ ] Code is readable and well-structured
- [ ] Functions are appropriately sized and focused
- [ ] Variable and function names are descriptive
- [ ] Error handling is appropriate and comprehensive
- [ ] No obvious performance issues
- [ ] Security best practices are followed

## Testing
- [ ] Adequate test coverage (>80% for new code)
- [ ] Tests are meaningful and test the right things
- [ ] Edge cases are covered
- [ ] Integration tests are included where appropriate

## Documentation
- [ ] Code is self-documenting or well-commented
- [ ] API documentation is updated (if applicable)
- [ ] README or other docs are updated (if applicable)
```

### 4. Development Standards

#### Go Development Standards

**Code Style & Formatting:**
```go
// Package documentation is required
// Package vm provides virtual machine management functionality.
//
// This package implements the core VM lifecycle operations including
// creation, deletion, migration, and monitoring of virtual machines
// across multiple hypervisor platforms.
package vm

import (
    "context"
    "fmt"
    "time"
    
    // Standard library imports first
    "database/sql"
    "encoding/json"
    
    // Third-party imports second
    "github.com/google/uuid"
    "github.com/sirupsen/logrus"
    
    // Internal imports last
    "github.com/novacron/backend/pkg/logger"
)

// Constants should be grouped and documented
const (
    // DefaultVMTimeout is the default timeout for VM operations
    DefaultVMTimeout = 30 * time.Second
    
    // MaxRetryAttempts defines the maximum number of retry attempts
    MaxRetryAttempts = 3
)

// VMManager handles virtual machine lifecycle operations
type VMManager struct {
    db     *sql.DB
    logger *logrus.Logger
    config Config
}

// NewVMManager creates a new VM manager instance
func NewVMManager(db *sql.DB, config Config) (*VMManager, error) {
    if db == nil {
        return nil, fmt.Errorf("database connection is required")
    }
    
    return &VMManager{
        db:     db,
        logger: logger.New("vm-manager"),
        config: config,
    }, nil
}

// CreateVM creates a new virtual machine with the specified configuration
//
// This method performs the following steps:
//   1. Validates the VM configuration
//   2. Checks resource availability
//   3. Creates the VM in the hypervisor
//   4. Stores VM metadata in the database
//
// Returns the created VM instance or an error if creation fails.
func (vm *VMManager) CreateVM(ctx context.Context, config VMConfig) (*VM, error) {
    // Validate input parameters
    if err := config.Validate(); err != nil {
        return nil, fmt.Errorf("invalid VM configuration: %w", err)
    }
    
    // Check context for cancellation
    select {
    case <-ctx.Done():
        return nil, ctx.Err()
    default:
    }
    
    // Implementation with proper error handling
    vmInstance, err := vm.createVMInstance(ctx, config)
    if err != nil {
        vm.logger.WithError(err).Error("Failed to create VM instance")
        return nil, fmt.Errorf("VM creation failed: %w", err)
    }
    
    vm.logger.WithField("vm_id", vmInstance.ID).Info("VM created successfully")
    return vmInstance, nil
}
```

**Error Handling Standards:**
```go
// Define package-specific error types
type VMError struct {
    Op   string // Operation that failed
    VM   string // VM identifier
    Err  error  // Underlying error
}

func (e *VMError) Error() string {
    return fmt.Sprintf("vm %s: %s: %v", e.VM, e.Op, e.Err)
}

func (e *VMError) Unwrap() error {
    return e.Err
}

// Use error wrapping for context
func (vm *VMManager) startVM(ctx context.Context, vmID string) error {
    if err := vm.hypervisor.Start(vmID); err != nil {
        return &VMError{
            Op:  "start",
            VM:  vmID,
            Err: err,
        }
    }
    return nil
}

// Implement proper error checking
func (vm *VMManager) ProcessVMOperation(ctx context.Context, vmID string, operation string) error {
    switch operation {
    case "start":
        if err := vm.startVM(ctx, vmID); err != nil {
            var vmErr *VMError
            if errors.As(err, &vmErr) {
                // Handle VM-specific error
                vm.logger.WithFields(logrus.Fields{
                    "vm_id":     vmErr.VM,
                    "operation": vmErr.Op,
                }).Error("VM operation failed")
            }
            return fmt.Errorf("failed to start VM %s: %w", vmID, err)
        }
    default:
        return fmt.Errorf("unsupported operation: %s", operation)
    }
    return nil
}
```

**Testing Standards:**
```go
func TestVMManager_CreateVM(t *testing.T) {
    // Test table pattern for multiple test cases
    tests := []struct {
        name        string
        config      VMConfig
        setup       func(*testing.T) *VMManager
        wantErr     bool
        wantErrType error
    }{
        {
            name: "valid configuration",
            config: VMConfig{
                Name:   "test-vm",
                CPU:    2,
                Memory: 4096,
                Disk:   50,
            },
            setup: func(t *testing.T) *VMManager {
                db := setupTestDB(t)
                return mustCreateVMManager(t, db)
            },
            wantErr: false,
        },
        {
            name: "invalid configuration - missing name",
            config: VMConfig{
                CPU:    2,
                Memory: 4096,
                Disk:   50,
            },
            setup: func(t *testing.T) *VMManager {
                db := setupTestDB(t)
                return mustCreateVMManager(t, db)
            },
            wantErr:     true,
            wantErrType: &ValidationError{},
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            vm := tt.setup(t)
            ctx := context.Background()
            
            result, err := vm.CreateVM(ctx, tt.config)
            
            if tt.wantErr {
                assert.Error(t, err)
                if tt.wantErrType != nil {
                    assert.ErrorAs(t, err, &tt.wantErrType)
                }
                assert.Nil(t, result)
            } else {
                assert.NoError(t, err)
                assert.NotNil(t, result)
                assert.Equal(t, tt.config.Name, result.Name)
            }
        })
    }
}

// Benchmark tests for performance-critical functions
func BenchmarkVMManager_ListVMs(b *testing.B) {
    vm := setupBenchmarkVMManager(b)
    ctx := context.Background()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        _, err := vm.ListVMs(ctx, ListOptions{Limit: 100})
        if err != nil {
            b.Fatal(err)
        }
    }
}
```

#### TypeScript/React Development Standards

**Component Structure:**
```typescript
// components/VMList.tsx
import React, { useMemo, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { VM, VMListOptions } from '@/types/vm';
import { Button } from '@/components/ui/button';
import { LoadingSpinner } from '@/components/ui/loading-spinner';
import { ErrorAlert } from '@/components/ui/error-alert';

interface VMListProps {
  /** Tenant ID to filter VMs */
  tenantId?: string;
  /** Called when a VM is selected */
  onVMSelect?: (vm: VM) => void;
  /** Whether to show VM actions */
  showActions?: boolean;
}

/**
 * VMList component displays a list of virtual machines with filtering and actions.
 * 
 * @example
 * ```tsx
 * <VMList
 *   tenantId="tenant-123"
 *   onVMSelect={handleVMSelect}
 *   showActions={true}
 * />
 * ```
 */
export const VMList: React.FC<VMListProps> = ({
  tenantId,
  onVMSelect,
  showActions = true,
}) => {
  const queryClient = useQueryClient();
  
  // Memoize query options to prevent unnecessary re-renders
  const queryOptions = useMemo((): VMListOptions => ({
    tenantId,
    limit: 50,
    sortBy: 'created_at',
    sortOrder: 'desc',
  }), [tenantId]);
  
  // Fetch VMs with proper error handling
  const {
    data: vms,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['vms', queryOptions],
    queryFn: () => fetchVMs(queryOptions),
    staleTime: 30 * 1000, // 30 seconds
    retry: (failureCount, error) => {
      // Don't retry on authentication errors
      if (error instanceof AuthError) return false;
      return failureCount < 3;
    },
  });
  
  // VM action mutations
  const startVMMutation = useMutation({
    mutationFn: (vmId: string) => startVM(vmId),
    onSuccess: () => {
      queryClient.invalidateQueries(['vms']);
    },
    onError: (error) => {
      console.error('Failed to start VM:', error);
    },
  });
  
  // Memoized callbacks to prevent child re-renders
  const handleStartVM = useCallback((vmId: string) => {
    startVMMutation.mutate(vmId);
  }, [startVMMutation]);
  
  const handleVMClick = useCallback((vm: VM) => {
    onVMSelect?.(vm);
  }, [onVMSelect]);
  
  // Loading state
  if (isLoading) {
    return (
      <div className="flex justify-center p-8">
        <LoadingSpinner size="lg" />
      </div>
    );
  }
  
  // Error state
  if (error) {
    return (
      <ErrorAlert
        title="Failed to load VMs"
        message={error.message}
        onRetry={refetch}
      />
    );
  }
  
  // Empty state
  if (!vms?.length) {
    return (
      <div className="text-center p-8">
        <p className="text-gray-500">No virtual machines found</p>
        <Button onClick={refetch} variant="outline" className="mt-4">
          Refresh
        </Button>
      </div>
    );
  }
  
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {vms.map((vm) => (
          <VMCard
            key={vm.id}
            vm={vm}
            onClick={handleVMClick}
            onStart={showActions ? handleStartVM : undefined}
            isStarting={startVMMutation.variables === vm.id}
          />
        ))}
      </div>
    </div>
  );
};

// Export component with display name for debugging
VMList.displayName = 'VMList';
```

**Custom Hooks Pattern:**
```typescript
// hooks/useVM.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { VM, VMConfig, VMUpdateInput } from '@/types/vm';
import { vmApi } from '@/lib/api/vm';

export interface UseVMOptions {
  /** Whether to fetch VM data on mount */
  enabled?: boolean;
  /** Polling interval in milliseconds */
  refetchInterval?: number;
}

export const useVM = (vmId: string, options: UseVMOptions = {}) => {
  const queryClient = useQueryClient();
  
  const vmQuery = useQuery({
    queryKey: ['vm', vmId],
    queryFn: () => vmApi.getVM(vmId),
    enabled: Boolean(vmId) && options.enabled !== false,
    refetchInterval: options.refetchInterval,
  });
  
  const updateMutation = useMutation({
    mutationFn: (input: VMUpdateInput) => vmApi.updateVM(vmId, input),
    onSuccess: (updatedVM) => {
      queryClient.setQueryData(['vm', vmId], updatedVM);
      queryClient.invalidateQueries(['vms']);
    },
  });
  
  const deleteMutation = useMutation({
    mutationFn: () => vmApi.deleteVM(vmId),
    onSuccess: () => {
      queryClient.removeQueries(['vm', vmId]);
      queryClient.invalidateQueries(['vms']);
    },
  });
  
  return {
    vm: vmQuery.data,
    isLoading: vmQuery.isLoading,
    error: vmQuery.error,
    refetch: vmQuery.refetch,
    
    updateVM: updateMutation.mutate,
    isUpdating: updateMutation.isPending,
    updateError: updateMutation.error,
    
    deleteVM: deleteMutation.mutate,
    isDeleting: deleteMutation.isPending,
    deleteError: deleteMutation.error,
  };
};
```

### 5. Testing Strategy

#### Backend Testing

**Unit Testing:**
```go
// vm_test.go
package vm_test

import (
    "context"
    "testing"
    "time"
    
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/testcontainers/testcontainers-go"
    
    "github.com/novacron/backend/core/vm"
)

func TestVMManager_Integration(t *testing.T) {
    // Setup test containers
    ctx := context.Background()
    pgContainer, err := testcontainers.GenericContainer(ctx, testcontainers.GenericContainerRequest{
        ContainerRequest: testcontainers.ContainerRequest{
            Image:        "postgres:15-alpine",
            ExposedPorts: []string{"5432/tcp"},
            Env: map[string]string{
                "POSTGRES_DB":       "test",
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_USER":     "test",
            },
        },
        Started: true,
    })
    require.NoError(t, err)
    defer pgContainer.Terminate(ctx)
    
    // Get container connection details
    host, err := pgContainer.Host(ctx)
    require.NoError(t, err)
    port, err := pgContainer.MappedPort(ctx, "5432")
    require.NoError(t, err)
    
    // Initialize VM manager with test database
    dsn := fmt.Sprintf("postgres://test:test@%s:%s/test?sslmode=disable", host, port.Port())
    vmManager, err := vm.NewVMManager(dsn)
    require.NoError(t, err)
    
    // Run integration tests
    t.Run("CreateAndDeleteVM", func(t *testing.T) {
        config := vm.VMConfig{
            Name:     "test-vm",
            CPU:      2,
            Memory:   4096,
            Disk:     50,
            TenantID: "test-tenant",
        }
        
        // Create VM
        createdVM, err := vmManager.CreateVM(ctx, config)
        require.NoError(t, err)
        assert.Equal(t, config.Name, createdVM.Name)
        assert.Equal(t, vm.StateCreating, createdVM.State)
        
        // Verify VM exists
        retrievedVM, err := vmManager.GetVM(ctx, createdVM.ID)
        require.NoError(t, err)
        assert.Equal(t, createdVM.ID, retrievedVM.ID)
        
        // Delete VM
        err = vmManager.DeleteVM(ctx, createdVM.ID)
        require.NoError(t, err)
        
        // Verify VM is deleted
        _, err = vmManager.GetVM(ctx, createdVM.ID)
        assert.ErrorIs(t, err, vm.ErrVMNotFound)
    })
}

func TestVMConfig_Validate(t *testing.T) {
    tests := []struct {
        name    string
        config  vm.VMConfig
        wantErr string
    }{
        {
            name: "valid config",
            config: vm.VMConfig{
                Name:     "test-vm",
                CPU:      2,
                Memory:   4096,
                Disk:     50,
                TenantID: "tenant-1",
            },
        },
        {
            name: "missing name",
            config: vm.VMConfig{
                CPU:      2,
                Memory:   4096,
                Disk:     50,
                TenantID: "tenant-1",
            },
            wantErr: "name is required",
        },
        {
            name: "invalid CPU",
            config: vm.VMConfig{
                Name:     "test-vm",
                CPU:      0,
                Memory:   4096,
                Disk:     50,
                TenantID: "tenant-1",
            },
            wantErr: "CPU must be at least 1",
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            err := tt.config.Validate()
            if tt.wantErr != "" {
                assert.Error(t, err)
                assert.Contains(t, err.Error(), tt.wantErr)
            } else {
                assert.NoError(t, err)
            }
        })
    }
}
```

#### Frontend Testing

**Component Testing:**
```typescript
// __tests__/components/VMList.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { VMList } from '@/components/VMList';
import { vmApi } from '@/lib/api/vm';

// Mock API
jest.mock('@/lib/api/vm');
const mockVmApi = vmApi as jest.Mocked<typeof vmApi>;

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('VMList', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });
  
  it('displays loading state initially', () => {
    mockVmApi.fetchVMs.mockImplementation(() => new Promise(() => {}));
    
    render(<VMList />, { wrapper: createWrapper() });
    
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });
  
  it('displays VMs when loaded', async () => {
    const mockVMs = [
      {
        id: 'vm-1',
        name: 'test-vm-1',
        state: 'running',
        cpu: 2,
        memory: 4096,
      },
      {
        id: 'vm-2',
        name: 'test-vm-2',
        state: 'stopped',
        cpu: 4,
        memory: 8192,
      },
    ];
    
    mockVmApi.fetchVMs.mockResolvedValue(mockVMs);
    
    render(<VMList />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('test-vm-1')).toBeInTheDocument();
      expect(screen.getByText('test-vm-2')).toBeInTheDocument();
    });
  });
  
  it('handles VM selection', async () => {
    const mockVMs = [
      {
        id: 'vm-1',
        name: 'test-vm-1',
        state: 'running',
        cpu: 2,
        memory: 4096,
      },
    ];
    
    const onVMSelect = jest.fn();
    mockVmApi.fetchVMs.mockResolvedValue(mockVMs);
    
    render(<VMList onVMSelect={onVMSelect} />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('test-vm-1')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('test-vm-1'));
    
    expect(onVMSelect).toHaveBeenCalledWith(mockVMs[0]);
  });
  
  it('displays error state on API failure', async () => {
    const error = new Error('Failed to fetch VMs');
    mockVmApi.fetchVMs.mockRejectedValue(error);
    
    render(<VMList />, { wrapper: createWrapper() });
    
    await waitFor(() => {
      expect(screen.getByText('Failed to load VMs')).toBeInTheDocument();
      expect(screen.getByText(error.message)).toBeInTheDocument();
    });
  });
});
```

**E2E Testing with Playwright:**
```typescript
// e2e/vm-management.spec.ts
import { test, expect } from '@playwright/test';

test.describe('VM Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login and navigate to VM management
    await page.goto('/auth/login');
    await page.fill('[data-testid="email"]', 'test@novacron.com');
    await page.fill('[data-testid="password"]', 'testpassword');
    await page.click('[data-testid="login-button"]');
    
    await page.waitForURL('/dashboard');
    await page.click('[data-testid="nav-vms"]');
  });
  
  test('creates a new VM', async ({ page }) => {
    // Click create VM button
    await page.click('[data-testid="create-vm-button"]');
    
    // Fill VM creation form
    await page.fill('[data-testid="vm-name"]', 'test-vm-e2e');
    await page.selectOption('[data-testid="vm-cpu"]', '2');
    await page.selectOption('[data-testid="vm-memory"]', '4096');
    await page.selectOption('[data-testid="vm-disk"]', '50');
    
    // Submit form
    await page.click('[data-testid="create-vm-submit"]');
    
    // Verify VM creation success
    await expect(page.locator('[data-testid="success-message"]')).toContainText('VM created successfully');
    
    // Verify VM appears in list
    await expect(page.locator('[data-testid="vm-list"]')).toContainText('test-vm-e2e');
  });
  
  test('starts and stops a VM', async ({ page }) => {
    // Locate VM in list
    const vmRow = page.locator('[data-testid="vm-row"]:has-text("test-vm")').first();
    
    // Start VM
    await vmRow.locator('[data-testid="start-vm-button"]').click();
    await expect(vmRow.locator('[data-testid="vm-status"]')).toContainText('Running');
    
    // Stop VM
    await vmRow.locator('[data-testid="stop-vm-button"]').click();
    await expect(vmRow.locator('[data-testid="vm-status"]')).toContainText('Stopped');
  });
  
  test('displays VM metrics', async ({ page }) => {
    // Click on VM to view details
    await page.click('[data-testid="vm-row"]:has-text("test-vm") [data-testid="view-details"]');
    
    // Verify metrics are displayed
    await expect(page.locator('[data-testid="cpu-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="memory-metric"]')).toBeVisible();
    await expect(page.locator('[data-testid="network-metric"]')).toBeVisible();
    
    // Verify metrics chart
    await expect(page.locator('[data-testid="metrics-chart"]')).toBeVisible();
  });
});
```

### 6. CI/CD Pipeline Configuration

#### GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  GO_VERSION: '1.23'
  NODE_VERSION: '18'

jobs:
  # Backend tests
  backend-test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
          POSTGRES_DB: novacron_test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Go
      uses: actions/setup-go@v4
      with:
        go-version: ${{ env.GO_VERSION }}
    
    - name: Cache Go modules
      uses: actions/cache@v3
      with:
        path: ~/go/pkg/mod
        key: ${{ runner.os }}-go-${{ hashFiles('**/go.sum') }}
        restore-keys: |
          ${{ runner.os }}-go-
    
    - name: Download dependencies
      run: go mod download
      working-directory: ./backend
    
    - name: Run linter
      uses: golangci/golangci-lint-action@v3
      with:
        version: latest
        working-directory: ./backend
    
    - name: Run tests
      run: |
        go test -v -race -coverprofile=coverage.out ./...
        go tool cover -html=coverage.out -o coverage.html
      working-directory: ./backend
      env:
        DATABASE_URL: postgres://postgres:testpass@localhost:5432/novacron_test?sslmode=disable
        REDIS_URL: redis://localhost:6379
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./backend/coverage.out
        flags: backend
    
    - name: Build binary
      run: go build -o bin/novacron-api cmd/api-server/main.go
      working-directory: ./backend

  # Frontend tests
  frontend-test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'
        cache-dependency-path: ./frontend/package-lock.json
    
    - name: Install dependencies
      run: npm ci
      working-directory: ./frontend
    
    - name: Run linter
      run: npm run lint
      working-directory: ./frontend
    
    - name: Run type check
      run: npm run type-check
      working-directory: ./frontend
    
    - name: Run unit tests
      run: npm run test:ci
      working-directory: ./frontend
    
    - name: Build application
      run: npm run build
      working-directory: ./frontend
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        directory: ./frontend/coverage
        flags: frontend

  # E2E tests
  e2e-test:
    runs-on: ubuntu-latest
    needs: [backend-test, frontend-test]
    if: github.event_name == 'pull_request'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Compose
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: ${{ env.NODE_VERSION }}
        cache: 'npm'
        cache-dependency-path: ./frontend/package-lock.json
    
    - name: Install dependencies
      run: npm ci
      working-directory: ./frontend
    
    - name: Install Playwright
      run: npx playwright install --with-deps
      working-directory: ./frontend
    
    - name: Run E2E tests
      run: npm run test:e2e
      working-directory: ./frontend
    
    - name: Upload E2E results
      uses: actions/upload-artifact@v3
      if: failure()
      with:
        name: playwright-report
        path: frontend/playwright-report/
        retention-days: 30

  # Security scanning
  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  # Build and push Docker images
  build:
    runs-on: ubuntu-latest
    needs: [backend-test, frontend-test]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Build and push backend
      uses: docker/build-push-action@v4
      with:
        context: ./backend
        push: true
        tags: |
          ghcr.io/novacron/api:latest
          ghcr.io/novacron/api:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
    
    - name: Build and push frontend
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        push: true
        tags: |
          ghcr.io/novacron/frontend:latest
          ghcr.io/novacron/frontend:${{ github.sha }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  # Deploy to staging
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add deployment script here
```

### 7. Code Quality & Security

#### Pre-commit Hooks
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: check-added-large-files

  - repo: https://github.com/dnephin/pre-commit-golang
    rev: v0.5.1
    hooks:
      - id: go-fmt
      - id: go-imports
      - id: go-cyclo
        args: [-over=15]
      - id: go-mod-tidy
      - id: go-unit-tests
      - id: golangci-lint

  - repo: https://github.com/pre-commit/mirrors-eslint
    rev: v8.44.0
    hooks:
      - id: eslint
        files: \.(js|jsx|ts|tsx)$
        types: [file]
        additional_dependencies:
          - eslint@8.44.0
          - typescript@5.1.6

  - repo: https://github.com/pre-commit/mirrors-prettier
    rev: v3.0.0
    hooks:
      - id: prettier
        files: \.(js|jsx|ts|tsx|json|css|md)$
```

#### Security Configuration

**Go Security (gosec configuration):**
```json
{
  "severity": "medium",
  "confidence": "medium",
  "rules": {
    "G101": "Look for hard coded credentials",
    "G102": "Bind to all interfaces",
    "G103": "Audit the use of unsafe block",
    "G104": "Audit errors not checked",
    "G106": "Audit the use of ssh.InsecureIgnoreHostKey",
    "G107": "Url provided to HTTP request as taint input",
    "G108": "Profiling endpoint automatically exposed on /debug/pprof",
    "G109": "Potential Integer overflow made by strconv.Atoi result conversion to int16/32",
    "G110": "Potential DoS vulnerability via decompression bomb",
    "G201": "SQL query construction using format string",
    "G202": "SQL query construction using string concatenation",
    "G203": "Use of unescaped data in HTML templates",
    "G204": "Audit use of command execution",
    "G301": "Poor file permissions used when creating a directory",
    "G302": "Poor file permissions used with chmod",
    "G303": "Creating tempfile using a predictable path",
    "G304": "File path provided as taint input",
    "G305": "File traversal when extracting zip/tar archive",
    "G306": "Poor file permissions used when writing to a new file",
    "G307": "Poor file permissions used when creating a file with os.Create",
    "G401": "Detect the usage of DES, RC4, MD5 or SHA1",
    "G402": "Look for bad TLS connection settings",
    "G403": "Ensure minimum RSA key length of 2048 bits",
    "G404": "Insecure random number source (rand)",
    "G501": "Import blacklist: crypto/md5",
    "G502": "Import blacklist: crypto/des",
    "G503": "Import blacklist: crypto/rc4",
    "G504": "Import blacklist: net/http/cgi",
    "G505": "Import blacklist: crypto/sha1",
    "G601": "Implicit memory aliasing of items from a range statement"
  }
}
```

### 8. Documentation Standards

#### Code Documentation Guidelines

**Go Documentation:**
```go
// Package vm provides comprehensive virtual machine management capabilities
// for the NovaCron platform.
//
// This package implements the core VM lifecycle operations including creation,
// deletion, migration, and monitoring across multiple hypervisor platforms
// including KVM/QEMU, VMware vSphere, and Microsoft Hyper-V.
//
// Basic usage:
//
//   manager, err := vm.NewVMManager(db, config)
//   if err != nil {
//       log.Fatal(err)
//   }
//
//   vm, err := manager.CreateVM(ctx, vm.VMConfig{
//       Name:   "my-vm",
//       CPU:    2,
//       Memory: 4096,
//       Disk:   50,
//   })
//
// The package provides comprehensive error handling and supports operation
// cancellation via context.Context.
package vm

// VMManager manages virtual machine lifecycle operations across multiple
// hypervisor platforms.
//
// The VMManager provides a unified interface for VM operations regardless
// of the underlying hypervisor technology. It handles resource validation,
// operation orchestration, and maintains VM metadata in the database.
//
// All operations are context-aware and support cancellation for long-running
// operations such as VM creation and migration.
type VMManager struct {
    // db provides database connectivity for VM metadata storage
    db *sql.DB
    
    // hypervisors contains registered hypervisor drivers
    hypervisors map[HypervisorType]HypervisorDriver
    
    // logger provides structured logging for operations
    logger *logrus.Entry
}

// CreateVM creates a new virtual machine with the specified configuration.
//
// This method performs comprehensive validation of the VM configuration,
// checks resource availability, and creates the VM using the appropriate
// hypervisor driver. The operation is atomic - if any step fails, the
// VM is not created and no resources are consumed.
//
// The method returns the created VM instance with its assigned ID and
// initial state, or an error if creation fails.
//
// Example:
//
//   vm, err := manager.CreateVM(ctx, vm.VMConfig{
//       Name:     "web-server",
//       CPU:      4,
//       Memory:   8192,
//       Disk:     100,
//       Image:    "ubuntu-22.04",
//       Network:  "production",
//   })
//   if err != nil {
//       return fmt.Errorf("failed to create VM: %w", err)
//   }
//
// Parameters:
//   ctx: Context for operation cancellation and timeout
//   config: VM configuration specifying resources and settings
//
// Returns:
//   *VM: Created virtual machine instance
//   error: Error if creation fails, nil on success
//
// Errors:
//   ErrInvalidConfig: Configuration validation failed
//   ErrInsufficientResources: Not enough resources available
//   ErrHypervisorError: Hypervisor operation failed
func (m *VMManager) CreateVM(ctx context.Context, config VMConfig) (*VM, error) {
    // Implementation...
}
```

**API Documentation:**
```go
// @title NovaCron API
// @version 1.0
// @description Comprehensive virtual machine management API
// @termsOfService https://novacron.com/terms

// @contact.name API Support
// @contact.url https://novacron.com/support
// @contact.email support@novacron.com

// @license.name Apache 2.0
// @license.url http://www.apache.org/licenses/LICENSE-2.0.html

// @host api.novacron.com
// @BasePath /api/v1

// @securityDefinitions.apikey BearerAuth
// @in header
// @name Authorization
// @description Type "Bearer" followed by a space and JWT token.

// ListVMs returns a paginated list of virtual machines for the authenticated user.
//
// This endpoint supports filtering by various VM attributes and provides
// comprehensive VM information including current status, resource usage,
// and configuration details.
//
// @Summary List virtual machines
// @Description Get paginated list of VMs with filtering options
// @Tags virtual-machines
// @Accept json
// @Produce json
// @Param tenant_id query string false "Filter by tenant ID"
// @Param state query string false "Filter by VM state" Enums(running,stopped,creating,error)
// @Param limit query int false "Number of VMs to return" minimum(1) maximum(100) default(20)
// @Param offset query int false "Number of VMs to skip" minimum(0) default(0)
// @Success 200 {object} VMListResponse
// @Failure 400 {object} ErrorResponse
// @Failure 401 {object} ErrorResponse
// @Failure 500 {object} ErrorResponse
// @Security BearerAuth
// @Router /vms [get]
func (h *VMHandler) ListVMs(c *gin.Context) {
    // Implementation...
}
```

#### README Template
```markdown
# NovaCron - Distributed VM Management Platform

[![CI/CD](https://github.com/novacron/novacron/workflows/CI/badge.svg)](https://github.com/novacron/novacron/actions)
[![Go Report Card](https://goreportcard.com/badge/github.com/novacron/novacron)](https://goreportcard.com/report/github.com/novacron/novacron)
[![Coverage Status](https://codecov.io/gh/novacron/novacron/branch/main/graph/badge.svg)](https://codecov.io/gh/novacron/novacron)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

NovaCron is an enterprise-grade distributed virtual machine management platform that provides comprehensive VM lifecycle management, real-time monitoring, and multi-cloud orchestration capabilities.

## 🚀 Features

- **Multi-Hypervisor Support**: KVM/QEMU, VMware vSphere, Hyper-V
- **Real-time Monitoring**: Comprehensive metrics and alerting
- **High Availability**: Distributed architecture with automatic failover
- **Multi-tenant**: Complete tenant isolation and resource management
- **RESTful APIs**: Comprehensive API with GraphQL support
- **Web Interface**: Modern React-based management dashboard
- **Security**: Enterprise-grade authentication and authorization

## 📋 Prerequisites

- **Go**: 1.23+ for backend development
- **Node.js**: 18+ for frontend development
- **PostgreSQL**: 15+ for data storage
- **Redis**: 7+ for caching and sessions
- **Docker**: 24+ for containerization
- **KVM/QEMU**: For virtualization support

## 🛠️ Quick Start

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/novacron/novacron.git
   cd novacron
   ```

2. **Setup development environment**
   ```bash
   make dev-setup
   ```

3. **Start services**
   ```bash
   docker-compose up -d postgres redis
   ```

4. **Run backend**
   ```bash
   cd backend
   go run cmd/api-server/main.go
   ```

5. **Run frontend**
   ```bash
   cd frontend
   npm run dev
   ```

6. **Access the application**
   - Frontend: http://localhost:3000
   - API: http://localhost:8080
   - API Docs: http://localhost:8080/docs

### Production Deployment

See our [Deployment Guide](docs/comprehensive/04-deployment-guide.md) for detailed production setup instructions.

## 📖 Documentation

- [Technical Architecture](docs/comprehensive/02-technical-architecture-guide.md)
- [API Documentation](docs/comprehensive/03-api-documentation.md)
- [Security & Compliance](docs/comprehensive/05-security-compliance-document.md)
- [Performance Optimization](docs/comprehensive/06-performance-optimization-guide.md)
- [Troubleshooting](docs/comprehensive/07-troubleshooting-handbook.md)
- [Development Workflow](docs/comprehensive/08-development-workflow-guide.md)

## 🧪 Testing

### Backend Tests
```bash
cd backend
go test -v -race ./...
```

### Frontend Tests
```bash
cd frontend
npm run test
npm run test:e2e
```

### Integration Tests
```bash
make test-integration
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Standards

- Follow [Go Code Review Guidelines](https://github.com/golang/go/wiki/CodeReviewComments)
- Use [Conventional Commits](https://conventionalcommits.org/)
- Ensure >80% test coverage for new code
- Run `make lint` before submitting PRs

## 📊 Performance

- **API Response Time**: <1s P95
- **System Uptime**: >99.9%
- **Throughput**: 1000+ RPS
- **VM Operations**: <30s average

## 🔒 Security

NovaCron implements enterprise-grade security including:

- JWT-based authentication with MFA support
- Role-based access control (RBAC)
- TLS encryption for all communications
- Comprehensive audit logging
- Regular security audits and updates

Report security vulnerabilities to security@novacron.com.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.novacron.com](https://docs.novacron.com)
- **Issues**: [GitHub Issues](https://github.com/novacron/novacron/issues)
- **Discussions**: [GitHub Discussions](https://github.com/novacron/novacron/discussions)
- **Email**: support@novacron.com

## 🙏 Acknowledgments

- The Go community for excellent tooling
- The React and Next.js teams for the frontend framework
- All our contributors and users

---

**Built with ❤️ by the NovaCron team**
```

## Team Collaboration

### 1. Communication Standards

#### Daily Standups
- **Time**: 9:00 AM local time
- **Duration**: 15 minutes maximum
- **Format**: What did you do yesterday? What will you do today? Any blockers?
- **Tool**: Slack/Teams with async updates for distributed team

#### Code Review Process
- **Minimum**: 1 approval required for PRs
- **Senior Review**: Required for architectural changes
- **Response Time**: 24 hours maximum for review
- **Conflict Resolution**: Team discussion or architect decision

#### Documentation Updates
- **API Changes**: Update OpenAPI specs immediately
- **Architecture Changes**: Update technical docs within 1 week
- **Process Changes**: Update team guides immediately
- **Knowledge Sharing**: Weekly tech talks and documentation reviews

### 2. Development Environment Best Practices

#### Local Development Setup
```bash
#!/bin/bash
# Development environment setup script

set -e

echo "Setting up NovaCron development environment..."

# Check prerequisites
command -v go >/dev/null 2>&1 || { echo "Go is required but not installed. Aborting." >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Node.js is required but not installed. Aborting." >&2; exit 1; }
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting." >&2; exit 1; }

# Setup Go environment
export GOPATH=$HOME/go
export PATH=$PATH:$GOPATH/bin

# Install development tools
go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest
go install github.com/swaggo/swag/cmd/swag@latest
go install github.com/golang-migrate/migrate/v4/cmd/migrate@latest

# Setup Node.js tools
npm install -g @playwright/test
npm install -g eslint prettier typescript

# Setup pre-commit hooks
pip3 install pre-commit
pre-commit install

# Start development services
docker-compose -f docker-compose.dev.yml up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 10

# Run database migrations
cd backend
migrate -path migrations -database "postgres://novacron:password@localhost:5432/novacron?sslmode=disable" up

echo "Development environment setup complete!"
echo "Backend: http://localhost:8080"
echo "Frontend: http://localhost:3000"
echo "Database: localhost:5432"
echo "Redis: localhost:6379"
```

#### Development Workflow Checklist

**Before Starting Work:**
- [ ] Pull latest changes from develop branch
- [ ] Create feature branch with descriptive name
- [ ] Verify all services are running locally
- [ ] Run existing tests to ensure clean baseline

**During Development:**
- [ ] Write tests for new functionality
- [ ] Follow coding standards and conventions
- [ ] Update documentation for API changes
- [ ] Commit frequently with meaningful messages
- [ ] Run linters and formatters regularly

**Before Creating PR:**
- [ ] Run full test suite locally
- [ ] Update API documentation if applicable
- [ ] Squash commits if necessary
- [ ] Write comprehensive PR description
- [ ] Add reviewers and labels

**After PR Approval:**
- [ ] Merge using squash and merge
- [ ] Delete feature branch
- [ ] Verify deployment in staging
- [ ] Monitor for any issues

---

**Document Classification**: Development Team - Internal Use  
**Last Updated**: September 2, 2025  
**Version**: 1.0  
**Review Schedule**: Quarterly  
**Team Lead**: development@novacron.com