# NovaCron Compilation Fixes TODO

## Critical Import Cycle Fixed ✅
- Fixed federation → backup → federation import cycle by creating shared interfaces

## Federation Package Issues Fixed ✅
- Resolved type redeclarations (ResourceAllocation, FederationManager, etc.)
- Fixed interface vs struct conflicts by renaming implementations  
- Added missing interface methods with stub implementations
- Fixed ResourceAllocation struct field alignment
- Updated method signatures and receivers

## Remaining Critical Issues

### 1. Type Redeclarations in Federation ✅ FIXED
- ResourceAllocation redeclared in types.go and federation_manager.go
- FederationManager redeclared in types.go and federation_manager.go  
- DiscoveryManager redeclared in types.go and registry.go
- HealthChecker redeclared in types.go and health.go

### 2. VM Manager Interface Issues
- vm.VM type not recognized in hypervisor
- vm.Manager undefined in multiple API packages
- ListMigrations method missing from VMManager
- Various VM method signature mismatches

### 3. Monitoring MetricBatch Issues  
- MetricBatch struct field mismatches
- metric.ID undefined issues

### 4. Multiple main() Functions
- Conflicting main functions in root directory
- Need to clean up test and example files

### 5. Missing API Endpoints
Still need to implement the requested endpoints after compilation fixes

## Fix Order Priority
1. Remove duplicate type declarations
2. Fix VM Manager interface consistency  
3. Fix MetricBatch struct alignment
4. Clean up duplicate main functions
5. Add missing VM Manager methods
6. Implement missing API endpoints