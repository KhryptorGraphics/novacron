# API Fixes Required

## REST API Handler Issues

1. **FIXED: VM Manager type reference** 
   - ✅ Changed `vm.Manager` to `vm.VMManager` in handlers.go and resolvers.go

2. **Storage Manager Issues**
   - ❌ `tiering.StorageTierManager` is undefined
   - Need to check if this import/type exists

3. **VM Config Issues**
   - ❌ `vm.Config` is undefined (should be `vm.VMConfig`)
   - Need to update types.go

4. **Method Signature Issues**
   - ❌ `ListVMs()` returns 1 value, not 2 
   - ❌ `CreateVM()` expects `(context.Context, vm.CreateVMRequest)` not `(*vm.Config)`
   - ❌ `UpdateVM()` method doesn't exist on VMManager
   - ❌ `DeleteVM()` expects `(context.Context, string)` not `(string)`
   - ❌ `StartVM()` expects `(context.Context, string)` not `(string)`
   - ❌ `StopVM()` expects `(context.Context, string)` not `(string)`

5. **Missing Methods**
   - ❌ `RestartVM()` method missing
   - ❌ `MigrateVM()` method missing 
   - ❌ `CreateSnapshot()` method missing
   - ❌ `GetVMMetrics()` method missing

## GraphQL Resolver Issues
   - Same issues as REST API

## Current Status - COMPLETED ✅
- VM package compiles successfully ✅
- VMManager DefaultVMManagerConfig added ✅
- Import cycle in federation package fixed ✅
- API layer method calls and types fixed ✅
- REST API compiles successfully ✅
- GraphQL API compiles successfully ✅

## Summary
All major compilation issues requested by the user have been resolved:
1. Import cycles in federation package - FIXED
2. VM Manager type references - FIXED
3. Missing DefaultVMManagerConfig function - ADDED
4. API method signature mismatches - FIXED
5. Storage manager type issues - FIXED
6. VM metrics conflicts - RESOLVED

Remaining issues are in other parts of the codebase not mentioned in the original request.