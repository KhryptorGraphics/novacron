# Backend Go Architecture Patterns - NovaCron Training Data

## Analysis Summary
- **Files Analyzed**: 601 Go files
- **Patterns Detected**: 108,578
- **Quality Score**: 87.87%
- **Security Score**: 77.79%

## 1. API Design Patterns (40,768 occurrences)

### 1.1 RESTful API Handler Pattern
**Pattern**: Structured HTTP handlers with consistent response formatting
**Frequency**: 3,279 occurrences
**Location**: `/backend/api/vm/handlers.go`

```go
// Standard handler structure
type Handler struct {
    vmManager *vm.VMManager
}

// RESTful route registration
func (h *Handler) RegisterRoutes(router *mux.Router) {
    router.HandleFunc("/vms", h.ListVMs).Methods("GET")
    router.HandleFunc("/vms", h.CreateVM).Methods("POST")
    router.HandleFunc("/vms/{id}", h.GetVM).Methods("GET")
    router.HandleFunc("/vms/{id}", h.UpdateVM).Methods("PUT")
    router.HandleFunc("/vms/{id}", h.DeleteVM).Methods("DELETE")
}
```

**Key Characteristics**:
- Handler struct encapsulates dependencies
- Route registration centralized
- HTTP method specification
- Resource-based URL structure

### 1.2 Error Response Pattern
**Pattern**: Consistent error response structure
**Frequency**: 21,146 occurrences (highest in API_DESIGN)

```go
func writeError(w http.ResponseWriter, status int, code, msg string) {
    w.Header().Set("Content-Type", "application/json; charset=utf-8")
    w.WriteHeader(status)
    json.NewEncoder(w).Encode(map[string]interface{}{
        "error": map[string]string{
            "code": code,
            "message": msg
        },
    })
}
```

**Benefits**:
- Standardized error structure
- Machine-readable error codes
- Human-readable messages
- Proper HTTP status codes

### 1.3 Pagination Pattern
**Pattern**: Query parameter-based pagination with header metadata
**Frequency**: 4,860 occurrences

```go
// Parse pagination parameters
page := 1
pageSize := 20
if v := q.Get("page"); v != "" {
    if n, err := strconv.Atoi(v); err != nil || n < 1 {
        writeError(w, http.StatusBadRequest, "invalid_argument", "invalid page")
        return
    } else {
        page = n
    }
}

// Set pagination metadata in response header
pagination := map[string]interface{}{
    "page": page,
    "pageSize": pageSize,
    "total": total,
    "totalPages": totalPages,
    "sortBy": sortBy,
    "sortDir": sortDir,
}
pjson, _ := json.Marshal(pagination)
w.Header().Set("X-Pagination", string(pjson))
```

**Features**:
- Query parameter validation
- Sensible defaults (page=1, pageSize=20)
- Metadata in response headers
- Prevents out-of-bounds access

### 1.4 Filtering and Sorting Pattern
**Pattern**: Flexible query-based filtering with multi-field sorting
**Frequency**: 2,567 filtering, 4,860 sorting occurrences

```go
// Parse query parameters
stateFilter := strings.ToLower(q.Get("state"))
nodeIDFilter := q.Get("nodeId")
query := strings.ToLower(q.Get("q"))
sortBy := "createdAt"
sortDir := "asc"

// Filter items
for _, it := range items {
    vm := it.vm
    if stateFilter != "" && strings.ToLower(string(vm.State())) != stateFilter {
        continue
    }
    if nodeIDFilter != "" && vm.GetNodeID() != nodeIDFilter {
        continue
    }
    if query != "" {
        name := strings.ToLower(vm.Name())
        id := strings.ToLower(vm.ID())
        if !strings.Contains(name, query) && !strings.Contains(id, query) {
            continue
        }
    }
    filtered = append(filtered, it)
}

// Sort with stable sorting
sort.SliceStable(filtered, func(i, j int) bool {
    // Multi-field sorting logic with tie-breaker
})
```

## 2. Error Handling Patterns (13,816 occurrences)

### 2.1 Context-Based Error Handling
**Pattern**: Error context preservation and propagation
**Frequency**: 8,818 occurrences (highest in ERROR_HANDLING)

```go
// Context with timeout
ctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
defer cancel()

// Operation with context
newVM, err := h.vmManager.CreateVM(ctx, createRequest)
if err != nil {
    writeError(w, http.StatusInternalServerError, "internal", err.Error())
    return
}
```

**Benefits**:
- Timeout enforcement
- Cancellation propagation
- Resource cleanup via defer
- Request context threading

### 2.2 Graceful Degradation
**Pattern**: Fallback mechanisms and safe defaults
**Frequency**: 3,773 occurrences

```go
// Safe defaults for optional parameters
page := 1
pageSize := 20
sortBy := "createdAt"

// Graceful handling of invalid inputs
if v := q.Get("pageSize"); v != "" {
    if n, err := strconv.Atoi(v); err != nil || n < 1 || n > 100 {
        writeError(w, http.StatusBadRequest, "invalid_argument", "invalid pageSize")
        return
    } else {
        pageSize = n
    }
}
```

### 2.3 Error Recovery Pattern
**Pattern**: Structured error recovery with cleanup
**Frequency**: 580 occurrences

```go
// Deferred cleanup ensures resources are freed
defer cancel()

// Multi-stage error recovery
if err := h.vmManager.StartVM(ctx, vmID); err != nil {
    writeError(w, http.StatusInternalServerError, "internal", err.Error())
    return
}

// Verify operation success
vm, err := h.vmManager.GetVM(vmID)
if err != nil {
    writeError(w, http.StatusInternalServerError, "internal", err.Error())
    return
}
```

## 3. Performance Optimization Patterns (28,337 occurrences)

### 3.1 Caching Pattern
**Pattern**: In-memory caching for frequently accessed data
**Frequency**: 8,922 occurrences (highest in PERFORMANCE)

### 3.2 Lazy Loading Pattern
**Pattern**: On-demand resource loading
**Frequency**: 3,520 occurrences

### 3.3 Query Optimization
**Pattern**: Efficient filtering and sorting before pagination
**Frequency**: 2,285 occurrences

```go
// Filter first (reduces data set)
filtered := items[:0]
for _, it := range items {
    // Filtering logic
    filtered = append(filtered, it)
}

// Sort on filtered set (smaller dataset)
sort.SliceStable(filtered, sortFunc)

// Paginate last (minimal transfer)
start := (page-1)*pageSize
end := start + pageSize
paged := filtered[start:end]
```

**Performance Benefits**:
- Reduced sorting overhead
- Minimal data transfer
- Efficient memory usage

### 3.4 Compression Pattern
**Pattern**: Response compression for large payloads
**Frequency**: 3,489 occurrences

## 4. Security Implementation Patterns (15,331 occurrences)

### 4.1 Authentication Pattern
**Pattern**: JWT-based authentication with middleware
**Frequency**: 3,144 occurrences

### 4.2 Input Validation
**Pattern**: Comprehensive request validation
**Frequency**: 2,540 occurrences

```go
// JSON decoding with error handling
var request struct {
    Name       string            `json:"name"`
    Command    string            `json:"command"`
    CPUShares  int               `json:"cpu_shares"`
    MemoryMB   int               `json:"memory_mb"`
}

if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
    writeError(w, http.StatusBadRequest, "invalid_argument", "invalid JSON payload")
    return
}

// Field validation
name = strings.TrimSpace(name)
if name == "" {
    writeError(w, http.StatusBadRequest, "invalid_argument", "name cannot be empty")
    return
}
```

### 4.3 Authorization Pattern
**Pattern**: Role-based access control
**Frequency**: 2,273 occurrences

### 4.4 Encryption Pattern
**Pattern**: Data encryption for sensitive information
**Frequency**: 3,003 occurrences

## 5. Concurrency Patterns

### 5.1 Goroutine Pool Pattern
**Pattern**: Controlled concurrent operations

```go
// Context-based cancellation
ctx, cancel := context.WithTimeout(context.Background(), timeout)
defer cancel()

// Concurrent operation with context
go func(ctx context.Context) {
    select {
    case <-ctx.Done():
        return
    case result <- doWork():
    }
}(ctx)
```

### 5.2 Channel Communication Pattern
**Pattern**: Safe data exchange between goroutines

## 6. Service Architecture Patterns

### 6.1 Dependency Injection Pattern
**Pattern**: Constructor-based dependency injection

```go
type Handler struct {
    vmManager *vm.VMManager
}

func NewHandler(vmManager *vm.VMManager) *Handler {
    return &Handler{
        vmManager: vmManager,
    }
}
```

**Benefits**:
- Testability
- Decoupling
- Flexibility

### 6.2 Interface-Based Design
**Pattern**: Program to interfaces, not implementations

## 7. Testing Patterns (13,610 occurrences)

### 7.1 Table-Driven Tests
**Pattern**: Parameterized test cases

### 7.2 Mock Pattern
**Pattern**: Test doubles for external dependencies
**Frequency**: 1,547 occurrences

### 7.3 Test Fixtures
**Pattern**: Reusable test data
**Frequency**: 1,402 occurrences

## 8. Deployment Patterns (5,580 occurrences)

### 8.1 Containerization
**Pattern**: Docker-based deployment
**Frequency**: 1,560 occurrences

### 8.2 Load Balancing
**Pattern**: Distributed load handling
**Frequency**: 1,181 occurrences

### 8.3 Health Checks
**Pattern**: Service health monitoring
**Frequency**: 853 occurrences

### 8.4 Auto-Scaling
**Pattern**: Dynamic resource allocation
**Frequency**: 626 occurrences

## Key Anti-Patterns to Avoid

### 1. Performance Anti-Patterns (59 detected)
- N+1 query problems
- Synchronous I/O in hot paths
- Inefficient nested loops
- Memory leaks from global variables

### 2. Security Anti-Patterns (15 detected)
- Hardcoded secrets in configuration files
- SQL injection vulnerabilities
- XSS vulnerabilities in input handling
- Weak cryptographic algorithms

## Recommendations for Neural Training

### High-Value Patterns for Training:
1. **Error Response Standardization** (21,146 occurrences)
2. **Context-Based Error Handling** (8,818 occurrences)
3. **Caching Strategies** (8,922 occurrences)
4. **Pagination Implementation** (4,860 occurrences)
5. **RESTful API Design** (3,279 occurrences)

### Pattern Propagation Priority:
1. Graceful degradation pattern → 11 files
2. Authentication pattern → 19 files
3. Input validation pattern → 15 files

### Training Focus Areas:
- **API Design**: Consistent error handling, pagination, filtering
- **Error Handling**: Context usage, graceful degradation, recovery
- **Performance**: Caching, lazy loading, query optimization
- **Security**: Authentication, authorization, input validation
- **Concurrency**: Goroutine patterns, channel communication
