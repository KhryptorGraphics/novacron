# DWCP v3 Production Best Practices

**Version**: 1.0.0  
**Last Updated**: 2025-11-10  
**Audience**: Engineering Teams, SREs, DevOps  
**Classification**: Internal Use

---

## Table of Contents

1. [Development Best Practices](#development-best-practices)
2. [Deployment Best Practices](#deployment-best-practices)
3. [Operational Best Practices](#operational-best-practices)
4. [Security Best Practices](#security-best-practices)
5. [Performance Best Practices](#performance-best-practices)
6. [Reliability Best Practices](#reliability-best-practices)
7. [Monitoring and Observability](#monitoring-and-observability)
8. [Database Best Practices](#database-best-practices)
9. [Anti-Patterns to Avoid](#anti-patterns-to-avoid)
10. [Code Review Checklist](#code-review-checklist)

---

## Development Best Practices

### 1. Code Organization

**Principle**: Keep code modular, maintainable, and testable

**Guidelines**:
```yaml
file_structure:
  max_lines_per_file: 500
  max_functions_per_file: 20
  max_function_length: 50 lines
  
naming_conventions:
  variables: snake_case
  functions: camelCase
  classes: PascalCase
  constants: UPPER_SNAKE_CASE
  
organization:
  - Group related functionality
  - Separate concerns (business logic, data access, presentation)
  - Use dependency injection
  - Avoid circular dependencies
```

**Example Structure**:
```
src/
├── controllers/       # HTTP request handlers
├── services/          # Business logic
├── repositories/      # Data access
├── models/            # Data structures
├── utils/             # Utility functions
├── middleware/        # Cross-cutting concerns
└── config/            # Configuration
```

### 2. Error Handling

**Principle**: Handle errors gracefully and provide meaningful context

**Best Practices**:
```go
// ✅ GOOD: Structured error handling
func ProcessOrder(orderID string) error {
    order, err := orderRepo.GetByID(orderID)
    if err != nil {
        if errors.Is(err, ErrNotFound) {
            return fmt.Errorf("order %s not found: %w", orderID, err)
        }
        return fmt.Errorf("failed to fetch order %s: %w", orderID, err)
    }
    
    if err := validateOrder(order); err != nil {
        return fmt.Errorf("invalid order %s: %w", orderID, err)
    }
    
    return nil
}

// ❌ BAD: Swallowing errors
func ProcessOrder(orderID string) {
    order, _ := orderRepo.GetByID(orderID)
    validateOrder(order)
}

// ❌ BAD: Generic error messages
func ProcessOrder(orderID string) error {
    order, err := orderRepo.GetByID(orderID)
    if err != nil {
        return errors.New("error")
    }
    return nil
}
```

**Error Handling Checklist**:
- [ ] All errors are checked and handled
- [ ] Error messages include context (what, why, where)
- [ ] Errors are logged with appropriate severity
- [ ] User-facing errors are sanitized (no internal details)
- [ ] Retry logic for transient failures
- [ ] Circuit breakers for external dependencies

### 3. Logging

**Principle**: Log meaningful events with appropriate context

**Logging Levels**:
```yaml
DEBUG:
  use: Development troubleshooting
  examples:
    - "Processing item 123"
    - "Cache hit for key user:456"
  
INFO:
  use: Normal operations, important events
  examples:
    - "Server started on port 8080"
    - "Order 789 processed successfully"
  
WARN:
  use: Recoverable issues, degraded performance
  examples:
    - "Cache miss rate high: 30%"
    - "Retry attempt 3/5 failed"
  
ERROR:
  use: Operation failures requiring attention
  examples:
    - "Failed to process payment: timeout"
    - "Database connection lost"
  
FATAL:
  use: Unrecoverable errors, service shutdown
  examples:
    - "Cannot connect to database"
    - "Critical configuration missing"
```

**Structured Logging**:
```go
// ✅ GOOD: Structured logging with context
log.Info("Order processed",
    "order_id", orderID,
    "user_id", userID,
    "amount", amount,
    "duration_ms", duration,
)

// ❌ BAD: Unstructured string logging
log.Info(fmt.Sprintf("Order %s processed for user %s", orderID, userID))
```

**Logging Best Practices**:
- Use structured logging (JSON format)
- Include request ID/trace ID in all logs
- Log at entry and exit points of important functions
- Don't log sensitive data (passwords, tokens, PII)
- Use appropriate log levels
- Include timing information for operations
- Aggregate errors before logging (avoid log spam)

### 4. Testing

**Testing Pyramid**:
```
       /\
      /  \  E2E (10%)
     /----\
    / Unit \ Integration (30%)
   /--------\
  /   Unit   \ (60%)
 /____________\
```

**Unit Testing Best Practices**:
```go
// ✅ GOOD: Clear, focused unit test
func TestOrderValidator_ValidateAmount(t *testing.T) {
    tests := []struct {
        name    string
        amount  float64
        wantErr bool
    }{
        {
            name:    "valid amount",
            amount:  100.50,
            wantErr: false,
        },
        {
            name:    "zero amount",
            amount:  0,
            wantErr: true,
        },
        {
            name:    "negative amount",
            amount:  -10.00,
            wantErr: true,
        },
    }
    
    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            validator := NewOrderValidator()
            err := validator.ValidateAmount(tt.amount)
            if (err != nil) != tt.wantErr {
                t.Errorf("ValidateAmount() error = %v, wantErr %v", err, tt.wantErr)
            }
        })
    }
}
```

**Test Coverage Goals**:
```yaml
coverage_targets:
  critical_paths: 100%
  business_logic: 90%
  overall: 80%
  
coverage_exceptions:
  - Generated code
  - Third-party integrations (use mocks)
  - Configuration files
```

### 5. Dependencies Management

**Principle**: Keep dependencies up-to-date and secure

**Best Practices**:
```yaml
dependency_management:
  versioning:
    - Use semantic versioning
    - Pin major/minor versions
    - Allow patch updates automatically
    
  security:
    - Run security scans daily
    - Update vulnerable dependencies within 48h
    - Review all dependency changes
    
  updates:
    - Update dependencies monthly
    - Test thoroughly before merging
    - Monitor deprecation notices
    
  selection:
    - Prefer well-maintained libraries
    - Check license compatibility
    - Evaluate security track record
```

**Example `package.json`**:
```json
{
  "dependencies": {
    "express": "^4.18.0",
    "pg": "^8.11.0"
  },
  "devDependencies": {
    "jest": "^29.5.0",
    "eslint": "^8.40.0"
  }
}
```

---

## Deployment Best Practices

### 1. Deployment Strategy

**Blue-Green Deployment**:
```yaml
approach: Blue-Green
use_case: Zero-downtime deployments
process:
  1: Deploy new version (green) alongside current (blue)
  2: Run health checks on green
  3: Switch traffic to green
  4: Monitor for issues
  5: Keep blue for quick rollback

pros:
  - Instant rollback
  - No downtime
  - Full testing in production environment

cons:
  - Requires double resources temporarily
  - Database migrations complex
```

**Canary Deployment**:
```yaml
approach: Canary
use_case: Risk mitigation for risky changes
process:
  1: Deploy to 5% of traffic
  2: Monitor metrics for 10 minutes
  3: Increase to 25% if healthy
  4: Increase to 50% if healthy
  5: Complete rollout to 100%

monitoring:
  - Error rate <0.5%
  - Latency p95 <500ms
  - No increase in 5xx errors

rollback_triggers:
  - Error rate >1%
  - Latency p95 >1000ms
  - Any spike in 5xx errors
```

### 2. Deployment Checklist

**Pre-Deployment**:
```markdown
- [ ] Code reviewed and approved
- [ ] All tests passing (unit, integration, E2E)
- [ ] Security scan passed
- [ ] Performance tests passed
- [ ] Documentation updated
- [ ] Runbook updated
- [ ] Monitoring/alerts configured
- [ ] Rollback plan documented
- [ ] Database migrations tested
- [ ] Feature flags configured
- [ ] Stakeholders notified
```

**During Deployment**:
```markdown
- [ ] Deployment started in log
- [ ] Health checks passing
- [ ] Metrics monitored in real-time
- [ ] Error rate within threshold
- [ ] Latency within threshold
- [ ] Logs reviewed for errors
- [ ] Database migrations completed
- [ ] Cache invalidated if needed
```

**Post-Deployment**:
```markdown
- [ ] Smoke tests executed
- [ ] End-to-end tests passed
- [ ] Monitoring dashboards checked
- [ ] No alerts firing
- [ ] Deployment documented
- [ ] Stakeholders notified of completion
- [ ] Post-deployment review scheduled
```

### 3. Rollback Procedures

**When to Rollback**:
```yaml
immediate_rollback:
  - Error rate >5%
  - Service unavailable
  - Data corruption detected
  - Security vulnerability introduced

consider_rollback:
  - Error rate >1%
  - Latency p95 >2x normal
  - User complaints spike
  - Unexpected behavior

monitoring_period:
  initial: 10 minutes (canary)
  full: 30 minutes (complete rollout)
```

**Rollback Script**:
```bash
#!/bin/bash
# Fast rollback procedure

DEPLOYMENT_ID=$1

echo "=== INITIATING ROLLBACK ==="
echo "Deployment: $DEPLOYMENT_ID"

# 1. Get previous version
PREVIOUS_VERSION=$(kubectl rollout history deployment/dwcp-api -n production | tail -2 | head -1 | awk '{print $1}')

# 2. Execute rollback
kubectl rollout undo deployment/dwcp-api -n production --to-revision=$PREVIOUS_VERSION

# 3. Wait for rollback to complete
kubectl rollout status deployment/dwcp-api -n production --timeout=5m

# 4. Verify health
./scripts/health-check.sh --comprehensive

# 5. Document rollback
./scripts/record-rollback.sh --deployment $DEPLOYMENT_ID --reason "Production issues"

echo "=== ROLLBACK COMPLETE ==="
```

---

## Operational Best Practices

### 1. On-Call Management

**On-Call Responsibilities**:
```yaml
primary_on_call:
  - Respond to alerts within 5 minutes
  - Triage and resolve incidents
  - Escalate if needed
  - Document all actions
  - Conduct post-incident reviews

secondary_on_call:
  - Backup for primary
  - Respond if primary unavailable (15 min)
  - Provide second opinion
  - Assist with complex incidents

on_call_rotation:
  duration: 1 week
  schedule: Monday 9am to Monday 9am
  handoff: Required handoff meeting
  compensation: On-call pay + time off for incidents
```

**On-Call Runbook**:
```markdown
# On-Call Runbook

## Before Your Shift
- [ ] Review recent incidents
- [ ] Check system status
- [ ] Test pager/alert channels
- [ ] Review runbooks
- [ ] Ensure access to all systems

## During Your Shift
- [ ] Acknowledge alerts within 5 minutes
- [ ] Document all actions in incident log
- [ ] Escalate if unresolved in 30 minutes
- [ ] Update stakeholders every 15 minutes
- [ ] Post incident summary after resolution

## After Your Shift
- [ ] Handoff to next on-call
- [ ] Share incident summaries
- [ ] Update runbooks with learnings
- [ ] File improvement tickets
```

### 2. Change Management

**Change Categories**:
```yaml
standard_change:
  approval: Automated
  examples:
    - Routine deployments
    - Configuration updates
    - Scaling operations
  process: Follow standard procedure

normal_change:
  approval: Change Advisory Board (CAB)
  lead_time: 48 hours
  examples:
    - Database schema changes
    - Infrastructure updates
    - Third-party integrations
  process:
    - Submit change request
    - CAB review
    - Schedule in maintenance window
    - Execute with approval

emergency_change:
  approval: Emergency CAB (within 2 hours)
  examples:
    - Security patches
    - Critical bug fixes
    - Incident remediation
  process:
    - Declare emergency
    - Emergency CAB approval
    - Document reason
    - Execute immediately
    - Post-implementation review
```

### 3. Capacity Planning

**Capacity Review Process**:
```bash
#!/bin/bash
# Monthly capacity review

MONTH=$(date +%Y-%m)

# 1. Collect current utilization
kubectl top nodes > /reports/capacity/${MONTH}/nodes.txt
kubectl top pods --all-namespaces > /reports/capacity/${MONTH}/pods.txt

# 2. Analyze trends
./scripts/capacity-trend.sh --months 3 --output /reports/capacity/${MONTH}/trends.json

# 3. Forecast future capacity
./scripts/forecast-capacity.sh --horizon 90d --output /reports/capacity/${MONTH}/forecast.json

# 4. Generate recommendations
./scripts/capacity-recommendations.sh \
    --current /reports/capacity/${MONTH}/nodes.txt \
    --trends /reports/capacity/${MONTH}/trends.json \
    --forecast /reports/capacity/${MONTH}/forecast.json \
    --output /reports/capacity/${MONTH}/recommendations.md

# 5. Present to leadership
./scripts/create-capacity-presentation.sh \
    --report /reports/capacity/${MONTH}/recommendations.md \
    --output /reports/capacity/${MONTH}/presentation.pdf
```

---

## Security Best Practices

### 1. Authentication and Authorization

**Authentication Best Practices**:
```yaml
authentication:
  method: OAuth 2.0 + JWT
  token_expiry: 1 hour (access), 7 days (refresh)
  mfa: Required for production access
  password_policy:
    min_length: 12
    complexity: uppercase + lowercase + numbers + symbols
    history: Cannot reuse last 10 passwords
    expiry: 90 days
```

**Authorization Best Practices**:
```yaml
authorization:
  model: RBAC (Role-Based Access Control)
  principle: Least privilege
  
  roles:
    read_only:
      - View dashboards
      - Read logs
      - Query metrics
    
    developer:
      - Deploy to development
      - View production metrics
      - Read production logs (sanitized)
    
    sre:
      - Deploy to staging/production
      - Modify infrastructure
      - Access production systems
    
    admin:
      - All permissions
      - User management
      - Security configuration

  review:
    frequency: Quarterly
    process: Remove unused permissions
```

### 2. Secrets Management

**Secrets Best Practices**:
```yaml
storage:
  - Use HashiCorp Vault or AWS Secrets Manager
  - Never commit secrets to Git
  - Rotate secrets every 90 days
  - Use different secrets per environment

access:
  - Limit access by role
  - Audit all secret access
  - Expire temporary credentials
  - Use service accounts for applications

rotation:
  - Automated rotation where possible
  - Zero-downtime rotation strategy
  - Verify rotation success
  - Maintain audit trail
```

**Secret Rotation Example**:
```bash
#!/bin/bash
# Automated secret rotation

SECRET_NAME=$1

# 1. Generate new secret
NEW_SECRET=$(./scripts/generate-secret.sh --type secure --length 32)

# 2. Store new secret in Vault
vault kv put secret/${SECRET_NAME}-new value="$NEW_SECRET"

# 3. Update applications to use new secret (blue-green)
kubectl set env deployment/app \
    --from secret/${SECRET_NAME}-new \
    --keys=value

# 4. Wait for rollout
kubectl rollout status deployment/app

# 5. Verify application health
./scripts/health-check.sh --app app

# 6. Delete old secret
vault kv delete secret/${SECRET_NAME}

# 7. Rename new secret
vault kv move secret/${SECRET_NAME}-new secret/${SECRET_NAME}

echo "Secret rotation complete"
```

### 3. Network Security

**Network Security Best Practices**:
```yaml
network_policies:
  default: Deny all
  allow: Explicit allow rules only
  
  example_policy:
    - Allow ingress from load balancer to API
    - Allow API to database
    - Allow API to cache
    - Deny all other traffic

encryption:
  in_transit:
    - TLS 1.3 for all external traffic
    - mTLS for internal service-to-service
    - Certificate rotation every 90 days
  
  at_rest:
    - AES-256 for all data
    - Key rotation every 180 days
    - Encrypt backups

firewalls:
  - WAF at edge (CloudFlare, AWS WAF)
  - Network firewall at VPC level
  - Host-based firewall on each node
  - Regular security group audits
```

---

## Performance Best Practices

### 1. Caching Strategy

**Caching Layers**:
```yaml
layer_1_client:
  type: Browser cache
  ttl: 5 minutes
  use: Static assets, API responses
  
layer_2_cdn:
  type: CDN (CloudFlare)
  ttl: 1 hour
  use: Images, CSS, JS, public APIs
  
layer_3_application:
  type: Redis
  ttl: 15 minutes
  use: User sessions, frequently accessed data
  
layer_4_database:
  type: PostgreSQL query cache
  ttl: N/A (automatic)
  use: Repeated queries
```

**Cache Invalidation**:
```go
// ✅ GOOD: Cache with invalidation
func UpdateUser(userID string, data UserData) error {
    // Update database
    if err := db.Update(userID, data); err != nil {
        return err
    }
    
    // Invalidate cache
    cache.Delete(fmt.Sprintf("user:%s", userID))
    
    // Invalidate related caches
    cache.Delete(fmt.Sprintf("user-profile:%s", userID))
    
    return nil
}

// ❌ BAD: Update without invalidation
func UpdateUser(userID string, data UserData) error {
    return db.Update(userID, data)
    // Cache now stale!
}
```

### 2. Database Optimization

**Query Optimization**:
```sql
-- ✅ GOOD: Indexed query with specific columns
SELECT id, name, email
FROM users
WHERE email = 'user@example.com'
AND status = 'active'
LIMIT 1;
-- Index: CREATE INDEX idx_users_email_status ON users(email, status);

-- ❌ BAD: Unindexed query selecting all columns
SELECT *
FROM users
WHERE LOWER(email) = 'user@example.com';
-- No index can be used due to LOWER()
```

**Connection Pooling**:
```yaml
connection_pool:
  min_connections: 10
  max_connections: 100
  idle_timeout: 5 minutes
  max_lifetime: 30 minutes
  
  monitoring:
    - Active connections
    - Idle connections
    - Wait time
    - Query duration
```

### 3. Async Processing

**Use Cases for Async**:
```yaml
async_processing:
  email_sending:
    method: Message queue
    reason: Non-blocking user experience
    
  report_generation:
    method: Background job
    reason: Long-running task
    
  data_export:
    method: Worker pool
    reason: Resource intensive
    
  notifications:
    method: Event-driven
    reason: Decoupling
```

**Example Implementation**:
```go
// ✅ GOOD: Async email sending
func CreateOrder(order Order) error {
    // Save order to database
    if err := db.Save(order); err != nil {
        return err
    }
    
    // Send confirmation email asynchronously
    go func() {
        sendConfirmationEmail(order)
    }()
    
    return nil
}

// Or using message queue
func CreateOrder(order Order) error {
    if err := db.Save(order); err != nil {
        return err
    }
    
    // Publish to message queue
    messageQueue.Publish("order.created", order)
    
    return nil
}
```

---

## Reliability Best Practices

### 1. Circuit Breakers

**Implementation**:
```go
import "github.com/sony/gobreaker"

var circuitBreaker = gobreaker.NewCircuitBreaker(gobreaker.Settings{
    Name:        "payment-service",
    MaxRequests: 3,
    Interval:    60 * time.Second,
    Timeout:     30 * time.Second,
    ReadyToTrip: func(counts gobreaker.Counts) bool {
        failureRatio := float64(counts.TotalFailures) / float64(counts.Requests)
        return counts.Requests >= 10 && failureRatio >= 0.5
    },
})

func ProcessPayment(payment Payment) error {
    result, err := circuitBreaker.Execute(func() (interface{}, error) {
        return paymentService.Process(payment)
    })
    
    if err != nil {
        if err == gobreaker.ErrOpenState {
            // Circuit breaker open - fail fast
            return errors.New("payment service unavailable")
        }
        return err
    }
    
    return nil
}
```

### 2. Retry Logic

**Retry Strategies**:
```yaml
retry_strategies:
  exponential_backoff:
    initial_delay: 100ms
    max_delay: 30s
    multiplier: 2
    max_attempts: 5
    jitter: true  # Add randomness to prevent thundering herd
    
  constant_backoff:
    delay: 1s
    max_attempts: 3
    
  no_retry:
    use_for:
      - User input errors (4xx)
      - Authentication failures
      - Resource not found
```

**Implementation**:
```go
func RetryWithExponentialBackoff(fn func() error, maxAttempts int) error {
    delay := 100 * time.Millisecond
    
    for attempt := 1; attempt <= maxAttempts; attempt++ {
        err := fn()
        if err == nil {
            return nil
        }
        
        if attempt == maxAttempts {
            return fmt.Errorf("max attempts reached: %w", err)
        }
        
        // Add jitter
        jitter := time.Duration(rand.Int63n(int64(delay)))
        time.Sleep(delay + jitter)
        
        // Exponential backoff
        delay *= 2
        if delay > 30*time.Second {
            delay = 30 * time.Second
        }
    }
    
    return nil
}
```

### 3. Graceful Degradation

**Degradation Strategies**:
```yaml
degradation:
  caching:
    - Serve stale cache if backend down
    - Extend TTL during outages
    
  feature_toggling:
    - Disable non-essential features
    - Reduce functionality to core only
    
  fallback_responses:
    - Return cached/default data
    - Partial responses instead of errors
    
  load_shedding:
    - Reject low-priority requests
    - Preserve capacity for critical operations
```

---

## Monitoring and Observability

### 1. The Three Pillars

**Metrics**:
```yaml
key_metrics:
  RED:  # Rate, Errors, Duration
    - Request rate
    - Error rate
    - Request duration (latency)
  
  USE:  # Utilization, Saturation, Errors
    - CPU/Memory utilization
    - Queue depth
    - Error count
  
  Custom:
    - Business metrics
    - User behavior
    - Feature usage
```

**Logs**:
```yaml
logging:
  structure: JSON
  fields:
    - timestamp
    - level
    - message
    - trace_id
    - span_id
    - user_id
    - request_id
    - service
    - environment
  
  retention:
    - Hot: 7 days (searchable)
    - Warm: 30 days (archived)
    - Cold: 90 days (compliance)
```

**Traces**:
```yaml
tracing:
  system: Jaeger
  sampling: 1% in production, 100% in dev
  
  instrumentation:
    - HTTP requests
    - Database queries
    - External API calls
    - Message queue operations
    - Cache operations
```

### 2. SLI/SLO/SLA

**Service Level Indicators (SLIs)**:
```yaml
slis:
  availability:
    measurement: successful_requests / total_requests
    
  latency:
    measurement: p95_request_duration
    
  error_rate:
    measurement: failed_requests / total_requests
```

**Service Level Objectives (SLOs)**:
```yaml
slos:
  availability: 99.95%
  latency_p95: <200ms
  error_rate: <0.1%
  
  measurement_window: 30 days
  error_budget: 0.05% (21.6 minutes/month)
```

**Service Level Agreements (SLAs)**:
```yaml
slas:
  availability:
    commitment: 99.9%
    measurement: Monthly uptime
    penalty: 10% credit for each 0.1% below
    
  support:
    response_time:
      critical: 15 minutes
      high: 1 hour
      medium: 4 hours
      low: 24 hours
```

---

## Database Best Practices

### 1. Schema Design

**Normalization vs Denormalization**:
```yaml
normalization:
  use_when:
    - Data integrity critical
    - Updates frequent
    - Storage cost high
  
  benefits:
    - No data redundancy
    - Easy to update
    - Enforced consistency

denormalization:
  use_when:
    - Read performance critical
    - Data mostly static
    - Joins expensive
  
  benefits:
    - Faster reads
    - Simpler queries
    - Reduced joins
```

### 2. Indexing Strategy

**Index Guidelines**:
```sql
-- ✅ GOOD: Compound index for common query
CREATE INDEX idx_orders_user_status_created 
ON orders(user_id, status, created_at);

-- Query benefits:
SELECT * FROM orders 
WHERE user_id = 123 
AND status = 'pending' 
ORDER BY created_at DESC;

-- ❌ BAD: Too many single-column indexes
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created ON orders(created_at);
```

**Index Maintenance**:
```sql
-- Check for unused indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0
AND indexname NOT LIKE 'pg_toast%';

-- Check for index bloat
SELECT
    schemaname,
    tablename,
    ROUND(CASE WHEN otta=0 THEN 0.0 
          ELSE sml.relpages/otta::numeric END,1) AS ratio
FROM (
    -- Complex query for bloat calculation
) AS sml
WHERE ratio > 2;

-- Rebuild bloated index
REINDEX INDEX CONCURRENTLY idx_orders_user_status_created;
```

### 3. Migration Best Practices

**Migration Checklist**:
```yaml
pre_migration:
  - [ ] Test migration on copy of production data
  - [ ] Estimate migration duration
  - [ ] Plan maintenance window
  - [ ] Create rollback script
  - [ ] Backup database
  - [ ] Notify stakeholders

during_migration:
  - [ ] Run migration in transaction (if possible)
  - [ ] Monitor for locks
  - [ ] Watch for performance impact
  - [ ] Keep stakeholders updated

post_migration:
  - [ ] Verify data integrity
  - [ ] Run application tests
  - [ ] Monitor performance
  - [ ] Document any issues
```

**Migration Template**:
```sql
-- Migration: 20250110_add_user_preferences
-- Description: Add preferences column to users table

BEGIN;

-- Add column with default value
ALTER TABLE users 
ADD COLUMN preferences JSONB DEFAULT '{}'::JSONB;

-- Create index for JSONB queries
CREATE INDEX CONCURRENTLY idx_users_preferences 
ON users USING GIN (preferences);

-- Verify migration
DO $$
DECLARE
    column_count INT;
BEGIN
    SELECT COUNT(*) INTO column_count
    FROM information_schema.columns
    WHERE table_name = 'users' AND column_name = 'preferences';
    
    IF column_count = 0 THEN
        RAISE EXCEPTION 'Migration failed: preferences column not created';
    END IF;
END $$;

COMMIT;
```

---

## Anti-Patterns to Avoid

### 1. Code Anti-Patterns

**God Object**:
```go
// ❌ BAD: One class does everything
type OrderManager struct {
    db *Database
}

func (om *OrderManager) CreateOrder() {}
func (om *OrderManager) SendEmail() {}
func (om *OrderManager) ProcessPayment() {}
func (om *OrderManager) UpdateInventory() {}
func (om *OrderManager) GenerateInvoice() {}
func (om *OrderManager) SendNotification() {}

// ✅ GOOD: Separation of concerns
type OrderService struct {
    emailService    *EmailService
    paymentService  *PaymentService
    inventoryService *InventoryService
}
```

**Premature Optimization**:
```go
// ❌ BAD: Optimizing before measuring
func GetUsers() []User {
    // Complex caching logic
    // Parallel processing
    // Memory pooling
    // ...when simple query works fine
}

// ✅ GOOD: Simple first, optimize if needed
func GetUsers() []User {
    return db.Query("SELECT * FROM users")
    // Profile first, then optimize if needed
}
```

**Magic Numbers**:
```go
// ❌ BAD: Magic numbers
if user.Age > 18 {
    // What does 18 mean?
}

if cache.TTL == 300 {
    // Why 300?
}

// ✅ GOOD: Named constants
const LegalAge = 18
const CacheTTL = 5 * time.Minute

if user.Age > LegalAge {
    // Clear intent
}
```

### 2. Architecture Anti-Patterns

**Distributed Monolith**:
```yaml
problem:
  - Microservices that can't be deployed independently
  - Tight coupling between services
  - Shared database across services
  
solution:
  - Define clear service boundaries
  - Use APIs for communication
  - Each service owns its data
  - Deploy independently
```

**Golden Hammer**:
```yaml
problem:
  - Using same solution for every problem
  - "We always use MongoDB"
  - Not evaluating alternatives
  
solution:
  - Choose tools based on requirements
  - Evaluate trade-offs
  - Use appropriate technology
```

### 3. Database Anti-Patterns

**N+1 Queries**:
```go
// ❌ BAD: N+1 query problem
orders := getOrders()  // 1 query
for _, order := range orders {
    order.User = getUser(order.UserID)  // N queries
}

// ✅ GOOD: Single query with JOIN
orders := db.Query(`
    SELECT o.*, u.* 
    FROM orders o 
    JOIN users u ON o.user_id = u.id
`)
```

**Using SELECT ***:
```sql
-- ❌ BAD: Select all columns
SELECT * FROM users WHERE id = 123;

-- ✅ GOOD: Select only needed columns
SELECT id, name, email FROM users WHERE id = 123;
```

---

## Code Review Checklist

### Functionality
- [ ] Code does what it's supposed to do
- [ ] Edge cases handled
- [ ] Error conditions handled
- [ ] No hardcoded values
- [ ] Feature flags used appropriately

### Code Quality
- [ ] Code is readable and well-organized
- [ ] Functions are small and focused
- [ ] Variable/function names are descriptive
- [ ] No code duplication
- [ ] Comments explain "why" not "what"

### Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added if needed
- [ ] Test coverage meets threshold (>80%)
- [ ] Tests are meaningful and test behavior
- [ ] Edge cases tested

### Performance
- [ ] No obvious performance issues
- [ ] Database queries optimized
- [ ] Appropriate caching used
- [ ] No n+1 query problems
- [ ] Resource usage reasonable

### Security
- [ ] No secrets in code
- [ ] Input validated
- [ ] SQL injection prevented
- [ ] XSS prevented
- [ ] CSRF protection in place
- [ ] Authentication/authorization correct

### Operations
- [ ] Logging added for important events
- [ ] Metrics/monitoring added
- [ ] Error tracking configured
- [ ] Documentation updated
- [ ] Runbook updated if needed

### Deployment
- [ ] Database migrations safe
- [ ] Feature flags configured
- [ ] Backward compatible
- [ ] Rollback plan documented
- [ ] Dependencies updated

---

**Document Control**:
- **Version**: 1.0.0
- **Last Updated**: 2025-11-10
- **Next Review**: 2025-12-10
- **Owner**: Engineering Team
- **Approver**: VP Engineering

---

*This document is classified as Internal Use.*
