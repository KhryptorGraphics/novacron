# Testing Strategies & Quality Assurance Framework

## Overview

Comprehensive testing framework for NovaCron enhancements, covering unit testing, integration testing, end-to-end testing, performance testing, security testing, and compliance validation. This framework ensures 95%+ code coverage and enterprise-grade quality standards.

## Table of Contents

1. [Testing Pyramid Strategy](#testing-pyramid-strategy)
2. [Unit Testing Framework](#unit-testing-framework)
3. [Integration Testing](#integration-testing)
4. [End-to-End Testing](#end-to-end-testing)
5. [Performance Testing](#performance-testing)
6. [Security Testing](#security-testing)
7. [Load & Stress Testing](#load--stress-testing)
8. [Chaos Engineering](#chaos-engineering)
9. [Test Data Management](#test-data-management)
10. [CI/CD Test Integration](#cicd-test-integration)

---

## Testing Pyramid Strategy

### Testing Distribution
- **Unit Tests**: 70% - Fast, isolated, deterministic
- **Integration Tests**: 20% - Component interactions
- **E2E Tests**: 10% - Critical user journeys

### Quality Gates
- **Unit Test Coverage**: ≥90%
- **Integration Test Coverage**: ≥80%
- **E2E Test Coverage**: ≥95% of critical paths
- **Performance Baseline**: All tests must pass within SLA thresholds

---

## Unit Testing Framework

### Go Unit Testing with Testify

```go
// internal/auth/jwt_service_test.go
package auth

import (
    "context"
    "testing"
    "time"

    "github.com/golang-jwt/jwt/v5"
    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/mock"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/suite"
    "go.uber.org/zap/zaptest"
)

type JWTServiceTestSuite struct {
    suite.Suite
    service     *JWTService
    mockBlacklist *MockTokenBlacklist
    mockRateLimiter *MockRateLimiter
    testUser    *User
    privateKey  string
    publicKey   string
}

func (suite *JWTServiceTestSuite) SetupTest() {
    // Generate test RSA keys
    suite.privateKey, suite.publicKey = generateTestKeys()
    
    // Setup mocks
    suite.mockBlacklist = new(MockTokenBlacklist)
    suite.mockRateLimiter = new(MockRateLimiter)
    
    // Create service
    config := JWTConfig{
        PrivateKey: suite.privateKey,
        PublicKey:  suite.publicKey,
        Issuer:     "test-issuer",
        AccessTTL:  15 * time.Minute,
        RefreshTTL: 24 * time.Hour,
    }
    
    service, err := NewJWTService(config, zaptest.NewLogger(suite.T()))
    require.NoError(suite.T(), err)
    
    // Inject mocks
    service.blacklist = suite.mockBlacklist
    service.rateLimiter = suite.mockRateLimiter
    
    suite.service = service
    
    // Test user
    suite.testUser = &User{
        ID:    123,
        Email: "test@example.com",
        Roles: []string{"user", "admin"},
    }
}

func (suite *JWTServiceTestSuite) TestGenerateTokenPair_Success() {
    // Arrange
    ctx := context.Background()
    suite.mockRateLimiter.On("Allow", mock.Anything, mock.Anything, 10, time.Hour).Return(true)
    
    // Act
    tokenPair, err := suite.service.GenerateTokenPair(ctx, suite.testUser)
    
    // Assert
    require.NoError(suite.T(), err)
    assert.NotEmpty(suite.T(), tokenPair.AccessToken)
    assert.NotEmpty(suite.T(), tokenPair.RefreshToken)
    assert.Equal(suite.T(), "Bearer", tokenPair.TokenType)
    assert.True(suite.T(), tokenPair.ExpiresAt.After(time.Now()))
    
    // Verify token structure
    token, err := jwt.Parse(tokenPair.AccessToken, func(token *jwt.Token) (interface{}, error) {
        return suite.service.publicKey, nil
    })
    require.NoError(suite.T(), err)
    
    claims, ok := token.Claims.(jwt.MapClaims)
    require.True(suite.T(), ok)
    assert.Equal(suite.T(), float64(suite.testUser.ID), claims["user_id"])
    assert.Equal(suite.T(), suite.testUser.Email, claims["email"])
    
    suite.mockRateLimiter.AssertExpectations(suite.T())
}

func (suite *JWTServiceTestSuite) TestGenerateTokenPair_RateLimited() {
    // Arrange
    ctx := context.Background()
    suite.mockRateLimiter.On("Allow", mock.Anything, mock.Anything, 10, time.Hour).Return(false)
    
    // Act
    tokenPair, err := suite.service.GenerateTokenPair(ctx, suite.testUser)
    
    // Assert
    assert.Nil(suite.T(), tokenPair)
    assert.ErrorIs(suite.T(), err, ErrRateLimited)
    
    suite.mockRateLimiter.AssertExpectations(suite.T())
}

func (suite *JWTServiceTestSuite) TestValidateToken_Success() {
    // Arrange
    ctx := context.Background()
    suite.mockRateLimiter.On("Allow", mock.Anything, mock.Anything, 10, time.Hour).Return(true)
    
    tokenPair, err := suite.service.GenerateTokenPair(ctx, suite.testUser)
    require.NoError(suite.T(), err)
    
    suite.mockBlacklist.On("IsBlacklisted", mock.Anything, tokenPair.AccessToken).Return(false)
    
    // Act
    claims, err := suite.service.ValidateToken(ctx, tokenPair.AccessToken)
    
    // Assert
    require.NoError(suite.T(), err)
    assert.Equal(suite.T(), suite.testUser.ID, claims.UserID)
    assert.Equal(suite.T(), suite.testUser.Email, claims.Email)
    assert.Equal(suite.T(), suite.testUser.Roles, claims.Roles)
    
    suite.mockBlacklist.AssertExpectations(suite.T())
}

func (suite *JWTServiceTestSuite) TestValidateToken_Blacklisted() {
    // Arrange
    ctx := context.Background()
    tokenString := "blacklisted.token.here"
    suite.mockBlacklist.On("IsBlacklisted", mock.Anything, tokenString).Return(true)
    
    // Act
    claims, err := suite.service.ValidateToken(ctx, tokenString)
    
    // Assert
    assert.Nil(suite.T(), claims)
    assert.ErrorIs(suite.T(), err, ErrTokenBlacklisted)
    
    suite.mockBlacklist.AssertExpectations(suite.T())
}

func (suite *JWTServiceTestSuite) TestValidateToken_InvalidSignature() {
    // Arrange
    ctx := context.Background()
    
    // Create token with wrong key
    wrongKey, _ := generateTestKeys()
    wrongService, _ := NewJWTService(JWTConfig{
        PrivateKey: wrongKey,
        PublicKey:  suite.publicKey,
        Issuer:     "test",
        AccessTTL:  time.Hour,
        RefreshTTL: time.Hour,
    }, zaptest.NewLogger(suite.T()))
    
    tokenPair, _ := wrongService.GenerateTokenPair(ctx, suite.testUser)
    
    suite.mockBlacklist.On("IsBlacklisted", mock.Anything, mock.Anything).Return(false)
    
    // Act
    claims, err := suite.service.ValidateToken(ctx, tokenPair.AccessToken)
    
    // Assert
    assert.Nil(suite.T(), claims)
    assert.Error(suite.T(), err)
}

// Benchmark tests
func BenchmarkJWTService_GenerateTokenPair(b *testing.B) {
    service := setupBenchmarkService(b)
    user := &User{ID: 123, Email: "test@example.com", Roles: []string{"user"}}
    ctx := context.Background()
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := service.GenerateTokenPair(ctx, user)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}

func BenchmarkJWTService_ValidateToken(b *testing.B) {
    service := setupBenchmarkService(b)
    user := &User{ID: 123, Email: "test@example.com", Roles: []string{"user"}}
    ctx := context.Background()
    
    // Pre-generate token
    tokenPair, err := service.GenerateTokenPair(ctx, user)
    if err != nil {
        b.Fatal(err)
    }
    
    b.ResetTimer()
    b.RunParallel(func(pb *testing.PB) {
        for pb.Next() {
            _, err := service.ValidateToken(ctx, tokenPair.AccessToken)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}

func TestJWTServiceSuite(t *testing.T) {
    suite.Run(t, new(JWTServiceTestSuite))
}

// Mock implementations
type MockTokenBlacklist struct {
    mock.Mock
}

func (m *MockTokenBlacklist) IsBlacklisted(ctx context.Context, token string) bool {
    args := m.Called(ctx, token)
    return args.Bool(0)
}

func (m *MockTokenBlacklist) BlacklistToken(ctx context.Context, token string, ttl time.Duration) error {
    args := m.Called(ctx, token, ttl)
    return args.Error(0)
}

type MockRateLimiter struct {
    mock.Mock
}

func (m *MockRateLimiter) Allow(ctx context.Context, key string, limit int, window time.Duration) bool {
    args := m.Called(ctx, key, limit, window)
    return args.Bool(0)
}

// Test utilities
func generateTestKeys() (privateKey, publicKey string) {
    // Implementation to generate test RSA keys
    // This is a simplified version - use crypto/rand in production
    return "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----",
           "-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
}

func setupBenchmarkService(b *testing.B) *JWTService {
    privateKey, publicKey := generateTestKeys()
    config := JWTConfig{
        PrivateKey: privateKey,
        PublicKey:  publicKey,
        Issuer:     "benchmark",
        AccessTTL:  time.Hour,
        RefreshTTL: 24 * time.Hour,
    }
    
    service, err := NewJWTService(config, zaptest.NewLogger(b))
    if err != nil {
        b.Fatal(err)
    }
    
    // Use no-op mocks for benchmarks
    service.blacklist = &NoOpTokenBlacklist{}
    service.rateLimiter = &NoOpRateLimiter{}
    
    return service
}
```

### Database Testing with Testcontainers

```go
// internal/database/repository_test.go
package database

import (
    "context"
    "database/sql"
    "fmt"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/suite"
    "github.com/testcontainers/testcontainers-go"
    "github.com/testcontainers/testcontainers-go/modules/postgres"
    "github.com/testcontainers/testcontainers-go/wait"
)

type RepositoryTestSuite struct {
    suite.Suite
    pgContainer  *postgres.PostgresContainer
    db          *sql.DB
    repository  *JobRepository
    ctx         context.Context
}

func (suite *RepositoryTestSuite) SetupSuite() {
    suite.ctx = context.Background()
    
    // Start PostgreSQL container
    pgContainer, err := postgres.RunContainer(
        suite.ctx,
        testcontainers.WithImage("postgres:15-alpine"),
        postgres.WithDatabase("testdb"),
        postgres.WithUsername("testuser"),
        postgres.WithPassword("testpass"),
        testcontainers.WithWaitStrategy(
            wait.ForLog("database system is ready to accept connections").
                WithOccurrence(2).
                WithStartupTimeout(5*time.Minute),
        ),
    )
    require.NoError(suite.T(), err)
    
    suite.pgContainer = pgContainer
    
    // Get connection string
    connStr, err := pgContainer.ConnectionString(suite.ctx, "sslmode=disable")
    require.NoError(suite.T(), err)
    
    // Connect to database
    db, err := sql.Open("postgres", connStr)
    require.NoError(suite.T(), err)
    
    suite.db = db
    
    // Run migrations
    err = suite.runMigrations()
    require.NoError(suite.T(), err)
    
    // Create repository
    suite.repository = NewJobRepository(db)
}

func (suite *RepositoryTestSuite) TearDownSuite() {
    if suite.db != nil {
        suite.db.Close()
    }
    if suite.pgContainer != nil {
        err := suite.pgContainer.Terminate(suite.ctx)
        require.NoError(suite.T(), err)
    }
}

func (suite *RepositoryTestSuite) SetupTest() {
    // Clean database before each test
    suite.cleanDatabase()
}

func (suite *RepositoryTestSuite) TestCreateJob_Success() {
    // Arrange
    job := &Job{
        UserID:      1,
        Name:        "Test Job",
        Description: "Test Description",
        Schedule:    "0 * * * *",
        Command:     "echo 'hello'",
        Status:      "active",
    }
    
    // Act
    err := suite.repository.Create(suite.ctx, job)
    
    // Assert
    require.NoError(suite.T(), err)
    assert.NotZero(suite.T(), job.ID)
    assert.NotZero(suite.T(), job.CreatedAt)
    
    // Verify in database
    var count int
    err = suite.db.QueryRow("SELECT COUNT(*) FROM jobs WHERE name = $1", job.Name).Scan(&count)
    require.NoError(suite.T(), err)
    assert.Equal(suite.T(), 1, count)
}

func (suite *RepositoryTestSuite) TestGetJobsByUser_WithPagination() {
    // Arrange
    userID := int64(1)
    
    // Create test jobs
    for i := 0; i < 15; i++ {
        job := &Job{
            UserID:   userID,
            Name:     fmt.Sprintf("Job %d", i),
            Schedule: "0 * * * *",
            Command:  "echo test",
            Status:   "active",
        }
        err := suite.repository.Create(suite.ctx, job)
        require.NoError(suite.T(), err)
    }
    
    // Act - First page
    jobs, total, err := suite.repository.GetByUser(suite.ctx, userID, 0, 10)
    
    // Assert
    require.NoError(suite.T(), err)
    assert.Len(suite.T(), jobs, 10)
    assert.Equal(suite.T(), int64(15), total)
    
    // Act - Second page
    jobs, total, err = suite.repository.GetByUser(suite.ctx, userID, 10, 10)
    
    // Assert
    require.NoError(suite.T(), err)
    assert.Len(suite.T(), jobs, 5)
    assert.Equal(suite.T(), int64(15), total)
}

func (suite *RepositoryTestSuite) TestGetJobsWithExecutions_NoNPlusOne() {
    // Arrange
    userID := int64(1)
    
    // Create jobs with executions
    for i := 0; i < 5; i++ {
        job := &Job{
            UserID:   userID,
            Name:     fmt.Sprintf("Job %d", i),
            Schedule: "0 * * * *",
            Command:  "echo test",
            Status:   "active",
        }
        err := suite.repository.Create(suite.ctx, job)
        require.NoError(suite.T(), err)
        
        // Create executions for each job
        for j := 0; j < 3; j++ {
            execution := &JobExecution{
                JobID:     job.ID,
                Status:    "success",
                StartedAt: time.Now().Add(-time.Duration(j) * time.Hour),
                EndedAt:   time.Now().Add(-time.Duration(j)*time.Hour + 5*time.Minute),
            }
            err := suite.repository.CreateExecution(suite.ctx, execution)
            require.NoError(suite.T(), err)
        }
    }
    
    // Act - Enable query logging to verify no N+1
    startQueries := suite.getQueryCount()
    
    jobs, err := suite.repository.GetJobsWithExecutions(suite.ctx, userID, 10)
    
    endQueries := suite.getQueryCount()
    
    // Assert
    require.NoError(suite.T(), err)
    assert.Len(suite.T(), jobs, 5)
    
    // Verify each job has executions
    for _, job := range jobs {
        assert.NotEmpty(suite.T(), job.Executions)
        assert.Len(suite.T(), job.Executions, 3)
    }
    
    // Verify query efficiency (should be 1 query, not 6 queries)
    queryDiff := endQueries - startQueries
    assert.LessOrEqual(suite.T(), queryDiff, int64(2)) // Allow for some flexibility
}

func (suite *RepositoryTestSuite) TestConcurrentJobCreation() {
    // Arrange
    userID := int64(1)
    numGoroutines := 10
    jobsPerGoroutine := 5
    
    errChan := make(chan error, numGoroutines)
    doneChan := make(chan bool, numGoroutines)
    
    // Act
    for i := 0; i < numGoroutines; i++ {
        go func(routineID int) {
            defer func() { doneChan <- true }()
            
            for j := 0; j < jobsPerGoroutine; j++ {
                job := &Job{
                    UserID:   userID,
                    Name:     fmt.Sprintf("Concurrent Job %d-%d", routineID, j),
                    Schedule: "0 * * * *",
                    Command:  "echo test",
                    Status:   "active",
                }
                
                if err := suite.repository.Create(suite.ctx, job); err != nil {
                    errChan <- err
                    return
                }
            }
        }(i)
    }
    
    // Wait for completion
    for i := 0; i < numGoroutines; i++ {
        select {
        case err := <-errChan:
            suite.T().Fatal("Concurrent creation failed:", err)
        case <-doneChan:
            // Success
        case <-time.After(10 * time.Second):
            suite.T().Fatal("Timeout waiting for concurrent operations")
        }
    }
    
    // Assert
    var count int
    err := suite.db.QueryRow("SELECT COUNT(*) FROM jobs WHERE user_id = $1", userID).Scan(&count)
    require.NoError(suite.T(), err)
    assert.Equal(suite.T(), numGoroutines*jobsPerGoroutine, count)
}

// Benchmark tests
func (suite *RepositoryTestSuite) TestJobRepository_Benchmarks() {
    // Setup benchmark data
    userID := int64(1)
    
    // Create jobs for benchmarking
    for i := 0; i < 100; i++ {
        job := &Job{
            UserID:   userID,
            Name:     fmt.Sprintf("Benchmark Job %d", i),
            Schedule: "0 * * * *",
            Command:  "echo test",
            Status:   "active",
        }
        err := suite.repository.Create(suite.ctx, job)
        require.NoError(suite.T(), err)
    }
    
    suite.T().Run("BenchmarkGetByUser", func(t *testing.T) {
        b := testing.B(*t)
        b.ResetTimer()
        
        for i := 0; i < b.N; i++ {
            _, _, err := suite.repository.GetByUser(suite.ctx, userID, 0, 20)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
    
    suite.T().Run("BenchmarkCreate", func(t *testing.T) {
        b := testing.B(*t)
        b.ResetTimer()
        
        for i := 0; i < b.N; i++ {
            job := &Job{
                UserID:   userID,
                Name:     fmt.Sprintf("Benchmark Create %d", i),
                Schedule: "0 * * * *",
                Command:  "echo test",
                Status:   "active",
            }
            
            err := suite.repository.Create(suite.ctx, job)
            if err != nil {
                b.Fatal(err)
            }
        }
    })
}

func TestRepositoryTestSuite(t *testing.T) {
    suite.Run(t, new(RepositoryTestSuite))
}

// Helper methods
func (suite *RepositoryTestSuite) runMigrations() error {
    migrations := []string{
        `CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        )`,
        `CREATE TABLE IF NOT EXISTS jobs (
            id BIGSERIAL PRIMARY KEY,
            user_id BIGINT NOT NULL REFERENCES users(id),
            name VARCHAR(255) NOT NULL,
            description TEXT,
            schedule VARCHAR(100) NOT NULL,
            command TEXT NOT NULL,
            status VARCHAR(50) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            deleted_at TIMESTAMP
        )`,
        `CREATE TABLE IF NOT EXISTS job_executions (
            id BIGSERIAL PRIMARY KEY,
            job_id BIGINT NOT NULL REFERENCES jobs(id),
            status VARCHAR(50) NOT NULL,
            started_at TIMESTAMP NOT NULL,
            ended_at TIMESTAMP,
            error_message TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        )`,
        `CREATE INDEX IF NOT EXISTS idx_jobs_user_id ON jobs(user_id)`,
        `CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)`,
        `CREATE INDEX IF NOT EXISTS idx_job_executions_job_id ON job_executions(job_id)`,
        `CREATE INDEX IF NOT EXISTS idx_job_executions_status ON job_executions(status)`,
    }
    
    for _, migration := range migrations {
        if _, err := suite.db.Exec(migration); err != nil {
            return fmt.Errorf("failed to run migration: %w", err)
        }
    }
    
    return nil
}

func (suite *RepositoryTestSuite) cleanDatabase() {
    tables := []string{"job_executions", "jobs", "users"}
    for _, table := range tables {
        _, err := suite.db.Exec(fmt.Sprintf("TRUNCATE TABLE %s RESTART IDENTITY CASCADE", table))
        require.NoError(suite.T(), err)
    }
}

func (suite *RepositoryTestSuite) getQueryCount() int64 {
    var count int64
    row := suite.db.QueryRow(`
        SELECT query_count 
        FROM pg_stat_database 
        WHERE datname = current_database()
    `)
    row.Scan(&count)
    return count
}
```

---

## Integration Testing

### API Integration Tests

```go
// tests/integration/api_test.go
package integration

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "net/http/httptest"
    "testing"
    "time"

    "github.com/stretchr/testify/assert"
    "github.com/stretchr/testify/require"
    "github.com/stretchr/testify/suite"
)

type APIIntegrationTestSuite struct {
    suite.Suite
    server     *httptest.Server
    client     *http.Client
    baseURL    string
    authToken  string
    testUser   *User
}

func (suite *APIIntegrationTestSuite) SetupSuite() {
    // Setup test server
    app := setupTestApplication()
    suite.server = httptest.NewServer(app)
    suite.baseURL = suite.server.URL
    suite.client = &http.Client{Timeout: 30 * time.Second}
    
    // Create test user and get auth token
    suite.createTestUser()
    suite.authenticateTestUser()
}

func (suite *APIIntegrationTestSuite) TearDownSuite() {
    if suite.server != nil {
        suite.server.Close()
    }
}

func (suite *APIIntegrationTestSuite) SetupTest() {
    // Clean test data before each test
    suite.cleanTestData()
}

func (suite *APIIntegrationTestSuite) TestJobLifecycle() {
    // Test complete job lifecycle: create -> update -> execute -> delete
    
    // 1. Create Job
    createJobReq := CreateJobRequest{
        Name:        "Integration Test Job",
        Description: "Test job for integration testing",
        Schedule:    "0 * * * *",
        Command:     "echo 'Hello Integration Test'",
    }
    
    job := suite.createJob(createJobReq)
    assert.NotZero(suite.T(), job.ID)
    assert.Equal(suite.T(), createJobReq.Name, job.Name)
    assert.Equal(suite.T(), "active", job.Status)
    
    // 2. Get Job
    retrievedJob := suite.getJob(job.ID)
    assert.Equal(suite.T(), job.ID, retrievedJob.ID)
    assert.Equal(suite.T(), job.Name, retrievedJob.Name)
    
    // 3. Update Job
    updateJobReq := UpdateJobRequest{
        Name:        "Updated Integration Test Job",
        Description: "Updated description",
        Schedule:    "0 */2 * * *",
    }
    
    updatedJob := suite.updateJob(job.ID, updateJobReq)
    assert.Equal(suite.T(), updateJobReq.Name, updatedJob.Name)
    assert.Equal(suite.T(), updateJobReq.Description, updatedJob.Description)
    assert.Equal(suite.T(), updateJobReq.Schedule, updatedJob.Schedule)
    
    // 4. Execute Job
    execution := suite.executeJob(job.ID)
    assert.NotZero(suite.T(), execution.ID)
    assert.Equal(suite.T(), job.ID, execution.JobID)
    
    // Wait for execution to complete
    suite.waitForExecutionCompletion(execution.ID, 30*time.Second)
    
    // 5. Get Job Executions
    executions := suite.getJobExecutions(job.ID)
    assert.Len(suite.T(), executions, 1)
    assert.Equal(suite.T(), "success", executions[0].Status)
    
    // 6. Delete Job
    suite.deleteJob(job.ID)
    
    // Verify job is deleted
    suite.assertJobNotFound(job.ID)
}

func (suite *APIIntegrationTestSuite) TestJobValidation() {
    // Test various validation scenarios
    
    testCases := []struct {
        name        string
        request     CreateJobRequest
        expectedErr string
    }{
        {
            name: "Empty name",
            request: CreateJobRequest{
                Name:     "",
                Schedule: "0 * * * *",
                Command:  "echo test",
            },
            expectedErr: "name is required",
        },
        {
            name: "Invalid cron schedule",
            request: CreateJobRequest{
                Name:     "Test Job",
                Schedule: "invalid cron",
                Command:  "echo test",
            },
            expectedErr: "invalid cron expression",
        },
        {
            name: "Dangerous command",
            request: CreateJobRequest{
                Name:     "Test Job",
                Schedule: "0 * * * *",
                Command:  "rm -rf /",
            },
            expectedErr: "command contains dangerous operations",
        },
    }
    
    for _, tc := range testCases {
        suite.T().Run(tc.name, func(t *testing.T) {
            resp := suite.makeRequest("POST", "/api/v1/jobs", tc.request)
            assert.Equal(t, http.StatusBadRequest, resp.StatusCode)
            
            var errorResp ErrorResponse
            suite.decodeResponse(resp, &errorResp)
            assert.Contains(t, errorResp.Message, tc.expectedErr)
        })
    }
}

func (suite *APIIntegrationTestSuite) TestConcurrentJobOperations() {
    // Test concurrent operations on jobs
    
    numJobs := 10
    jobIDs := make([]int64, numJobs)
    
    // Create jobs concurrently
    jobChan := make(chan *Job, numJobs)
    errChan := make(chan error, numJobs)
    
    for i := 0; i < numJobs; i++ {
        go func(index int) {
            req := CreateJobRequest{
                Name:     fmt.Sprintf("Concurrent Job %d", index),
                Schedule: "0 * * * *",
                Command:  "echo test",
            }
            
            job, err := suite.createJobAsync(req)
            if err != nil {
                errChan <- err
                return
            }
            jobChan <- job
        }(i)
    }
    
    // Collect results
    for i := 0; i < numJobs; i++ {
        select {
        case job := <-jobChan:
            jobIDs[i] = job.ID
        case err := <-errChan:
            suite.T().Fatal("Concurrent job creation failed:", err)
        case <-time.After(10 * time.Second):
            suite.T().Fatal("Timeout waiting for concurrent job creation")
        }
    }
    
    // Verify all jobs were created
    jobs := suite.getUserJobs()
    assert.Len(suite.T(), jobs, numJobs)
    
    // Execute all jobs concurrently
    executionChan := make(chan *JobExecution, numJobs)
    
    for _, jobID := range jobIDs {
        go func(id int64) {
            execution, err := suite.executeJobAsync(id)
            if err != nil {
                errChan <- err
                return
            }
            executionChan <- execution
        }(jobID)
    }
    
    // Wait for all executions
    executions := make([]*JobExecution, 0, numJobs)
    for i := 0; i < numJobs; i++ {
        select {
        case execution := <-executionChan:
            executions = append(executions, execution)
        case err := <-errChan:
            suite.T().Fatal("Concurrent job execution failed:", err)
        case <-time.After(30 * time.Second):
            suite.T().Fatal("Timeout waiting for concurrent job execution")
        }
    }
    
    assert.Len(suite.T(), executions, numJobs)
    
    // Wait for all executions to complete
    for _, execution := range executions {
        suite.waitForExecutionCompletion(execution.ID, 30*time.Second)
    }
}

func (suite *APIIntegrationTestSuite) TestPaginationAndFiltering() {
    // Create test data
    for i := 0; i < 25; i++ {
        status := "active"
        if i%5 == 0 {
            status = "paused"
        }
        
        req := CreateJobRequest{
            Name:     fmt.Sprintf("Pagination Test Job %02d", i),
            Schedule: "0 * * * *",
            Command:  "echo test",
        }
        
        job := suite.createJob(req)
        if status == "paused" {
            suite.pauseJob(job.ID)
        }
    }
    
    // Test pagination
    firstPage := suite.getUserJobsPaginated(0, 10)
    assert.Len(suite.T(), firstPage.Jobs, 10)
    assert.Equal(suite.T(), int64(25), firstPage.Total)
    
    secondPage := suite.getUserJobsPaginated(10, 10)
    assert.Len(suite.T(), secondPage.Jobs, 10)
    assert.Equal(suite.T(), int64(25), secondPage.Total)
    
    thirdPage := suite.getUserJobsPaginated(20, 10)
    assert.Len(suite.T(), thirdPage.Jobs, 5)
    assert.Equal(suite.T(), int64(25), thirdPage.Total)
    
    // Test filtering
    activeJobs := suite.getUserJobsFiltered("status=active")
    assert.Len(suite.T(), activeJobs.Jobs, 20)
    
    pausedJobs := suite.getUserJobsFiltered("status=paused")
    assert.Len(suite.T(), pausedJobs.Jobs, 5)
    
    // Test search
    searchResults := suite.searchJobs("Pagination")
    assert.Equal(suite.T(), int64(25), searchResults.Total)
    
    specificSearch := suite.searchJobs("Job 01")
    assert.Equal(suite.T(), int64(1), specificSearch.Total)
    assert.Contains(suite.T(), specificSearch.Jobs[0].Name, "01")
}

func TestAPIIntegrationTestSuite(t *testing.T) {
    suite.Run(t, new(APIIntegrationTestSuite))
}

// Helper methods
func (suite *APIIntegrationTestSuite) createJob(req CreateJobRequest) *Job {
    resp := suite.makeRequest("POST", "/api/v1/jobs", req)
    require.Equal(suite.T(), http.StatusCreated, resp.StatusCode)
    
    var job Job
    suite.decodeResponse(resp, &job)
    return &job
}

func (suite *APIIntegrationTestSuite) getJob(jobID int64) *Job {
    resp := suite.makeRequest("GET", fmt.Sprintf("/api/v1/jobs/%d", jobID), nil)
    require.Equal(suite.T(), http.StatusOK, resp.StatusCode)
    
    var job Job
    suite.decodeResponse(resp, &job)
    return &job
}

func (suite *APIIntegrationTestSuite) updateJob(jobID int64, req UpdateJobRequest) *Job {
    resp := suite.makeRequest("PUT", fmt.Sprintf("/api/v1/jobs/%d", jobID), req)
    require.Equal(suite.T(), http.StatusOK, resp.StatusCode)
    
    var job Job
    suite.decodeResponse(resp, &job)
    return &job
}

func (suite *APIIntegrationTestSuite) executeJob(jobID int64) *JobExecution {
    resp := suite.makeRequest("POST", fmt.Sprintf("/api/v1/jobs/%d/execute", jobID), nil)
    require.Equal(suite.T(), http.StatusCreated, resp.StatusCode)
    
    var execution JobExecution
    suite.decodeResponse(resp, &execution)
    return &execution
}

func (suite *APIIntegrationTestSuite) makeRequest(method, path string, body interface{}) *http.Response {
    var reqBody io.Reader
    if body != nil {
        jsonBody, _ := json.Marshal(body)
        reqBody = bytes.NewReader(jsonBody)
    }
    
    req, _ := http.NewRequest(method, suite.baseURL+path, reqBody)
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Authorization", "Bearer "+suite.authToken)
    
    resp, err := suite.client.Do(req)
    require.NoError(suite.T(), err)
    
    return resp
}

func (suite *APIIntegrationTestSuite) decodeResponse(resp *http.Response, v interface{}) {
    defer resp.Body.Close()
    
    body, err := io.ReadAll(resp.Body)
    require.NoError(suite.T(), err)
    
    err = json.Unmarshal(body, v)
    require.NoError(suite.T(), err)
}
```

---

## End-to-End Testing

### Playwright E2E Tests

```typescript
// tests/e2e/job-management.spec.ts
import { test, expect, Page } from '@playwright/test';
import { APIRequestContext } from '@playwright/test';

interface TestUser {
    email: string;
    password: string;
    accessToken?: string;
}

interface TestJob {
    id?: number;
    name: string;
    schedule: string;
    command: string;
}

class JobManagementPage {
    constructor(private page: Page) {}

    async navigateToJobs() {
        await this.page.goto('/jobs');
        await this.page.waitForSelector('[data-testid="jobs-list"]');
    }

    async createJob(job: TestJob) {
        await this.page.click('[data-testid="create-job-button"]');
        
        // Fill job form
        await this.page.fill('[data-testid="job-name-input"]', job.name);
        await this.page.fill('[data-testid="job-schedule-input"]', job.schedule);
        await this.page.fill('[data-testid="job-command-input"]', job.command);
        
        // Submit form
        await this.page.click('[data-testid="create-job-submit"]');
        
        // Wait for success notification
        await this.page.waitForSelector('[data-testid="success-notification"]');
    }

    async editJob(jobName: string, updates: Partial<TestJob>) {
        // Find and click edit button for the job
        await this.page.click(`[data-testid="edit-job-${jobName}"]`);
        
        if (updates.name) {
            await this.page.fill('[data-testid="job-name-input"]', updates.name);
        }
        if (updates.schedule) {
            await this.page.fill('[data-testid="job-schedule-input"]', updates.schedule);
        }
        if (updates.command) {
            await this.page.fill('[data-testid="job-command-input"]', updates.command);
        }
        
        await this.page.click('[data-testid="update-job-submit"]');
        await this.page.waitForSelector('[data-testid="success-notification"]');
    }

    async executeJob(jobName: string) {
        await this.page.click(`[data-testid="execute-job-${jobName}"]`);
        await this.page.click('[data-testid="confirm-execute"]');
        await this.page.waitForSelector('[data-testid="execution-started-notification"]');
    }

    async waitForJobExecution(jobName: string, timeout: number = 30000) {
        // Wait for execution status to update
        await this.page.waitForFunction(
            (name) => {
                const statusElement = document.querySelector(`[data-testid="job-status-${name}"]`);
                return statusElement?.textContent !== 'running';
            },
            jobName,
            { timeout }
        );
    }

    async getJobStatus(jobName: string): Promise<string> {
        const statusElement = await this.page.$(`[data-testid="job-status-${jobName}"]`);
        return await statusElement?.textContent() || '';
    }

    async getJobsList(): Promise<string[]> {
        const jobElements = await this.page.$$('[data-testid^="job-item-"]');
        const jobNames = await Promise.all(
            jobElements.map(async (element) => {
                const nameElement = await element.$('[data-testid="job-name"]');
                return await nameElement?.textContent() || '';
            })
        );
        return jobNames.filter(name => name !== '');
    }

    async deleteJob(jobName: string) {
        await this.page.click(`[data-testid="delete-job-${jobName}"]`);
        await this.page.click('[data-testid="confirm-delete"]');
        await this.page.waitForSelector('[data-testid="success-notification"]');
    }

    async searchJobs(searchTerm: string) {
        await this.page.fill('[data-testid="job-search-input"]', searchTerm);
        await this.page.press('[data-testid="job-search-input"]', 'Enter');
        await this.page.waitForTimeout(500); // Wait for search to complete
    }

    async filterJobsByStatus(status: string) {
        await this.page.selectOption('[data-testid="status-filter"]', status);
        await this.page.waitForTimeout(500);
    }
}

class AuthenticationHelper {
    constructor(private page: Page, private apiContext: APIRequestContext) {}

    async login(user: TestUser): Promise<void> {
        await this.page.goto('/login');
        
        await this.page.fill('[data-testid="email-input"]', user.email);
        await this.page.fill('[data-testid="password-input"]', user.password);
        await this.page.click('[data-testid="login-button"]');
        
        // Wait for redirect to dashboard
        await this.page.waitForURL('/dashboard');
        
        // Get auth token from local storage
        const token = await this.page.evaluate(() => 
            localStorage.getItem('authToken')
        );
        user.accessToken = token;
    }

    async createTestUser(user: TestUser): Promise<void> {
        await this.apiContext.post('/api/v1/auth/register', {
            data: {
                email: user.email,
                password: user.password,
                firstName: 'Test',
                lastName: 'User',
            }
        });
    }

    async logout(): Promise<void> {
        await this.page.click('[data-testid="user-menu"]');
        await this.page.click('[data-testid="logout-button"]');
        await this.page.waitForURL('/login');
    }
}

test.describe('Job Management E2E Tests', () => {
    let testUser: TestUser;
    let jobManagementPage: JobManagementPage;
    let authHelper: AuthenticationHelper;

    test.beforeEach(async ({ page, request }) => {
        testUser = {
            email: `test.user.${Date.now()}@example.com`,
            password: 'TestPassword123!',
        };

        jobManagementPage = new JobManagementPage(page);
        authHelper = new AuthenticationHelper(page, request);

        // Create and login test user
        await authHelper.createTestUser(testUser);
        await authHelper.login(testUser);
    });

    test('Complete job lifecycle - create, execute, monitor, delete', async ({ page }) => {
        const testJob: TestJob = {
            name: 'E2E Test Job',
            schedule: '0 * * * *',
            command: 'echo "Hello E2E Test"',
        };

        await jobManagementPage.navigateToJobs();

        // Create job
        await jobManagementPage.createJob(testJob);

        // Verify job appears in list
        const jobsList = await jobManagementPage.getJobsList();
        expect(jobsList).toContain(testJob.name);

        // Execute job
        await jobManagementPage.executeJob(testJob.name);

        // Wait for execution to complete
        await jobManagementPage.waitForJobExecution(testJob.name);

        // Verify execution success
        const status = await jobManagementPage.getJobStatus(testJob.name);
        expect(status).toBe('success');

        // Delete job
        await jobManagementPage.deleteJob(testJob.name);

        // Verify job is removed
        const updatedJobsList = await jobManagementPage.getJobsList();
        expect(updatedJobsList).not.toContain(testJob.name);
    });

    test('Job validation and error handling', async ({ page }) => {
        await jobManagementPage.navigateToJobs();

        // Test empty form validation
        await page.click('[data-testid="create-job-button"]');
        await page.click('[data-testid="create-job-submit"]');

        // Verify validation errors
        await expect(page.locator('[data-testid="name-error"]')).toBeVisible();
        await expect(page.locator('[data-testid="schedule-error"]')).toBeVisible();
        await expect(page.locator('[data-testid="command-error"]')).toBeVisible();

        // Test invalid cron expression
        await page.fill('[data-testid="job-name-input"]', 'Invalid Cron Job');
        await page.fill('[data-testid="job-schedule-input"]', 'invalid cron');
        await page.fill('[data-testid="job-command-input"]', 'echo test');
        await page.click('[data-testid="create-job-submit"]');

        await expect(page.locator('[data-testid="schedule-error"]')).toContainText('Invalid cron expression');

        // Test dangerous command
        await page.fill('[data-testid="job-schedule-input"]', '0 * * * *');
        await page.fill('[data-testid="job-command-input"]', 'rm -rf /');
        await page.click('[data-testid="create-job-submit"]');

        await expect(page.locator('[data-testid="command-error"]')).toContainText('dangerous operations');
    });

    test('Job search and filtering', async ({ page }) => {
        // Create multiple test jobs
        const testJobs = [
            { name: 'Search Test Job 1', schedule: '0 * * * *', command: 'echo test1' },
            { name: 'Search Test Job 2', schedule: '0 */2 * * *', command: 'echo test2' },
            { name: 'Different Name', schedule: '0 */3 * * *', command: 'echo test3' },
        ];

        await jobManagementPage.navigateToJobs();

        for (const job of testJobs) {
            await jobManagementPage.createJob(job);
        }

        // Test search functionality
        await jobManagementPage.searchJobs('Search Test');
        let jobsList = await jobManagementPage.getJobsList();
        expect(jobsList).toHaveLength(2);
        expect(jobsList.every(name => name.includes('Search Test'))).toBe(true);

        // Clear search
        await page.fill('[data-testid="job-search-input"]', '');
        await page.press('[data-testid="job-search-input"]', 'Enter');

        // Test status filtering
        // First execute one job and pause another
        await jobManagementPage.executeJob(testJobs[0].name);
        await page.click(`[data-testid="pause-job-${testJobs[1].name}"]`);

        // Filter by status
        await jobManagementPage.filterJobsByStatus('paused');
        jobsList = await jobManagementPage.getJobsList();
        expect(jobsList).toContain(testJobs[1].name);
        expect(jobsList).not.toContain(testJobs[0].name);
    });

    test('Concurrent job operations', async ({ page, context }) => {
        await jobManagementPage.navigateToJobs();

        // Create multiple jobs concurrently by opening multiple tabs
        const numJobs = 5;
        const pages = [];
        
        for (let i = 0; i < numJobs; i++) {
            const newPage = await context.newPage();
            await new AuthenticationHelper(newPage, context.request).login(testUser);
            pages.push(newPage);
        }

        // Create jobs concurrently
        const createPromises = pages.map(async (p, index) => {
            const jobPage = new JobManagementPage(p);
            await jobPage.navigateToJobs();
            await jobPage.createJob({
                name: `Concurrent Job ${index + 1}`,
                schedule: '0 * * * *',
                command: `echo "Concurrent test ${index + 1}"`,
            });
        });

        await Promise.all(createPromises);

        // Verify all jobs were created
        await page.reload();
        const finalJobsList = await jobManagementPage.getJobsList();
        
        for (let i = 1; i <= numJobs; i++) {
            expect(finalJobsList).toContain(`Concurrent Job ${i}`);
        }

        // Execute all jobs concurrently
        const executePromises = Array.from({ length: numJobs }, (_, index) =>
            jobManagementPage.executeJob(`Concurrent Job ${index + 1}`)
        );

        await Promise.all(executePromises);

        // Wait for all executions to complete
        const waitPromises = Array.from({ length: numJobs }, (_, index) =>
            jobManagementPage.waitForJobExecution(`Concurrent Job ${index + 1}`)
        );

        await Promise.all(waitPromises);

        // Verify all executions succeeded
        for (let i = 1; i <= numJobs; i++) {
            const status = await jobManagementPage.getJobStatus(`Concurrent Job ${i}`);
            expect(status).toBe('success');
        }

        // Close additional pages
        for (const p of pages) {
            await p.close();
        }
    });

    test('Real-time updates and notifications', async ({ page }) => {
        await jobManagementPage.navigateToJobs();

        const testJob: TestJob = {
            name: 'Real-time Test Job',
            schedule: '0 * * * *',
            command: 'sleep 5 && echo "Real-time test completed"',
        };

        await jobManagementPage.createJob(testJob);

        // Execute job and watch real-time updates
        await jobManagementPage.executeJob(testJob.name);

        // Verify status changes to "running"
        await expect(page.locator(`[data-testid="job-status-${testJob.name}"]`))
            .toContainText('running');

        // Verify progress indicator appears
        await expect(page.locator(`[data-testid="job-progress-${testJob.name}"]`))
            .toBeVisible();

        // Wait for completion
        await jobManagementPage.waitForJobExecution(testJob.name, 10000);

        // Verify final status
        const finalStatus = await jobManagementPage.getJobStatus(testJob.name);
        expect(finalStatus).toBe('success');

        // Verify progress indicator disappears
        await expect(page.locator(`[data-testid="job-progress-${testJob.name}"]`))
            .not.toBeVisible();
    });

    test('Mobile responsive design', async ({ page }) => {
        // Test mobile viewport
        await page.setViewportSize({ width: 375, height: 667 });
        
        await jobManagementPage.navigateToJobs();

        // Verify mobile navigation works
        await page.click('[data-testid="mobile-menu-toggle"]');
        await expect(page.locator('[data-testid="mobile-nav-menu"]')).toBeVisible();

        // Create job on mobile
        await page.click('[data-testid="mobile-create-job-fab"]');
        
        const mobileJob: TestJob = {
            name: 'Mobile Test Job',
            schedule: '0 * * * *',
            command: 'echo "Mobile test"',
        };

        await jobManagementPage.createJob(mobileJob);

        // Verify job appears in mobile list
        const jobsList = await jobManagementPage.getJobsList();
        expect(jobsList).toContain(mobileJob.name);

        // Test mobile job actions
        await page.click(`[data-testid="mobile-job-actions-${mobileJob.name}"]`);
        await expect(page.locator('[data-testid="mobile-actions-menu"]')).toBeVisible();

        await page.click('[data-testid="mobile-execute-action"]');
        await page.click('[data-testid="confirm-execute"]');

        // Verify execution on mobile
        await jobManagementPage.waitForJobExecution(mobileJob.name);
        const status = await jobManagementPage.getJobStatus(mobileJob.name);
        expect(status).toBe('success');
    });

    test.afterEach(async ({ page }) => {
        await authHelper.logout();
    });
});

// Performance tests
test.describe('Performance Tests', () => {
    test('Page load performance', async ({ page }) => {
        // Monitor page load metrics
        await page.goto('/jobs', { waitUntil: 'networkidle' });

        const performanceMetrics = await page.evaluate(() => {
            const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
            return {
                domContentLoaded: navigation.domContentLoadedEventEnd - navigation.domContentLoadedEventStart,
                loadComplete: navigation.loadEventEnd - navigation.loadEventStart,
                firstContentfulPaint: performance.getEntriesByName('first-contentful-paint')[0]?.startTime || 0,
            };
        });

        // Assert performance thresholds
        expect(performanceMetrics.domContentLoaded).toBeLessThan(2000); // 2 seconds
        expect(performanceMetrics.loadComplete).toBeLessThan(3000); // 3 seconds
        expect(performanceMetrics.firstContentfulPaint).toBeLessThan(1500); // 1.5 seconds
    });

    test('Large dataset performance', async ({ page }) => {
        // This would typically be done via API to create large dataset
        // Then test UI performance with large number of jobs
        
        await page.goto('/jobs');
        
        // Measure time to render large job list
        const startTime = Date.now();
        await page.waitForSelector('[data-testid="jobs-list"]');
        const renderTime = Date.now() - startTime;
        
        expect(renderTime).toBeLessThan(5000); // 5 seconds max for large dataset
    });
});
```

This comprehensive testing strategy document provides detailed examples of unit testing, integration testing, and end-to-end testing frameworks. The examples include proper mocking, test data management, concurrent testing scenarios, and performance validation to ensure the NovaCron enhancements meet enterprise-grade quality standards.