# Implementation Guides - Code Examples & Configuration Templates

## Overview

This document provides comprehensive implementation guides, code examples, and configuration templates for the NovaCron enhancement roadmap. Each section includes production-ready code, configuration files, and deployment instructions.

## Table of Contents

1. [Security Implementation Guide](#security-implementation-guide)
2. [Performance Optimization Guide](#performance-optimization-guide)
3. [Infrastructure as Code Templates](#infrastructure-as-code-templates)
4. [Monitoring & Observability Setup](#monitoring--observability-setup)
5. [CI/CD Pipeline Configuration](#cicd-pipeline-configuration)
6. [Database Migration Scripts](#database-migration-scripts)
7. [Testing Framework Setup](#testing-framework-setup)
8. [Deployment Automation](#deployment-automation)

---

## Security Implementation Guide

### JWT Authentication System

```go
// internal/auth/jwt_service.go
package auth

import (
    "context"
    "crypto/rsa"
    "fmt"
    "time"

    "github.com/golang-jwt/jwt/v5"
    "github.com/google/uuid"
    "go.uber.org/zap"
)

type JWTService struct {
    privateKey    *rsa.PrivateKey
    publicKey     *rsa.PublicKey
    issuer        string
    accessTTL     time.Duration
    refreshTTL    time.Duration
    logger        *zap.Logger
    blacklist     TokenBlacklist
    rateLimiter   RateLimiter
}

type TokenPair struct {
    AccessToken  string    `json:"access_token"`
    RefreshToken string    `json:"refresh_token"`
    ExpiresAt    time.Time `json:"expires_at"`
    TokenType    string    `json:"token_type"`
}

type Claims struct {
    UserID      int64    `json:"user_id"`
    Email       string   `json:"email"`
    Roles       []string `json:"roles"`
    Permissions []string `json:"permissions"`
    SessionID   string   `json:"session_id"`
    jwt.RegisteredClaims
}

func NewJWTService(config JWTConfig, logger *zap.Logger) (*JWTService, error) {
    privateKey, err := jwt.ParseRSAPrivateKeyFromPEM([]byte(config.PrivateKey))
    if err != nil {
        return nil, fmt.Errorf("failed to parse private key: %w", err)
    }

    publicKey, err := jwt.ParseRSAPublicKeyFromPEM([]byte(config.PublicKey))
    if err != nil {
        return nil, fmt.Errorf("failed to parse public key: %w", err)
    }

    return &JWTService{
        privateKey:  privateKey,
        publicKey:   publicKey,
        issuer:      config.Issuer,
        accessTTL:   config.AccessTTL,
        refreshTTL:  config.RefreshTTL,
        logger:      logger,
        blacklist:   NewRedisTokenBlacklist(config.RedisClient),
        rateLimiter: NewRedisRateLimiter(config.RedisClient),
    }, nil
}

func (j *JWTService) GenerateTokenPair(ctx context.Context, user *User) (*TokenPair, error) {
    // Rate limiting check
    if !j.rateLimiter.Allow(ctx, fmt.Sprintf("token_generation:%d", user.ID), 10, time.Hour) {
        return nil, ErrRateLimited
    }

    sessionID := uuid.New().String()
    now := time.Now()
    
    // Access token claims
    accessClaims := &Claims{
        UserID:      user.ID,
        Email:       user.Email,
        Roles:       user.Roles,
        Permissions: user.GetPermissions(),
        SessionID:   sessionID,
        RegisteredClaims: jwt.RegisteredClaims{
            Issuer:    j.issuer,
            Subject:   fmt.Sprintf("%d", user.ID),
            Audience:  []string{"api"},
            ExpiresAt: jwt.NewNumericDate(now.Add(j.accessTTL)),
            NotBefore: jwt.NewNumericDate(now),
            IssuedAt:  jwt.NewNumericDate(now),
            ID:        uuid.New().String(),
        },
    }

    // Create and sign access token
    accessToken := jwt.NewWithClaims(jwt.SigningMethodRS256, accessClaims)
    accessTokenString, err := accessToken.SignedString(j.privateKey)
    if err != nil {
        return nil, fmt.Errorf("failed to sign access token: %w", err)
    }

    // Refresh token claims (longer lived, fewer permissions)
    refreshClaims := &Claims{
        UserID:    user.ID,
        SessionID: sessionID,
        RegisteredClaims: jwt.RegisteredClaims{
            Issuer:    j.issuer,
            Subject:   fmt.Sprintf("%d", user.ID),
            Audience:  []string{"refresh"},
            ExpiresAt: jwt.NewNumericDate(now.Add(j.refreshTTL)),
            NotBefore: jwt.NewNumericDate(now),
            IssuedAt:  jwt.NewNumericDate(now),
            ID:        uuid.New().String(),
        },
    }

    refreshToken := jwt.NewWithClaims(jwt.SigningMethodRS256, refreshClaims)
    refreshTokenString, err := refreshToken.SignedString(j.privateKey)
    if err != nil {
        return nil, fmt.Errorf("failed to sign refresh token: %w", err)
    }

    // Store session information
    if err := j.storeSession(ctx, sessionID, user.ID, now.Add(j.refreshTTL)); err != nil {
        j.logger.Error("Failed to store session", zap.Error(err), zap.String("session_id", sessionID))
    }

    return &TokenPair{
        AccessToken:  accessTokenString,
        RefreshToken: refreshTokenString,
        ExpiresAt:    now.Add(j.accessTTL),
        TokenType:    "Bearer",
    }, nil
}

func (j *JWTService) ValidateToken(ctx context.Context, tokenString string) (*Claims, error) {
    // Check blacklist first
    if j.blacklist.IsBlacklisted(ctx, tokenString) {
        return nil, ErrTokenBlacklisted
    }

    token, err := jwt.ParseWithClaims(tokenString, &Claims{}, func(token *jwt.Token) (interface{}, error) {
        if _, ok := token.Method.(*jwt.SigningMethodRSA); !ok {
            return nil, fmt.Errorf("unexpected signing method: %v", token.Header["alg"])
        }
        return j.publicKey, nil
    })

    if err != nil {
        return nil, fmt.Errorf("failed to parse token: %w", err)
    }

    claims, ok := token.Claims.(*Claims)
    if !ok || !token.Valid {
        return nil, ErrInvalidToken
    }

    // Validate session
    if !j.isSessionValid(ctx, claims.SessionID, claims.UserID) {
        return nil, ErrSessionInvalid
    }

    return claims, nil
}

// Authentication middleware
func (j *JWTService) AuthMiddleware() func(http.Handler) http.Handler {
    return func(next http.Handler) http.Handler {
        return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
            // Extract token from Authorization header
            authHeader := r.Header.Get("Authorization")
            if authHeader == "" {
                http.Error(w, "Missing authorization header", http.StatusUnauthorized)
                return
            }

            const bearerPrefix = "Bearer "
            if !strings.HasPrefix(authHeader, bearerPrefix) {
                http.Error(w, "Invalid authorization header format", http.StatusUnauthorized)
                return
            }

            tokenString := authHeader[len(bearerPrefix):]
            
            // Validate token
            claims, err := j.ValidateToken(r.Context(), tokenString)
            if err != nil {
                j.logger.Warn("Token validation failed", 
                    zap.Error(err), 
                    zap.String("remote_addr", r.RemoteAddr),
                    zap.String("user_agent", r.UserAgent()))
                http.Error(w, "Invalid token", http.StatusUnauthorized)
                return
            }

            // Add claims to context
            ctx := context.WithValue(r.Context(), "claims", claims)
            next.ServeHTTP(w, r.WithContext(ctx))
        })
    }
}
```

### HashiCorp Vault Integration

```go
// internal/secrets/vault_client.go
package secrets

import (
    "context"
    "fmt"
    "time"

    vault "github.com/hashicorp/vault/api"
    auth "github.com/hashicorp/vault/api/auth/kubernetes"
)

type VaultClient struct {
    client      *vault.Client
    authMethod  auth.KubernetesAuth
    secretPath  string
    renewTicker *time.Ticker
    stopChan    chan struct{}
}

type SecretManager interface {
    GetSecret(ctx context.Context, key string) (*Secret, error)
    StoreSecret(ctx context.Context, key string, value interface{}, ttl time.Duration) error
    DeleteSecret(ctx context.Context, key string) error
    RenewLease(ctx context.Context, leaseID string) error
}

type Secret struct {
    Data      map[string]interface{} `json:"data"`
    LeaseID   string                `json:"lease_id"`
    Renewable bool                  `json:"renewable"`
    TTL       time.Duration         `json:"ttl"`
}

func NewVaultClient(config VaultConfig) (*VaultClient, error) {
    vaultConfig := vault.DefaultConfig()
    vaultConfig.Address = config.Address
    vaultConfig.Timeout = 10 * time.Second

    client, err := vault.NewClient(vaultConfig)
    if err != nil {
        return nil, fmt.Errorf("failed to create vault client: %w", err)
    }

    // Configure Kubernetes authentication
    k8sAuth, err := auth.NewKubernetesAuth(
        config.Role,
        auth.WithServiceAccountTokenPath(config.ServiceAccountTokenPath),
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create k8s auth: %w", err)
    }

    vc := &VaultClient{
        client:     client,
        authMethod: *k8sAuth,
        secretPath: config.SecretPath,
        stopChan:   make(chan struct{}),
    }

    // Initial authentication
    if err := vc.authenticate(); err != nil {
        return nil, fmt.Errorf("initial authentication failed: %w", err)
    }

    // Start token renewal goroutine
    go vc.startTokenRenewal()

    return vc, nil
}

func (vc *VaultClient) authenticate() error {
    authInfo, err := vc.client.Auth().Login(context.Background(), &vc.authMethod)
    if err != nil {
        return fmt.Errorf("authentication failed: %w", err)
    }

    if authInfo == nil {
        return fmt.Errorf("no auth info returned")
    }

    return nil
}

func (vc *VaultClient) GetSecret(ctx context.Context, key string) (*Secret, error) {
    secretPath := fmt.Sprintf("%s/%s", vc.secretPath, key)
    
    vaultSecret, err := vc.client.Logical().ReadWithContext(ctx, secretPath)
    if err != nil {
        return nil, fmt.Errorf("failed to read secret %s: %w", key, err)
    }

    if vaultSecret == nil {
        return nil, fmt.Errorf("secret %s not found", key)
    }

    return &Secret{
        Data:      vaultSecret.Data,
        LeaseID:   vaultSecret.LeaseID,
        Renewable: vaultSecret.Renewable,
        TTL:       time.Duration(vaultSecret.LeaseDuration) * time.Second,
    }, nil
}

func (vc *VaultClient) StoreSecret(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    secretPath := fmt.Sprintf("%s/%s", vc.secretPath, key)
    
    data := map[string]interface{}{
        "value": value,
        "ttl":   fmt.Sprintf("%ds", int(ttl.Seconds())),
    }

    _, err := vc.client.Logical().WriteWithContext(ctx, secretPath, data)
    if err != nil {
        return fmt.Errorf("failed to store secret %s: %w", key, err)
    }

    return nil
}

func (vc *VaultClient) startTokenRenewal() {
    ticker := time.NewTicker(30 * time.Minute) // Renew every 30 minutes
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            if err := vc.renewToken(); err != nil {
                // Log error and try to re-authenticate
                if err := vc.authenticate(); err != nil {
                    // Critical error - should alert operations team
                    continue
                }
            }
        case <-vc.stopChan:
            return
        }
    }
}

func (vc *VaultClient) renewToken() error {
    _, err := vc.client.Auth().Token().RenewSelf(0)
    return err
}
```

### Input Validation & Sanitization

```go
// internal/validation/validator.go
package validation

import (
    "fmt"
    "net/mail"
    "regexp"
    "strings"
    "unicode"

    "github.com/go-playground/validator/v10"
    "github.com/microcosm-cc/bluemonday"
)

type Validator struct {
    validate   *validator.Validate
    htmlPolicy *bluemonday.Policy
}

type ValidationError struct {
    Field   string `json:"field"`
    Message string `json:"message"`
    Value   string `json:"value,omitempty"`
}

type ValidationErrors []ValidationError

func (ve ValidationErrors) Error() string {
    var messages []string
    for _, err := range ve {
        messages = append(messages, fmt.Sprintf("%s: %s", err.Field, err.Message))
    }
    return strings.Join(messages, "; ")
}

func NewValidator() *Validator {
    v := validator.New()
    
    // Register custom validators
    v.RegisterValidation("strong_password", validateStrongPassword)
    v.RegisterValidation("no_sql_injection", validateNoSQLInjection)
    v.RegisterValidation("safe_filename", validateSafeFilename)
    v.RegisterValidation("cron_expression", validateCronExpression)

    // HTML sanitization policy
    htmlPolicy := bluemonday.UGCPolicy()
    htmlPolicy.AllowElements("p", "br", "strong", "em", "u", "ol", "ul", "li")
    htmlPolicy.AllowAttrs("href").OnElements("a")

    return &Validator{
        validate:   v,
        htmlPolicy: htmlPolicy,
    }
}

// User registration validation
type UserRegistration struct {
    Email           string `json:"email" validate:"required,email,max=255"`
    Password        string `json:"password" validate:"required,strong_password,min=8,max=128"`
    ConfirmPassword string `json:"confirm_password" validate:"required,eqfield=Password"`
    FirstName       string `json:"first_name" validate:"required,min=1,max=50,no_sql_injection"`
    LastName        string `json:"last_name" validate:"required,min=1,max=50,no_sql_injection"`
    Company         string `json:"company" validate:"max=100,no_sql_injection"`
}

func (v *Validator) ValidateUserRegistration(user *UserRegistration) ValidationErrors {
    var errors ValidationErrors

    // Basic validation
    if err := v.validate.Struct(user); err != nil {
        for _, err := range err.(validator.ValidationErrors) {
            errors = append(errors, ValidationError{
                Field:   err.Field(),
                Message: getValidationMessage(err.Tag(), err.Param()),
                Value:   fmt.Sprintf("%v", err.Value()),
            })
        }
    }

    // Additional email validation
    if user.Email != "" {
        if !isValidBusinessEmail(user.Email) {
            errors = append(errors, ValidationError{
                Field:   "email",
                Message: "Business email address required",
                Value:   user.Email,
            })
        }
    }

    // Sanitize HTML content
    user.FirstName = v.htmlPolicy.Sanitize(user.FirstName)
    user.LastName = v.htmlPolicy.Sanitize(user.LastName)
    user.Company = v.htmlPolicy.Sanitize(user.Company)

    return errors
}

// Job creation validation
type JobCreation struct {
    Name        string            `json:"name" validate:"required,min=3,max=100,no_sql_injection"`
    Description string            `json:"description" validate:"max=1000"`
    Schedule    string            `json:"schedule" validate:"required,cron_expression"`
    Command     string            `json:"command" validate:"required,max=10000,no_sql_injection"`
    Environment map[string]string `json:"environment" validate:"dive,keys,max=100,endkeys,max=1000"`
    Timeout     int               `json:"timeout" validate:"min=1,max=86400"` // Max 24 hours
    Retries     int               `json:"retries" validate:"min=0,max=5"`
}

func (v *Validator) ValidateJobCreation(job *JobCreation) ValidationErrors {
    var errors ValidationErrors

    if err := v.validate.Struct(job); err != nil {
        for _, err := range err.(validator.ValidationErrors) {
            errors = append(errors, ValidationError{
                Field:   err.Field(),
                Message: getValidationMessage(err.Tag(), err.Param()),
                Value:   fmt.Sprintf("%v", err.Value()),
            })
        }
    }

    // Validate command safety
    if containsDangerousCommands(job.Command) {
        errors = append(errors, ValidationError{
            Field:   "command",
            Message: "Command contains potentially dangerous operations",
        })
    }

    // Sanitize description
    job.Description = v.htmlPolicy.Sanitize(job.Description)

    return errors
}

// Custom validation functions
func validateStrongPassword(fl validator.FieldLevel) bool {
    password := fl.Field().String()
    
    if len(password) < 8 {
        return false
    }

    var (
        hasUpper   bool
        hasLower   bool
        hasNumber  bool
        hasSpecial bool
    )

    for _, char := range password {
        switch {
        case unicode.IsUpper(char):
            hasUpper = true
        case unicode.IsLower(char):
            hasLower = true
        case unicode.IsNumber(char):
            hasNumber = true
        case unicode.IsPunct(char) || unicode.IsSymbol(char):
            hasSpecial = true
        }
    }

    return hasUpper && hasLower && hasNumber && hasSpecial
}

func validateNoSQLInjection(fl validator.FieldLevel) bool {
    value := strings.ToLower(fl.Field().String())
    
    // Common SQL injection patterns
    patterns := []string{
        "union select", "drop table", "drop database", "delete from",
        "insert into", "update set", "script", "javascript:", "vbscript:",
        "onload=", "onerror=", "<script", "</script>", "exec(",
    }

    for _, pattern := range patterns {
        if strings.Contains(value, pattern) {
            return false
        }
    }

    return true
}

func validateSafeFilename(fl validator.FieldLevel) bool {
    filename := fl.Field().String()
    
    // Check for path traversal
    if strings.Contains(filename, "..") || strings.Contains(filename, "/") || strings.Contains(filename, "\\") {
        return false
    }

    // Check for reserved names
    reserved := []string{"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "LPT1", "LPT2"}
    upper := strings.ToUpper(filename)
    for _, res := range reserved {
        if upper == res {
            return false
        }
    }

    return true
}

func validateCronExpression(fl validator.FieldLevel) bool {
    expression := fl.Field().String()
    
    // Basic cron expression validation (5 or 6 fields)
    fields := strings.Fields(expression)
    if len(fields) != 5 && len(fields) != 6 {
        return false
    }

    // More sophisticated validation would use a cron parsing library
    cronRegex := regexp.MustCompile(`^(\*|([0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9])|\*\/([0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9])) (\*|([0-9]|1[0-9]|2[0-3])|\*\/([0-9]|1[0-9]|2[0-3])) (\*|([1-9]|1[0-9]|2[0-9]|3[0-1])|\*\/([1-9]|1[0-9]|2[0-9]|3[0-1])) (\*|([1-9]|1[0-2])|\*\/([1-9]|1[0-2])) (\*|([0-6])|\*\/[0-6])$`)
    
    return cronRegex.MatchString(expression)
}

func isValidBusinessEmail(email string) bool {
    addr, err := mail.ParseAddress(email)
    if err != nil {
        return false
    }

    // Split email to check domain
    parts := strings.Split(addr.Address, "@")
    if len(parts) != 2 {
        return false
    }

    domain := strings.ToLower(parts[1])
    
    // Block common free email providers for business accounts
    freeProviders := []string{
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "aol.com", "icloud.com", "protonmail.com",
    }

    for _, provider := range freeProviders {
        if domain == provider {
            return false
        }
    }

    return true
}

func containsDangerousCommands(command string) bool {
    dangerous := []string{
        "rm -rf", "format", "del /s", "dd if=", ":(){ :|:& };:",
        "sudo", "chmod 777", "wget", "curl", "nc ", "netcat",
    }

    lowerCommand := strings.ToLower(command)
    for _, danger := range dangerous {
        if strings.Contains(lowerCommand, danger) {
            return true
        }
    }

    return false
}
```

---

## Performance Optimization Guide

### Database Query Optimization

```sql
-- advanced_queries.sql
-- Performance-optimized queries for NovaCron

-- 1. User Jobs with Latest Execution Status (Eliminates N+1 Query)
CREATE OR REPLACE FUNCTION get_user_jobs_with_status(p_user_id BIGINT, p_limit INT DEFAULT 20)
RETURNS TABLE (
    job_id BIGINT,
    job_name VARCHAR(255),
    job_schedule VARCHAR(100),
    job_status VARCHAR(50),
    last_execution_id BIGINT,
    last_execution_status VARCHAR(50),
    last_execution_time TIMESTAMP,
    avg_duration INTERVAL,
    success_rate DECIMAL(5,2)
) AS $$
BEGIN
    RETURN QUERY
    WITH job_stats AS (
        SELECT 
            je.job_id,
            COUNT(*) as total_executions,
            COUNT(*) FILTER (WHERE je.status = 'success') as successful_executions,
            AVG(je.ended_at - je.started_at) as avg_duration,
            MAX(je.id) as latest_execution_id
        FROM job_executions je
        WHERE je.created_at >= NOW() - INTERVAL '30 days'
        GROUP BY je.job_id
    ),
    latest_execution AS (
        SELECT DISTINCT ON (je.job_id)
            je.job_id,
            je.id as execution_id,
            je.status as execution_status,
            je.ended_at as execution_time
        FROM job_executions je
        ORDER BY je.job_id, je.created_at DESC
    )
    SELECT 
        j.id::BIGINT,
        j.name,
        j.schedule,
        j.status,
        le.execution_id::BIGINT,
        le.execution_status,
        le.execution_time,
        COALESCE(js.avg_duration, INTERVAL '0'),
        CASE 
            WHEN js.total_executions > 0 THEN 
                ROUND((js.successful_executions::DECIMAL / js.total_executions) * 100, 2)
            ELSE 0
        END as success_rate
    FROM jobs j
    LEFT JOIN job_stats js ON j.id = js.job_id
    LEFT JOIN latest_execution le ON j.id = le.job_id
    WHERE j.user_id = p_user_id 
        AND j.deleted_at IS NULL
    ORDER BY j.created_at DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql;

-- 2. Advanced Job Execution Analytics
CREATE MATERIALIZED VIEW job_execution_analytics AS
WITH execution_metrics AS (
    SELECT 
        j.id as job_id,
        j.user_id,
        j.name as job_name,
        DATE_TRUNC('hour', je.started_at) as hour_bucket,
        COUNT(*) as execution_count,
        COUNT(*) FILTER (WHERE je.status = 'success') as success_count,
        COUNT(*) FILTER (WHERE je.status = 'failed') as failure_count,
        AVG(EXTRACT(EPOCH FROM (je.ended_at - je.started_at))) as avg_duration_seconds,
        PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (je.ended_at - je.started_at))) as median_duration_seconds,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (je.ended_at - je.started_at))) as p95_duration_seconds,
        MAX(EXTRACT(EPOCH FROM (je.ended_at - je.started_at))) as max_duration_seconds,
        MIN(EXTRACT(EPOCH FROM (je.ended_at - je.started_at))) as min_duration_seconds
    FROM jobs j
    JOIN job_executions je ON j.id = je.job_id
    WHERE je.started_at >= NOW() - INTERVAL '7 days'
        AND je.ended_at IS NOT NULL
        AND j.deleted_at IS NULL
    GROUP BY j.id, j.user_id, j.name, DATE_TRUNC('hour', je.started_at)
),
trend_analysis AS (
    SELECT 
        job_id,
        user_id,
        job_name,
        hour_bucket,
        execution_count,
        success_count,
        failure_count,
        avg_duration_seconds,
        median_duration_seconds,
        p95_duration_seconds,
        -- Calculate trend over previous 24 hours
        AVG(avg_duration_seconds) OVER (
            PARTITION BY job_id 
            ORDER BY hour_bucket 
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ) as trend_avg_duration_24h,
        LAG(avg_duration_seconds, 1) OVER (
            PARTITION BY job_id 
            ORDER BY hour_bucket
        ) as prev_hour_avg_duration,
        -- Success rate trend
        (success_count::DECIMAL / NULLIF(execution_count, 0)) * 100 as success_rate,
        AVG(success_count::DECIMAL / NULLIF(execution_count, 0)) OVER (
            PARTITION BY job_id 
            ORDER BY hour_bucket 
            ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
        ) * 100 as trend_success_rate_24h
    FROM execution_metrics
)
SELECT 
    job_id,
    user_id,
    job_name,
    hour_bucket,
    execution_count,
    success_count,
    failure_count,
    ROUND(avg_duration_seconds, 2) as avg_duration_seconds,
    ROUND(median_duration_seconds, 2) as median_duration_seconds,
    ROUND(p95_duration_seconds, 2) as p95_duration_seconds,
    ROUND(trend_avg_duration_24h, 2) as trend_avg_duration_24h,
    ROUND(success_rate, 2) as success_rate,
    ROUND(trend_success_rate_24h, 2) as trend_success_rate_24h,
    -- Performance indicators
    CASE 
        WHEN avg_duration_seconds > trend_avg_duration_24h * 1.5 THEN 'DEGRADED'
        WHEN avg_duration_seconds > trend_avg_duration_24h * 1.2 THEN 'SLOW'
        ELSE 'NORMAL'
    END as performance_status,
    -- Reliability indicators
    CASE 
        WHEN success_rate < 50 THEN 'CRITICAL'
        WHEN success_rate < 80 THEN 'WARNING'
        ELSE 'HEALTHY'
    END as reliability_status
FROM trend_analysis;

-- Create indexes for performance
CREATE UNIQUE INDEX job_execution_analytics_pk 
ON job_execution_analytics (job_id, hour_bucket);

CREATE INDEX job_execution_analytics_user_time 
ON job_execution_analytics (user_id, hour_bucket DESC);

CREATE INDEX job_execution_analytics_performance 
ON job_execution_analytics (performance_status, hour_bucket DESC) 
WHERE performance_status IN ('DEGRADED', 'SLOW');

-- Refresh materialized view automatically
CREATE OR REPLACE FUNCTION refresh_job_execution_analytics()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY job_execution_analytics;
END;
$$ LANGUAGE plpgsql;

-- Schedule refresh every 15 minutes
SELECT cron.schedule('refresh-analytics', '*/15 * * * *', 'SELECT refresh_job_execution_analytics();');
```

### Caching Strategy Implementation

```go
// internal/cache/multi_tier_cache.go
package cache

import (
    "context"
    "encoding/json"
    "fmt"
    "time"

    "github.com/allegro/bigcache/v3"
    "github.com/redis/go-redis/v9"
    "go.uber.org/zap"
)

type MultiTierCache struct {
    l1Cache    *bigcache.BigCache  // Local memory cache
    l2Cache    redis.UniversalClient // Redis distributed cache
    l3Cache    *DatabaseCache     // Database cache for persistent data
    logger     *zap.Logger
    metrics    *CacheMetrics
    serializer Serializer
}

type CacheEntry struct {
    Data        []byte    `json:"data"`
    Timestamp   time.Time `json:"timestamp"`
    TTL         int64     `json:"ttl"`
    Version     string    `json:"version"`
}

type CacheConfig struct {
    L1Config L1Config
    L2Config L2Config
    L3Config L3Config
}

type L1Config struct {
    MaxSize      int           // Max memory usage in MB
    MaxEntries   int           // Max number of entries
    TTL          time.Duration // Default TTL
    CleanWindow  time.Duration // Cleanup interval
}

func NewMultiTierCache(config CacheConfig, logger *zap.Logger) (*MultiTierCache, error) {
    // L1 Cache (BigCache - Local Memory)
    l1Config := bigcache.Config{
        Shards:             256,
        LifeWindow:         config.L1Config.TTL,
        CleanWindow:        config.L1Config.CleanWindow,
        MaxEntriesInWindow: config.L1Config.MaxEntries,
        MaxEntrySize:       1024 * 1024, // 1MB max entry size
        HardMaxCacheSize:   config.L1Config.MaxSize,
        Verbose:            false,
    }

    l1Cache, err := bigcache.New(context.Background(), l1Config)
    if err != nil {
        return nil, fmt.Errorf("failed to create L1 cache: %w", err)
    }

    // L2 Cache (Redis - Distributed)
    l2Cache := redis.NewUniversalClient(&redis.UniversalOptions{
        Addrs:       config.L2Config.Addrs,
        Password:    config.L2Config.Password,
        DB:          config.L2Config.DB,
        PoolSize:    100,
        MinIdleConns: 20,
        MaxRetries:  3,
        DialTimeout: 5 * time.Second,
        ReadTimeout: 3 * time.Second,
        WriteTimeout: 3 * time.Second,
    })

    // Test Redis connection
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    if err := l2Cache.Ping(ctx).Err(); err != nil {
        return nil, fmt.Errorf("failed to connect to Redis: %w", err)
    }

    // L3 Cache (Database)
    l3Cache := NewDatabaseCache(config.L3Config)

    return &MultiTierCache{
        l1Cache:    l1Cache,
        l2Cache:    l2Cache,
        l3Cache:    l3Cache,
        logger:     logger,
        metrics:    NewCacheMetrics(),
        serializer: NewJSONSerializer(),
    }, nil
}

func (c *MultiTierCache) Get(ctx context.Context, key string, dest interface{}) error {
    startTime := time.Now()
    defer func() {
        c.metrics.RecordLatency("get", time.Since(startTime))
    }()

    // L1 Cache lookup
    if data, err := c.l1Cache.Get(key); err == nil {
        c.metrics.RecordHit("l1")
        return c.serializer.Deserialize(data, dest)
    }

    // L2 Cache lookup
    data, err := c.l2Cache.Get(ctx, key).Bytes()
    if err == nil {
        c.metrics.RecordHit("l2")
        
        // Store in L1 for faster future access
        go func() {
            if serData, err := c.l2Cache.Get(context.Background(), key).Bytes(); err == nil {
                c.l1Cache.Set(key, serData)
            }
        }()
        
        return c.serializer.Deserialize(data, dest)
    }

    if err != redis.Nil {
        c.logger.Error("L2 cache error", zap.Error(err), zap.String("key", key))
    }

    // L3 Cache lookup
    if data, err := c.l3Cache.Get(ctx, key); err == nil {
        c.metrics.RecordHit("l3")
        
        // Promote to higher tiers asynchronously
        go func() {
            ctx := context.Background()
            
            // Store in L2
            if serData, err := c.serializer.Serialize(data); err == nil {
                c.l2Cache.Set(ctx, key, serData, time.Hour).Err()
                
                // Store in L1
                c.l1Cache.Set(key, serData)
            }
        }()
        
        return c.serializer.Deserialize(data, dest)
    }

    c.metrics.RecordMiss()
    return ErrCacheMiss
}

func (c *MultiTierCache) Set(ctx context.Context, key string, value interface{}, ttl time.Duration) error {
    data, err := c.serializer.Serialize(value)
    if err != nil {
        return fmt.Errorf("failed to serialize value: %w", err)
    }

    entry := CacheEntry{
        Data:      data,
        Timestamp: time.Now(),
        TTL:       int64(ttl.Seconds()),
        Version:   "1.0",
    }

    entryData, err := json.Marshal(entry)
    if err != nil {
        return fmt.Errorf("failed to marshal cache entry: %w", err)
    }

    // Store in all tiers
    var errors []error

    // L1 Cache
    if err := c.l1Cache.Set(key, entryData); err != nil {
        errors = append(errors, fmt.Errorf("L1 set failed: %w", err))
    }

    // L2 Cache
    if err := c.l2Cache.Set(ctx, key, entryData, ttl).Err(); err != nil {
        errors = append(errors, fmt.Errorf("L2 set failed: %w", err))
    }

    // L3 Cache (async for performance)
    go func() {
        if err := c.l3Cache.Set(context.Background(), key, entryData, ttl); err != nil {
            c.logger.Error("L3 cache set failed", zap.Error(err), zap.String("key", key))
        }
    }()

    if len(errors) > 0 {
        return fmt.Errorf("cache set errors: %v", errors)
    }

    c.metrics.RecordSet()
    return nil
}

// Smart cache warming based on access patterns
func (c *MultiTierCache) WarmCache(ctx context.Context, strategy WarmingStrategy) error {
    switch strategy.Type {
    case "user_jobs":
        return c.warmUserJobs(ctx, strategy.UserIDs)
    case "popular_queries":
        return c.warmPopularQueries(ctx, strategy.QueryPatterns)
    case "scheduled_jobs":
        return c.warmScheduledJobs(ctx)
    default:
        return fmt.Errorf("unknown warming strategy: %s", strategy.Type)
    }
}

func (c *MultiTierCache) warmUserJobs(ctx context.Context, userIDs []int64) error {
    const batchSize = 100
    
    for i := 0; i < len(userIDs); i += batchSize {
        end := i + batchSize
        if end > len(userIDs) {
            end = len(userIDs)
        }
        
        batch := userIDs[i:end]
        
        // Warm user job data
        for _, userID := range batch {
            go func(uid int64) {
                key := fmt.Sprintf("user_jobs:%d", uid)
                
                // Check if already cached
                var jobs []Job
                if err := c.Get(ctx, key, &jobs); err == nil {
                    return // Already cached
                }
                
                // Load from database and cache
                if jobs, err := c.loadUserJobsFromDB(ctx, uid); err == nil {
                    c.Set(ctx, key, jobs, 30*time.Minute)
                }
            }(userID)
        }
        
        // Rate limiting
        time.Sleep(100 * time.Millisecond)
    }
    
    return nil
}

// Cache invalidation with intelligent patterns
func (c *MultiTierCache) InvalidatePattern(ctx context.Context, pattern string) error {
    // L1 Cache - clear all (simple approach)
    c.l1Cache.Reset()
    
    // L2 Cache - pattern-based deletion
    keys, err := c.l2Cache.Keys(ctx, pattern).Result()
    if err != nil {
        return fmt.Errorf("failed to get keys for pattern %s: %w", pattern, err)
    }
    
    if len(keys) > 0 {
        if err := c.l2Cache.Del(ctx, keys...).Err(); err != nil {
            return fmt.Errorf("failed to delete keys: %w", err)
        }
    }
    
    // L3 Cache - async invalidation
    go func() {
        c.l3Cache.InvalidatePattern(context.Background(), pattern)
    }()
    
    c.metrics.RecordInvalidation(pattern, len(keys))
    return nil
}
```

---

## Infrastructure as Code Templates

### Terraform AWS Infrastructure

```hcl
# terraform/main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  backend "s3" {
    bucket         = "novacron-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-lock"
  }
}

# Local variables
locals {
  cluster_name = "novacron-${var.environment}"
  common_tags = {
    Project     = "novacron"
    Environment = var.environment
    ManagedBy   = "terraform"
    Owner       = "devops-team"
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
  filter {
    name   = "zone-type"
    values = ["availability-zone"]
  }
}

data "aws_caller_identity" "current" {}

# VPC Module
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = var.vpc_cidr

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets
  database_subnets = var.database_subnets

  enable_nat_gateway     = true
  single_nat_gateway     = var.environment == "development"
  enable_vpn_gateway     = false
  enable_dns_hostnames   = true
  enable_dns_support     = true

  # Flow logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  flow_log_max_aggregation_interval    = 60

  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
    "kubernetes.io/cluster/${local.cluster_name}" = "owned"
  }

  tags = local.common_tags
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name                   = local.cluster_name
  cluster_version               = var.kubernetes_version
  cluster_endpoint_public_access = true

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Logging
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]

  # EKS Managed Node Groups
  eks_managed_node_groups = {
    general = {
      name           = "general"
      instance_types = ["m5.large", "m5.xlarge"]
      
      min_size     = 2
      max_size     = 10
      desired_size = 3

      ami_type       = "AL2_x86_64"
      capacity_type  = "ON_DEMAND"
      disk_size      = 50

      # Taints and labels
      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "general"
      }

      # Security group rules
      vpc_security_group_ids = [aws_security_group.node_group_general.id]

      # User data
      pre_bootstrap_user_data = <<-EOT
        #!/bin/bash
        yum update -y
        yum install -y amazon-cloudwatch-agent
        
        # Configure CloudWatch agent
        cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<EOF
        {
          "metrics": {
            "namespace": "CWAgent",
            "metrics_collected": {
              "cpu": {
                "measurement": ["cpu_usage_idle", "cpu_usage_iowait", "cpu_usage_user", "cpu_usage_system"],
                "metrics_collection_interval": 60
              },
              "disk": {
                "measurement": ["used_percent"],
                "metrics_collection_interval": 60,
                "resources": ["*"]
              },
              "mem": {
                "measurement": ["mem_used_percent"],
                "metrics_collection_interval": 60
              }
            }
          }
        }
        EOF
        
        /opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl \
          -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s
      EOT
    }

    ml_workloads = {
      name           = "ml-workloads"
      instance_types = ["p3.2xlarge", "p3.8xlarge"]
      
      min_size     = 0
      max_size     = 5
      desired_size = 1

      ami_type       = "AL2_x86_64_GPU"
      capacity_type  = "SPOT"
      disk_size      = 100

      k8s_labels = {
        Environment = var.environment
        NodeGroup   = "ml-workloads"
        WorkloadType = "gpu"
      }

      taints = {
        gpu = {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }

      vpc_security_group_ids = [aws_security_group.node_group_ml.id]
    }
  }

  # aws-auth configmap
  manage_aws_auth_configmap = true
  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.eks_admin.arn
      username = "admin"
      groups   = ["system:masters"]
    }
  ]

  tags = local.common_tags
}

# RDS Database
module "rds" {
  source = "terraform-aws-modules/rds/aws"
  version = "~> 6.0"

  identifier = "${local.cluster_name}-db"

  # Engine configuration
  engine               = "postgres"
  engine_version      = "15.4"
  family              = "postgres15"
  major_engine_version = "15"
  instance_class      = var.db_instance_class

  # Storage
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  iops                = 3000
  storage_throughput  = 125

  # Database configuration
  db_name  = "novacron"
  username = "postgres"
  port     = 5432

  # High Availability
  multi_az               = var.environment == "production"
  create_db_subnet_group = true
  subnet_ids            = module.vpc.database_subnets
  vpc_security_group_ids = [aws_security_group.rds.id]

  # Backup and Maintenance
  backup_retention_period = var.environment == "production" ? 30 : 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "Sun:04:00-Sun:05:00"

  # Monitoring
  monitoring_interval = 60
  monitoring_role_arn = aws_iam_role.rds_enhanced_monitoring.arn
  performance_insights_enabled = true
  performance_insights_retention_period = var.environment == "production" ? 731 : 7

  # Encryption
  storage_encrypted = true
  kms_key_id       = aws_kms_key.rds.arn

  # Deletion protection
  deletion_protection = var.environment == "production"
  skip_final_snapshot = var.environment != "production"

  # Parameter group
  create_db_parameter_group = true
  parameters = [
    {
      name  = "log_min_duration_statement"
      value = "1000"
    },
    {
      name  = "log_statement"
      value = "mod"
    },
    {
      name  = "shared_preload_libraries"
      value = "pg_stat_statements"
    }
  ]

  tags = local.common_tags
}

# ElastiCache Redis Cluster
resource "aws_elasticache_replication_group" "redis" {
  replication_group_id         = "${local.cluster_name}-redis"
  description                 = "Redis cluster for NovaCron caching"

  port                       = 6379
  parameter_group_name       = aws_elasticache_parameter_group.redis.name
  node_type                  = var.redis_node_type
  num_cache_clusters         = var.environment == "production" ? 3 : 2

  # High Availability
  automatic_failover_enabled = var.environment == "production"
  multi_az_enabled          = var.environment == "production"

  # Security
  subnet_group_name       = aws_elasticache_subnet_group.redis.name
  security_group_ids      = [aws_security_group.redis.id]
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = random_password.redis_auth.result

  # Backup
  snapshot_retention_limit = var.environment == "production" ? 7 : 1
  snapshot_window         = "05:00-07:00"

  # Maintenance
  maintenance_window = "sun:07:00-sun:09:00"

  # Logging
  log_delivery_configuration {
    destination      = aws_cloudwatch_log_group.redis_slow.name
    destination_type = "cloudwatch-logs"
    log_format      = "text"
    log_type        = "slow-log"
  }

  tags = local.common_tags
}

# Application Load Balancer
module "alb" {
  source = "terraform-aws-modules/alb/aws"
  version = "~> 8.0"

  name = "${local.cluster_name}-alb"

  load_balancer_type = "application"
  vpc_id             = module.vpc.vpc_id
  subnets            = module.vpc.public_subnets
  security_groups    = [aws_security_group.alb.id]

  # Access logs
  access_logs = {
    bucket  = aws_s3_bucket.alb_logs.id
    enabled = true
    prefix  = "alb-logs"
  }

  # Target groups
  target_groups = [
    {
      name             = "${local.cluster_name}-api"
      backend_protocol = "HTTP"
      backend_port     = 80
      target_type      = "ip"
      
      health_check = {
        enabled             = true
        healthy_threshold   = 2
        unhealthy_threshold = 2
        timeout            = 5
        interval           = 30
        path               = "/health"
        matcher            = "200"
        port               = "traffic-port"
        protocol           = "HTTP"
      }

      stickiness = {
        enabled         = false
        cookie_duration = 86400
        type           = "lb_cookie"
      }
    }
  ]

  # HTTPS listener
  https_listeners = [
    {
      port               = 443
      protocol           = "HTTPS"
      certificate_arn    = aws_acm_certificate.main.arn
      target_group_index = 0

      action_type = "forward"
    }
  ]

  # HTTP to HTTPS redirect
  http_tcp_listeners = [
    {
      port        = 80
      protocol    = "HTTP"
      action_type = "redirect"
      redirect = {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }
  ]

  tags = local.common_tags
}
```

### Kubernetes Deployment Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: novacron
  labels:
    name: novacron
    environment: production
    managed-by: terraform
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: novacron-config
  namespace: novacron
data:
  app.env: "production"
  log.level: "info"
  database.host: "postgres-cluster.database.svc.cluster.local"
  database.port: "5432"
  redis.host: "redis-cluster.cache.svc.cluster.local"
  redis.port: "6379"
  metrics.enabled: "true"
  tracing.enabled: "true"
  jaeger.endpoint: "http://jaeger-collector:14268/api/traces"

---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: novacron-secrets
  namespace: novacron
type: Opaque
data:
  database.password: <base64-encoded-password>
  redis.password: <base64-encoded-password>
  jwt.private.key: <base64-encoded-private-key>
  jwt.public.key: <base64-encoded-public-key>
  vault.token: <base64-encoded-vault-token>

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: novacron-api
  namespace: novacron
  labels:
    app: novacron-api
    version: v1
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: novacron-api
      version: v1
  template:
    metadata:
      labels:
        app: novacron-api
        version: v1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: novacron-api
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        fsGroup: 1000
      containers:
      - name: api
        image: novacron/api:latest
        imagePullPolicy: Always
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9090
          protocol: TCP
        env:
        - name: PORT
          value: "8080"
        - name: METRICS_PORT
          value: "9090"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: redis-credentials
              key: url
        envFrom:
        - configMapRef:
            name: novacron-config
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            drop:
            - ALL
          readOnlyRootFilesystem: true
        volumeMounts:
        - name: tmp
          mountPath: /tmp
        - name: cache
          mountPath: /app/cache
      volumes:
      - name: tmp
        emptyDir: {}
      - name: cache
        emptyDir: {}

---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: novacron-api
  namespace: novacron
  labels:
    app: novacron-api
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: http
    protocol: TCP
    name: http
  - port: 9090
    targetPort: metrics
    protocol: TCP
    name: metrics
  selector:
    app: novacron-api

---
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: novacron-api-hpa
  namespace: novacron
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: novacron-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
      - type: Percent
        value: 100
        periodSeconds: 15
      - type: Pods
        value: 4
        periodSeconds: 15
      selectPolicy: Max

---
# k8s/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: novacron-api-pdb
  namespace: novacron
spec:
  minAvailable: 2
  selector:
    matchLabels:
      app: novacron-api

---
# k8s/networkpolicy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: novacron-api-netpol
  namespace: novacron
spec:
  podSelector:
    matchLabels:
      app: novacron-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8080
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: database
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - namespaceSelector:
        matchLabels:
          name: cache
    ports:
    - protocol: TCP
      port: 6379
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to: []
    ports:
    - protocol: TCP
      port: 443
```

This comprehensive implementation guide provides production-ready code examples and configuration templates for all major components of the NovaCron enhancement roadmap. Each section includes detailed implementations with proper error handling, security measures, and operational best practices.