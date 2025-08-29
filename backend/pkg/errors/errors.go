package errors

import (
	"encoding/json"
	"fmt"
	"net/http"
	"runtime"
	"time"
)

// ErrorCode represents a structured error code
type ErrorCode string

// Standard error codes for the system
const (
	// General errors
	ErrInternal      ErrorCode = "INTERNAL_ERROR"
	ErrInvalidInput  ErrorCode = "INVALID_INPUT"
	ErrNotFound      ErrorCode = "NOT_FOUND"
	ErrUnauthorized  ErrorCode = "UNAUTHORIZED"
	ErrForbidden     ErrorCode = "FORBIDDEN"
	ErrConflict      ErrorCode = "CONFLICT"
	ErrTimeout       ErrorCode = "TIMEOUT"
	
	// VM-specific errors
	ErrVMNotFound       ErrorCode = "VM_NOT_FOUND"
	ErrVMAlreadyRunning ErrorCode = "VM_ALREADY_RUNNING"
	ErrVMNotRunning     ErrorCode = "VM_NOT_RUNNING"
	ErrVMCreateFailed   ErrorCode = "VM_CREATE_FAILED"
	ErrVMStartFailed    ErrorCode = "VM_START_FAILED"
	ErrVMStopFailed     ErrorCode = "VM_STOP_FAILED"
	ErrVMMigrationFailed ErrorCode = "VM_MIGRATION_FAILED"
	
	// Auth-specific errors
	ErrInvalidCredentials ErrorCode = "INVALID_CREDENTIALS"
	ErrSessionExpired     ErrorCode = "SESSION_EXPIRED"
	ErrInvalidToken       ErrorCode = "INVALID_TOKEN"
	ErrPasswordPolicy     ErrorCode = "PASSWORD_POLICY_VIOLATION"
	
	// Resource errors
	ErrResourceNotFound    ErrorCode = "RESOURCE_NOT_FOUND"
	ErrResourceExhausted   ErrorCode = "RESOURCE_EXHAUSTED"
	ErrResourceConstraint  ErrorCode = "RESOURCE_CONSTRAINT"
	
	// Database errors
	ErrDatabaseConnection ErrorCode = "DATABASE_CONNECTION"
	ErrDatabaseQuery      ErrorCode = "DATABASE_QUERY"
	ErrDatabaseConstraint ErrorCode = "DATABASE_CONSTRAINT"
)

// AppError represents a structured application error
type AppError struct {
	Code        ErrorCode   `json:"code"`
	Message     string      `json:"message"`
	Details     string      `json:"details,omitempty"`
	Timestamp   time.Time   `json:"timestamp"`
	RequestID   string      `json:"request_id,omitempty"`
	UserID      string      `json:"user_id,omitempty"`
	StackTrace  []string    `json:"stack_trace,omitempty"`
	Cause       error       `json:"-"`
	HTTPStatus  int         `json:"-"`
	Metadata    interface{} `json:"metadata,omitempty"`
}

// Error implements the error interface
func (e *AppError) Error() string {
	if e.Details != "" {
		return fmt.Sprintf("%s: %s - %s", e.Code, e.Message, e.Details)
	}
	return fmt.Sprintf("%s: %s", e.Code, e.Message)
}

// Unwrap returns the underlying cause
func (e *AppError) Unwrap() error {
	return e.Cause
}

// MarshalJSON implements json.Marshaler
func (e *AppError) MarshalJSON() ([]byte, error) {
	type Alias AppError
	return json.Marshal(&struct {
		*Alias
		Error string `json:"error"`
	}{
		Alias: (*Alias)(e),
		Error: e.Error(),
	})
}

// NewAppError creates a new application error
func NewAppError(code ErrorCode, message string) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		Timestamp:  time.Now().UTC(),
		HTTPStatus: getHTTPStatusFromCode(code),
	}
}

// NewAppErrorWithDetails creates a new application error with details
func NewAppErrorWithDetails(code ErrorCode, message, details string) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		Details:    details,
		Timestamp:  time.Now().UTC(),
		HTTPStatus: getHTTPStatusFromCode(code),
	}
}

// NewAppErrorWithCause creates a new application error with an underlying cause
func NewAppErrorWithCause(code ErrorCode, message string, cause error) *AppError {
	return &AppError{
		Code:       code,
		Message:    message,
		Cause:      cause,
		Timestamp:  time.Now().UTC(),
		HTTPStatus: getHTTPStatusFromCode(code),
		StackTrace: captureStackTrace(),
	}
}

// WithRequestID adds a request ID to the error
func (e *AppError) WithRequestID(requestID string) *AppError {
	e.RequestID = requestID
	return e
}

// WithUserID adds a user ID to the error
func (e *AppError) WithUserID(userID string) *AppError {
	e.UserID = userID
	return e
}

// WithMetadata adds metadata to the error
func (e *AppError) WithMetadata(metadata interface{}) *AppError {
	e.Metadata = metadata
	return e
}

// WithHTTPStatus sets a custom HTTP status code
func (e *AppError) WithHTTPStatus(status int) *AppError {
	e.HTTPStatus = status
	return e
}

// IsCode checks if the error has a specific code
func (e *AppError) IsCode(code ErrorCode) bool {
	return e.Code == code
}

// HTTPStatusCode returns the HTTP status code for this error
func (e *AppError) HTTPStatusCode() int {
	if e.HTTPStatus != 0 {
		return e.HTTPStatus
	}
	return getHTTPStatusFromCode(e.Code)
}

// getHTTPStatusFromCode maps error codes to HTTP status codes
func getHTTPStatusFromCode(code ErrorCode) int {
	switch code {
	case ErrInvalidInput, ErrPasswordPolicy:
		return http.StatusBadRequest
	case ErrUnauthorized, ErrInvalidCredentials, ErrSessionExpired, ErrInvalidToken:
		return http.StatusUnauthorized
	case ErrForbidden:
		return http.StatusForbidden
	case ErrNotFound, ErrVMNotFound, ErrResourceNotFound:
		return http.StatusNotFound
	case ErrConflict, ErrVMAlreadyRunning, ErrDatabaseConstraint:
		return http.StatusConflict
	case ErrTimeout:
		return http.StatusRequestTimeout
	case ErrResourceExhausted:
		return http.StatusTooManyRequests
	case ErrResourceConstraint:
		return http.StatusUnprocessableEntity
	default:
		return http.StatusInternalServerError
	}
}

// captureStackTrace captures the current stack trace
func captureStackTrace() []string {
	const depth = 32
	var pcs [depth]uintptr
	n := runtime.Callers(3, pcs[:])
	
	var st []string
	frames := runtime.CallersFrames(pcs[:n])
	for {
		frame, more := frames.Next()
		st = append(st, fmt.Sprintf("%s:%d %s", frame.File, frame.Line, frame.Function))
		if !more {
			break
		}
	}
	return st
}

// WrapError wraps an existing error with additional context
func WrapError(err error, code ErrorCode, message string) *AppError {
	if err == nil {
		return nil
	}
	
	if appErr, ok := err.(*AppError); ok {
		// If it's already an AppError, add context
		return &AppError{
			Code:       code,
			Message:    message,
			Cause:      appErr,
			Timestamp:  time.Now().UTC(),
			HTTPStatus: getHTTPStatusFromCode(code),
			StackTrace: captureStackTrace(),
		}
	}
	
	return NewAppErrorWithCause(code, message, err)
}

// IsAppError checks if an error is an AppError
func IsAppError(err error) bool {
	_, ok := err.(*AppError)
	return ok
}

// GetAppError extracts an AppError from any error
func GetAppError(err error) *AppError {
	if appErr, ok := err.(*AppError); ok {
		return appErr
	}
	
	// Convert generic error to AppError
	return NewAppErrorWithCause(ErrInternal, "Internal server error", err)
}

// Common error constructors for frequently used patterns

// NewValidationError creates a validation error
func NewValidationError(field, message string) *AppError {
	return NewAppErrorWithDetails(ErrInvalidInput, 
		"Validation failed", 
		fmt.Sprintf("Field '%s': %s", field, message))
}

// NewNotFoundError creates a not found error
func NewNotFoundError(resource, id string) *AppError {
	return NewAppErrorWithDetails(ErrNotFound,
		fmt.Sprintf("%s not found", resource),
		fmt.Sprintf("Resource '%s' with ID '%s' does not exist", resource, id))
}

// NewUnauthorizedError creates an unauthorized error
func NewUnauthorizedError(message string) *AppError {
	return NewAppError(ErrUnauthorized, message)
}

// NewForbiddenError creates a forbidden error
func NewForbiddenError(resource string) *AppError {
	return NewAppError(ErrForbidden, fmt.Sprintf("Access to %s is forbidden", resource))
}

// NewTimeoutError creates a timeout error
func NewTimeoutError(operation string) *AppError {
	return NewAppError(ErrTimeout, fmt.Sprintf("Operation '%s' timed out", operation))
}

// NewResourceExhaustedError creates a resource exhausted error
func NewResourceExhaustedError(resource string) *AppError {
	return NewAppError(ErrResourceExhausted, fmt.Sprintf("Resource '%s' exhausted", resource))
}