package middleware

import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/khryptorgraphics/novacron/backend/pkg/errors"
)

// ErrorResponse represents the structure of error responses
type ErrorResponse struct {
	Success   bool                `json:"success"`
	Error     *errors.AppError    `json:"error"`
	Timestamp time.Time           `json:"timestamp"`
	Path      string              `json:"path"`
	RequestID string              `json:"request_id,omitempty"`
	TraceID   string              `json:"trace_id,omitempty"`
}

// ErrorHandler is a middleware that handles panics and converts them to proper error responses
func ErrorHandler(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Set up panic recovery
		defer func() {
			if err := recover(); err != nil {
				var appErr *errors.AppError
				
				switch v := err.(type) {
				case *errors.AppError:
					appErr = v
				case error:
					appErr = errors.WrapError(v, errors.ErrInternal, "Internal server error")
				default:
					appErr = errors.NewAppError(errors.ErrInternal, "An unexpected error occurred")
				}
				
				// Add request context if available
				if reqID := GetRequestID(r); reqID != "" {
					appErr = appErr.WithRequestID(reqID)
				}
				
				SendErrorResponse(w, r, appErr)
			}
		}()
		
		// Continue with the next handler
		next.ServeHTTP(w, r)
	})
}

// SendErrorResponse sends a standardized error response
func SendErrorResponse(w http.ResponseWriter, r *http.Request, err *errors.AppError) {
	// Set content type
	w.Header().Set("Content-Type", "application/json")
	
	// Set status code
	statusCode := err.HTTPStatusCode()
	w.WriteHeader(statusCode)
	
	// Create response
	response := ErrorResponse{
		Success:   false,
		Error:     err,
		Timestamp: time.Now().UTC(),
		Path:      r.URL.Path,
		RequestID: GetRequestID(r),
		TraceID:   GetTraceID(r),
	}
	
	// Encode and send response
	if encErr := json.NewEncoder(w).Encode(response); encErr != nil {
		// If we can't encode the error response, send a basic error
		http.Error(w, "Internal server error", http.StatusInternalServerError)
	}
}

// GetRequestID extracts request ID from context or headers
func GetRequestID(r *http.Request) string {
	// First try to get from context (if set by request ID middleware)
	if reqID := r.Context().Value("request_id"); reqID != nil {
		if id, ok := reqID.(string); ok {
			return id
		}
	}
	
	// Fall back to header
	return r.Header.Get("X-Request-ID")
}

// GetTraceID extracts trace ID from context or headers
func GetTraceID(r *http.Request) string {
	// First try to get from context
	if traceID := r.Context().Value("trace_id"); traceID != nil {
		if id, ok := traceID.(string); ok {
			return id
		}
	}
	
	// Fall back to header
	return r.Header.Get("X-Trace-ID")
}

// JSONErrorHandler creates an HTTP handler that returns JSON errors
type JSONErrorHandler func(w http.ResponseWriter, r *http.Request) error

// ToHTTPHandler converts a JSONErrorHandler to a standard http.Handler
func (h JSONErrorHandler) ToHTTPHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if err := h(w, r); err != nil {
			var appErr *errors.AppError
			
			// Convert error to AppError if needed
			if errors.IsAppError(err) {
				appErr = err.(*errors.AppError)
			} else {
				appErr = errors.WrapError(err, errors.ErrInternal, "Request processing failed")
			}
			
			// Add request context
			if reqID := GetRequestID(r); reqID != "" {
				appErr = appErr.WithRequestID(reqID)
			}
			
			SendErrorResponse(w, r, appErr)
		}
	})
}

// ValidationErrorHandler handles validation errors specifically
func ValidationErrorHandler(field, message string) *errors.AppError {
	return errors.NewValidationError(field, message)
}

// DatabaseErrorHandler wraps database errors with appropriate context
func DatabaseErrorHandler(err error, operation string) *errors.AppError {
	if err == nil {
		return nil
	}
	
	// Check for specific database errors
	errMsg := err.Error()
	
	if containsAny(errMsg, []string{"connection", "timeout", "refused"}) {
		return errors.WrapError(err, errors.ErrDatabaseConnection, 
			"Database connection failed during "+operation)
	}
	
	if containsAny(errMsg, []string{"constraint", "duplicate", "unique"}) {
		return errors.WrapError(err, errors.ErrDatabaseConstraint,
			"Database constraint violation during "+operation)
	}
	
	// Generic database error
	return errors.WrapError(err, errors.ErrDatabaseQuery,
		"Database query failed during "+operation)
}

// VMErrorHandler wraps VM-related errors with appropriate context
func VMErrorHandler(err error, vmID, operation string) *errors.AppError {
	if err == nil {
		return nil
	}
	
	errMsg := err.Error()
	
	if containsAny(errMsg, []string{"not found", "does not exist"}) {
		return errors.NewNotFoundError("VM", vmID).WithMetadata(map[string]string{
			"operation": operation,
		})
	}
	
	if containsAny(errMsg, []string{"already running", "running"}) {
		return errors.WrapError(err, errors.ErrVMAlreadyRunning,
			"VM is already running").WithMetadata(map[string]string{
			"vm_id": vmID,
			"operation": operation,
		})
	}
	
	if containsAny(errMsg, []string{"not running", "stopped"}) {
		return errors.WrapError(err, errors.ErrVMNotRunning,
			"VM is not running").WithMetadata(map[string]string{
			"vm_id": vmID,
			"operation": operation,
		})
	}
	
	// Determine appropriate VM error code based on operation
	var code errors.ErrorCode
	switch operation {
	case "start":
		code = errors.ErrVMStartFailed
	case "stop":
		code = errors.ErrVMStopFailed
	case "create":
		code = errors.ErrVMCreateFailed
	case "migrate":
		code = errors.ErrVMMigrationFailed
	default:
		code = errors.ErrInternal
	}
	
	return errors.WrapError(err, code, "VM operation failed").WithMetadata(map[string]string{
		"vm_id": vmID,
		"operation": operation,
	})
}

// containsAny checks if a string contains any of the given substrings
func containsAny(s string, substrings []string) bool {
	for _, substr := range substrings {
		if len(s) >= len(substr) {
			for i := 0; i <= len(s)-len(substr); i++ {
				if s[i:i+len(substr)] == substr {
					return true
				}
			}
		}
	}
	return false
}