package logger

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strings"
	"time"
)

// Level represents a logging level
type Level int

const (
	DebugLevel Level = iota
	InfoLevel
	WarnLevel
	ErrorLevel
	FatalLevel
)

// String returns the string representation of a level
func (l Level) String() string {
	switch l {
	case DebugLevel:
		return "debug"
	case InfoLevel:
		return "info"
	case WarnLevel:
		return "warn"
	case ErrorLevel:
		return "error"
	case FatalLevel:
		return "fatal"
	default:
		return "unknown"
	}
}

// LevelFromString converts a string to a Level
func LevelFromString(s string) Level {
	switch strings.ToLower(s) {
	case "debug":
		return DebugLevel
	case "info":
		return InfoLevel
	case "warn", "warning":
		return WarnLevel
	case "error":
		return ErrorLevel
	case "fatal":
		return FatalLevel
	default:
		return InfoLevel
	}
}

// Logger represents a structured logger
type Logger struct {
	level      Level
	output     io.Writer
	structured bool
	service    string
	version    string
}

// Entry represents a log entry
type Entry struct {
	Timestamp   time.Time              `json:"timestamp"`
	Level       string                 `json:"level"`
	Message     string                 `json:"message"`
	Service     string                 `json:"service,omitempty"`
	Version     string                 `json:"version,omitempty"`
	RequestID   string                 `json:"request_id,omitempty"`
	TraceID     string                 `json:"trace_id,omitempty"`
	UserID      string                 `json:"user_id,omitempty"`
	TenantID    string                 `json:"tenant_id,omitempty"`
	File        string                 `json:"file,omitempty"`
	Function    string                 `json:"function,omitempty"`
	Line        int                    `json:"line,omitempty"`
	Fields      map[string]interface{} `json:"fields,omitempty"`
	Error       string                 `json:"error,omitempty"`
	StackTrace  []string               `json:"stack_trace,omitempty"`
}

// Config holds logger configuration
type Config struct {
	Level      string `json:"level"`
	Format     string `json:"format"`
	Output     string `json:"output"`
	Structured bool   `json:"structured"`
	Service    string `json:"service"`
	Version    string `json:"version"`
}

// New creates a new logger with the given configuration
func New(config Config) *Logger {
	var output io.Writer
	switch config.Output {
	case "stderr":
		output = os.Stderr
	case "stdout", "":
		output = os.Stdout
	default:
		// Assume it's a file path
		if file, err := os.OpenFile(config.Output, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666); err == nil {
			output = file
		} else {
			output = os.Stdout
		}
	}
	
	return &Logger{
		level:      LevelFromString(config.Level),
		output:     output,
		structured: config.Structured,
		service:    config.Service,
		version:    config.Version,
	}
}

// Default creates a logger with default configuration
func Default() *Logger {
	return New(Config{
		Level:      "info",
		Format:     "json",
		Output:     "stdout",
		Structured: true,
		Service:    "novacron",
		Version:    "1.0.0",
	})
}

// WithContext creates a new logger with context information
func (l *Logger) WithContext(ctx context.Context) *Logger {
	return &Logger{
		level:      l.level,
		output:     l.output,
		structured: l.structured,
		service:    l.service,
		version:    l.version,
	}
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, keysAndValues ...interface{}) {
	if l.level > DebugLevel {
		return
	}
	l.log(DebugLevel, msg, keysAndValues...)
}

// Info logs an info message
func (l *Logger) Info(msg string, keysAndValues ...interface{}) {
	if l.level > InfoLevel {
		return
	}
	l.log(InfoLevel, msg, keysAndValues...)
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, keysAndValues ...interface{}) {
	if l.level > WarnLevel {
		return
	}
	l.log(WarnLevel, msg, keysAndValues...)
}

// Error logs an error message
func (l *Logger) Error(msg string, keysAndValues ...interface{}) {
	if l.level > ErrorLevel {
		return
	}
	l.log(ErrorLevel, msg, keysAndValues...)
}

// Fatal logs a fatal message and exits
func (l *Logger) Fatal(msg string, keysAndValues ...interface{}) {
	l.log(FatalLevel, msg, keysAndValues...)
	os.Exit(1)
}

// log is the internal logging method
func (l *Logger) log(level Level, msg string, keysAndValues ...interface{}) {
	entry := Entry{
		Timestamp: time.Now().UTC(),
		Level:     level.String(),
		Message:   msg,
		Service:   l.service,
		Version:   l.version,
		Fields:    make(map[string]interface{}),
	}
	
	// Add caller information
	if pc, file, line, ok := runtime.Caller(2); ok {
		entry.File = file
		entry.Line = line
		if fn := runtime.FuncForPC(pc); fn != nil {
			entry.Function = fn.Name()
		}
	}
	
	// Process key-value pairs
	for i := 0; i < len(keysAndValues); i += 2 {
		if i+1 < len(keysAndValues) {
			key := fmt.Sprintf("%v", keysAndValues[i])
			value := keysAndValues[i+1]
			
			// Handle special keys
			switch key {
			case "request_id":
				entry.RequestID = fmt.Sprintf("%v", value)
			case "trace_id":
				entry.TraceID = fmt.Sprintf("%v", value)
			case "user_id":
				entry.UserID = fmt.Sprintf("%v", value)
			case "tenant_id":
				entry.TenantID = fmt.Sprintf("%v", value)
			case "error":
				if err, ok := value.(error); ok {
					entry.Error = err.Error()
					// Add stack trace for errors
					if level >= ErrorLevel {
						entry.StackTrace = getStackTrace()
					}
				} else {
					entry.Error = fmt.Sprintf("%v", value)
				}
			default:
				entry.Fields[key] = value
			}
		}
	}
	
	// Remove empty fields map
	if len(entry.Fields) == 0 {
		entry.Fields = nil
	}
	
	// Output the log entry
	if l.structured {
		if data, err := json.Marshal(entry); err == nil {
			fmt.Fprintln(l.output, string(data))
		}
	} else {
		l.outputPlain(entry)
	}
}

// outputPlain outputs a plain text log entry
func (l *Logger) outputPlain(entry Entry) {
	var parts []string
	
	// Timestamp
	parts = append(parts, entry.Timestamp.Format("2006-01-02 15:04:05"))
	
	// Level
	parts = append(parts, fmt.Sprintf("[%s]", strings.ToUpper(entry.Level)))
	
	// Request ID
	if entry.RequestID != "" {
		parts = append(parts, fmt.Sprintf("[%s]", entry.RequestID))
	}
	
	// Message
	parts = append(parts, entry.Message)
	
	// Fields
	if entry.Fields != nil {
		for key, value := range entry.Fields {
			parts = append(parts, fmt.Sprintf("%s=%v", key, value))
		}
	}
	
	// Error
	if entry.Error != "" {
		parts = append(parts, fmt.Sprintf("error=%s", entry.Error))
	}
	
	// File and line
	if entry.File != "" {
		parts = append(parts, fmt.Sprintf("%s:%d", entry.File, entry.Line))
	}
	
	fmt.Fprintln(l.output, strings.Join(parts, " "))
	
	// Stack trace on separate lines
	if entry.StackTrace != nil {
		for _, trace := range entry.StackTrace {
			fmt.Fprintln(l.output, "  ", trace)
		}
	}
}

// getStackTrace captures the current stack trace
func getStackTrace() []string {
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
		if len(st) >= 10 { // Limit stack trace depth
			break
		}
	}
	return st
}

// ContextLogger extracts logger with context information
func ContextLogger(ctx context.Context, logger *Logger) *Logger {
	// This could be enhanced to extract context information
	// and create a logger with pre-populated fields
	return logger.WithContext(ctx)
}

// GlobalLogger is the default logger instance
var GlobalLogger = Default()

// SetGlobalLogger sets the global logger instance
func SetGlobalLogger(logger *Logger) {
	GlobalLogger = logger
}

// Debug logs to the global logger
func Debug(msg string, keysAndValues ...interface{}) {
	GlobalLogger.Debug(msg, keysAndValues...)
}

// Info logs to the global logger
func Info(msg string, keysAndValues ...interface{}) {
	GlobalLogger.Info(msg, keysAndValues...)
}

// Warn logs to the global logger
func Warn(msg string, keysAndValues ...interface{}) {
	GlobalLogger.Warn(msg, keysAndValues...)
}

// Error logs to the global logger
func Error(msg string, keysAndValues ...interface{}) {
	GlobalLogger.Error(msg, keysAndValues...)
}

// Fatal logs to the global logger and exits
func Fatal(msg string, keysAndValues ...interface{}) {
	GlobalLogger.Fatal(msg, keysAndValues...)
}

// Sync flushes any buffered log entries
func Sync() {
	// In this implementation, we write directly, so no sync needed
	// This could be enhanced for buffered loggers
}

// SetLevel sets the logging level for the global logger
func SetLevel(level string) {
	GlobalLogger.level = LevelFromString(level)
}

// NewFromConfig creates a logger from a configuration struct
func NewFromConfig(level, format, output string, structured bool) *Logger {
	return New(Config{
		Level:      level,
		Format:     format,
		Output:     output,
		Structured: structured,
		Service:    "novacron",
		Version:    "1.0.0",
	})
}

// Compatible with standard log package
var _ = log.Logger{}