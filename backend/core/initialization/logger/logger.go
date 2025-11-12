// Package logger provides structured logging for NovaCron initialization
package logger

import (
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"
)

// Level represents log level
type Level int

const (
	LevelDebug Level = iota
	LevelInfo
	LevelWarn
	LevelError
)

// String returns string representation of level
func (l Level) String() string {
	switch l {
	case LevelDebug:
		return "DEBUG"
	case LevelInfo:
		return "INFO"
	case LevelWarn:
		return "WARN"
	case LevelError:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

// Logger provides structured logging
type Logger struct {
	mu       sync.RWMutex
	level    Level
	output   io.Writer
	fields   map[string]interface{}
	prefix   string
	loggers  []*log.Logger
}

// Config defines logger configuration
type Config struct {
	Level      string
	Output     io.Writer
	File       string
	EnableJSON bool
}

// NewLogger creates a new logger
func NewLogger(config Config) (*Logger, error) {
	level := parseLevel(config.Level)

	output := config.Output
	if output == nil {
		output = os.Stdout
	}

	// Open log file if specified
	if config.File != "" {
		f, err := os.OpenFile(config.File, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return nil, fmt.Errorf("failed to open log file: %w", err)
		}
		output = io.MultiWriter(output, f)
	}

	return &Logger{
		level:   level,
		output:  output,
		fields:  make(map[string]interface{}),
		loggers: make([]*log.Logger, 4),
	}, nil
}

// NewDefaultLogger creates a logger with default settings
func NewDefaultLogger() *Logger {
	logger, _ := NewLogger(Config{
		Level:  "info",
		Output: os.Stdout,
	})
	return logger
}

// WithFields returns a new logger with additional fields
func (l *Logger) WithFields(fields map[string]interface{}) *Logger {
	l.mu.RLock()
	defer l.mu.RUnlock()

	newFields := make(map[string]interface{})
	for k, v := range l.fields {
		newFields[k] = v
	}
	for k, v := range fields {
		newFields[k] = v
	}

	return &Logger{
		level:   l.level,
		output:  l.output,
		fields:  newFields,
		prefix:  l.prefix,
		loggers: l.loggers,
	}
}

// WithPrefix returns a new logger with a prefix
func (l *Logger) WithPrefix(prefix string) *Logger {
	l.mu.RLock()
	defer l.mu.RUnlock()

	return &Logger{
		level:   l.level,
		output:  l.output,
		fields:  l.fields,
		prefix:  prefix,
		loggers: l.loggers,
	}
}

// Debug logs a debug message
func (l *Logger) Debug(msg string, keysAndValues ...interface{}) {
	if l.level > LevelDebug {
		return
	}
	l.log(LevelDebug, msg, keysAndValues...)
}

// Info logs an info message
func (l *Logger) Info(msg string, keysAndValues ...interface{}) {
	if l.level > LevelInfo {
		return
	}
	l.log(LevelInfo, msg, keysAndValues...)
}

// Warn logs a warning message
func (l *Logger) Warn(msg string, keysAndValues ...interface{}) {
	if l.level > LevelWarn {
		return
	}
	l.log(LevelWarn, msg, keysAndValues...)
}

// Error logs an error message
func (l *Logger) Error(msg string, err error, keysAndValues ...interface{}) {
	if l.level > LevelError {
		return
	}
	// Prepend error to keys and values
	allKV := append([]interface{}{"error", err.Error()}, keysAndValues...)
	l.log(LevelError, msg, allKV...)
}

// log performs the actual logging
func (l *Logger) log(level Level, msg string, keysAndValues ...interface{}) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	// Build log entry
	timestamp := time.Now().Format("2006-01-02T15:04:05.000Z07:00")
	prefix := l.prefix
	if prefix != "" {
		prefix = "[" + prefix + "] "
	}

	// Combine fields and keysAndValues
	allFields := make(map[string]interface{})
	for k, v := range l.fields {
		allFields[k] = v
	}

	for i := 0; i < len(keysAndValues); i += 2 {
		if i+1 < len(keysAndValues) {
			key := fmt.Sprintf("%v", keysAndValues[i])
			allFields[key] = keysAndValues[i+1]
		}
	}

	// Format fields
	fieldsStr := ""
	if len(allFields) > 0 {
		fieldsStr = " "
		for k, v := range allFields {
			fieldsStr += fmt.Sprintf("%s=%v ", k, v)
		}
	}

	// Write log entry
	entry := fmt.Sprintf("%s [%s] %s%s%s\n", timestamp, level.String(), prefix, msg, fieldsStr)
	l.output.Write([]byte(entry))
}

// SetLevel sets the log level
func (l *Logger) SetLevel(level string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.level = parseLevel(level)
}

// GetLevel returns current log level
func (l *Logger) GetLevel() Level {
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.level
}

// parseLevel parses string level to Level
func parseLevel(level string) Level {
	switch level {
	case "debug":
		return LevelDebug
	case "info":
		return LevelInfo
	case "warn":
		return LevelWarn
	case "error":
		return LevelError
	default:
		return LevelInfo
	}
}

// Sync flushes any buffered log entries
func (l *Logger) Sync() error {
	if syncer, ok := l.output.(interface{ Sync() error }); ok {
		return syncer.Sync()
	}
	return nil
}

// Close closes the logger (if using file output)
func (l *Logger) Close() error {
	if closer, ok := l.output.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
