"""
Logging configuration for the AI Operations Engine.

Provides structured logging with JSON formatting, correlation IDs,
and integration with monitoring systems.
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

import structlog


def setup_logging(level: str = "INFO", format: str = "json", 
                 log_file: Optional[str] = None) -> None:
    """
    Setup structured logging for the AI Operations Engine.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Log format ('json' or 'console')
        log_file: Optional log file path
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if format == "json" else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )
    
    # Create custom formatter
    formatter = JSONFormatter() if format == "json" else ConsoleFormatter()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        root_logger.addHandler(file_handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def __init__(self):
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
        
        # Add request ID if available
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        
        # Add service context
        if hasattr(record, 'service_type'):
            log_entry['service_type'] = record.service_type
        
        if hasattr(record, 'model_id'):
            log_entry['model_id'] = record.model_id
        
        # Add extra fields
        if hasattr(record, 'extra') and record.extra:
            log_entry.update(record.extra)
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add stack trace if present
        if record.stack_info:
            log_entry['stack_info'] = record.stack_info
        
        return json.dumps(log_entry, default=str)


class ConsoleFormatter(logging.Formatter):
    """Custom console formatter for human-readable logging."""
    
    # Color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record for console output."""
        
        # Get color for level
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Format timestamp
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        # Base format
        log_parts = [
            f"{timestamp}",
            f"{color}{record.levelname:8s}{reset}",
            f"{record.name}",
            f"{record.getMessage()}"
        ]
        
        # Add context if available
        context_parts = []
        if hasattr(record, 'correlation_id'):
            context_parts.append(f"correlation_id={record.correlation_id}")
        
        if hasattr(record, 'request_id'):
            context_parts.append(f"request_id={record.request_id}")
        
        if hasattr(record, 'service_type'):
            context_parts.append(f"service={record.service_type}")
        
        if hasattr(record, 'model_id'):
            context_parts.append(f"model={record.model_id}")
        
        if context_parts:
            log_parts.append(f"[{', '.join(context_parts)}]")
        
        log_line = " | ".join(log_parts)
        
        # Add exception info if present
        if record.exc_info:
            log_line += "\n" + self.formatException(record.exc_info)
        
        return log_line


class AIEngineLogger:
    """Custom logger with AI engine specific functionality."""
    
    def __init__(self, name: str):
        """Initialize AI engine logger."""
        self.logger = structlog.get_logger(name)
        self.name = name
    
    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with context."""
        self.logger.info(message, **kwargs)
    
    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with context."""
        self.logger.debug(message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with context."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with context."""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with context."""
        self.logger.critical(message, **kwargs)
    
    def log_prediction(self, service_type: str, model_id: str, request_id: str,
                      response_time: float, success: bool, **kwargs: Any) -> None:
        """Log prediction request with standardized format."""
        self.logger.info(
            "Prediction request completed",
            service_type=service_type,
            model_id=model_id,
            request_id=request_id,
            response_time=response_time,
            success=success,
            **kwargs
        )
    
    def log_model_training(self, model_type: str, model_id: str, 
                          training_duration: float, metrics: Dict[str, Any],
                          success: bool, **kwargs: Any) -> None:
        """Log model training with standardized format."""
        self.logger.info(
            "Model training completed",
            model_type=model_type,
            model_id=model_id,
            training_duration=training_duration,
            metrics=metrics,
            success=success,
            **kwargs
        )
    
    def log_anomaly_detection(self, request_id: str, anomaly_types: list,
                            severity: str, confidence: float, **kwargs: Any) -> None:
        """Log anomaly detection with standardized format."""
        self.logger.warning(
            "Anomaly detected",
            request_id=request_id,
            anomaly_types=anomaly_types,
            severity=severity,
            confidence=confidence,
            **kwargs
        )
    
    def log_failure_prediction(self, request_id: str, node_id: str,
                             failure_probability: float, time_to_failure: int,
                             **kwargs: Any) -> None:
        """Log failure prediction with standardized format."""
        self.logger.warning(
            "Failure prediction alert",
            request_id=request_id,
            node_id=node_id,
            failure_probability=failure_probability,
            time_to_failure_minutes=time_to_failure,
            **kwargs
        )
    
    def log_resource_optimization(self, recommendation_id: str, action: str,
                                target_resources: Dict[str, float],
                                expected_savings: float, **kwargs: Any) -> None:
        """Log resource optimization recommendation."""
        self.logger.info(
            "Resource optimization recommendation",
            recommendation_id=recommendation_id,
            action=action,
            target_resources=target_resources,
            expected_savings=expected_savings,
            **kwargs
        )
    
    def log_placement_optimization(self, workload_id: str, recommended_node: str,
                                 placement_score: float, reasoning: list,
                                 **kwargs: Any) -> None:
        """Log workload placement optimization."""
        self.logger.info(
            "Workload placement recommendation",
            workload_id=workload_id,
            recommended_node=recommended_node,
            placement_score=placement_score,
            reasoning=reasoning,
            **kwargs
        )
    
    def log_performance_metrics(self, service_type: str, metrics: Dict[str, Any],
                              **kwargs: Any) -> None:
        """Log performance metrics."""
        self.logger.info(
            "Performance metrics",
            service_type=service_type,
            metrics=metrics,
            **kwargs
        )


class CorrelationIdFilter(logging.Filter):
    """Logging filter to add correlation IDs to log records."""
    
    def __init__(self):
        super().__init__()
        self._correlation_id: Optional[str] = None
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for current request."""
        self._correlation_id = correlation_id
    
    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        self._correlation_id = None
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        if self._correlation_id:
            record.correlation_id = self._correlation_id
        return True


# Global correlation ID filter instance
correlation_filter = CorrelationIdFilter()


def get_logger(name: str) -> AIEngineLogger:
    """Get AI engine logger instance."""
    return AIEngineLogger(name)


def set_correlation_id(correlation_id: str) -> None:
    """Set correlation ID for current request."""
    correlation_filter.set_correlation_id(correlation_id)


def clear_correlation_id() -> None:
    """Clear correlation ID."""
    correlation_filter.clear_correlation_id()


# Add correlation filter to root logger
logging.getLogger().addFilter(correlation_filter)