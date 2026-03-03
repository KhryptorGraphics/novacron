# NovaCron Configuration Schema

**Document Type**: Configuration Reference
**Version**: 1.0
**Date**: 2025-11-14

---

## Overview

This document defines the complete configuration schema for NovaCron initialization, including all configuration sections, validation rules, and environment variable mappings.

---

## Configuration File Locations

### Development
- **Default Config**: `src/config/config.default.json`
- **Environment Config**: `src/config/config.development.json`

### Production
- **Default Config**: `/etc/novacron/config.default.json`
- **Environment Config**: `/etc/novacron/config.production.json`
- **Secrets**: Via environment variables or secrets manager

---

## Complete Configuration Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "NovaCron Configuration",
  "type": "object",
  "required": ["environment", "platform", "database"],
  "properties": {
    "environment": {
      "type": "string",
      "enum": ["development", "staging", "production"],
      "description": "Deployment environment"
    },
    "platform": {
      "type": "object",
      "required": ["name", "version"],
      "properties": {
        "name": {
          "type": "string",
          "const": "NovaCron"
        },
        "version": {
          "type": "string",
          "pattern": "^\\d+\\.\\d+\\.\\d+$"
        },
        "nodeId": {
          "type": "string",
          "description": "Unique node identifier in cluster"
        },
        "description": {
          "type": "string"
        }
      }
    },
    "system": {
      "type": "object",
      "properties": {
        "dataDir": {
          "type": "string",
          "default": "/var/lib/novacron"
        },
        "logLevel": {
          "type": "string",
          "enum": ["debug", "info", "warn", "error"],
          "default": "info"
        },
        "maxConcurrency": {
          "type": "integer",
          "minimum": 1,
          "maximum": 16,
          "default": 4
        },
        "shutdownTimeout": {
          "type": "string",
          "pattern": "^\\d+[smh]$",
          "default": "30s"
        }
      }
    },
    "database": {
      "type": "object",
      "required": ["postgres"],
      "properties": {
        "postgres": {
          "type": "object",
          "required": ["host", "port", "database", "user"],
          "properties": {
            "host": {
              "type": "string"
            },
            "port": {
              "type": "integer",
              "minimum": 1,
              "maximum": 65535,
              "default": 5432
            },
            "database": {
              "type": "string"
            },
            "user": {
              "type": "string"
            },
            "password": {
              "type": "string",
              "description": "Use ${ENV_VAR} for environment variable substitution"
            },
            "poolSize": {
              "type": "integer",
              "minimum": 1,
              "maximum": 100,
              "default": 10
            },
            "maxIdleConns": {
              "type": "integer",
              "minimum": 0,
              "default": 5
            },
            "connMaxLifetime": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "1h"
            },
            "sslMode": {
              "type": "string",
              "enum": ["disable", "require", "verify-ca", "verify-full"],
              "default": "require"
            }
          }
        },
        "redis": {
          "type": "object",
          "properties": {
            "host": {
              "type": "string",
              "default": "localhost"
            },
            "port": {
              "type": "integer",
              "minimum": 1,
              "maximum": 65535,
              "default": 6379
            },
            "password": {
              "type": "string"
            },
            "database": {
              "type": "integer",
              "minimum": 0,
              "maximum": 15,
              "default": 0
            },
            "poolSize": {
              "type": "integer",
              "minimum": 1,
              "default": 10
            }
          }
        }
      }
    },
    "dwcp": {
      "type": "object",
      "properties": {
        "enabled": {
          "type": "boolean",
          "default": false
        },
        "transport": {
          "type": "object",
          "properties": {
            "minStreams": {
              "type": "integer",
              "minimum": 1,
              "default": 16
            },
            "maxStreams": {
              "type": "integer",
              "maximum": 1024,
              "default": 256
            },
            "initialStreams": {
              "type": "integer",
              "default": 32
            },
            "streamScalingFactor": {
              "type": "number",
              "minimum": 1.0,
              "default": 1.5
            },
            "congestionAlgorithm": {
              "type": "string",
              "enum": ["bbr", "cubic", "reno"],
              "default": "bbr"
            },
            "enableECN": {
              "type": "boolean",
              "default": true
            },
            "sendBufferSize": {
              "type": "integer",
              "default": 16777216
            },
            "recvBufferSize": {
              "type": "integer",
              "default": 16777216
            },
            "enableRDMA": {
              "type": "boolean",
              "default": false
            },
            "rdmaDevice": {
              "type": "string"
            }
          }
        },
        "compression": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "algorithm": {
              "type": "string",
              "enum": ["zstd", "lz4", "snappy"],
              "default": "zstd"
            },
            "level": {
              "type": "string",
              "enum": ["none", "fast", "balanced", "max"],
              "default": "balanced"
            },
            "enableDeltaEncoding": {
              "type": "boolean",
              "default": true
            },
            "enableDictionary": {
              "type": "boolean",
              "default": true
            },
            "dictionaryUpdateInterval": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "1h"
            }
          }
        },
        "prediction": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "modelType": {
              "type": "string",
              "enum": ["lstm", "arima", "prophet"],
              "default": "lstm"
            },
            "predictionHorizon": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "5m"
            },
            "updateInterval": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "1m"
            }
          }
        },
        "consensus": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "algorithm": {
              "type": "string",
              "enum": ["raft", "gossip", "byzantine"],
              "default": "raft"
            },
            "quorumSize": {
              "type": "integer",
              "minimum": 1,
              "default": 3
            },
            "electionTimeout": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "5s"
            },
            "heartbeatInterval": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "1s"
            }
          }
        }
      }
    },
    "security": {
      "type": "object",
      "properties": {
        "zeroTrust": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "continuousAuthentication": {
              "type": "boolean",
              "default": true
            },
            "maxTrustDuration": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "4h"
            },
            "microSegmentation": {
              "type": "boolean",
              "default": true
            }
          }
        },
        "encryption": {
          "type": "object",
          "properties": {
            "algorithm": {
              "type": "string",
              "enum": ["AES-256-GCM", "ChaCha20-Poly1305"],
              "default": "AES-256-GCM"
            },
            "keyRotationEnabled": {
              "type": "boolean",
              "default": true
            },
            "keyRotationInterval": {
              "type": "string",
              "pattern": "^\\d+[smh]$",
              "default": "24h"
            }
          }
        },
        "audit": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "retentionDays": {
              "type": "integer",
              "minimum": 1,
              "default": 90
            },
            "logSigning": {
              "type": "boolean",
              "default": true
            }
          }
        },
        "jwt": {
          "type": "object",
          "properties": {
            "expiresIn": {
              "type": "string",
              "default": "24h"
            },
            "refreshExpiresIn": {
              "type": "string",
              "default": "7d"
            }
          }
        }
      }
    },
    "api": {
      "type": "object",
      "properties": {
        "host": {
          "type": "string",
          "default": "0.0.0.0"
        },
        "port": {
          "type": "integer",
          "minimum": 1,
          "maximum": 65535,
          "default": 8080
        },
        "cors": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "allowedOrigins": {
              "type": "array",
              "items": {
                "type": "string"
              },
              "default": ["*"]
            }
          }
        },
        "rateLimit": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "requestsPerMinute": {
              "type": "integer",
              "minimum": 1,
              "default": 1000
            }
          }
        }
      }
    },
    "monitoring": {
      "type": "object",
      "properties": {
        "metricsPort": {
          "type": "integer",
          "default": 9090
        },
        "healthCheckPort": {
          "type": "integer",
          "default": 8081
        },
        "prometheusEnabled": {
          "type": "boolean",
          "default": true
        }
      }
    },
    "logging": {
      "type": "object",
      "properties": {
        "level": {
          "type": "string",
          "enum": ["debug", "info", "warn", "error"],
          "default": "info"
        },
        "format": {
          "type": "string",
          "enum": ["json", "text"],
          "default": "json"
        },
        "destination": {
          "type": "string",
          "enum": ["console", "file", "both"],
          "default": "console"
        },
        "file": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": false
            },
            "path": {
              "type": "string",
              "default": "./logs"
            },
            "maxSize": {
              "type": "string",
              "default": "10m"
            },
            "maxFiles": {
              "type": "integer",
              "default": 5
            }
          }
        }
      }
    },
    "services": {
      "type": "object",
      "properties": {
        "cache": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "ttl": {
              "type": "integer",
              "default": 3600
            },
            "maxSize": {
              "type": "integer",
              "default": 1000
            }
          }
        },
        "workload-monitor": {
          "type": "object",
          "properties": {
            "enabled": {
              "type": "boolean",
              "default": true
            },
            "interval": {
              "type": "integer",
              "default": 5000
            },
            "thresholds": {
              "type": "object",
              "properties": {
                "cpu": {
                  "type": "integer",
                  "default": 80
                },
                "memory": {
                  "type": "integer",
                  "default": 85
                },
                "disk": {
                  "type": "integer",
                  "default": 90
                }
              }
            }
          }
        }
      }
    },
    "features": {
      "type": "object",
      "properties": {
        "mlEngineering": {
          "type": "boolean",
          "default": true
        },
        "vmManagement": {
          "type": "boolean",
          "default": true
        },
        "distributedSystems": {
          "type": "boolean",
          "default": true
        },
        "autoScaling": {
          "type": "boolean",
          "default": true
        },
        "monitoring": {
          "type": "boolean",
          "default": true
        }
      }
    }
  }
}
```

---

## Environment Variable Mapping

### Naming Convention

**Pattern**: `NOVACRON_<SECTION>_<SUBSECTION>_<KEY>`

**Examples**:
```bash
# Platform
NOVACRON_PLATFORM_NODEID="node-001"

# Database
NOVACRON_DATABASE_POSTGRES_HOST="db.example.com"
NOVACRON_DATABASE_POSTGRES_PORT="5432"
NOVACRON_DATABASE_POSTGRES_PASSWORD="secure_password"

# DWCP
NOVACRON_DWCP_ENABLED="true"
NOVACRON_DWCP_TRANSPORT_MAXSTREAMS="512"
NOVACRON_DWCP_COMPRESSION_LEVEL="max"

# Security
NOVACRON_SECURITY_ZEROTRUST_ENABLED="true"
NOVACRON_SECURITY_ENCRYPTION_ALGORITHM="AES-256-GCM"

# API
NOVACRON_API_PORT="8080"
NOVACRON_API_CORS_ALLOWEDORIGINS='["https://app.example.com"]'
```

### Variable Substitution in Config Files

Use `${ENV_VAR}` syntax in JSON/YAML files:

```json
{
  "database": {
    "postgres": {
      "password": "${POSTGRES_PASSWORD}",
      "host": "${POSTGRES_HOST:-localhost}"
    }
  }
}
```

**Default values**: Use `${VAR:-default}` syntax

---

## Configuration Validation

### Required Fields

**Critical**:
- `environment`
- `platform.name`
- `platform.version`
- `database.postgres.host`
- `database.postgres.database`
- `database.postgres.user`

### Validation Rules

1. **Port Numbers**: 1-65535
2. **Durations**: Must match pattern `\d+[smh]` (e.g., "30s", "5m", "2h")
3. **Pool Sizes**: Minimum 1, reasonable maximums
4. **SSL Mode**: Must be valid PostgreSQL SSL mode
5. **Algorithms**: Must be from enumerated list

### Cross-Field Validation

```javascript
// DWCP: minStreams <= initialStreams <= maxStreams
if (config.dwcp.transport.minStreams > config.dwcp.transport.initialStreams) {
    throw new Error("minStreams cannot exceed initialStreams");
}

// Database: maxIdleConns <= poolSize
if (config.database.postgres.maxIdleConns > config.database.postgres.poolSize) {
    throw new Error("maxIdleConns cannot exceed poolSize");
}
```

---

## Configuration Examples

### Development Configuration

```json
{
  "environment": "development",
  "platform": {
    "name": "NovaCron",
    "version": "1.0.0",
    "nodeId": "dev-node-1"
  },
  "database": {
    "postgres": {
      "host": "localhost",
      "port": 5432,
      "database": "novacron_dev",
      "user": "dev_user",
      "password": "dev_password",
      "sslMode": "disable"
    }
  },
  "dwcp": {
    "enabled": false
  },
  "security": {
    "zeroTrust": {
      "enabled": false
    }
  },
  "logging": {
    "level": "debug",
    "format": "text"
  }
}
```

### Production Configuration

```json
{
  "environment": "production",
  "platform": {
    "name": "NovaCron",
    "version": "1.0.0",
    "nodeId": "${NODE_ID}"
  },
  "database": {
    "postgres": {
      "host": "${POSTGRES_HOST}",
      "port": 5432,
      "database": "novacron",
      "user": "novacron",
      "password": "${POSTGRES_PASSWORD}",
      "poolSize": 50,
      "sslMode": "verify-full"
    },
    "redis": {
      "host": "${REDIS_HOST}",
      "password": "${REDIS_PASSWORD}",
      "poolSize": 20
    }
  },
  "dwcp": {
    "enabled": true,
    "transport": {
      "maxStreams": 512,
      "congestionAlgorithm": "bbr",
      "enableRDMA": true
    },
    "compression": {
      "level": "max",
      "enableDictionary": true
    }
  },
  "security": {
    "zeroTrust": {
      "enabled": true,
      "continuousAuthentication": true
    },
    "encryption": {
      "keyRotationEnabled": true
    }
  },
  "monitoring": {
    "prometheusEnabled": true
  },
  "logging": {
    "level": "info",
    "format": "json",
    "destination": "both"
  }
}
```

---

## Document Control

**Version**: 1.0
**Author**: System Architecture Designer
**Date**: 2025-11-14
