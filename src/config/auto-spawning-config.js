/**
 * Smart Agent Auto-Spawning Configuration
 * Project-specific rules and settings for NovaCron
 */

const autoSpawningConfig = {
  // Global settings
  global: {
    enabled: true,
    maxAgents: 8,
    minAgents: 1,
    defaultTopology: 'auto', // auto, mesh, hierarchical, adaptive
    defaultStrategy: 'balanced', // minimal, balanced, optimal
    enableMCP: true
  },

  // File type detection rules
  fileTypeRules: {
    // Go backend files
    'backend/**/*.go': {
      agents: ['coder'],
      specialization: 'go-backend',
      capabilities: ['go-lang', 'distributed-systems', 'vm-management'],
      priority: 'high'
    },
    
    // Scheduler specific
    'backend/core/scheduler/**/*.go': {
      agents: ['coder', 'architect'],
      specialization: 'scheduler-expert',
      capabilities: ['go-lang', 'scheduling-algorithms', 'resource-allocation'],
      priority: 'critical'
    },
    
    // VM management
    'backend/core/vm/**/*.go': {
      agents: ['coder', 'system-architect'],
      specialization: 'vm-specialist',
      capabilities: ['go-lang', 'kvm', 'libvirt', 'virtualization'],
      priority: 'critical'
    },
    
    // Frontend React/TypeScript
    'frontend/**/*.tsx': {
      agents: ['coder'],
      specialization: 'react-frontend',
      capabilities: ['typescript', 'react', 'nextjs', 'websockets'],
      priority: 'high'
    },
    
    // API endpoints
    'backend/api/**/*.go': {
      agents: ['coder', 'api-specialist'],
      specialization: 'api-development',
      capabilities: ['go-lang', 'rest-api', 'grpc', 'authentication'],
      priority: 'high'
    },
    
    // Database migrations
    'database/migrations/**/*.sql': {
      agents: ['analyst', 'database-expert'],
      specialization: 'database-migration',
      capabilities: ['sql', 'postgresql', 'schema-design'],
      priority: 'high'
    },
    
    // Infrastructure as Code
    'deployment/**/*.yaml': {
      agents: ['coder', 'devops-engineer'],
      specialization: 'infrastructure',
      capabilities: ['kubernetes', 'docker', 'iac'],
      priority: 'medium'
    },
    
    // Tests
    'tests/**/*.go': {
      agents: ['tester'],
      specialization: 'go-testing',
      capabilities: ['go-lang', 'unit-testing', 'integration-testing'],
      priority: 'high'
    },
    
    'frontend/tests/**/*.{ts,tsx}': {
      agents: ['tester'],
      specialization: 'frontend-testing',
      capabilities: ['typescript', 'jest', 'playwright', 'e2e-testing'],
      priority: 'high'
    }
  },

  // Task complexity patterns
  complexityPatterns: {
    simple: {
      keywords: ['fix typo', 'update comment', 'rename', 'format', 'style'],
      agents: ['coordinator'],
      maxAgents: 1,
      topology: 'single'
    },
    
    medium: {
      keywords: ['add feature', 'refactor', 'optimize', 'update', 'modify'],
      agents: ['coordinator', 'coder'],
      maxAgents: 2,
      topology: 'mesh'
    },
    
    complex: {
      keywords: ['implement', 'design', 'architect', 'migrate', 'integrate'],
      agents: ['coordinator', 'architect', 'coder', 'tester'],
      maxAgents: 4,
      topology: 'hierarchical'
    },
    
    veryComplex: {
      keywords: ['oauth', 'authentication', 'distributed', 'microservices', 'real-time', 'scaling', 'multi-cloud'],
      agents: ['coordinator', 'architect', 'coder', 'tester', 'researcher', 'security-auditor'],
      maxAgents: 8,
      topology: 'adaptive'
    }
  },

  // Dynamic scaling thresholds
  scaling: {
    checkInterval: 5000, // 5 seconds
    scaleUpThreshold: 0.75, // 75% utilization
    scaleDownThreshold: 0.25, // 25% utilization
    cooldownPeriod: 30000, // 30 seconds between scaling actions
    maxScaleUpRate: 2, // Max 2x agents per scaling action
    maxScaleDownRate: 0.5 // Max 50% reduction per scaling action
  },

  // Agent specializations for NovaCron
  specializations: {
    'vm-management': {
      requiredCapabilities: ['kvm', 'libvirt', 'qemu', 'virtualization'],
      preferredAgents: ['system-architect', 'coder']
    },
    
    'migration-system': {
      requiredCapabilities: ['live-migration', 'wan-optimization', 'distributed-systems'],
      preferredAgents: ['architect', 'coder', 'performance-optimizer']
    },
    
    'scheduler-optimization': {
      requiredCapabilities: ['scheduling-algorithms', 'resource-allocation', 'constraint-solving'],
      preferredAgents: ['architect', 'coder', 'performance-optimizer']
    },
    
    'real-time-dashboard': {
      requiredCapabilities: ['react', 'websockets', 'real-time-ui', 'data-visualization'],
      preferredAgents: ['coder', 'ux-expert']
    },
    
    'api-development': {
      requiredCapabilities: ['rest-api', 'grpc', 'authentication', 'rate-limiting'],
      preferredAgents: ['coder', 'api-specialist', 'security-auditor']
    }
  },

  // Monitoring and metrics
  monitoring: {
    enabled: true,
    metricsRetention: 1000, // Keep last 1000 decisions
    logLevel: 'info', // debug, info, warn, error
    exportMetrics: true,
    metricsExportPath: './metrics/auto-spawning-metrics.json'
  }
};

module.exports = autoSpawningConfig;

