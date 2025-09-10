/**
 * NovaCron Claude-Flow Deployment Automation Configuration
 * Enterprise-grade canary deployment with comprehensive monitoring and rollback capabilities
 */

module.exports = {
  // Core Deployment Configuration
  deployment: {
    strategy: 'canary',
    application: 'novacron',
    version: process.env.BUILD_VERSION || 'latest',
    namespace: 'novacron-system',
    
    // Canary Deployment Settings
    canary: {
      enabled: true,
      analysis: 'automatic',
      trafficSplit: [5, 10, 25, 50, 100], // Percentage stages
      rolloutDuration: '1h',
      successThreshold: 95, // % success rate required
      
      // Traffic splitting configuration
      trafficSplitting: {
        method: 'header-based', // header-based, weight-based, geo-based
        duration: {
          '5%': '10m',   // 5% traffic for 10 minutes
          '10%': '10m',  // 10% traffic for 10 minutes  
          '25%': '15m',  // 25% traffic for 15 minutes
          '50%': '15m',  // 50% traffic for 15 minutes
          '100%': '10m'  // 100% traffic for 10 minutes (final)
        },
        
        // Canary analysis metrics
        metrics: {
          successRate: {
            threshold: 99.0,
            interval: '5m'
          },
          latency: {
            p95: 500, // ms
            p99: 1000, // ms
            interval: '5m'
          },
          errorRate: {
            threshold: 1.0, // %
            interval: '5m'
          }
        }
      }
    }
  },
  
  // Multi-Environment Pipeline
  environments: {
    dev: {
      enabled: true,
      cluster: 'novacron-dev-cluster',
      namespace: 'novacron-dev',
      replicas: 2,
      resources: {
        requests: { cpu: '100m', memory: '128Mi' },
        limits: { cpu: '500m', memory: '512Mi' }
      },
      autoDeploy: true,
      requiresApproval: false
    },
    
    qa: {
      enabled: true,
      cluster: 'novacron-qa-cluster', 
      namespace: 'novacron-qa',
      replicas: 3,
      resources: {
        requests: { cpu: '200m', memory: '256Mi' },
        limits: { cpu: '1000m', memory: '1Gi' }
      },
      autoDeploy: false,
      requiresApproval: true,
      approvers: ['qa-team@novacron.com']
    },
    
    staging: {
      enabled: true,
      cluster: 'novacron-staging-cluster',
      namespace: 'novacron-staging', 
      replicas: 5,
      resources: {
        requests: { cpu: '500m', memory: '512Mi' },
        limits: { cpu: '2000m', memory: '2Gi' }
      },
      autoDeploy: false,
      requiresApproval: true,
      approvers: ['staging-team@novacron.com', 'product-team@novacron.com']
    },
    
    production: {
      enabled: true,
      cluster: 'novacron-prod-cluster',
      namespace: 'novacron-prod',
      replicas: 10,
      resources: {
        requests: { cpu: '1000m', memory: '1Gi' },
        limits: { cpu: '4000m', memory: '4Gi' }
      },
      autoDeploy: false,
      requiresApproval: true,
      approvers: ['ops-team@novacron.com', 'security-team@novacron.com', 'cto@novacron.com'],
      canaryDeployment: true,
      blueGreenFallback: true
    }
  },
  
  // Infrastructure as Code (Terraform)
  infrastructureAsCode: {
    enabled: true,
    provider: 'terraform',
    version: '1.6.0',
    
    // Terraform Configuration  
    terraform: {
      backendType: 's3',
      backendConfig: {
        bucket: 'novacron-terraform-state',
        key: 'infrastructure/terraform.tfstate',
        region: 'us-west-2',
        dynamodbTable: 'novacron-terraform-locks',
        encrypt: true
      },
      
      // Resource definitions
      resources: {
        kubernetes: {
          clusters: {
            dev: { nodeCount: 3, machineType: 'e2-standard-4' },
            qa: { nodeCount: 3, machineType: 'e2-standard-4' },
            staging: { nodeCount: 5, machineType: 'e2-standard-8' },
            production: { nodeCount: 10, machineType: 'e2-standard-16' }
          }
        },
        
        networking: {
          vpc: true,
          subnets: ['public', 'private'],
          loadBalancers: true,
          cdn: true
        },
        
        storage: {
          persistentVolumes: true,
          backupStorage: true,
          archiveStorage: true
        },
        
        monitoring: {
          prometheus: true,
          grafana: true,
          elkStack: true,
          jaeger: true
        }
      }
    }
  },
  
  // Container Registry Configuration
  containerRegistry: {
    provider: 'gcr', // Google Container Registry
    registry: 'gcr.io/novacron-project',
    
    repositories: {
      frontend: 'novacron-frontend',
      backend: 'novacron-backend', 
      api: 'novacron-api',
      database: 'novacron-database',
      cache: 'novacron-cache'
    },
    
    // Image scanning and security
    security: {
      vulnerabilityScanning: true,
      binaryAuthorization: true,
      signedImages: true,
      allowedRegistries: [
        'gcr.io/novacron-project',
        'gcr.io/distroless'
      ]
    },
    
    // Image lifecycle management
    lifecycle: {
      retention: {
        dev: '7d',
        qa: '30d',
        staging: '90d', 
        production: '1y'
      },
      cleanup: {
        untaggedImages: '24h',
        oldImages: true
      }
    }
  },
  
  // Kubernetes Orchestration
  kubernetes: {
    enabled: true,
    version: '1.28.0',
    
    // Cluster configuration
    cluster: {
      nodeGroups: {
        system: { minSize: 3, maxSize: 5, instanceType: 'e2-standard-4' },
        application: { minSize: 5, maxSize: 20, instanceType: 'e2-standard-8' },
        monitoring: { minSize: 2, maxSize: 3, instanceType: 'e2-standard-4' }
      },
      
      networking: {
        cni: 'calico',
        serviceCIDR: '10.96.0.0/12',
        podCIDR: '10.244.0.0/16'
      }
    },
    
    // Workload configuration
    workloads: {
      deployments: {
        frontend: {
          replicas: 3,
          image: 'gcr.io/novacron-project/novacron-frontend',
          ports: [{ name: 'http', containerPort: 8092 }],
          resources: {
            requests: { cpu: '100m', memory: '256Mi' },
            limits: { cpu: '500m', memory: '512Mi' }
          }
        },
        
        backend: {
          replicas: 5,
          image: 'gcr.io/novacron-project/novacron-backend',
          ports: [{ name: 'http', containerPort: 8080 }],
          resources: {
            requests: { cpu: '500m', memory: '512Mi' },
            limits: { cpu: '2000m', memory: '2Gi' }
          }
        },
        
        api: {
          replicas: 7,
          image: 'gcr.io/novacron-project/novacron-api',
          ports: [{ name: 'http', containerPort: 3000 }],
          resources: {
            requests: { cpu: '300m', memory: '256Mi' },
            limits: { cpu: '1000m', memory: '1Gi' }
          }
        }
      },
      
      services: {
        frontend: { type: 'ClusterIP', port: 80, targetPort: 8092 },
        backend: { type: 'ClusterIP', port: 80, targetPort: 8080 },
        api: { type: 'ClusterIP', port: 80, targetPort: 3000 }
      },
      
      // Horizontal Pod Autoscaler
      hpa: {
        enabled: true,
        minReplicas: 3,
        maxReplicas: 20,
        targetCPUUtilizationPercentage: 70,
        targetMemoryUtilizationPercentage: 80
      }
    }
  },
  
  // Service Mesh (Istio)
  serviceMesh: {
    enabled: true,
    provider: 'istio',
    version: '1.19.0',
    
    // Istio configuration
    istio: {
      components: {
        pilot: true,
        gateway: true,
        egressGateway: true,
        istiod: true,
        proxy: true
      },
      
      // Traffic management
      trafficManagement: {
        virtualServices: {
          frontend: {
            hosts: ['novacron.com'],
            http: [{
              route: [{ destination: { host: 'frontend-service' } }]
            }]
          },
          
          api: {
            hosts: ['api.novacron.com'],
            http: [{
              route: [{ destination: { host: 'api-service' } }]
            }]
          }
        },
        
        destinationRules: {
          frontend: {
            host: 'frontend-service',
            trafficPolicy: {
              loadBalancer: { simple: 'LEAST_CONN' }
            }
          }
        },
        
        gateways: {
          public: {
            servers: [{
              port: { number: 443, name: 'https', protocol: 'HTTPS' },
              hosts: ['novacron.com', 'api.novacron.com'],
              tls: { mode: 'SIMPLE', credentialName: 'novacron-tls' }
            }]
          }
        }
      },
      
      // Security policies
      security: {
        authorizationPolicies: {
          'deny-all': {
            action: 'DENY',
            rules: [{}]
          },
          'allow-frontend': {
            action: 'ALLOW',
            rules: [{
              from: [{ source: { principals: ['cluster.local/ns/novacron-system/sa/frontend'] } }]
            }]
          }
        },
        
        peerAuthentication: {
          default: { mtls: { mode: 'STRICT' } }
        }
      }
    }
  },
  
  // Ingress Controller (NGINX)
  ingress: {
    enabled: true,
    controller: 'nginx',
    version: '1.8.0',
    
    // NGINX Ingress configuration
    nginx: {
      replicas: 3,
      resources: {
        requests: { cpu: '100m', memory: '90Mi' },
        limits: { cpu: '200m', memory: '256Mi' }
      },
      
      config: {
        'use-forwarded-headers': 'true',
        'compute-full-forwarded-for': 'true',
        'use-proxy-protocol': 'false',
        'enable-real-ip': 'true',
        'proxy-real-ip-cidr': '0.0.0.0/0'
      },
      
      // Ingress resources
      ingresses: {
        main: {
          host: 'novacron.com',
          paths: [{
            path: '/',
            pathType: 'Prefix',
            backend: {
              service: { name: 'frontend-service', port: { number: 80 } }
            }
          }],
          tls: [{
            secretName: 'novacron-tls',
            hosts: ['novacron.com']
          }]
        },
        
        api: {
          host: 'api.novacron.com',
          paths: [{
            path: '/',
            pathType: 'Prefix',
            backend: {
              service: { name: 'api-service', port: { number: 80 } }
            }
          }],
          tls: [{
            secretName: 'novacron-api-tls',
            hosts: ['api.novacron.com']
          }]
        }
      }
    }
  },
  
  // SSL Certificates (Let's Encrypt)
  ssl: {
    enabled: true,
    provider: 'letsencrypt',
    
    certManager: {
      version: '1.13.0',
      
      issuers: {
        letsencryptStaging: {
          server: 'https://acme-staging-v02.api.letsencrypt.org/directory',
          email: 'ssl@novacron.com',
          privateKeySecretRef: { name: 'letsencrypt-staging' }
        },
        
        letsencryptProd: {
          server: 'https://acme-v02.api.letsencrypt.org/directory',
          email: 'ssl@novacron.com',
          privateKeySecretRef: { name: 'letsencrypt-prod' }
        }
      },
      
      certificates: {
        novacron: {
          secretName: 'novacron-tls',
          issuerRef: { name: 'letsencrypt-prod' },
          dnsNames: ['novacron.com', 'www.novacron.com']
        },
        
        api: {
          secretName: 'novacron-api-tls',
          issuerRef: { name: 'letsencrypt-prod' },
          dnsNames: ['api.novacron.com']
        }
      }
    }
  },
  
  // DNS Provider (Cloudflare)
  dns: {
    enabled: true,
    provider: 'cloudflare',
    
    cloudflare: {
      apiToken: process.env.CLOUDFLARE_API_TOKEN,
      zoneId: process.env.CLOUDFLARE_ZONE_ID,
      
      records: {
        '@': { type: 'A', content: 'load-balancer-ip', ttl: 300 },
        'www': { type: 'CNAME', content: 'novacron.com', ttl: 300 },
        'api': { type: 'A', content: 'api-load-balancer-ip', ttl: 300 }
      },
      
      settings: {
        ssl: 'flexible',
        securityLevel: 'medium',
        cacheLevel: 'aggressive'
      }
    }
  },
  
  // CDN Provider (Fastly)
  cdn: {
    enabled: true,
    provider: 'fastly',
    
    fastly: {
      serviceId: process.env.FASTLY_SERVICE_ID,
      apiToken: process.env.FASTLY_API_TOKEN,
      
      backends: {
        origin: {
          address: 'novacron.com',
          port: 443,
          useSSL: true
        }
      },
      
      domains: {
        primary: 'novacron.com',
        www: 'www.novacron.com'
      },
      
      caching: {
        defaultTTL: 3600,
        staticAssets: 86400,
        api: 0
      },
      
      compression: {
        enabled: true,
        types: ['text/html', 'application/json', 'text/css', 'application/javascript']
      }
    }
  },
  
  // Comprehensive Monitoring Stack
  monitoring: {
    enabled: true,
    stack: ['prometheus', 'grafana', 'jaeger', 'elk'],
    
    // Prometheus configuration
    prometheus: {
      enabled: true,
      version: '2.47.0',
      retention: '30d',
      scrapeInterval: '30s',
      
      targets: [
        'kubernetes-api-server',
        'kubernetes-nodes', 
        'kubernetes-pods',
        'istio-mesh',
        'application-metrics'
      ],
      
      rules: {
        recording: [
          'instance:node_cpu_utilisation:rate1m',
          'instance:node_memory_utilisation:ratio'
        ],
        alerting: [
          'NodeCPUUsage',
          'NodeMemoryUsage',
          'PodCrashLooping',
          'KubeletDown'
        ]
      }
    },
    
    // Grafana configuration  
    grafana: {
      enabled: true,
      version: '10.2.0',
      
      datasources: [
        { name: 'Prometheus', type: 'prometheus', url: 'http://prometheus:9090' },
        { name: 'Jaeger', type: 'jaeger', url: 'http://jaeger-query:16686' },
        { name: 'Elasticsearch', type: 'elasticsearch', url: 'http://elasticsearch:9200' }
      ],
      
      dashboards: [
        'kubernetes-cluster-overview',
        'istio-service-mesh',
        'application-performance',
        'infrastructure-monitoring',
        'business-metrics'
      ]
    },
    
    // Jaeger distributed tracing
    jaeger: {
      enabled: true,
      version: '1.50.0',
      
      components: {
        agent: true,
        collector: true,
        query: true,
        ingester: true
      },
      
      storage: {
        type: 'elasticsearch',
        options: {
          serverUrls: 'http://elasticsearch:9200',
          indexPrefix: 'jaeger'
        }
      }
    },
    
    // ELK Stack (Elasticsearch, Logstash, Kibana)
    elk: {
      enabled: true,
      
      elasticsearch: {
        version: '8.11.0',
        replicas: 3,
        storage: '100Gi',
        retention: '30d'
      },
      
      logstash: {
        version: '8.11.0',
        replicas: 2,
        pipelines: ['kubernetes', 'application', 'istio']
      },
      
      kibana: {
        version: '8.11.0',
        dashboards: ['kubernetes', 'application-logs', 'security']
      },
      
      filebeat: {
        version: '8.11.0',
        inputs: ['kubernetes', 'docker']
      }
    }
  },
  
  // Comprehensive Alerting Rules
  alerting: {
    enabled: true,
    rules: 'comprehensive',
    
    // Alert manager configuration
    alertManager: {
      version: '0.26.0',
      
      global: {
        smtpSmarthost: 'localhost:587',
        smtpFrom: 'alerts@novacron.com'
      },
      
      routes: [
        {
          match: { severity: 'critical' },
          receiver: 'critical-alerts',
          groupWait: '10s',
          groupInterval: '5m',
          repeatInterval: '1h'
        },
        {
          match: { severity: 'warning' },
          receiver: 'warning-alerts',
          groupWait: '30s',
          groupInterval: '10m',
          repeatInterval: '4h'
        }
      ],
      
      receivers: [
        {
          name: 'critical-alerts',
          slackConfigs: [{ 
            apiUrl: process.env.SLACK_WEBHOOK_URL,
            channel: '#alerts-critical'
          }],
          pagerdutyConfigs: [{
            serviceKey: process.env.PAGERDUTY_SERVICE_KEY
          }]
        },
        {
          name: 'warning-alerts',
          emailConfigs: [{
            to: 'ops-team@novacron.com',
            subject: 'NovaCron Warning Alert'
          }]
        }
      ]
    },
    
    // Alert rules
    rules: {
      infrastructure: [
        {
          alert: 'NodeCPUUsage',
          expr: '100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80',
          for: '5m',
          labels: { severity: 'warning' },
          annotations: { summary: 'High CPU usage detected' }
        },
        {
          alert: 'NodeMemoryUsage', 
          expr: '(1 - node_memory_MemAvailable_bytes/node_memory_MemTotal_bytes) * 100 > 85',
          for: '5m',
          labels: { severity: 'warning' },
          annotations: { summary: 'High memory usage detected' }
        }
      ],
      
      application: [
        {
          alert: 'HighErrorRate',
          expr: 'rate(http_requests_total{status=~"5.."}[5m]) > 0.01',
          for: '2m',
          labels: { severity: 'critical' },
          annotations: { summary: 'High error rate detected' }
        },
        {
          alert: 'HighLatency',
          expr: 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 0.5',
          for: '5m',
          labels: { severity: 'warning' },
          annotations: { summary: 'High latency detected' }
        }
      ]
    }
  },
  
  // Pre-deployment Operations
  preDeployment: {
    // Backup before deployment
    backup: {
      enabled: true,
      targets: ['database', 'persistent-volumes', 'configurations'],
      
      database: {
        type: 'pg_dump',
        retention: '7d',
        compression: true,
        encryption: true
      },
      
      volumes: {
        type: 'snapshot',
        retention: '3d'
      }
    },
    
    // Database migrations
    databaseMigration: {
      enabled: true,
      automatic: true,
      
      migrations: {
        framework: 'flyway',
        location: 'db/migrations',
        validateOnMigrate: true,
        cleanDisabled: true
      },
      
      rollback: {
        enabled: true,
        strategy: 'automatic',
        maxRollbacks: 3
      }
    }
  },
  
  // Post-deployment Operations
  postDeployment: {
    // Cache warming
    cacheWarming: {
      enabled: true,
      
      strategies: [
        {
          name: 'static-assets',
          urls: ['/css/*', '/js/*', '/images/*'],
          concurrency: 10
        },
        {
          name: 'api-endpoints',
          urls: ['/api/v1/health', '/api/v1/status'],
          concurrency: 5
        }
      ]
    },
    
    // Health checks
    healthChecks: {
      types: ['liveness', 'readiness', 'startup'],
      
      liveness: {
        httpGet: { path: '/health/live', port: 8080 },
        initialDelaySeconds: 30,
        periodSeconds: 10,
        timeoutSeconds: 5,
        successThreshold: 1,
        failureThreshold: 3
      },
      
      readiness: {
        httpGet: { path: '/health/ready', port: 8080 },
        initialDelaySeconds: 5,
        periodSeconds: 5,
        timeoutSeconds: 3,
        successThreshold: 1,
        failureThreshold: 3
      },
      
      startup: {
        httpGet: { path: '/health/startup', port: 8080 },
        initialDelaySeconds: 10,
        periodSeconds: 5,
        timeoutSeconds: 3,
        successThreshold: 1,
        failureThreshold: 30
      }
    },
    
    // Smoke tests
    smokeTests: {
      enabled: true,
      
      tests: [
        { name: 'homepage-load', url: 'https://novacron.com', expectedStatus: 200 },
        { name: 'api-health', url: 'https://api.novacron.com/health', expectedStatus: 200 },
        { name: 'login-form', url: 'https://novacron.com/login', expectedStatus: 200 },
        { name: 'dashboard-access', url: 'https://novacron.com/dashboard', requiresAuth: true }
      ]
    },
    
    // Synthetic monitoring
    syntheticMonitoring: {
      enabled: true,
      
      checks: [
        {
          name: 'user-registration-flow',
          type: 'browser',
          frequency: '5m',
          locations: ['us-east-1', 'eu-west-1'],
          steps: [
            'navigate to /register',
            'fill registration form',
            'submit form',
            'verify success page'
          ]
        },
        {
          name: 'api-response-time',
          type: 'api',
          frequency: '1m',
          url: 'https://api.novacron.com/v1/vms',
          expectedResponseTime: 500
        }
      ]
    }
  },
  
  // Automated Rollback Triggers
  rollbackTriggers: [
    {
      metric: 'error-rate',
      threshold: '1%',
      duration: '2m',
      action: 'rollback',
      severity: 'critical'
    },
    {
      metric: 'p99-latency', 
      threshold: '500ms',
      duration: '5m',
      action: 'rollback',
      severity: 'warning'
    },
    {
      metric: 'cpu-usage',
      threshold: '80%',
      duration: '10m',
      action: 'alert',
      severity: 'warning'
    },
    {
      metric: 'memory-usage',
      threshold: '85%',
      duration: '5m', 
      action: 'rollback',
      severity: 'critical'
    },
    {
      metric: 'disk-usage',
      threshold: '90%',
      duration: '1m',
      action: 'rollback',
      severity: 'critical'
    }
  ],
  
  // Notification Channels
  notifications: {
    channels: ['slack', 'pagerduty', 'email', 'sms'],
    
    slack: {
      enabled: true,
      webhookUrl: process.env.SLACK_WEBHOOK_URL,
      channels: {
        deployments: '#deployments',
        alerts: '#alerts',
        critical: '#critical-alerts'
      }
    },
    
    pagerduty: {
      enabled: true,
      serviceKey: process.env.PAGERDUTY_SERVICE_KEY,
      escalationPolicy: 'novacron-ops',
      severity: ['critical', 'high']
    },
    
    email: {
      enabled: true,
      smtp: {
        host: 'smtp.novacron.com',
        port: 587,
        secure: false
      },
      recipients: {
        deployments: ['ops-team@novacron.com'],
        alerts: ['ops-team@novacron.com', 'dev-team@novacron.com'],
        critical: ['ops-team@novacron.com', 'oncall@novacron.com', 'cto@novacron.com']
      }
    },
    
    sms: {
      enabled: true,
      provider: 'twilio',
      recipients: ['+1-555-0100', '+1-555-0101'], // On-call numbers
      severity: ['critical']
    }
  },
  
  // Approval Gates
  approvalGates: [
    {
      name: 'security-scan',
      type: 'automated',
      required: true,
      tools: ['snyk', 'trivy', 'sonarqube'],
      criteria: {
        criticalVulnerabilities: 0,
        highVulnerabilities: 5,
        securityRating: 'A'
      }
    },
    {
      name: 'performance-test',
      type: 'automated', 
      required: true,
      criteria: {
        responseTime: '< 500ms',
        errorRate: '< 1%',
        throughput: '> 1000 rps'
      }
    },
    {
      name: 'manual-approval',
      type: 'manual',
      required: true,
      environments: ['staging', 'production'],
      approvers: ['ops-team@novacron.com', 'product-team@novacron.com'],
      requiredApprovals: 2,
      timeout: '24h'
    }
  ]
};