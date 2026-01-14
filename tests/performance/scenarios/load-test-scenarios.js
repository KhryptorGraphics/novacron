/**
 * Database Load Testing Scenarios
 * Comprehensive test scenarios for different database workload patterns
 */

const loadTestScenarios = {
  // OLTP (Online Transaction Processing) Scenarios
  oltp: [
    {
      name: "User Authentication Load",
      type: "oltp",
      concurrency: 50,
      duration: 30000,
      operation: {
        type: "select",
        complexity: "low",
        resultSize: "small",
        query: "SELECT * FROM users WHERE email = ? AND password_hash = ?",
        indexUsage: true
      },
      delayMs: 10,
      description: "Simulates user login requests with email/password lookup"
    },
    {
      name: "Order Processing",
      type: "oltp",
      concurrency: 30,
      duration: 45000,
      operation: {
        type: "transaction",
        operations: [
          {
            type: "insert",
            batchSize: 1,
            table: "orders"
          },
          {
            type: "insert",
            batchSize: 3,
            table: "order_items"
          },
          {
            type: "update",
            table: "inventory",
            whereCondition: true
          }
        ]
      },
      description: "Complete order processing with inventory updates"
    },
    {
      name: "Real-time Inventory Check",
      type: "oltp",
      concurrency: 80,
      duration: 20000,
      operation: {
        type: "select",
        complexity: "low",
        resultSize: "small",
        query: "SELECT stock_quantity FROM inventory WHERE product_id = ?",
        indexUsage: true
      },
      delayMs: 5,
      description: "High-frequency inventory availability checks"
    },
    {
      name: "User Profile Updates",
      type: "oltp",
      concurrency: 25,
      duration: 35000,
      operation: {
        type: "update",
        table: "users",
        whereCondition: true,
        complexity: "low"
      },
      delayMs: 100,
      description: "User profile modification operations"
    }
  ],

  // OLAP (Online Analytical Processing) Scenarios
  olap: [
    {
      name: "Sales Analytics Dashboard",
      type: "olap",
      concurrency: 5,
      duration: 60000,
      operation: {
        type: "aggregate",
        aggregateType: "SUM",
        groupByColumns: ["date", "region", "product_category"],
        complexity: "high",
        query: "SELECT date, region, SUM(amount) FROM sales GROUP BY date, region"
      },
      delayMs: 2000,
      description: "Complex sales aggregation for dashboard"
    },
    {
      name: "Customer Behavior Analysis",
      type: "olap",
      concurrency: 3,
      duration: 90000,
      operation: {
        type: "join",
        tableCount: 4,
        complexity: "high",
        query: "Complex multi-table join for customer analytics"
      },
      delayMs: 5000,
      description: "Multi-table customer behavior analysis"
    },
    {
      name: "Revenue Reporting",
      type: "olap",
      concurrency: 2,
      duration: 120000,
      operation: {
        type: "aggregate",
        aggregateType: "AVG",
        groupByColumns: ["month", "year", "department"],
        complexity: "high"
      },
      delayMs: 10000,
      description: "Monthly/yearly revenue calculations"
    }
  ],

  // Mixed Workload Scenarios
  mixed: [
    {
      name: "E-commerce Peak Traffic",
      type: "mixed",
      concurrency: 100,
      duration: 45000,
      operation: {
        type: "mixed",
        operations: [
          { type: "select", weight: 60, complexity: "low" },
          { type: "insert", weight: 25, batchSize: 1 },
          { type: "update", weight: 10, whereCondition: true },
          { type: "delete", weight: 5 }
        ]
      },
      delayMs: 20,
      description: "Peak e-commerce traffic with mixed operations"
    },
    {
      name: "Social Media Platform Load",
      type: "mixed",
      concurrency: 150,
      duration: 30000,
      operation: {
        type: "mixed",
        operations: [
          { type: "select", weight: 70, complexity: "low" }, // Feed reads
          { type: "insert", weight: 20, batchSize: 1 }, // Posts/comments
          { type: "update", weight: 8, whereCondition: true }, // Likes/reactions
          { type: "join", weight: 2, tableCount: 2 } // Social graph queries
        ]
      },
      delayMs: 5,
      description: "Social media platform with high read/write ratio"
    }
  ],

  // Stress Testing Scenarios
  stress: [
    {
      name: "Connection Pool Stress",
      type: "stress",
      concurrency: 200,
      duration: 60000,
      operation: {
        type: "select",
        complexity: "medium",
        resultSize: "medium"
      },
      delayMs: 1,
      description: "Test connection pool limits and management"
    },
    {
      name: "Memory Pressure Test",
      type: "stress",
      concurrency: 50,
      duration: 90000,
      operation: {
        type: "select",
        complexity: "high",
        resultSize: "large",
        query: "SELECT * FROM large_table ORDER BY random_column LIMIT 10000"
      },
      delayMs: 500,
      description: "Test memory usage with large result sets"
    },
    {
      name: "Transaction Deadlock Test",
      type: "stress",
      concurrency: 20,
      duration: 45000,
      operation: {
        type: "transaction",
        operations: [
          { type: "update", table: "accounts", whereCondition: true },
          { type: "update", table: "transactions", whereCondition: true },
          { type: "insert", table: "audit_log", batchSize: 1 }
        ]
      },
      delayMs: 100,
      description: "Test deadlock detection and resolution"
    }
  ],

  // Performance Regression Scenarios
  regression: [
    {
      name: "Index Performance Test",
      type: "regression",
      concurrency: 40,
      duration: 30000,
      operation: {
        type: "select",
        complexity: "medium",
        resultSize: "small",
        useIndex: true,
        query: "SELECT * FROM products WHERE category_id = ? AND price BETWEEN ? AND ?"
      },
      delayMs: 10,
      baseline: {
        throughput: 800, // queries per second
        avgLatency: 15,  // milliseconds
        p95Latency: 30   // milliseconds
      },
      description: "Verify index usage and performance"
    },
    {
      name: "Query Optimization Validation",
      type: "regression",
      concurrency: 20,
      duration: 45000,
      operation: {
        type: "join",
        tableCount: 3,
        complexity: "medium",
        optimized: true
      },
      baseline: {
        throughput: 200,
        avgLatency: 50,
        p95Latency: 100
      },
      description: "Validate query optimization improvements"
    }
  ],

  // Scalability Testing Scenarios
  scalability: [
    {
      name: "Horizontal Scaling Test",
      type: "scalability",
      concurrency: [10, 25, 50, 100, 200], // Progressive load
      duration: 60000,
      operation: {
        type: "select",
        complexity: "medium",
        resultSize: "medium"
      },
      delayMs: 50,
      description: "Test performance across different load levels"
    },
    {
      name: "Read Replica Distribution",
      type: "scalability",
      concurrency: 80,
      duration: 45000,
      operation: {
        type: "mixed",
        operations: [
          { type: "select", weight: 90, readOnly: true },
          { type: "insert", weight: 10, writeOnly: true }
        ]
      },
      description: "Test read/write splitting performance"
    }
  ],

  // Database-Specific Scenarios
  postgresql: [
    {
      name: "PostgreSQL JSONB Performance",
      type: "postgresql",
      concurrency: 30,
      duration: 40000,
      operation: {
        type: "select",
        complexity: "medium",
        resultSize: "medium",
        query: "SELECT * FROM documents WHERE metadata @> ?::jsonb",
        features: ["jsonb", "gin_index"]
      },
      description: "Test PostgreSQL JSONB operations"
    },
    {
      name: "PostgreSQL Full-Text Search",
      type: "postgresql",
      concurrency: 15,
      duration: 35000,
      operation: {
        type: "select",
        complexity: "high",
        query: "SELECT * FROM articles WHERE to_tsvector('english', content) @@ to_tsquery(?)",
        features: ["full_text_search", "tsvector"]
      },
      description: "Test full-text search performance"
    }
  ],

  mysql: [
    {
      name: "MySQL InnoDB Performance",
      type: "mysql",
      concurrency: 40,
      duration: 35000,
      operation: {
        type: "transaction",
        operations: [
          { type: "insert", batchSize: 5 },
          { type: "select", complexity: "low" }
        ],
        engine: "InnoDB"
      },
      description: "Test InnoDB transaction performance"
    }
  ],

  mongodb: [
    {
      name: "MongoDB Document Insert",
      type: "mongodb",
      concurrency: 50,
      duration: 30000,
      operation: {
        type: "insert",
        batchSize: 10,
        documentSize: "medium",
        collection: "events"
      },
      description: "Test MongoDB document insertion performance"
    },
    {
      name: "MongoDB Aggregation Pipeline",
      type: "mongodb",
      concurrency: 10,
      duration: 60000,
      operation: {
        type: "aggregate",
        pipeline: [
          { $match: { status: "active" } },
          { $group: { _id: "$category", total: { $sum: "$amount" } } },
          { $sort: { total: -1 } }
        ],
        complexity: "high"
      },
      description: "Test MongoDB aggregation performance"
    }
  ]
};

/**
 * Scenario selection utilities
 */
const scenarioUtils = {
  /**
   * Get scenarios by type
   */
  getByType(type) {
    return loadTestScenarios[type] || [];
  },

  /**
   * Get all scenarios
   */
  getAll() {
    return Object.values(loadTestScenarios).flat();
  },

  /**
   * Get scenarios by complexity
   */
  getByComplexity(complexity) {
    const allScenarios = this.getAll();
    return allScenarios.filter(scenario => {
      const op = scenario.operation;
      return op.complexity === complexity || 
             (op.operations && op.operations.some(subOp => subOp.complexity === complexity));
    });
  },

  /**
   * Get baseline scenarios for comparison
   */
  getBaselineScenarios() {
    return [
      ...loadTestScenarios.oltp.slice(0, 2),
      ...loadTestScenarios.olap.slice(0, 1),
      ...loadTestScenarios.mixed.slice(0, 1),
      ...loadTestScenarios.regression
    ];
  },

  /**
   * Get stress test scenarios
   */
  getStressScenarios() {
    return [
      ...loadTestScenarios.stress,
      ...loadTestScenarios.scalability
    ];
  },

  /**
   * Create custom scenario
   */
  createCustom(config) {
    const defaults = {
      concurrency: 10,
      duration: 30000,
      delayMs: 100,
      operation: {
        type: "select",
        complexity: "medium",
        resultSize: "medium"
      }
    };

    return {
      ...defaults,
      ...config,
      operation: {
        ...defaults.operation,
        ...config.operation
      }
    };
  },

  /**
   * Validate scenario configuration
   */
  validate(scenario) {
    const errors = [];

    if (!scenario.name) {
      errors.push("Scenario must have a name");
    }

    if (!scenario.type) {
      errors.push("Scenario must have a type");
    }

    if (!scenario.operation) {
      errors.push("Scenario must have an operation");
    }

    if (scenario.concurrency && scenario.concurrency <= 0) {
      errors.push("Concurrency must be positive");
    }

    if (scenario.duration && scenario.duration <= 0) {
      errors.push("Duration must be positive");
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
};

module.exports = {
  loadTestScenarios,
  scenarioUtils
};