/**
 * Training data for ML-based task classifier
 * Real-world examples with labeled complexity
 */

const trainingData = [
  // Simple tasks
  { description: 'Fix typo in README', complexity: 'simple', level: 1 },
  { description: 'Update comment in code', complexity: 'simple', level: 1 },
  { description: 'Rename variable for clarity', complexity: 'simple', level: 1 },
  { description: 'Format code with prettier', complexity: 'simple', level: 1 },
  { description: 'Add missing semicolon', complexity: 'simple', level: 1 },
  { description: 'Update version number', complexity: 'simple', level: 1 },
  { description: 'Fix indentation', complexity: 'simple', level: 1 },
  { description: 'Remove unused import', complexity: 'simple', level: 1 },
  { description: 'Update copyright year', complexity: 'simple', level: 1 },
  { description: 'Fix spelling error', complexity: 'simple', level: 1 },
  
  // Medium tasks
  { description: 'Add new feature to user dashboard', complexity: 'medium', level: 2 },
  { description: 'Refactor authentication logic', complexity: 'medium', level: 2 },
  { description: 'Update API endpoint to support pagination', complexity: 'medium', level: 2 },
  { description: 'Optimize database query performance', complexity: 'medium', level: 2 },
  { description: 'Add validation to form inputs', complexity: 'medium', level: 2 },
  { description: 'Implement caching for API responses', complexity: 'medium', level: 2 },
  { description: 'Create new React component for notifications', complexity: 'medium', level: 2 },
  { description: 'Add error handling to payment processing', complexity: 'medium', level: 2 },
  { description: 'Update user profile page with new fields', complexity: 'medium', level: 2 },
  { description: 'Refactor code to use async/await', complexity: 'medium', level: 2 },
  
  // Complex tasks
  { description: 'Implement new API endpoint with authentication and rate limiting', complexity: 'complex', level: 3 },
  { description: 'Design and implement database schema for multi-tenancy', complexity: 'complex', level: 3 },
  { description: 'Migrate legacy system to microservices architecture', complexity: 'complex', level: 3 },
  { description: 'Implement real-time notifications using WebSockets', complexity: 'complex', level: 3 },
  { description: 'Design and build payment processing system', complexity: 'complex', level: 3 },
  { description: 'Create comprehensive testing framework for API', complexity: 'complex', level: 3 },
  { description: 'Implement role-based access control system', complexity: 'complex', level: 3 },
  { description: 'Build data analytics dashboard with real-time updates', complexity: 'complex', level: 3 },
  { description: 'Architect and implement caching layer for distributed system', complexity: 'complex', level: 3 },
  { description: 'Design API gateway with load balancing and service discovery', complexity: 'complex', level: 3 },
  
  // Very complex tasks
  { description: 'Implement OAuth 2.0 authentication with Google and GitHub', complexity: 'very-complex', level: 4 },
  { description: 'Design and implement distributed transaction system across microservices', complexity: 'very-complex', level: 4 },
  { description: 'Build real-time collaborative editing system with conflict resolution', complexity: 'very-complex', level: 4 },
  { description: 'Architect multi-cloud deployment strategy with automatic failover', complexity: 'very-complex', level: 4 },
  { description: 'Implement distributed caching with Redis cluster and consistency guarantees', complexity: 'very-complex', level: 4 },
  { description: 'Design and build event-driven architecture with Kafka and stream processing', complexity: 'very-complex', level: 4 },
  { description: 'Implement end-to-end encryption for messaging system with key management', complexity: 'very-complex', level: 4 },
  { description: 'Build ML-powered recommendation engine with real-time training', complexity: 'very-complex', level: 4 },
  { description: 'Architect and implement distributed tracing system across all services', complexity: 'very-complex', level: 4 },
  { description: 'Design multi-region database replication with conflict resolution', complexity: 'very-complex', level: 4 },
  
  // NovaCron-specific tasks
  { description: 'Implement live VM migration with memory pre-copy', complexity: 'very-complex', level: 4 },
  { description: 'Add WAN optimization for cross-region VM migration', complexity: 'complex', level: 3 },
  { description: 'Create VM snapshot scheduling system', complexity: 'medium', level: 2 },
  { description: 'Update VM dashboard with resource metrics', complexity: 'medium', level: 2 },
  { description: 'Implement AI-powered VM placement scheduler', complexity: 'very-complex', level: 4 },
  { description: 'Add support for new hypervisor type', complexity: 'complex', level: 3 },
  { description: 'Optimize VM startup time', complexity: 'complex', level: 3 },
  { description: 'Fix VM console connection issue', complexity: 'medium', level: 2 },
  { description: 'Add VM tagging functionality', complexity: 'simple', level: 1 },
  { description: 'Implement multi-cloud VM federation', complexity: 'very-complex', level: 4 },
  
  // Backend tasks
  { description: 'Add new Go API endpoint for user management', complexity: 'medium', level: 2 },
  { description: 'Implement gRPC service for inter-service communication', complexity: 'complex', level: 3 },
  { description: 'Optimize Go scheduler performance', complexity: 'complex', level: 3 },
  { description: 'Add database connection pooling', complexity: 'medium', level: 2 },
  { description: 'Implement distributed locking with Redis', complexity: 'complex', level: 3 },
  
  // Frontend tasks
  { description: 'Create new React dashboard component', complexity: 'medium', level: 2 },
  { description: 'Implement real-time updates with WebSockets in React', complexity: 'complex', level: 3 },
  { description: 'Add TypeScript types to existing components', complexity: 'medium', level: 2 },
  { description: 'Build interactive data visualization with D3.js', complexity: 'complex', level: 3 },
  { description: 'Implement responsive design for mobile', complexity: 'medium', level: 2 },
  
  // Database tasks
  { description: 'Add database index for performance', complexity: 'simple', level: 1 },
  { description: 'Design schema for new feature', complexity: 'medium', level: 2 },
  { description: 'Implement database migration strategy', complexity: 'complex', level: 3 },
  { description: 'Optimize slow SQL queries', complexity: 'medium', level: 2 },
  { description: 'Design multi-tenant database architecture', complexity: 'very-complex', level: 4 },
  
  // DevOps tasks
  { description: 'Update Docker configuration', complexity: 'simple', level: 1 },
  { description: 'Create Kubernetes deployment manifests', complexity: 'medium', level: 2 },
  { description: 'Implement CI/CD pipeline with automated testing', complexity: 'complex', level: 3 },
  { description: 'Design multi-region deployment strategy', complexity: 'very-complex', level: 4 },
  { description: 'Add monitoring and alerting with Prometheus', complexity: 'complex', level: 3 },
  
  // Security tasks
  { description: 'Update dependency versions', complexity: 'simple', level: 1 },
  { description: 'Implement JWT authentication', complexity: 'complex', level: 3 },
  { description: 'Add rate limiting to API endpoints', complexity: 'medium', level: 2 },
  { description: 'Implement end-to-end encryption', complexity: 'very-complex', level: 4 },
  { description: 'Design zero-trust security architecture', complexity: 'very-complex', level: 4 },
  
  // Testing tasks
  { description: 'Add unit test for new function', complexity: 'simple', level: 1 },
  { description: 'Create integration tests for API', complexity: 'medium', level: 2 },
  { description: 'Implement E2E testing framework', complexity: 'complex', level: 3 },
  { description: 'Build comprehensive test automation suite', complexity: 'very-complex', level: 4 },
  { description: 'Add performance testing with load generation', complexity: 'complex', level: 3 }
];

/**
 * Get training data split for ML model
 */
function getTrainTestSplit(splitRatio = 0.8) {
  const shuffled = [...trainingData].sort(() => Math.random() - 0.5);
  const splitIndex = Math.floor(shuffled.length * splitRatio);
  
  return {
    train: shuffled.slice(0, splitIndex),
    test: shuffled.slice(splitIndex),
    total: shuffled.length
  };
}

/**
 * Get training data by complexity level
 */
function getByComplexity(complexity) {
  return trainingData.filter(sample => sample.complexity === complexity);
}

/**
 * Get statistics about training data
 */
function getStatistics() {
  const stats = {
    total: trainingData.length,
    byComplexity: {
      simple: getByComplexity('simple').length,
      medium: getByComplexity('medium').length,
      complex: getByComplexity('complex').length,
      'very-complex': getByComplexity('very-complex').length
    }
  };
  
  return stats;
}

module.exports = {
  trainingData,
  getTrainTestSplit,
  getByComplexity,
  getStatistics
};

