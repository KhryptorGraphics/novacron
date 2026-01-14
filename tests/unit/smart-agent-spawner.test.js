/**
 * Unit tests for Smart Agent Spawner
 */

const SmartAgentSpawner = require('../../src/services/smart-agent-spawner');

describe('SmartAgentSpawner', () => {
  let spawner;

  beforeEach(() => {
    spawner = new SmartAgentSpawner({
      maxAgents: 8,
      enableMCP: false // Disable MCP for unit tests
    });
  });

  describe('File Type Detection', () => {
    test('should detect Go backend agent for .go files', () => {
      const result = spawner.detectAgentFromFileType('backend/api/server.go');
      expect(result.type).toBe('coder');
      expect(result.specialization).toBe('go-backend');
      expect(result.capabilities).toContain('go-lang');
    });

    test('should detect React frontend agent for .tsx files', () => {
      const result = spawner.detectAgentFromFileType('frontend/components/Dashboard.tsx');
      expect(result.type).toBe('coder');
      expect(result.specialization).toBe('react-frontend');
      expect(result.capabilities).toContain('typescript');
      expect(result.capabilities).toContain('react');
    });

    test('should detect analyst for YAML config files', () => {
      const result = spawner.detectAgentFromFileType('config/settings.yaml');
      expect(result.type).toBe('analyst');
      expect(result.specialization).toBe('config-manager');
    });

    test('should detect researcher for markdown files', () => {
      const result = spawner.detectAgentFromFileType('docs/README.md');
      expect(result.type).toBe('researcher');
      expect(result.specialization).toBe('documentation');
    });

    test('should detect tester for test files', () => {
      const result = spawner.detectAgentFromFileType('tests/api.test.js');
      expect(result.type).toBe('tester');
      expect(result.specialization).toBe('unit-testing');
    });
  });

  describe('Task Complexity Analysis', () => {
    test('should classify simple tasks correctly', () => {
      const result = spawner.analyzeTaskComplexity('Fix typo in README');
      expect(result.complexity).toBe('simple');
      expect(result.score).toBe(1);
      expect(result.recommendedAgents).toContain('coordinator');
    });

    test('should classify medium complexity tasks', () => {
      const result = spawner.analyzeTaskComplexity('Add feature to user dashboard');
      expect(result.complexity).toBe('medium');
      expect(result.score).toBe(2);
      expect(result.recommendedAgents).toContain('coder');
    });

    test('should classify complex tasks', () => {
      const result = spawner.analyzeTaskComplexity('Implement new API endpoint with validation');
      expect(result.complexity).toBe('complex');
      expect(result.score).toBe(3);
      expect(result.recommendedAgents).toContain('architect');
      expect(result.recommendedAgents).toContain('tester');
    });

    test('should classify very complex tasks', () => {
      const result = spawner.analyzeTaskComplexity('Implement OAuth authentication with Google');
      expect(result.complexity).toBe('very-complex');
      expect(result.score).toBe(4);
      expect(result.recommendedAgents.length).toBeGreaterThan(3);
    });

    test('should add security agent for auth-related tasks', () => {
      const result = spawner.analyzeTaskComplexity('Implement authentication system');
      expect(result.recommendedAgents).toContain('security-auditor');
    });

    test('should add database expert for database tasks', () => {
      const result = spawner.analyzeTaskComplexity('Design database schema for users');
      expect(result.recommendedAgents).toContain('database-expert');
    });

    test('should mark complex tasks as parallelizable', () => {
      const result = spawner.analyzeTaskComplexity('Implement microservices architecture');
      expect(result.parallelizable).toBe(true);
    });
  });

  describe('Topology Selection', () => {
    test('should select single topology for simple tasks', () => {
      const analysis = { score: 1, complexity: 'simple' };
      const topology = spawner.selectTopology(analysis, 1);
      expect(topology).toBe('single');
    });

    test('should select mesh topology for medium tasks', () => {
      const analysis = { score: 2, complexity: 'medium' };
      const topology = spawner.selectTopology(analysis, 2);
      expect(topology).toBe('mesh');
    });

    test('should select hierarchical topology for complex tasks', () => {
      const analysis = { score: 3, complexity: 'complex' };
      const topology = spawner.selectTopology(analysis, 4);
      expect(topology).toBe('hierarchical');
    });

    test('should select adaptive topology for very complex tasks', () => {
      const analysis = { score: 4, complexity: 'very-complex' };
      const topology = spawner.selectTopology(analysis, 6);
      expect(topology).toBe('adaptive');
    });

    test('should respect manual topology configuration', () => {
      const customSpawner = new SmartAgentSpawner({ topology: 'mesh', enableMCP: false });
      const analysis = { score: 4, complexity: 'very-complex' };
      const topology = customSpawner.selectTopology(analysis, 6);
      expect(topology).toBe('mesh');
    });
  });

  describe('Agent Recommendation Combination', () => {
    test('should combine task and file-based recommendations', () => {
      const taskAgents = ['coordinator', 'coder'];
      const fileAgents = [
        { type: 'tester' },
        { type: 'coder' } // Duplicate
      ];
      
      const result = spawner.combineAgentRecommendations(taskAgents, fileAgents);
      expect(result).toContain('coordinator');
      expect(result).toContain('coder');
      expect(result).toContain('tester');
      expect(result.length).toBe(3); // No duplicates
    });

    test('should respect maxAgents limit', () => {
      const customSpawner = new SmartAgentSpawner({ maxAgents: 3, enableMCP: false });
      const taskAgents = ['coordinator', 'coder', 'tester', 'architect'];
      const fileAgents = [{ type: 'researcher' }, { type: 'analyst' }];
      
      const result = customSpawner.combineAgentRecommendations(taskAgents, fileAgents);
      expect(result.length).toBeLessThanOrEqual(3);
    });
  });

  describe('Auto-Spawn', () => {
    test('should create comprehensive spawn plan', async () => {
      const plan = await spawner.autoSpawn({
        task: 'Implement OAuth with Google',
        files: ['backend/auth/oauth.go', 'frontend/components/Login.tsx'],
        context: { priority: 'high' }
      });

      expect(plan).toHaveProperty('agents');
      expect(plan).toHaveProperty('topology');
      expect(plan).toHaveProperty('complexity');
      expect(plan).toHaveProperty('timestamp');
      expect(plan.agents.length).toBeGreaterThan(0);
    });

    test('should emit spawn-plan event', async () => {
      const eventPromise = new Promise(resolve => {
        spawner.once('spawn-plan', resolve);
      });

      await spawner.autoSpawn({
        task: 'Add new feature',
        files: ['src/feature.js']
      });

      const plan = await eventPromise;
      expect(plan).toBeDefined();
    });
  });

  describe('Metrics', () => {
    test('should track spawning metrics', async () => {
      await spawner.autoSpawn({ task: 'Task 1', files: [] });
      await spawner.autoSpawn({ task: 'Task 2', files: [] });

      const metrics = spawner.getMetrics();
      expect(metrics.spawningDecisions.length).toBe(2);
    });
  });
});

