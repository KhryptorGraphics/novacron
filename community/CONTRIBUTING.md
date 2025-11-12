# Contributing to DWCP v3

Thank you for your interest in contributing to the Distributed Workspace Communication Protocol (DWCP) v3! This document provides comprehensive guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Contributing Workflow](#contributing-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Requirements](#testing-requirements)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)
9. [Community](#community)

---

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

### Our Pledge

We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

---

## Getting Started

### Ways to Contribute

- **Code**: Bug fixes, features, performance improvements
- **Documentation**: Guides, tutorials, API docs, examples
- **Testing**: Bug reports, test cases, QA
- **Community**: Answering questions, helping users, organizing events
- **Design**: UI/UX improvements, diagrams, visual assets

### Before You Start

1. **Check existing issues**: Search for similar issues or features
2. **Create an issue**: Discuss your idea before implementation
3. **Read documentation**: Familiarize yourself with the architecture
4. **Join Discord**: Connect with the community

---

## Development Setup

### Prerequisites

```bash
# Required tools
node --version  # v20.11.0+
rustc --version  # 1.75.0+
docker --version  # 24.0.0+
git --version  # 2.30.0+

# Optional but recommended
kubectl version  # 1.28.0+
psql --version  # 15.0+
redis-cli --version  # 7.2.0+
```

### Clone and Build

```bash
# Clone repository
git clone https://github.com/dwcp/dwcp.git
cd dwcp

# Install dependencies
npm install

# Build native modules
cd native
cargo build --release
cd ..

# Build TypeScript
npm run build

# Run tests
npm test

# Start development cluster
npm run dev
```

### Development Environment

```bash
# Set up development environment
cp .env.example .env.development

# Start dependencies (PostgreSQL, Redis)
docker-compose -f docker-compose.dev.yml up -d

# Initialize database
npm run db:migrate

# Start development server with hot reload
npm run dev -- --watch

# Run in debug mode
npm run dev -- --debug
```

---

## Contributing Workflow

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/dwcp.git
cd dwcp

# Add upstream remote
git remote add upstream https://github.com/dwcp/dwcp.git

# Verify remotes
git remote -v
```

### 2. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-feature

# Or bug fix branch
git checkout -b fix/issue-123
```

**Branch Naming Conventions**:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test improvements
- `perf/` - Performance improvements

### 3. Make Changes

```bash
# Make your changes
# Follow coding standards (see below)
# Add tests
# Update documentation

# Run linter
npm run lint

# Run formatter
npm run format

# Run type checker
npm run typecheck

# Run tests
npm test
```

### 4. Commit Changes

```bash
# Stage changes
git add .

# Commit with conventional commit message
git commit -m "feat: add support for quantum-resistant encryption"

# Or
git commit -m "fix: resolve consensus deadlock on network partition"
```

**Commit Message Format**:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Add/update tests
- `chore`: Maintenance tasks
- `perf`: Performance improvement

**Examples**:
```
feat(consensus): implement Byzantine fault tolerance
fix(storage): prevent data race in CRDT merge
docs(api): add GraphQL schema documentation
test(cluster): add integration tests for scaling
perf(network): optimize QUIC connection handling
```

### 5. Push and Create PR

```bash
# Push to your fork
git push origin feature/my-feature

# Create pull request on GitHub
# Fill out PR template
# Link related issues
# Add reviewers
```

---

## Coding Standards

### TypeScript Style Guide

```typescript
// Use explicit types
function calculateLatency(startTime: number, endTime: number): number {
  return endTime - startTime;
}

// Use interfaces for objects
interface NodeConfig {
  id: string;
  address: string;
  role: 'leader' | 'follower';
}

// Use async/await for promises
async function fetchData(): Promise<Data> {
  const response = await fetch('/api/data');
  return response.json();
}

// Use descriptive names
const maxConnectionTimeout = 30000;  // Good
const mct = 30000;  // Bad

// Document complex logic
/**
 * Implements the Byzantine consensus algorithm.
 *
 * @param proposal - The proposed operation
 * @param nodes - List of participating nodes
 * @returns Promise resolving to consensus result
 */
async function byzantineConsensus(
  proposal: Proposal,
  nodes: Node[]
): Promise<ConsensusResult> {
  // Implementation
}
```

### Rust Style Guide

```rust
// Use descriptive names
fn calculate_merkle_root(transactions: &[Transaction]) -> Hash {
    // Implementation
}

// Use Result for error handling
fn connect_to_peer(address: &str) -> Result<Connection, ConnectionError> {
    // Implementation
}

// Document public APIs
/// Implements the Raft consensus algorithm.
///
/// # Arguments
/// * `config` - Raft configuration
/// * `storage` - Persistent storage backend
///
/// # Returns
/// A new Raft instance
pub fn new(config: RaftConfig, storage: Storage) -> Self {
    // Implementation
}

// Use traits for abstractions
pub trait ConsensusProtocol {
    fn propose(&mut self, operation: Operation) -> Result<(), Error>;
    fn commit(&mut self, index: u64) -> Result<(), Error>;
}
```

### Code Organization

```
src/
├── core/           # Core functionality
│   ├── consensus/  # Consensus protocols
│   ├── network/    # Networking layer
│   ├── storage/    # Storage layer
│   └── security/   # Security features
├── services/       # Application services
├── utils/          # Utility functions
└── types/          # Type definitions

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── e2e/            # End-to-end tests
```

### Best Practices

1. **Keep functions small**: Max 50 lines
2. **Single responsibility**: One function, one purpose
3. **DRY principle**: Don't repeat yourself
4. **SOLID principles**: Follow object-oriented design principles
5. **Error handling**: Always handle errors explicitly
6. **Logging**: Use structured logging
7. **Comments**: Explain why, not what
8. **Performance**: Profile before optimizing

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80%
- **Critical paths**: 100%
- **New features**: Must include tests

### Unit Tests

```typescript
// tests/unit/consensus/raft.test.ts

import { describe, it, expect } from '@dwcp/testing';
import { RaftConsensus } from '../../../src/core/consensus/raft';

describe('RaftConsensus', () => {
  describe('leader election', () => {
    it('should elect leader when election timeout expires', async () => {
      const nodes = createTestNodes(5);
      const raft = new RaftConsensus(nodes[0]);

      await raft.start();
      await sleep(5000); // Wait for election

      const leader = nodes.find(n => n.state === 'leader');
      expect(leader).toBeDefined();
    });

    it('should maintain single leader', async () => {
      const nodes = createTestNodes(5);

      for (const node of nodes) {
        await node.start();
      }

      await sleep(10000);

      const leaders = nodes.filter(n => n.state === 'leader');
      expect(leaders.length).toBe(1);
    });
  });
});
```

### Integration Tests

```typescript
// tests/integration/cluster.test.ts

import { TestCluster } from '@dwcp/testing';

describe('Cluster Operations', () => {
  let cluster: TestCluster;

  beforeAll(async () => {
    cluster = await TestCluster.create({
      nodes: 7,
      consensus: 'raft'
    });
  });

  afterAll(async () => {
    await cluster.destroy();
  });

  it('should handle node failure gracefully', async () => {
    // Write data
    await cluster.write('key1', 'value1');

    // Kill a node
    await cluster.killNode(2);

    // Verify data is still accessible
    const value = await cluster.read('key1');
    expect(value).toBe('value1');

    // Verify cluster is still functional
    await cluster.write('key2', 'value2');
    expect(await cluster.read('key2')).toBe('value2');
  });
});
```

### Performance Tests

```typescript
// tests/performance/throughput.test.ts

import { TestCluster } from '@dwcp/testing';

describe('Performance Benchmarks', () => {
  it('should handle 100k requests per second', async () => {
    const cluster = await TestCluster.create({ nodes: 7 });

    const results = await cluster.benchmark({
      duration: 60000,
      requestsPerSecond: 100000,
      operation: 'write'
    });

    expect(results.throughput).toBeGreaterThan(100000);
    expect(results.p99Latency).toBeLessThan(10);
  });
});
```

### Running Tests

```bash
# Run all tests
npm test

# Run specific test file
npm test -- tests/unit/consensus/raft.test.ts

# Run with coverage
npm test -- --coverage

# Run in watch mode
npm test -- --watch

# Run integration tests only
npm run test:integration

# Run performance tests
npm run test:perf
```

---

## Documentation

### Code Documentation

```typescript
/**
 * Implements the Byzantine fault-tolerant consensus protocol.
 *
 * This implementation follows the PBFT algorithm with optimizations
 * for high-throughput scenarios.
 *
 * @example
 * ```typescript
 * const consensus = new ByzantineConsensus({
 *   nodes: 7,
 *   faultTolerance: 2
 * });
 *
 * await consensus.propose({
 *   operation: 'write',
 *   data: { key: 'foo', value: 'bar' }
 * });
 * ```
 *
 * @param config - Byzantine consensus configuration
 * @returns A new Byzantine consensus instance
 */
export class ByzantineConsensus implements ConsensusProtocol {
  // Implementation
}
```

### API Documentation

- Update OpenAPI spec for API changes
- Add examples for new endpoints
- Document error responses
- Update GraphQL schema

### User Documentation

- Update relevant guides
- Add tutorials for new features
- Update troubleshooting guide
- Add migration guides for breaking changes

---

## Pull Request Process

### Before Submitting

- [ ] Code follows style guidelines
- [ ] All tests pass
- [ ] Code coverage is maintained/improved
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] No merge conflicts
- [ ] PR description is complete

### PR Description Template

```markdown
## Description
Brief description of changes

## Related Issues
Fixes #123
Related to #456

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Screenshots
If applicable

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Breaking changes documented
```

### Review Process

1. **Automated checks**: CI/CD pipeline runs
2. **Code review**: At least 2 approvals required
3. **Testing**: QA team tests changes
4. **Documentation**: Docs team reviews
5. **Approval**: Maintainer approves
6. **Merge**: Squash and merge to main

### After Merge

- PR is automatically deployed to staging
- Performance tests run
- Documentation is published
- Release notes updated

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General discussions, Q&A
- **Discord**: Real-time chat, community support
- **Forum**: Long-form discussions
- **Twitter**: Announcements, updates
- **Email**: security@dwcp.io (security issues)

### Getting Help

1. **Search documentation**: https://docs.dwcp.io
2. **Check FAQ**: Common questions answered
3. **Ask on Discord**: Community support
4. **Create discussion**: For longer questions
5. **File issue**: For bugs or feature requests

### Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project website
- Annual contributor report

### Licensing

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## Thank You!

Your contributions make DWCP better for everyone. We appreciate your time and effort!

For questions, contact: contribute@dwcp.io

---

*Last Updated: 2025-11-10*
*Version: 3.0.0*
