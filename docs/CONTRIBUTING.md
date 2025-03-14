# Contributing to NovaCron

Thank you for your interest in contributing to NovaCron! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and considerate of others when participating in our community.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone https://github.com/yourusername/novacron.git
   cd novacron
   ```
3. Add the original repository as an upstream remote
   ```bash
   git remote add upstream https://github.com/novacron/novacron.git
   ```
4. Create a new branch for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Environment

### Prerequisites

- Go 1.19+
- Node.js 18+
- Python 3.8+
- Docker and Docker Compose
- For VM functionality: KVM/QEMU (Linux) or Hyper-V (Windows)

### Setup

1. Run the setup script to create necessary configuration files:
   ```bash
   # For Linux/macOS
   ./scripts/setup.sh
   
   # For Windows
   .\scripts\setup.ps1
   ```

2. Build and run the development environment:
   ```bash
   docker-compose up -d
   ```

### Code Organization

The codebase is organized as follows:

- `backend/core`: Core Go components for VM management and migration
- `backend/services`: Auxiliary Python services
- `frontend`: Next.js web interface
- `docker`: Dockerfiles and container configurations
- `scripts`: Setup and utility scripts
- `docs`: Documentation

## Making Changes

1. Make your changes, following our coding conventions
2. Write or update tests as necessary
3. Ensure all tests pass
4. Update documentation as needed
5. Commit your changes with meaningful commit messages

### Coding Conventions

#### Go Code

- Follow the [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
- Use `gofmt` to format your code
- Document all exported types, functions, and methods
- Add unit tests for new functionality

#### Python Code

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use type hints when possible
- Document functions and classes with docstrings

#### TypeScript/React Code

- Follow the project's ESLint and Prettier configurations
- Use TypeScript for all new components
- Follow React best practices

### Testing

Before submitting a PR, ensure all tests pass:

```bash
# Go tests
cd backend/core
go test ./...

# Python tests
cd backend/services
pytest

# Frontend tests
cd frontend
npm test
```

## Pull Request Process

1. Update your fork with the latest changes from upstream:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a pull request on GitHub from your fork to the main repository
4. Fill out the PR template with details about your changes
5. Address any feedback from code reviewers
6. Once approved, a maintainer will merge your PR

### PR Guidelines

- Keep PRs focused on a single feature or bug fix
- Link to related issues
- Provide tests for new functionality
- Ensure CI checks pass

## Development Best Practices

### VM Migration

When working on VM migration functionality:

1. Consider the three migration types (cold, warm, live) and their specific requirements
2. Ensure robust error handling and rollback mechanisms
3. Add comprehensive logging for debugging
4. Consider performance implications, especially for live migrations
5. Add metrics for monitoring migration progress

### Security Considerations

- Use proper authentication and authorization
- Avoid hardcoding credentials
- Follow secure coding practices
- Consider potential attack vectors in your implementation

## Documentation

Documentation is crucial for this project. When adding new features, please:

1. Update relevant README files
2. Add code comments for non-obvious parts
3. Update API documentation if needed
4. For complex features, consider adding a dedicated doc in the `/docs` directory

## Releasing

Maintainers will handle releases according to the following process:

1. Update version numbers in relevant files
2. Create release notes
3. Tag the release in git
4. Build and publish Docker images

## Questions?

If you have questions or need help, please:

1. Check existing issues first
2. Open a new issue if needed
3. Join our community discussions

Thank you for contributing to NovaCron!
