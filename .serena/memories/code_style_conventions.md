# NovaCron Code Style and Conventions

## Go Backend Conventions
- **Package naming**: Lowercase, single word preferred
- **File naming**: snake_case (e.g., vm_manager.go)
- **Function/Method naming**: PascalCase for exported, camelCase for internal
- **Interface naming**: Suffix with "er" when possible (e.g., Manager, Provider)
- **Error handling**: Always check and handle errors explicitly
- **Comments**: Godoc style for exported functions
- **Testing**: Test files alongside source (e.g., vm_test.go)

## TypeScript/React Frontend Conventions
- **File naming**: kebab-case for files, PascalCase for components
- **Component structure**: Functional components with hooks
- **State management**: Jotai for global state, React Query for server state
- **Styling**: Tailwind CSS with component variants (class-variance-authority)
- **Type safety**: Strict TypeScript, Zod for runtime validation
- **Testing**: Jest with React Testing Library

## General Patterns
- **Architecture**: Clean architecture with separation of concerns
- **API Design**: RESTful with consistent naming
- **Error handling**: Comprehensive error types and proper propagation
- **Logging**: Structured logging with appropriate levels
- **Security**: Input validation, authentication checks, secure defaults
- **Documentation**: Clear comments, README files, API documentation