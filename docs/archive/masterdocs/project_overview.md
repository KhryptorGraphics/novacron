# NovaCron Project Overview

## Purpose
NovaCron is a sophisticated distributed virtual machine management platform with:
- Advanced migration capabilities
- Real-time monitoring and alerting
- Intelligent resource scheduling
- Multi-cloud orchestration support
- Enterprise-grade security features

## Tech Stack
### Backend
- **Language**: Go 1.21+
- **API Framework**: Gorilla/mux
- **Dependencies**: libvirt (KVM), PostgreSQL, Redis
- **Architecture**: Microservices, distributed consensus

### Frontend  
- **Framework**: Next.js 13.5.6 with React 18.2
- **Language**: TypeScript
- **State Management**: Jotai, React Query
- **UI Components**: Radix UI, Tailwind CSS
- **Real-time**: WebSocket (react-use-websocket)
- **Charts**: Chart.js, D3.js

## Project Structure
- `/backend` - Go backend services
  - `/api` - HTTP handlers and routes
  - `/core` - Core business logic
  - `/cmd` - Executable commands
- `/frontend` - Next.js frontend
  - `/src/app` - App router pages
  - `/src/components` - React components
  - `/src/lib` - Utilities and hooks
- `/docs` - Documentation
- `/scripts` - Deployment scripts
- `/docker` - Docker configurations

## Current Status
~85% complete - Production-ready core functionality with advanced monitoring dashboard