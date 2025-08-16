# API Integration Implementation Plan

## Task: Replace Mock Data with Real Backend API Calls

### Objective:
Replace all mock data and simulated API calls in the frontend dashboard with real connections to the backend API services, ensuring seamless integration between frontend UI components and backend functionality.

### Current State Analysis:

The frontend dashboard currently uses mock data for:
1. VM management (listing, creating, starting, stopping, deleting VMs)
2. Job management (listing, creating, executing jobs)
3. Workflow management (listing, creating, executing workflows)
4. Monitoring data (VM metrics, system status, alerts)
5. User authentication (currently missing entirely)

### Backend API Services Available:

1. VM Management API (Go backend - port 8090)
   - GET `/api/v1/vms` - List VMs
   - POST `/api/v1/vms` - Create VM
   - GET `/api/v1/vms/{id}` - Get VM details
   - POST `/api/v1/vms/{id}/start` - Start VM
   - POST `/api/v1/vms/{id}/stop` - Stop VM
   - DELETE `/api/v1/vms/{id}` - Delete VM
   - GET `/api/v1/vms/{id}/info` - Get VM info
   - GET `/api/v1/vms/{id}/stats` - Get VM stats

2. Jobs API (Node.js backend - port 8090)
   - GET `/api/jobs` - List jobs
   - POST `/api/jobs` - Create job
   - GET `/api/jobs/{id}` - Get job details
   - PUT `/api/jobs/{id}` - Update job
   - DELETE `/api/jobs/{id}` - Delete job
   - POST `/api/jobs/{id}/execute` - Execute job
   - GET `/api/jobs/{id}/executions` - Get job executions

3. Workflows API (Node.js backend - port 8090)
   - GET `/api/workflows` - List workflows
   - POST `/api/workflows` - Create workflow
   - GET `/api/workflows/{id}` - Get workflow details
   - PUT `/api/workflows/{id}` - Update workflow
   - DELETE `/api/workflows/{id}` - Delete workflow
   - POST `/api/workflows/{id}/execute` - Execute workflow
   - GET `/api/workflows/executions/{id}` - Get execution status

### Frontend Components Requiring API Integration:

1. Dashboard Overview (`/frontend/src/app/dashboard/page-updated.tsx`)
   - System status indicators
   - Real-time WebSocket updates
   - Execution statistics

2. VM Management (`/frontend/src/components/dashboard/vm-list.tsx`)
   - VM listing with real data
   - VM creation form
   - VM control actions (start, stop, delete)

3. Job Management (`/frontend/src/components/dashboard/job-list.tsx`)
   - Job listing with real data
   - Job creation form
   - Job execution controls

4. Workflow Management (`/frontend/src/components/dashboard/workflow-list.tsx`)
   - Workflow listing with real data
   - Workflow creation form
   - Workflow execution controls

5. Monitoring Dashboard (`/frontend/src/components/monitoring/MonitoringDashboard.tsx`)
   - Real-time metrics visualization
   - Alert management
   - Performance charts

6. Visualization Components (`/frontend/src/components/visualizations/`)
   - Network topology with real data
   - Resource heatmaps
   - Predictive charts
   - Alert correlation

### Implementation Requirements:

1. Update `/frontend/src/lib/api.ts` to use real API endpoints
2. Remove all mock data and simulated responses
3. Implement proper error handling for API failures
4. Add loading states for all API requests
5. Implement WebSocket connections for real-time updates
6. Add request caching for improved performance
7. Implement retry logic for failed requests
8. Add proper typing for all API responses

### Specific Implementation Tasks:

#### Task 1: Update API Service Layer
1. Modify `/frontend/src/lib/api.ts` to connect to real backend endpoints
2. Remove mock data implementations
3. Add proper error handling with user-friendly messages
4. Implement loading state management
5. Add request/response logging for debugging

#### Task 2: Implement WebSocket Connections
1. Create WebSocket connection service
2. Implement real-time updates for VM status
3. Add real-time updates for job/workflow executions
4. Implement WebSocket reconnection logic
5. Add proper error handling for WebSocket connections

#### Task 3: Update Dashboard Components
1. Modify `VMList` component to fetch real VM data
2. Update `JobList` component to fetch real job data
3. Update `WorkflowList` component to fetch real workflow data
4. Connect monitoring components to real metric data
5. Add proper loading and error states to all components

#### Task 4: Implement Data Validation
1. Add input validation for all form submissions
2. Implement proper error handling for validation failures
3. Add user feedback for successful operations
4. Implement confirmation dialogs for destructive actions

#### Task 5: Performance Optimization
1. Implement request caching to reduce API calls
2. Add pagination for large data sets
3. Implement data virtualization for large lists
4. Optimize real-time update frequency
5. Add lazy loading for non-critical data

### File Modifications Required:

1. `/frontend/src/lib/api.ts` - Update all API service functions
2. `/frontend/src/components/dashboard/vm-list.tsx` - Connect to real VM API
3. `/frontend/src/components/dashboard/job-list.tsx` - Connect to real Jobs API
4. `/frontend/src/components/dashboard/workflow-list.tsx` - Connect to real Workflows API
5. `/frontend/src/components/monitoring/MonitoringDashboard.tsx` - Connect to real monitoring data
6. `/frontend/src/components/visualizations/*` - Connect to real analytics data
7. `/frontend/src/hooks/useAPI.ts` - Update hooks to use real API services

### Testing Requirements:

1. Unit tests for all API service functions
2. Component tests for all dashboard components
3. Integration tests for API connections
4. E2E tests for complete user workflows
5. Performance testing for API response times
6. Error handling tests for API failures

### Success Criteria:

1. All dashboard components display real backend data
2. All user actions trigger actual backend operations
3. Real-time updates work through WebSocket connections
4. Proper error handling for all API failures
5. Loading states provide good user experience
6. Performance meets response time requirements (< 200ms for 95% of requests)
7. No mock data or simulated responses remain
8. System is production-ready with full functionality