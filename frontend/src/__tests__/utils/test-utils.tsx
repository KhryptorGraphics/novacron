import React, { ReactElement } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Create a new QueryClient for each test
const createTestQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });
};

// Custom render function that includes providers
const AllTheProviders = ({ children }: { children: React.ReactNode }) => {
  const queryClient = createTestQueryClient();

  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

const customRender = (
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) => render(ui, { wrapper: AllTheProviders, ...options });

// Re-export everything
export * from '@testing-library/react';
export { customRender as render };

// Mock data helpers
export const mockVMData = {
  vm1: {
    id: 'vm-1',
    name: 'Test VM 1',
    state: 'running',
    cpu: 2,
    memory: 1024,
    disk: 20,
    os: 'ubuntu-24.04',
    ipAddress: '192.168.1.100',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  vm2: {
    id: 'vm-2',
    name: 'Test VM 2',
    state: 'stopped',
    cpu: 4,
    memory: 2048,
    disk: 40,
    os: 'centos-8',
    ipAddress: '192.168.1.101',
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
};

export const mockUserData = {
  user1: {
    id: 'user-1',
    email: 'test@test.local',
    firstName: 'Test',
    lastName: 'User',
    accountType: 'personal',
    createdAt: '2024-01-01T00:00:00Z',
  },
  user2: {
    id: 'user-2',
    email: 'admin@example.com',
    firstName: 'Admin',
    lastName: 'User',
    accountType: 'organization',
    organizationName: 'Test Org',
    createdAt: '2024-01-01T00:00:00Z',
  },
};

export const mockMetricsData = {
  cpuUsage: [
    { timestamp: '2024-01-01T00:00:00Z', value: 45.2 },
    { timestamp: '2024-01-01T00:01:00Z', value: 52.1 },
    { timestamp: '2024-01-01T00:02:00Z', value: 38.9 },
  ],
  memoryUsage: [
    { timestamp: '2024-01-01T00:00:00Z', value: 67.8 },
    { timestamp: '2024-01-01T00:01:00Z', value: 71.2 },
    { timestamp: '2024-01-01T00:02:00Z', value: 69.5 },
  ],
};

// API mocking helpers
export const mockApiResponse = (data: any, status = 200, ok = true) => ({
  ok,
  status,
  json: async () => data,
  text: async () => JSON.stringify(data),
  headers: new Headers({
    'content-type': 'application/json',
  }),
});

export const mockApiError = (message: string, status = 500) => ({
  ok: false,
  status,
  json: async () => ({ error: message }),
  text: async () => JSON.stringify({ error: message }),
  headers: new Headers({
    'content-type': 'application/json',
  }),
});

// Form testing helpers
export const fillForm = async (user: any, fields: Record<string, string>) => {
  for (const [label, value] of Object.entries(fields)) {
    const field = screen.getByLabelText(new RegExp(label, 'i'));
    await user.clear(field);
    await user.type(field, value);
  }
};

export const submitForm = async (user: any, buttonText = /submit/i) => {
  const submitButton = screen.getByRole('button', { name: buttonText });
  await user.click(submitButton);
};
