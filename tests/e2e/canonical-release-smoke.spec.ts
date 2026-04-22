import { expect, test } from '@playwright/test';

type CanonicalVm = {
  id: string;
  name: string;
  state: string;
  node_id: string;
  created_at: string;
  updated_at: string;
};

type CanonicalNetwork = {
  id: string;
  name: string;
  type: string;
  subnet: string;
  gateway?: string;
  status: string;
  created_at: string;
  updated_at: string;
};

type CanonicalVmInterface = {
  id: string;
  vm_id: string;
  network_id?: string;
  name: string;
  mac_address: string;
  ip_address?: string;
  status: string;
  created_at: string;
  updated_at: string;
};

type CanonicalUser = {
  id: number;
  username: string;
  email: string;
  role: string;
  active: boolean;
  created_at: string;
  updated_at: string;
};

type CanonicalVolume = {
  id: string;
  name: string;
  size: number;
  tier: 'HOT' | 'WARM' | 'COLD';
  vmId?: string;
  createdAt: string;
  updatedAt: string;
};

function encodeBase64Url(value: unknown) {
  return Buffer.from(JSON.stringify(value))
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/g, '');
}

function makeAdminToken() {
  const now = Math.floor(Date.now() / 1000);

  return [
    encodeBase64Url({ alg: 'HS256', typ: 'JWT' }),
    encodeBase64Url({
      sub: 'admin-1',
      email: 'admin@novacron.dev',
      first_name: 'Release',
      last_name: 'Admin',
      role: 'admin',
      roles: ['admin'],
      tenant_id: 'default',
      exp: now + 3600,
    }),
    'signature',
  ].join('.');
}

test.describe.configure({ mode: 'serial' });

test('canonical release routes load and support live rehabbed flows', async ({ page }) => {
  const now = '2026-04-18T00:00:00Z';
  const token = makeAdminToken();
  let complianceChecks = 0;
  const adminUser = {
    id: 'admin-1',
    email: 'admin@novacron.dev',
    firstName: 'Release',
    lastName: 'Admin',
    tenantId: 'default',
    status: 'active',
    role: 'admin',
    roles: ['admin'],
  };
  const selectedCluster = {
    id: 'cluster-a',
    name: 'Primary Fabric',
    tier: 'production',
    performanceScore: 98,
    interconnectLatencyMs: 12,
    interconnectBandwidthMbps: 20000,
    currentNodeCount: 3,
    maxSupportedNodeCount: 12,
    growthState: 'stable',
    federationState: 'healthy',
    degraded: false,
    lastEvaluatedAt: now,
  };
  const memberships = [
    {
      admitted: true,
      state: 'active',
      clusterId: selectedCluster.id,
      role: 'admin',
      source: 'smoke-fixture',
      admittedAt: now,
      selected: true,
      cluster: selectedCluster,
    },
  ];
  const session = {
    id: 'session-admin-1',
    expiresAt: '2026-04-18T12:00:00Z',
    createdAt: now,
    lastAccessedAt: now,
    selectedClusterId: selectedCluster.id,
  };

  const vms: CanonicalVm[] = [
    {
      id: 'vm-1',
      name: 'Alpha',
      state: 'running',
      node_id: 'node-a',
      created_at: now,
      updated_at: now,
    },
    {
      id: 'vm-2',
      name: 'Beta',
      state: 'stopped',
      node_id: 'node-b',
      created_at: now,
      updated_at: now,
    },
  ];

  const networks: CanonicalNetwork[] = [
    {
      id: 'net-1',
      name: 'Production',
      type: 'bridged',
      subnet: '192.168.10.0/24',
      gateway: '192.168.10.1',
      status: 'active',
      created_at: now,
      updated_at: now,
    },
  ];

  const interfacesByVm: Record<string, CanonicalVmInterface[]> = {
    'vm-1': [
      {
        id: 'eth0',
        vm_id: 'vm-1',
        network_id: 'net-1',
        name: 'eth0',
        mac_address: '00:16:3e:12:34:56',
        ip_address: '192.168.10.25',
        status: 'active',
        created_at: now,
        updated_at: now,
      },
    ],
    'vm-2': [],
  };

  const users: CanonicalUser[] = [
    {
      id: 1,
      username: 'admin',
      email: 'admin@novacron.dev',
      role: 'admin',
      active: true,
      created_at: now,
      updated_at: now,
    },
  ];

  const roleAssignments: Record<string, string[]> = {
    '1': ['admin'],
  };

  const volumes: CanonicalVolume[] = [
    {
      id: 'vol-1',
      name: 'primary-data',
      size: 100,
      tier: 'HOT',
      vmId: 'vm-1',
      createdAt: now,
      updatedAt: now,
    },
  ];

  await page.addInitScript((seed) => {
    window.localStorage.setItem('novacron_token', seed.token);
    window.localStorage.setItem('authUser', JSON.stringify(seed.user));
    window.localStorage.setItem('authMemberships', JSON.stringify(seed.memberships));
    window.localStorage.setItem('selectedCluster', JSON.stringify(seed.selectedCluster));
    window.localStorage.setItem('authSession', JSON.stringify(seed.session));
  }, {
    token,
    user: adminUser,
    memberships,
    selectedCluster,
    session,
  });

  await page.route('**/*', async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const path = url.pathname;
    const method = request.method();

    const fulfillJson = async (body: unknown, status = 200) => {
      await route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(body),
      });
    };

    if (path === '/api/auth/me' && method === 'GET') {
      await fulfillJson({
        user: adminUser,
        admission: memberships[0],
        memberships,
        selectedCluster,
        session,
      });
      return;
    }

    if (path === '/graphql' && method === 'POST') {
      const payload = request.postDataJSON() as { query?: string; variables?: Record<string, unknown> };
      const query = payload.query || '';

      if (query.includes('query Volumes')) {
        await fulfillJson({ data: { volumes } });
        return;
      }

      if (query.includes('mutation CreateVolume')) {
        const input = payload.variables?.input as Record<string, unknown>;
        const createdVolume: CanonicalVolume = {
          id: `vol-${volumes.length + 1}`,
          name: String(input.name),
          size: Number(input.size),
          tier: String(input.tier) as CanonicalVolume['tier'],
          vmId: typeof input.vmId === 'string' ? input.vmId : undefined,
          createdAt: now,
          updatedAt: now,
        };
        volumes.unshift(createdVolume);
        await fulfillJson({ data: { createVolume: createdVolume } });
        return;
      }

      if (query.includes('mutation ChangeVolumeTier')) {
        const id = String(payload.variables?.id);
        const tier = String(payload.variables?.tier) as CanonicalVolume['tier'];
        const volume = volumes.find((entry) => entry.id === id);
        if (volume) {
          volume.tier = tier;
          volume.updatedAt = now;
        }
        await fulfillJson({ data: { changeVolumeTier: volume } });
        return;
      }

      await fulfillJson({ errors: [{ message: 'Unsupported GraphQL operation' }] }, 400);
      return;
    }

    if (!path.startsWith('/api/')) {
      await route.continue();
      return;
    }

    if (path === '/api/v1/vms' && method === 'GET') {
      await fulfillJson({
        data: vms,
        error: null,
        pagination: { page: 1, pageSize: 50, total: vms.length, totalPages: 1 },
      });
      return;
    }

    const vmActionMatch = path.match(/^\/api\/v1\/vms\/([^/]+)\/(start|stop)$/);
    if (vmActionMatch && method === 'POST') {
      const [, vmId, action] = vmActionMatch;
      const vm = vms.find((entry) => entry.id === vmId);
      if (vm) {
        vm.state = action === 'start' ? 'running' : 'stopped';
        vm.updated_at = now;
      }
      await fulfillJson({ data: vm, error: null });
      return;
    }

    const vmInterfacesMatch = path.match(/^\/api\/v1\/vms\/([^/]+)\/interfaces(?:\/([^/]+))?$/);
    if (vmInterfacesMatch && method === 'GET') {
      const [, vmId] = vmInterfacesMatch;
      await fulfillJson(interfacesByVm[vmId] || []);
      return;
    }

    if (vmInterfacesMatch && method === 'POST') {
      const [, vmId] = vmInterfacesMatch;
      const payload = request.postDataJSON() as Record<string, unknown>;
      const createdInterface: CanonicalVmInterface = {
        id: `iface-${(interfacesByVm[vmId] || []).length + 1}`,
        vm_id: vmId,
        network_id: typeof payload.network_id === 'string' ? payload.network_id : undefined,
        name: String(payload.name),
        mac_address: String(payload.mac_address),
        ip_address: typeof payload.ip_address === 'string' ? payload.ip_address : undefined,
        status: 'active',
        created_at: now,
        updated_at: now,
      };
      interfacesByVm[vmId] = [...(interfacesByVm[vmId] || []), createdInterface];
      await fulfillJson(createdInterface, 201);
      return;
    }

    if (vmInterfacesMatch && method === 'DELETE') {
      const [, vmId, interfaceId] = vmInterfacesMatch;
      interfacesByVm[vmId] = (interfacesByVm[vmId] || []).filter((entry) => entry.id !== interfaceId);
      await route.fulfill({ status: 204, body: '' });
      return;
    }

    if (path === '/api/v1/monitoring/metrics' && method === 'GET') {
      await fulfillJson({
        currentCpuUsage: 42.5,
        currentMemoryUsage: 61.2,
        currentDiskUsage: 55.1,
        currentNetworkUsage: 128.4,
      });
      return;
    }

    if (path === '/api/v1/networks' && method === 'GET') {
      await fulfillJson(networks);
      return;
    }

    if (path === '/api/v1/networks' && method === 'POST') {
      const payload = request.postDataJSON() as Record<string, unknown>;
      const createdNetwork: CanonicalNetwork = {
        id: `net-${networks.length + 1}`,
        name: String(payload.name),
        type: String(payload.type),
        subnet: String(payload.subnet),
        gateway: typeof payload.gateway === 'string' ? payload.gateway : undefined,
        status: 'active',
        created_at: now,
        updated_at: now,
      };
      networks.unshift(createdNetwork);
      await fulfillJson(createdNetwork, 201);
      return;
    }

    const networkIdMatch = path.match(/^\/api\/v1\/networks\/([^/]+)$/);
    if (networkIdMatch && method === 'DELETE') {
      const [, networkId] = networkIdMatch;
      const index = networks.findIndex((entry) => entry.id === networkId);
      if (index >= 0) {
        networks.splice(index, 1);
      }
      await route.fulfill({ status: 204, body: '' });
      return;
    }

    if (path === '/api/admin/users' && method === 'GET') {
      await fulfillJson({
        users,
        total: users.length,
        page: 1,
        page_size: 100,
        total_pages: 1,
      });
      return;
    }

    if (path === '/api/admin/users' && method === 'POST') {
      const payload = request.postDataJSON() as Record<string, unknown>;
      const createdUser: CanonicalUser = {
        id: users.length + 1,
        username: String(payload.username),
        email: String(payload.email),
        role: typeof payload.role === 'string' ? payload.role : 'user',
        active: true,
        created_at: now,
        updated_at: now,
      };
      users.unshift(createdUser);
      roleAssignments[String(createdUser.id)] = [createdUser.role];
      await fulfillJson(createdUser, 201);
      return;
    }

    const adminUserMatch = path.match(/^\/api\/admin\/users\/(\d+)(?:\/roles)?$/);
    if (adminUserMatch && method === 'PUT') {
      const userId = Number(adminUserMatch[1]);
      const payload = request.postDataJSON() as Record<string, unknown>;
      const userRecord = users.find((entry) => entry.id === userId);
      if (userRecord) {
        userRecord.username = String(payload.username ?? userRecord.username);
        userRecord.email = String(payload.email ?? userRecord.email);
        userRecord.role = String(payload.role ?? userRecord.role);
        userRecord.active = typeof payload.active === 'boolean' ? payload.active : userRecord.active;
        userRecord.updated_at = now;
      }
      await fulfillJson(userRecord);
      return;
    }

    if (adminUserMatch && method === 'DELETE' && !path.endsWith('/roles')) {
      const userId = Number(adminUserMatch[1]);
      const index = users.findIndex((entry) => entry.id === userId);
      if (index >= 0) {
        users.splice(index, 1);
      }
      await route.fulfill({ status: 204, body: '' });
      return;
    }

    if (adminUserMatch && method === 'POST' && path.endsWith('/roles')) {
      const userId = adminUserMatch[1];
      const payload = request.postDataJSON() as { roles?: string[] };
      roleAssignments[userId] = Array.isArray(payload.roles) ? payload.roles : [];
      const userRecord = users.find((entry) => String(entry.id) === userId);
      if (userRecord && roleAssignments[userId][0]) {
        userRecord.role = roleAssignments[userId][0];
      }
      await fulfillJson({});
      return;
    }

    if (path.startsWith('/api/security/events') && method === 'GET') {
      await fulfillJson({
        events: [
          {
            id: 'evt-1',
            timestamp: now,
            type: 'threat',
            severity: 'high',
            source: 'sensor-a',
            user: 'admin@novacron.dev',
            resource: 'vm-1',
            action: 'Threat detected',
            result: 'blocked',
            details: 'Blocked suspicious login attempt',
            ip: '203.0.113.10',
          },
        ],
        total: 1,
      });
      return;
    }

    if (path === '/api/security/compliance' && method === 'GET') {
      await fulfillJson({
        compliance_score: 91,
        last_updated: now,
        frameworks: [
          { id: 'soc2', name: 'SOC 2', status: 'compliant' },
          { id: 'cis', name: 'CIS', status: 'partial' },
        ],
      });
      return;
    }

    if (path === '/api/security/compliance/check' && method === 'POST') {
      complianceChecks += 1;
      await fulfillJson({ jobId: `compliance-job-${complianceChecks}` });
      return;
    }

    if (path === '/api/security/vulnerabilities' && method === 'GET') {
      await fulfillJson({
        last_scan: now,
        summary: {
          critical: 0,
          high: 1,
          medium: 2,
          low: 1,
          info: 0,
        },
        vulnerabilities: [
          {
            id: 'finding-1',
            title: 'Outdated package',
            severity: 'high',
            component: 'api-server',
            description: 'A high-severity package version is present.',
            remediation: 'Upgrade the package.',
            exploitable: false,
          },
        ],
      });
      return;
    }

    if (path === '/api/security/threats' && method === 'GET') {
      await fulfillJson({
        threats: [
          { id: 'threat-1', severity: 'high' },
        ],
      });
      return;
    }

    if (path === '/api/security/rbac/roles' && method === 'GET') {
      await fulfillJson({
        roles: [
          {
            id: 'admin',
            name: 'Administrator',
            description: 'Full release-surface administration.',
            permissions: ['users:create', 'users:update', 'security:read'],
          },
          {
            id: 'operator',
            name: 'Operator',
            description: 'Operational access to VMs and monitoring.',
            permissions: ['vms:read', 'monitoring:read'],
          },
        ],
      });
      return;
    }

    if (path === '/api/security/rbac/permissions' && method === 'GET') {
      await fulfillJson({
        permissions: [
          {
            id: 'users:create',
            name: 'Create users',
            description: 'Create canonical admin users.',
          },
          {
            id: 'users:update',
            name: 'Update users',
            description: 'Update canonical admin users.',
          },
          {
            id: 'security:read',
            name: 'Read security',
            description: 'View security dashboards.',
          },
        ],
      });
      return;
    }

    const securityRoleMatch = path.match(/^\/api\/security\/rbac\/user\/([^/]+)\/roles$/);
    if (securityRoleMatch && method === 'GET') {
      const userId = securityRoleMatch[1];
      await fulfillJson({ roles: roleAssignments[userId] || [] });
      return;
    }

    if (securityRoleMatch && method === 'POST') {
      const userId = securityRoleMatch[1];
      const payload = request.postDataJSON() as { roles?: string[] };
      roleAssignments[userId] = Array.isArray(payload.roles) ? payload.roles : [];
      await fulfillJson({ roles: roleAssignments[userId] });
      return;
    }

    if (path.startsWith('/api/security/audit/events') && method === 'GET') {
      await fulfillJson({
        events: [
          {
            id: 'audit-1',
            timestamp: now,
            type: 'access',
            severity: 'info',
            source: 'audit',
            user: 'admin@novacron.dev',
            action: 'Role assignment updated',
            result: 'success',
            details: 'Administrator role assignment refreshed.',
          },
        ],
        total: 1,
      });
      return;
    }

    if (path === '/api/security/scan' && method === 'POST') {
      await fulfillJson({ scan_id: 'scan-1' }, 202);
      return;
    }

    await fulfillJson({ error: `Unhandled route for ${method} ${path}` }, 500);
  });

  await page.goto('/dashboard');
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible();
  await expect(page.getByText('Live operational overview for the release-candidate surface.')).toBeVisible();
  await expect(page.getByRole('link', { name: /Users/i })).toBeVisible();

  await page.goto('/vms');
  await expect(page.getByRole('heading', { name: 'Virtual Machines' })).toBeVisible();
  await expect(page.getByText('Alpha')).toBeVisible();

  await page.goto('/core/vms');
  await expect(page.getByRole('heading', { name: 'Virtual Machines' })).toBeVisible();
  await expect(page.getByText('Beta')).toBeVisible();

  await page.goto('/analytics');
  await expect(page.getByRole('heading', { name: 'Analytics' })).toBeVisible();
  await expect(page.getByText('Historical trends are intentionally unavailable')).toBeVisible();

  await page.goto('/settings');
  await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible();
  await expect(page.getByText('admin@novacron.dev')).toBeVisible();

  await page.goto('/users');
  await expect(page.getByRole('heading', { name: 'Users', exact: true })).toBeVisible();
  await expect(page.getByRole('cell', { name: 'admin@novacron.dev' })).toBeVisible();
  await page.getByRole('button', { name: 'Add User' }).click();
  await page.getByLabel('Username').fill('ops-user');
  await page.getByLabel('Email').fill('ops-user@novacron.dev');
  await page.getByLabel('Password').fill('StrongPass123!');
  await page.getByRole('button', { name: 'Create User' }).click();
  await expect(page.getByRole('cell', { name: 'ops-user', exact: true })).toBeVisible();

  await page.goto('/network');
  await expect(page.getByRole('heading', { name: 'Network', exact: true })).toBeVisible();
  await expect(page.getByText('Production', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Add Network' }).click();
  await page.getByLabel('Name').fill('Staging');
  await page.getByLabel('Subnet').fill('10.10.0.0/24');
  await page.getByLabel('Gateway').fill('10.10.0.1');
  await page.getByRole('button', { name: 'Create Network' }).click();
  await expect(page.getByText('Staging', { exact: true })).toBeVisible();

  await page.goto('/storage');
  await expect(page.getByRole('heading', { name: 'Storage' })).toBeVisible();
  await page.getByLabel('Name').fill('logs-volume');
  await page.getByLabel('Size (GiB)').fill('250');
  await page.getByRole('button', { name: 'Create Volume' }).click();
  await expect(page.getByText('logs-volume', { exact: true })).toBeVisible();

  await page.goto('/security');
  await expect(page.getByRole('heading', { name: 'Security & Compliance' })).toBeVisible();
  await page.getByRole('tab', { name: 'Compliance' }).click();
  await page.getByRole('button', { name: 'Recheck' }).first().click();
  await expect.poll(() => complianceChecks).toBe(1);

  await page.goto('/admin');
  await expect(page.getByRole('heading', { name: 'Admin Dashboard' })).toBeVisible();
});
