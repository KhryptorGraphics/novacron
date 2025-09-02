const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const jwt = require('jsonwebtoken');
const redis = require('redis');
const WebSocket = require('ws');
const { v4: uuidv4 } = require('uuid');
const morgan = require('morgan');
const compression = require('compression');
const http = require('http');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Configuration
const config = {
  port: process.env.PORT || 3000,
  jwtSecret: process.env.JWT_SECRET || 'mock-jwt-secret',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:15560',
  useMemoryDB: process.env.USE_MEMORY_DB === 'true'
};

// In-memory database
const memoryDB = {
  users: [
    { id: '1', username: 'admin', email: 'admin@novacron.local', role: 'admin', password: 'admin' },
    { id: '2', username: 'operator1', email: 'op1@novacron.local', role: 'operator', password: 'password' },
    { id: '3', username: 'user1', email: 'user1@novacron.local', role: 'user', password: 'password' }
  ],
  vms: [
    { id: '1', name: 'web-server-1', status: 'running', cpu_cores: 2, memory_mb: 2048, disk_gb: 40, os_type: 'Ubuntu 22.04 LTS', ip_address: '192.168.1.201', host_node: 'hypervisor-01', owner_id: '3' },
    { id: '2', name: 'web-server-2', status: 'stopped', cpu_cores: 4, memory_mb: 4096, disk_gb: 60, os_type: 'Ubuntu 22.04 LTS', ip_address: '192.168.1.202', host_node: 'hypervisor-02', owner_id: '3' },
    { id: '3', name: 'web-server-3', status: 'paused', cpu_cores: 2, memory_mb: 2048, disk_gb: 80, os_type: 'Ubuntu 22.04 LTS', ip_address: '192.168.1.203', host_node: 'hypervisor-03', owner_id: '3' },
    { id: '4', name: 'db-server-1', status: 'running', cpu_cores: 4, memory_mb: 8192, disk_gb: 100, os_type: 'CentOS 8', ip_address: '192.168.1.211', host_node: 'hypervisor-01', owner_id: '2' },
    { id: '5', name: 'db-server-2', status: 'running', cpu_cores: 4, memory_mb: 8192, disk_gb: 100, os_type: 'CentOS 8', ip_address: '192.168.1.212', host_node: 'hypervisor-02', owner_id: '2' }
  ],
  hypervisors: [
    { id: '1', name: 'hypervisor-01', hostname: 'hv01.novacron.local', ip_address: '192.168.1.101', status: 'online', cpu_cores: 16, memory_mb: 65536, disk_gb: 2048 },
    { id: '2', name: 'hypervisor-02', hostname: 'hv02.novacron.local', ip_address: '192.168.1.102', status: 'online', cpu_cores: 24, memory_mb: 131072, disk_gb: 4096 },
    { id: '3', name: 'hypervisor-03', hostname: 'hv03.novacron.local', ip_address: '192.168.1.103', status: 'maintenance', cpu_cores: 32, memory_mb: 262144, disk_gb: 8192 }
  ]
};

// Redis client
let redisClient;
const initRedis = async () => {
  try {
    redisClient = redis.createClient({ url: config.redisUrl });
    await redisClient.connect();
    console.log('✓ Connected to Redis');
  } catch (error) {
    console.log('⚠ Redis connection failed, using fallback mode');
    redisClient = null;
  }
};

// Middleware
app.use(helmet());
app.use(cors());
app.use(compression());
app.use(morgan('combined'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Mock data
const mockMetrics = {
  system: {
    cpu: Math.random() * 100,
    memory: Math.random() * 100,
    disk: Math.random() * 100,
    network_in: Math.random() * 1000000,
    network_out: Math.random() * 1000000
  }
};

// WebSocket connections
const wsClients = new Set();

wss.on('connection', (ws) => {
  wsClients.add(ws);
  console.log('WebSocket client connected');
  
  ws.send(JSON.stringify({
    type: 'system_metrics',
    data: mockMetrics.system,
    timestamp: new Date().toISOString()
  }));
  
  ws.on('close', () => wsClients.delete(ws));
  ws.on('error', () => wsClients.delete(ws));
});

// Broadcast to all WebSocket clients
const broadcast = (data) => {
  const message = JSON.stringify(data);
  wsClients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(message);
    }
  });
};

// Generate real-time metrics
setInterval(() => {
  mockMetrics.system = {
    cpu: Math.random() * 100,
    memory: Math.random() * 100,
    disk: Math.random() * 100,
    network_in: Math.random() * 1000000,
    network_out: Math.random() * 1000000
  };
  
  broadcast({
    type: 'system_metrics',
    data: mockMetrics.system,
    timestamp: new Date().toISOString()
  });
}, 5000);

// Authentication middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }
  
  jwt.verify(token, config.jwtSecret, (err, user) => {
    if (err) {
      return res.status(403).json({ error: 'Invalid token' });
    }
    req.user = user;
    next();
  });
};

// Health check
app.get('/health', async (req, res) => {
  let redisStatus = 'disconnected';
  try {
    if (redisClient) {
      await redisClient.ping();
      redisStatus = 'connected';
    }
  } catch (err) {
    redisStatus = 'error';
  }
  
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    services: {
      database: config.useMemoryDB ? 'memory' : 'external',
      redis: redisStatus,
      websocket: `${wsClients.size} clients`,
      memory_db: `${memoryDB.users.length} users, ${memoryDB.vms.length} vms`
    }
  });
});

// Authentication endpoints
app.post('/auth/login', (req, res) => {
  const { username, password } = req.body;
  
  if (!username || !password) {
    return res.status(400).json({ error: 'Username and password required' });
  }
  
  const user = memoryDB.users.find(u => u.username === username);
  
  if (!user || user.password !== password) {
    return res.status(401).json({ error: 'Invalid credentials' });
  }
  
  const token = jwt.sign(
    { 
      userId: user.id,
      username: user.username,
      role: user.role
    },
    config.jwtSecret,
    { expiresIn: '24h' }
  );
  
  res.json({
    token,
    user: {
      id: user.id,
      username: user.username,
      email: user.email,
      role: user.role
    }
  });
});

// VM Management endpoints
app.get('/api/vms', authenticateToken, (req, res) => {
  const { status, limit = 50, offset = 0 } = req.query;
  
  let vms = [...memoryDB.vms];
  
  if (status) {
    vms = vms.filter(vm => vm.status === status);
  }
  
  const startIndex = parseInt(offset);
  const endIndex = startIndex + parseInt(limit);
  const paginatedVMs = vms.slice(startIndex, endIndex);
  
  // Add owner username
  const enrichedVMs = paginatedVMs.map(vm => ({
    ...vm,
    owner_username: memoryDB.users.find(u => u.id === vm.owner_id)?.username || 'unknown',
    host_name: vm.host_node
  }));
  
  res.json({
    vms: enrichedVMs,
    total: vms.length,
    offset: startIndex,
    limit: parseInt(limit)
  });
});

app.post('/api/vms', authenticateToken, (req, res) => {
  const { name, cpu_cores, memory_mb, disk_gb, os_type } = req.body;
  
  if (!name || !cpu_cores || !memory_mb || !disk_gb || !os_type) {
    return res.status(400).json({ error: 'All VM parameters required' });
  }
  
  const newVM = {
    id: uuidv4(),
    name,
    cpu_cores: parseInt(cpu_cores),
    memory_mb: parseInt(memory_mb),
    disk_gb: parseInt(disk_gb),
    os_type,
    status: 'creating',
    ip_address: `192.168.1.${200 + Math.floor(Math.random() * 50)}`,
    host_node: 'hypervisor-01',
    owner_id: req.user.userId,
    created_at: new Date().toISOString()
  };
  
  memoryDB.vms.push(newVM);
  
  // Simulate VM creation process
  setTimeout(() => {
    const vm = memoryDB.vms.find(v => v.id === newVM.id);
    if (vm) {
      vm.status = 'stopped';
      broadcast({
        type: 'vm_status_change',
        data: { vm_id: vm.id, status: 'stopped' },
        timestamp: new Date().toISOString()
      });
    }
  }, 3000);
  
  res.status(201).json(newVM);
});

app.get('/api/vms/:id', authenticateToken, (req, res) => {
  const vm = memoryDB.vms.find(v => v.id === req.params.id);
  
  if (!vm) {
    return res.status(404).json({ error: 'VM not found' });
  }
  
  res.json(vm);
});

app.patch('/api/vms/:id', authenticateToken, (req, res) => {
  const vm = memoryDB.vms.find(v => v.id === req.params.id);
  
  if (!vm) {
    return res.status(404).json({ error: 'VM not found' });
  }
  
  const { status, cpu_cores, memory_mb, disk_gb } = req.body;
  
  if (status) vm.status = status;
  if (cpu_cores) vm.cpu_cores = parseInt(cpu_cores);
  if (memory_mb) vm.memory_mb = parseInt(memory_mb);
  if (disk_gb) vm.disk_gb = parseInt(disk_gb);
  vm.updated_at = new Date().toISOString();
  
  if (status) {
    broadcast({
      type: 'vm_status_change',
      data: { vm_id: vm.id, status: vm.status },
      timestamp: new Date().toISOString()
    });
  }
  
  res.json(vm);
});

app.delete('/api/vms/:id', authenticateToken, (req, res) => {
  const vmIndex = memoryDB.vms.findIndex(v => v.id === req.params.id);
  
  if (vmIndex === -1) {
    return res.status(404).json({ error: 'VM not found' });
  }
  
  memoryDB.vms.splice(vmIndex, 1);
  
  broadcast({
    type: 'vm_deleted',
    data: { vm_id: req.params.id },
    timestamp: new Date().toISOString()
  });
  
  res.status(204).send();
});

// VM Operations
app.post('/api/vms/:id/start', authenticateToken, (req, res) => {
  const vm = memoryDB.vms.find(v => v.id === req.params.id);
  
  if (!vm) {
    return res.status(404).json({ error: 'VM not found' });
  }
  
  vm.status = 'running';
  vm.updated_at = new Date().toISOString();
  
  broadcast({
    type: 'vm_status_change',
    data: { vm_id: req.params.id, status: 'running' },
    timestamp: new Date().toISOString()
  });
  
  res.json({ message: 'VM started successfully', vm });
});

app.post('/api/vms/:id/stop', authenticateToken, (req, res) => {
  const vm = memoryDB.vms.find(v => v.id === req.params.id);
  
  if (!vm) {
    return res.status(404).json({ error: 'VM not found' });
  }
  
  vm.status = 'stopped';
  vm.updated_at = new Date().toISOString();
  
  broadcast({
    type: 'vm_status_change',
    data: { vm_id: req.params.id, status: 'stopped' },
    timestamp: new Date().toISOString()
  });
  
  res.json({ message: 'VM stopped successfully', vm });
});

// Metrics endpoints
app.get('/api/metrics/system', authenticateToken, (req, res) => {
  res.json({
    ...mockMetrics.system,
    timestamp: new Date().toISOString()
  });
});

app.get('/api/metrics/vms/:id', authenticateToken, (req, res) => {
  // Generate mock metrics
  const metrics = [];
  for (let i = 0; i < 24; i++) {
    metrics.push({
      timestamp: new Date(Date.now() - i * 3600000).toISOString(),
      cpu_usage: Math.random() * 100,
      memory_usage: Math.random() * 100,
      disk_usage: Math.random() * 100,
      network_in_bytes: Math.floor(Math.random() * 1000000000),
      network_out_bytes: Math.floor(Math.random() * 1000000000)
    });
  }
  
  res.json({
    vm_id: req.params.id,
    metrics: metrics,
    latest: metrics[0] || null
  });
});

// Hypervisor nodes
app.get('/api/hypervisors', authenticateToken, (req, res) => {
  res.json(memoryDB.hypervisors);
});

// Users endpoint
app.get('/api/users', authenticateToken, (req, res) => {
  if (req.user.role !== 'admin') {
    return res.status(403).json({ error: 'Admin access required' });
  }
  
  const users = memoryDB.users.map(({ password, ...user }) => user);
  res.json(users);
});

// Dashboard stats
app.get('/api/dashboard/stats', authenticateToken, (req, res) => {
  const vmStats = memoryDB.vms.reduce((acc, vm) => {
    acc[vm.status] = (acc[vm.status] || 0) + 1;
    return acc;
  }, {});
  
  const nodeStats = memoryDB.hypervisors.reduce((acc, node) => {
    acc[node.status] = (acc[node.status] || 0) + 1;
    return acc;
  }, {});
  
  res.json({
    vms: {
      total: memoryDB.vms.length,
      running: vmStats.running || 0,
      stopped: vmStats.stopped || 0,
      paused: vmStats.paused || 0,
      error: vmStats.error || 0,
      creating: vmStats.creating || 0
    },
    users: {
      total: memoryDB.users.length
    },
    nodes: nodeStats,
    system: mockMetrics.system
  });
});

// Error handling
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ 
    error: 'Internal server error',
    message: err.message 
  });
});

app.use('*', (req, res) => {
  res.status(404).json({ error: 'Endpoint not found' });
});

// Start server
const startServer = async () => {
  try {
    if (!config.useMemoryDB) {
      await initRedis();
    }
    
    server.listen(config.port, () => {
      console.log(`🚀 NovaCron Mock API Server running on port ${config.port}`);
      console.log(`📊 WebSocket server ready for real-time updates`);
      console.log(`🔗 Health check: http://localhost:${config.port}/health`);
      console.log(`💾 Using ${config.useMemoryDB ? 'in-memory' : 'external'} database`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
};

startServer();