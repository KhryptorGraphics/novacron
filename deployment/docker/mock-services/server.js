const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const { Pool } = require('pg');
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
  dbUrl: process.env.DATABASE_URL || 'postgresql://novacron:novacron123@localhost:15555/novacron',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:15560'
};

// Database connection
const pool = new Pool({
  connectionString: config.dbUrl,
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});

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
  },
  vms: {},
  alerts: []
};

// WebSocket connections
const wsClients = new Set();

wss.on('connection', (ws) => {
  wsClients.add(ws);
  console.log('WebSocket client connected');
  
  // Send initial data
  ws.send(JSON.stringify({
    type: 'system_metrics',
    data: mockMetrics.system,
    timestamp: new Date().toISOString()
  }));
  
  ws.on('close', () => {
    wsClients.delete(ws);
    console.log('WebSocket client disconnected');
  });
  
  ws.on('error', (error) => {
    console.error('WebSocket error:', error);
    wsClients.delete(ws);
  });
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
  try {
    // Test database connection
    const dbResult = await pool.query('SELECT NOW()');
    
    // Test Redis connection
    let redisStatus = 'connected';
    try {
      if (redisClient) {
        await redisClient.ping();
      } else {
        redisStatus = 'disconnected';
      }
    } catch (err) {
      redisStatus = 'error';
    }
    
    res.json({
      status: 'healthy',
      timestamp: new Date().toISOString(),
      uptime: process.uptime(),
      services: {
        database: 'connected',
        redis: redisStatus,
        websocket: `${wsClients.size} clients`
      }
    });
  } catch (error) {
    res.status(503).json({
      status: 'unhealthy',
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

// Authentication endpoints
app.post('/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }
    
    // Get user from database
    const result = await pool.query('SELECT * FROM users WHERE username = $1', [username]);
    
    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    
    const user = result.rows[0];
    
    // For demo purposes, accept any password for existing users
    const token = jwt.sign(
      { 
        userId: user.id,
        username: user.username,
        role: user.role
      },
      config.jwtSecret,
      { expiresIn: '24h' }
    );
    
    // Update last login
    await pool.query('UPDATE users SET last_login = NOW() WHERE id = $1', [user.id]);
    
    res.json({
      token,
      user: {
        id: user.id,
        username: user.username,
        email: user.email,
        role: user.role
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/auth/register', async (req, res) => {
  try {
    const { username, email, password } = req.body;
    
    if (!username || !email || !password) {
      return res.status(400).json({ error: 'Username, email and password required' });
    }
    
    const hashedPassword = await bcrypt.hash(password, 10);
    
    const result = await pool.query(
      'INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3) RETURNING id, username, email, role',
      [username, email, hashedPassword]
    );
    
    const user = result.rows[0];
    
    const token = jwt.sign(
      { 
        userId: user.id,
        username: user.username,
        role: user.role
      },
      config.jwtSecret,
      { expiresIn: '24h' }
    );
    
    res.status(201).json({
      token,
      user
    });
  } catch (error) {
    if (error.code === '23505') {
      res.status(409).json({ error: 'Username or email already exists' });
    } else {
      res.status(500).json({ error: error.message });
    }
  }
});

// VM Management endpoints
app.get('/api/vms', authenticateToken, async (req, res) => {
  try {
    const { status, limit = 50, offset = 0 } = req.query;
    
    let query = `
      SELECT vm.*, u.username as owner_username, hn.name as host_name
      FROM virtual_machines vm
      LEFT JOIN users u ON vm.owner_id = u.id
      LEFT JOIN hypervisor_nodes hn ON vm.host_node = hn.name
    `;
    const params = [];
    
    if (status) {
      query += ' WHERE vm.status = $1';
      params.push(status);
    }
    
    query += ` ORDER BY vm.created_at DESC LIMIT $${params.length + 1} OFFSET $${params.length + 2}`;
    params.push(parseInt(limit), parseInt(offset));
    
    const result = await pool.query(query, params);
    
    res.json({
      vms: result.rows,
      total: result.rows.length,
      offset: parseInt(offset),
      limit: parseInt(limit)
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/vms', authenticateToken, async (req, res) => {
  try {
    const { name, cpu_cores, memory_mb, disk_gb, os_type } = req.body;
    
    if (!name || !cpu_cores || !memory_mb || !disk_gb || !os_type) {
      return res.status(400).json({ error: 'All VM parameters required' });
    }
    
    // Get available host
    const hostResult = await pool.query(
      'SELECT name FROM hypervisor_nodes WHERE status = $1 ORDER BY RANDOM() LIMIT 1',
      ['online']
    );
    
    const host_node = hostResult.rows.length > 0 ? hostResult.rows[0].name : 'hypervisor-01';
    
    const result = await pool.query(
      `INSERT INTO virtual_machines 
       (name, cpu_cores, memory_mb, disk_gb, os_type, host_node, owner_id, status) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8) 
       RETURNING *`,
      [name, cpu_cores, memory_mb, disk_gb, os_type, host_node, req.user.userId, 'creating']
    );
    
    const vm = result.rows[0];
    
    // Simulate VM creation process
    setTimeout(async () => {
      try {
        await pool.query('UPDATE virtual_machines SET status = $1 WHERE id = $2', ['stopped', vm.id]);
        
        broadcast({
          type: 'vm_status_change',
          data: { vm_id: vm.id, status: 'stopped' },
          timestamp: new Date().toISOString()
        });
      } catch (err) {
        console.error('Error updating VM status:', err);
      }
    }, 3000);
    
    res.status(201).json(vm);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/vms/:id', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(
      'SELECT * FROM virtual_machines WHERE id = $1',
      [req.params.id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'VM not found' });
    }
    
    res.json(result.rows[0]);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.patch('/api/vms/:id', authenticateToken, async (req, res) => {
  try {
    const { status, cpu_cores, memory_mb, disk_gb } = req.body;
    const updates = [];
    const params = [];
    let paramCount = 1;
    
    if (status) {
      updates.push(`status = $${paramCount++}`);
      params.push(status);
    }
    if (cpu_cores) {
      updates.push(`cpu_cores = $${paramCount++}`);
      params.push(cpu_cores);
    }
    if (memory_mb) {
      updates.push(`memory_mb = $${paramCount++}`);
      params.push(memory_mb);
    }
    if (disk_gb) {
      updates.push(`disk_gb = $${paramCount++}`);
      params.push(disk_gb);
    }
    
    if (updates.length === 0) {
      return res.status(400).json({ error: 'No valid updates provided' });
    }
    
    updates.push(`updated_at = NOW()`);
    params.push(req.params.id);
    
    const result = await pool.query(
      `UPDATE virtual_machines SET ${updates.join(', ')} WHERE id = $${paramCount} RETURNING *`,
      params
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'VM not found' });
    }
    
    const vm = result.rows[0];
    
    // Broadcast status change
    if (status) {
      broadcast({
        type: 'vm_status_change',
        data: { vm_id: vm.id, status: vm.status },
        timestamp: new Date().toISOString()
      });
    }
    
    res.json(vm);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.delete('/api/vms/:id', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(
      'DELETE FROM virtual_machines WHERE id = $1 RETURNING *',
      [req.params.id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'VM not found' });
    }
    
    broadcast({
      type: 'vm_deleted',
      data: { vm_id: req.params.id },
      timestamp: new Date().toISOString()
    });
    
    res.status(204).send();
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// VM Operations
app.post('/api/vms/:id/start', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(
      'UPDATE virtual_machines SET status = $1, updated_at = NOW() WHERE id = $2 RETURNING *',
      ['running', req.params.id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'VM not found' });
    }
    
    broadcast({
      type: 'vm_status_change',
      data: { vm_id: req.params.id, status: 'running' },
      timestamp: new Date().toISOString()
    });
    
    res.json({ message: 'VM started successfully', vm: result.rows[0] });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/vms/:id/stop', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(
      'UPDATE virtual_machines SET status = $1, updated_at = NOW() WHERE id = $2 RETURNING *',
      ['stopped', req.params.id]
    );
    
    if (result.rows.length === 0) {
      return res.status(404).json({ error: 'VM not found' });
    }
    
    broadcast({
      type: 'vm_status_change',
      data: { vm_id: req.params.id, status: 'stopped' },
      timestamp: new Date().toISOString()
    });
    
    res.json({ message: 'VM stopped successfully', vm: result.rows[0] });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Metrics endpoints
app.get('/api/metrics/system', authenticateToken, (req, res) => {
  res.json({
    ...mockMetrics.system,
    timestamp: new Date().toISOString()
  });
});

app.get('/api/metrics/vms/:id', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query(
      `SELECT * FROM vm_metrics 
       WHERE vm_id = $1 
       ORDER BY timestamp DESC 
       LIMIT 100`,
      [req.params.id]
    );
    
    res.json({
      vm_id: req.params.id,
      metrics: result.rows,
      latest: result.rows[0] || null
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Hypervisor nodes
app.get('/api/hypervisors', authenticateToken, async (req, res) => {
  try {
    const result = await pool.query('SELECT * FROM hypervisor_nodes ORDER BY name');
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Users endpoint
app.get('/api/users', authenticateToken, async (req, res) => {
  try {
    if (req.user.role !== 'admin') {
      return res.status(403).json({ error: 'Admin access required' });
    }
    
    const result = await pool.query(
      'SELECT id, username, email, role, created_at, last_login, is_active FROM users ORDER BY created_at'
    );
    res.json(result.rows);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Dashboard stats
app.get('/api/dashboard/stats', authenticateToken, async (req, res) => {
  try {
    const vmCountResult = await pool.query('SELECT status, COUNT(*) as count FROM virtual_machines GROUP BY status');
    const userCountResult = await pool.query('SELECT COUNT(*) as count FROM users');
    const nodeCountResult = await pool.query('SELECT status, COUNT(*) as count FROM hypervisor_nodes GROUP BY status');
    
    const vmStats = {};
    vmCountResult.rows.forEach(row => {
      vmStats[row.status] = parseInt(row.count);
    });
    
    res.json({
      vms: {
        total: vmCountResult.rows.reduce((sum, row) => sum + parseInt(row.count), 0),
        running: vmStats.running || 0,
        stopped: vmStats.stopped || 0,
        paused: vmStats.paused || 0,
        error: vmStats.error || 0
      },
      users: {
        total: parseInt(userCountResult.rows[0].count)
      },
      nodes: nodeCountResult.rows.reduce((acc, row) => {
        acc[row.status] = parseInt(row.count);
        return acc;
      }, {}),
      system: mockMetrics.system
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
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
    // Initialize Redis
    await initRedis();
    
    // Test database connection
    await pool.query('SELECT NOW()');
    console.log('✓ Database connected');
    
    server.listen(config.port, () => {
      console.log(`🚀 NovaCron Mock API Server running on port ${config.port}`);
      console.log(`📊 WebSocket server ready for real-time updates`);
      console.log(`🔗 Health check: http://localhost:${config.port}/health`);
    });
  } catch (error) {
    console.error('❌ Failed to start server:', error);
    process.exit(1);
  }
};

startServer();