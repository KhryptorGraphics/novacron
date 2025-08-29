-- NovaCron Database Initialization
-- This script sets up the basic database structure for NovaCron

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create basic tables for the system
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    address VARCHAR(255) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    cpu_cores INTEGER,
    memory_mb INTEGER,
    disk_gb INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS vms (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    node_id UUID REFERENCES nodes(id),
    status VARCHAR(50) DEFAULT 'stopped',
    cpu_cores INTEGER DEFAULT 1,
    memory_mb INTEGER DEFAULT 512,
    disk_gb INTEGER DEFAULT 10,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_nodes_status ON nodes(status);
CREATE INDEX IF NOT EXISTS idx_vms_status ON vms(status);
CREATE INDEX IF NOT EXISTS idx_vms_node_id ON vms(node_id);

-- Insert a default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, role) 
VALUES (
    'admin', 
    'admin@novacron.local', 
    crypt('admin123', gen_salt('bf')), 
    'admin'
) ON CONFLICT (username) DO NOTHING;

-- Insert a sample node
INSERT INTO nodes (name, address, cpu_cores, memory_mb, disk_gb) 
VALUES (
    'node-01', 
    '127.0.0.1', 
    4, 
    8192, 
    100
) ON CONFLICT DO NOTHING;