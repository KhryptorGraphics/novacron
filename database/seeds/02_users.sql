-- Seed Users
-- Development and test users with different roles
-- Default password for all users: Password123!

INSERT INTO users (id, email, username, password_hash, first_name, last_name, organization_id, role, status, email_verified) VALUES
    -- Admin users
    ('a1eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'admin@novacron.io', 'admin', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Admin', 'User', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'admin', 'active', true),
    ('a2eebc99-9c0b-4ef8-bb6d-6bb9bd380a12', 'superadmin@novacron.io', 'superadmin', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Super', 'Admin', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'admin', 'active', true),
    
    -- Operator users
    ('b1eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', 'operator1@novacron.io', 'operator1', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'John', 'Operator', 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'operator', 'active', true),
    ('b2eebc99-9c0b-4ef8-bb6d-6bb9bd380a14', 'operator2@acme.com', 'operator2', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Jane', 'Smith', 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a12', 'operator', 'active', true),
    
    -- Viewer users
    ('c1eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', 'viewer1@techstart.io', 'viewer1', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Bob', 'Viewer', 'c0eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', 'viewer', 'active', true),
    ('c2eebc99-9c0b-4ef8-bb6d-6bb9bd380a16', 'viewer2@enterprise.com', 'viewer2', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Alice', 'Monitor', 'd0eebc99-9c0b-4ef8-bb6d-6bb9bd380a14', 'viewer', 'active', true),
    
    -- Test users with various states
    ('d1eebc99-9c0b-4ef8-bb6d-6bb9bd380a17', 'inactive@test.com', 'inactive_user', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Inactive', 'User', 'e0eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', 'viewer', 'inactive', false),
    ('d2eebc99-9c0b-4ef8-bb6d-6bb9bd380a18', 'suspended@test.com', 'suspended_user', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Suspended', 'Account', 'e0eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', 'viewer', 'suspended', false),
    ('d3eebc99-9c0b-4ef8-bb6d-6bb9bd380a19', 'pending@test.com', 'pending_user', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', 'Pending', 'Verification', 'e0eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', 'viewer', 'pending', false)
ON CONFLICT (id) DO NOTHING;

-- Create some API keys for testing
INSERT INTO api_keys (id, user_id, name, key_hash, scopes) VALUES
    ('e1eebc99-9c0b-4ef8-bb6d-6bb9bd380a20', 'a1eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'Admin API Key', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', ARRAY['read', 'write', 'admin']),
    ('e2eebc99-9c0b-4ef8-bb6d-6bb9bd380a21', 'b1eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', 'Operator API Key', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', ARRAY['read', 'write']),
    ('e3eebc99-9c0b-4ef8-bb6d-6bb9bd380a22', 'c1eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', 'Viewer API Key', '$2a$10$rBV2JDeWW3.vKyeQcM8fFO4777l4bVeQgDL6VZkZJer5pNlaW1lKu', ARRAY['read'])
ON CONFLICT (id) DO NOTHING;