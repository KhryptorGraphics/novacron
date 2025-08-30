-- Seed Organizations
-- Development and test organizations

INSERT INTO organizations (id, name, slug, size) VALUES
    ('a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11', 'NovaCron Inc', 'novacron', '50-100'),
    ('b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a12', 'Acme Corporation', 'acme', '100-500'),
    ('c0eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', 'TechStart', 'techstart', '1-10'),
    ('d0eebc99-9c0b-4ef8-bb6d-6bb9bd380a14', 'Enterprise Co', 'enterprise', '1000+'),
    ('e0eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', 'Dev Team', 'dev-team', '10-50')
ON CONFLICT (id) DO NOTHING;