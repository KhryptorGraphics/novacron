-- Migration: cluster_node_identity
-- Created: 2026-09-20
-- Direction: DOWN

ALTER TABLE migrations ALTER COLUMN target_node_id TYPE UUID USING NULLIF(target_node_id, '')::uuid;
ALTER TABLE migrations ALTER COLUMN source_node_id TYPE UUID USING NULLIF(source_node_id, '')::uuid;
ALTER TABLE migrations ADD CONSTRAINT migrations_target_node_id_fkey FOREIGN KEY (target_node_id) REFERENCES nodes(id);
ALTER TABLE migrations ADD CONSTRAINT migrations_source_node_id_fkey FOREIGN KEY (source_node_id) REFERENCES nodes(id);

ALTER TABLE vms DROP COLUMN IF EXISTS requested_owner_id;
ALTER TABLE vms ALTER COLUMN node_id TYPE UUID USING NULLIF(node_id, '')::uuid;
ALTER TABLE vms ADD CONSTRAINT vms_node_id_fkey FOREIGN KEY (node_id) REFERENCES nodes(id) ON DELETE SET NULL;
