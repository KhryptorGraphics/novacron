-- Migration: networks
-- Direction: DOWN
-- Drops the networks catalog and the vms.network_id attachment. Host bridges
-- already provisioned for catalog rows are NOT removed by this migration;
-- delete the networks through the API first (DELETE /api/v1/networks/{id})
-- or remove the ncbr-* links by hand.

DROP INDEX IF EXISTS idx_vms_network_id;
ALTER TABLE vms DROP COLUMN IF EXISTS network_id;
DROP TABLE IF EXISTS networks;
