-- Migration: networks
-- Direction: UP
-- Networks catalog (novacron-ed7). Each row is one NovaCron-managed Linux
-- bridge on this node (backend/core/network/provision): `bridge` is derived
-- from the id (ncbr-<first 10 hex of the id>), `gateway` (when set) is the
-- host address on the bridge, `vlan_id` (when set) tags the bridge onto the
-- node uplink. A KVM guest created with network_id gets a virtio-net NIC on
-- that bridge; vms.network_id records the attachment and blocks deleting a
-- network that still has VMs (ON DELETE RESTRICT).

CREATE TABLE networks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(63) NOT NULL,
    bridge VARCHAR(15) NOT NULL,
    cidr CIDR NOT NULL,
    gateway INET,
    vlan_id INTEGER,
    mtu INTEGER NOT NULL DEFAULT 1500,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    CONSTRAINT networks_name_format CHECK (name ~ '^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$'),
    CONSTRAINT networks_bridge_key UNIQUE (bridge),
    CONSTRAINT networks_bridge_format CHECK (bridge ~ '^ncbr-[0-9a-f]{10}$'),
    CONSTRAINT networks_vlan_range CHECK (vlan_id IS NULL OR vlan_id BETWEEN 1 AND 4094),
    CONSTRAINT networks_mtu_range CHECK (mtu BETWEEN 576 AND 9000),
    CONSTRAINT networks_gateway_in_cidr CHECK (gateway IS NULL OR (masklen(gateway) = CASE family(gateway) WHEN 4 THEN 32 ELSE 128 END AND gateway << cidr)),
    -- Two bridges routing overlapping prefixes on one host would black-hole
    -- one of them; the exclusion constraint makes that race-proof.
    CONSTRAINT networks_cidr_no_overlap EXCLUDE USING gist (cidr inet_ops WITH &&)
);

-- Names are unique case-insensitively ("Prod" and "prod" are one network).
CREATE UNIQUE INDEX networks_name_lower_key ON networks (lower(name));

CREATE TRIGGER update_networks_updated_at BEFORE UPDATE ON networks
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

ALTER TABLE vms ADD COLUMN network_id UUID REFERENCES networks(id) ON DELETE RESTRICT;
CREATE INDEX idx_vms_network_id ON vms(network_id) WHERE network_id IS NOT NULL;

COMMENT ON TABLE networks IS 'NovaCron-managed host bridges (the /api/v1/networks catalog); provisioned by backend/core/network/provision.';
COMMENT ON COLUMN networks.bridge IS 'Host bridge interface name, ncbr-<first 10 hex digits of id>; links carry the alias novacron-network:<id>.';
COMMENT ON COLUMN networks.gateway IS 'Host address assigned to the bridge (NULL = pure L2 bridge).';
COMMENT ON COLUMN networks.vlan_id IS '802.1Q tag on the node uplink (NOVACRON_NETWORK_UPLINK) enslaved to the bridge; NULL = untagged, bridge-local.';
COMMENT ON COLUMN vms.network_id IS 'Catalog network the VM''s primary NIC is bridged onto (NULL = isolated user-mode NIC).';
