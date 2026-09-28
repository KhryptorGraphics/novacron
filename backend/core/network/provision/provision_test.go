package provision

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const testNetID = "0f3c2a9e-51d4-4b7a-9c2e-7d1f00aa1234"

func TestBridgeNameIsDerivedAndFitsIFNAMSIZ(t *testing.T) {
	name, err := BridgeName(testNetID)
	if err != nil {
		t.Fatal(err)
	}
	if name != "ncbr-0f3c2a9e51" {
		t.Fatalf("BridgeName = %q", name)
	}
	if len(name) > 15 || len(vlanLinkName(name)) > 15 {
		t.Fatalf("%q / %q exceed IFNAMSIZ-1", name, vlanLinkName(name))
	}
	upper, _ := BridgeName(strings.ToUpper(testNetID))
	if upper != name {
		t.Fatalf("BridgeName is case-sensitive: %q vs %q", upper, name)
	}
	if _, err := BridgeName("not-a-uuid"); err == nil {
		t.Fatal("non-UUID network id accepted")
	}
}

func TestNewSpecValidation(t *testing.T) {
	cases := []struct {
		name    string
		cidr    string
		gateway string
		vlan    int
		mtu     int
		wantErr string // "" = valid
	}{
		{"ipv4 with gateway", "10.20.0.0/24", "10.20.0.1", 0, 1500, ""},
		{"ipv4 pure L2", "10.20.0.0/24", "", 0, 1500, ""},
		{"vlan bounds low", "10.20.0.0/24", "", 1, 1500, ""},
		{"vlan bounds high", "10.20.0.0/24", "", 4094, 9000, ""},
		{"mtu floor", "10.20.0.0/24", "", 0, 576, ""},
		{"ipv6", "fd00:10::/64", "fd00:10::1", 0, 1500, ""},
		{"host bits set", "10.20.0.5/24", "", 0, 1500, "host bits"},
		{"garbage cidr", "10.20.0.0", "", 0, 1500, "not a valid network prefix"},
		{"prefix too long", "10.20.0.0/31", "", 0, 1500, "between /8 and /30"},
		{"gateway outside", "10.20.0.0/24", "10.21.0.1", 0, 1500, "not inside"},
		{"gateway is network address", "10.20.0.0/24", "10.20.0.0", 0, 1500, "network address"},
		{"gateway is broadcast", "10.20.0.0/24", "10.20.0.255", 0, 1500, "broadcast"},
		{"gateway wrong family", "10.20.0.0/24", "fd00::1", 0, 1500, "address family"},
		{"gateway garbage", "10.20.0.0/24", "10.20.0.x", 0, 1500, "not a valid IP"},
		{"vlan 4095", "10.20.0.0/24", "", 4095, 1500, "vlan_id"},
		{"vlan negative", "10.20.0.0/24", "", -1, 1500, "vlan_id"},
		{"mtu too small", "10.20.0.0/24", "", 0, 575, "mtu"},
		{"mtu too large", "10.20.0.0/24", "", 0, 9001, "mtu"},
		{"ipv6 below 1280", "fd00:10::/64", "", 0, 1000, "IPv6 minimum"},
		{"loopback", "127.1.0.0/16", "", 0, 1500, "reserved range"},
		{"link-local", "169.254.10.0/24", "", 0, 1500, "reserved range"},
		{"multicast", "239.1.0.0/16", "", 0, 1500, "reserved range"},
		{"contains loopback", "::/16", "", 0, 1500, "reserved range"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec, err := NewSpec(testNetID, tc.cidr, tc.gateway, tc.vlan, tc.mtu)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if spec.Bridge != "ncbr-0f3c2a9e51" {
					t.Fatalf("bridge = %q", spec.Bridge)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("err = %v, want containing %q", err, tc.wantErr)
			}
		})
	}
}

func TestSpecValidateRejectsMismatchedBridge(t *testing.T) {
	spec, err := NewSpec(testNetID, "10.20.0.0/24", "", 0, 1500)
	if err != nil {
		t.Fatal(err)
	}
	spec.Bridge = "docker0"
	if err := spec.Validate(); err == nil {
		t.Fatal("spec naming a foreign bridge validated")
	}
}

func TestBridgeACLAllowRevoke(t *testing.T) {
	dir := t.TempDir()
	helperConf := filepath.Join(dir, "bridge.conf")
	// An operator's existing ACL must survive untouched apart from the include.
	if err := os.WriteFile(helperConf, []byte("allow virbr0"), 0o644); err != nil {
		t.Fatal(err)
	}
	acl := &BridgeACL{Dir: dir}
	for i := 0; i < 2; i++ { // idempotent
		if err := acl.Allow("ncbr-0f3c2a9e51"); err != nil {
			t.Fatal(err)
		}
	}
	if err := acl.Allow("ncbr-aaaaaaaaaa"); err != nil {
		t.Fatal(err)
	}
	listPath := filepath.Join(dir, "novacron-bridges.conf")
	wantInclude := "allow virbr0\ninclude " + listPath + "\n"
	if got := readFile(t, helperConf); got != wantInclude {
		t.Fatalf("bridge.conf = %q, want %q", got, wantInclude)
	}
	if got := readFile(t, listPath); !strings.HasSuffix(got, "allow ncbr-0f3c2a9e51\nallow ncbr-aaaaaaaaaa\n") {
		t.Fatalf("allow list = %q", got)
	}

	if err := acl.Revoke("ncbr-0f3c2a9e51"); err != nil {
		t.Fatal(err)
	}
	got := readFile(t, listPath)
	if strings.Contains(got, "ncbr-0f3c2a9e51") || !strings.Contains(got, "allow ncbr-aaaaaaaaaa\n") {
		t.Fatalf("allow list after revoke = %q", got)
	}
	if got := readFile(t, helperConf); got != wantInclude {
		t.Fatalf("revoke changed bridge.conf: %q", got)
	}

	if err := acl.Allow("virbr0"); err == nil {
		t.Fatal("ACL accepted a non-NovaCron bridge")
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}
