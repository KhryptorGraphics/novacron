package provision

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/google/uuid"
	"github.com/vishvananda/netlink"
)

// These tests drive the real netlink provisioner and therefore need
// CAP_NET_ADMIN. They only create links with fresh random names, but run them
// in a throwaway network namespace anyway:
//
//	cd backend/core && go test -c -o /tmp/provision.test ./network/provision/
//	sudo unshare --net /tmp/provision.test -test.run Linux -test.v
func requireNetAdmin(t *testing.T) {
	t.Helper()
	status, err := os.ReadFile("/proc/self/status")
	if err != nil {
		t.Skipf("cannot read capabilities: %v", err)
	}
	for _, line := range strings.Split(string(status), "\n") {
		if hex, ok := strings.CutPrefix(line, "CapEff:"); ok {
			caps, err := strconv.ParseUint(strings.TrimSpace(hex), 16, 64)
			if err == nil && caps&(1<<12) != 0 { // CAP_NET_ADMIN
				return
			}
		}
	}
	t.Skip("needs CAP_NET_ADMIN (e.g. sudo unshare --net <test binary>)")
}

// testSpec returns a fresh network spec plus cleanup of anything left behind.
func testSpec(t *testing.T, cidr, gateway string, vlan, mtu int) Spec {
	t.Helper()
	spec, err := NewSpec(uuid.NewString(), cidr, gateway, vlan, mtu)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		for _, name := range []string{vlanLinkName(spec.Bridge), spec.Bridge} {
			if link, err := netlink.LinkByName(name); err == nil {
				_ = netlink.LinkDel(link)
			}
		}
	})
	return spec
}

// addDummy creates a dummy link (removed at test end), optionally with addr.
func addDummy(t *testing.T, prefix, addr string) netlink.Link {
	t.Helper()
	name := prefix + strings.ReplaceAll(uuid.NewString(), "-", "")[:6]
	if err := netlink.LinkAdd(&netlink.Dummy{LinkAttrs: netlink.LinkAttrs{Name: name}}); err != nil {
		t.Fatalf("add dummy %s: %v", name, err)
	}
	link, err := netlink.LinkByName(name)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = netlink.LinkDel(link) })
	if addr != "" {
		a, err := netlink.ParseAddr(addr)
		if err != nil {
			t.Fatal(err)
		}
		if err := netlink.AddrAdd(link, a); err != nil {
			t.Fatal(err)
		}
	}
	if err := netlink.LinkSetUp(link); err != nil {
		t.Fatal(err)
	}
	return link
}

func mustLink(t *testing.T, name string) netlink.Link {
	t.Helper()
	link, err := netlink.LinkByName(name)
	if err != nil {
		t.Fatalf("link %s: %v", name, err)
	}
	return link
}

func assertAbsent(t *testing.T, name string) {
	t.Helper()
	if _, err := netlink.LinkByName(name); err == nil {
		t.Fatalf("link %s still exists", name)
	}
}

func globalAddrs(t *testing.T, link netlink.Link) []string {
	t.Helper()
	addrs, err := netlink.AddrList(link, netlink.FAMILY_ALL)
	if err != nil {
		t.Fatal(err)
	}
	var out []string
	for _, a := range addrs {
		if !a.IP.IsLinkLocalUnicast() {
			out = append(out, a.IPNet.String())
		}
	}
	return out
}

func TestLinuxEnsureConvergesAndRemoves(t *testing.T) {
	requireNetAdmin(t)
	ctx := context.Background()
	spec := testSpec(t, "198.18.77.0/24", "198.18.77.1", 0, 1400)
	aclDir := t.TempDir()
	p := &Linux{ACL: &BridgeACL{Dir: aclDir}}

	if err := p.Ensure(ctx, spec); err != nil {
		t.Fatalf("Ensure: %v", err)
	}
	br := mustLink(t, spec.Bridge)
	if br.Type() != "bridge" || br.Attrs().Alias != "novacron-network:"+spec.NetworkID {
		t.Fatalf("bridge type=%s alias=%q", br.Type(), br.Attrs().Alias)
	}
	if br.Attrs().MTU != 1400 || br.Attrs().Flags&net.FlagUp == 0 {
		t.Fatalf("bridge mtu=%d flags=%v", br.Attrs().MTU, br.Attrs().Flags)
	}
	if got := globalAddrs(t, br); fmt.Sprint(got) != "[198.18.77.1/24]" {
		t.Fatalf("bridge addrs = %v", got)
	}
	if acl := readFile(t, aclDir+"/novacron-bridges.conf"); !strings.Contains(acl, "allow "+spec.Bridge+"\n") {
		t.Fatalf("ACL missing bridge: %q", acl)
	}

	// Drift (stray address, changed MTU) is converged by a repeat Ensure.
	stray, _ := netlink.ParseAddr("198.18.78.5/24")
	if err := netlink.AddrAdd(br, stray); err != nil {
		t.Fatal(err)
	}
	if err := netlink.LinkSetMTU(br, 1500); err != nil {
		t.Fatal(err)
	}
	if err := p.Ensure(ctx, spec); err != nil {
		t.Fatalf("second Ensure: %v", err)
	}
	br = mustLink(t, spec.Bridge)
	if got := globalAddrs(t, br); fmt.Sprint(got) != "[198.18.77.1/24]" || br.Attrs().MTU != 1400 {
		t.Fatalf("after converge addrs=%v mtu=%d", got, br.Attrs().MTU)
	}

	if err := p.Remove(ctx, spec); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	assertAbsent(t, spec.Bridge)
	if acl := readFile(t, aclDir+"/novacron-bridges.conf"); strings.Contains(acl, spec.Bridge) {
		t.Fatalf("ACL still allows removed bridge: %q", acl)
	}
	if err := p.Remove(ctx, spec); err != nil {
		t.Fatalf("Remove of an absent network: %v", err)
	}
}

func TestLinuxVLANNetwork(t *testing.T) {
	requireNetAdmin(t)
	ctx := context.Background()
	uplink := addDummy(t, "nctu", "")
	spec := testSpec(t, "198.18.80.0/24", "", 123, 1500)

	if err := (&Linux{}).Ensure(ctx, spec); !errors.Is(err, ErrNoUplink) {
		t.Fatalf("Ensure without uplink: %v, want ErrNoUplink", err)
	}
	assertAbsent(t, spec.Bridge)

	p := &Linux{Uplink: uplink.Attrs().Name}
	for i := 0; i < 2; i++ {
		if err := p.Ensure(ctx, spec); err != nil {
			t.Fatalf("Ensure #%d: %v", i+1, err)
		}
	}
	br := mustLink(t, spec.Bridge)
	vl, ok := mustLink(t, vlanLinkName(spec.Bridge)).(*netlink.Vlan)
	if !ok {
		t.Fatal("vlan port is not a vlan link")
	}
	if vl.VlanId != 123 || vl.ParentIndex != uplink.Attrs().Index || vl.MasterIndex != br.Attrs().Index {
		t.Fatalf("vlan id=%d parent=%d master=%d, want 123/%d/%d", vl.VlanId, vl.ParentIndex, vl.MasterIndex, uplink.Attrs().Index, br.Attrs().Index)
	}
	if got := globalAddrs(t, br); len(got) != 0 {
		t.Fatalf("pure L2 bridge has addresses %v", got)
	}

	// The VLAN port is ours and does not count as "in use".
	if err := p.Remove(ctx, spec); err != nil {
		t.Fatalf("Remove: %v", err)
	}
	assertAbsent(t, spec.Bridge)
	assertAbsent(t, vlanLinkName(spec.Bridge))
	mustLink(t, uplink.Attrs().Name)
}

func TestLinuxNeverTouchesForeignLinks(t *testing.T) {
	requireNetAdmin(t)
	ctx := context.Background()
	spec := testSpec(t, "198.18.82.0/24", "198.18.82.1", 0, 1500)
	// Someone else's bridge that happens to carry the reserved name.
	if err := netlink.LinkAdd(&netlink.Bridge{LinkAttrs: netlink.LinkAttrs{Name: spec.Bridge, MTU: 1500}}); err != nil {
		t.Fatal(err)
	}
	p := &Linux{}

	if err := p.Ensure(ctx, spec); !errors.Is(err, ErrForeignLink) {
		t.Fatalf("Ensure over a foreign bridge: %v, want ErrForeignLink", err)
	}
	if err := p.Remove(ctx, spec); !errors.Is(err, ErrForeignLink) {
		t.Fatalf("Remove of a foreign bridge: %v, want ErrForeignLink", err)
	}
	br := mustLink(t, spec.Bridge)
	if br.Attrs().Alias != "" || len(globalAddrs(t, br)) != 0 || br.Attrs().Flags&net.FlagUp != 0 {
		t.Fatalf("foreign bridge was modified: alias=%q addrs=%v flags=%v", br.Attrs().Alias, globalAddrs(t, br), br.Attrs().Flags)
	}
}

func TestLinuxRemoveRefusesBridgeWithPorts(t *testing.T) {
	requireNetAdmin(t)
	ctx := context.Background()
	spec := testSpec(t, "198.18.84.0/24", "198.18.84.1", 0, 1500)
	p := &Linux{}
	if err := p.Ensure(ctx, spec); err != nil {
		t.Fatal(err)
	}
	guestTap := addDummy(t, "nctp", "")
	if err := netlink.LinkSetMaster(guestTap, mustLink(t, spec.Bridge)); err != nil {
		t.Fatal(err)
	}

	err := p.Remove(ctx, spec)
	if !errors.Is(err, ErrBridgeInUse) || !strings.Contains(err.Error(), guestTap.Attrs().Name) {
		t.Fatalf("Remove with an attached port: %v, want ErrBridgeInUse naming %s", err, guestTap.Attrs().Name)
	}
	mustLink(t, spec.Bridge)

	if err := netlink.LinkDel(guestTap); err != nil {
		t.Fatal(err)
	}
	if err := p.Remove(ctx, spec); err != nil {
		t.Fatalf("Remove after the port left: %v", err)
	}
	assertAbsent(t, spec.Bridge)
}

func TestLinuxRefusesGatewayOverlappingHostAddress(t *testing.T) {
	requireNetAdmin(t)
	ctx := context.Background()
	addDummy(t, "ncth", "198.18.90.1/24")
	spec := testSpec(t, "198.18.90.0/24", "198.18.90.254", 0, 1500)
	p := &Linux{}

	if err := p.Ensure(ctx, spec); !errors.Is(err, ErrAddressConflict) {
		t.Fatalf("Ensure: %v, want ErrAddressConflict", err)
	}
	assertAbsent(t, spec.Bridge)

	// Without a gateway no host route is added, so a pure L2 bridge is fine.
	l2 := spec
	l2.Gateway = netip.Addr{}
	if err := p.Ensure(ctx, l2); err != nil {
		t.Fatalf("pure L2 Ensure: %v", err)
	}
	if err := p.Remove(ctx, l2); err != nil {
		t.Fatal(err)
	}
}
