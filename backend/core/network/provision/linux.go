package provision

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/netip"
	"sort"
	"strings"
	"sync"
	"syscall"

	"github.com/vishvananda/netlink"
)

// ErrAddressConflict: the network has a gateway and its prefix overlaps an
// address already configured on another host interface, so assigning the
// gateway would add a competing route for that prefix.
var ErrAddressConflict = errors.New("cidr overlaps an address on another host interface")

// Linux realises catalog networks on this host's network stack through
// netlink. It needs CAP_NET_ADMIN, and write access to ACL.Dir when an ACL is
// configured.
type Linux struct {
	// Uplink is the host interface VLAN networks are tagged onto; empty means
	// this node cannot provision VLAN networks (Ensure returns ErrNoUplink).
	Uplink string
	// ACL, when non-nil, keeps qemu-bridge-helper's allow list in sync:
	// Ensure allows the bridge, Remove revokes it.
	ACL *BridgeACL

	mu sync.Mutex // serialises Ensure/Remove so a converge never interleaves
}

var _ Provisioner = (*Linux)(nil)

// Ensure creates or converges the bridge (MTU, owner alias, gateway address,
// link up), the VLAN sub-interface of Uplink enslaved to it when
// spec.VLANID != 0, and the qemu-bridge-helper ACL entry. Nothing is changed
// when a reserved name belongs to someone else (ErrForeignLink) or the
// gateway prefix collides with another interface (ErrAddressConflict).
func (l *Linux) Ensure(ctx context.Context, spec Spec) error {
	if err := spec.Validate(); err != nil {
		return err
	}
	if spec.VLANID != 0 && l.Uplink == "" {
		return ErrNoUplink
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	alias := ownerAlias(spec.NetworkID)
	vlanName := vlanLinkName(spec.Bridge)
	// Refuse before mutating anything: foreign links under our names and
	// gateway prefix collisions leave the host exactly as it was.
	br, err := lookupLink(spec.Bridge)
	if err != nil {
		return err
	}
	if br != nil {
		if err := checkOwned(br, "bridge", alias); err != nil {
			return err
		}
	}
	vl, err := lookupLink(vlanName)
	if err != nil {
		return err
	}
	if vl != nil {
		if err := checkOwned(vl, "vlan", alias); err != nil {
			return err
		}
	}
	if spec.Gateway.IsValid() {
		if err := checkAddressConflict(spec); err != nil {
			return err
		}
	}

	if br == nil {
		if br, err = addOwnedLink(&netlink.Bridge{LinkAttrs: netlink.LinkAttrs{Name: spec.Bridge, MTU: spec.MTU, Alias: alias}}, alias); err != nil {
			return err
		}
	}
	if err := l.ensureVLAN(spec, br, vl); err != nil {
		return err
	}
	// Re-read: enslaving a port can move the bridge MTU. Setting it
	// explicitly pins the bridge instead of letting it follow its smallest
	// port.
	if br, err = lookupLink(spec.Bridge); err != nil {
		return err
	}
	if br == nil {
		return fmt.Errorf("bridge %s vanished while being provisioned", spec.Bridge)
	}
	if br.Attrs().MTU != spec.MTU {
		if err := netlink.LinkSetMTU(br, spec.MTU); err != nil {
			return linkErr("set mtu on", spec.Bridge, err)
		}
	}
	if err := ensureGateway(spec, br); err != nil {
		return err
	}
	if err := netlink.LinkSetUp(br); err != nil {
		return linkErr("bring up", spec.Bridge, err)
	}
	if l.ACL != nil {
		return l.ACL.Allow(spec.Bridge)
	}
	return nil
}

// ensureVLAN converges the tagged uplink port: absent when untagged, else a
// vlan link on Uplink with spec.VLANID, enslaved to br. An owned link with a
// stale tag or parent is recreated.
func (l *Linux) ensureVLAN(spec Spec, br, vl netlink.Link) error {
	name := vlanLinkName(spec.Bridge)
	alias := ownerAlias(spec.NetworkID)
	if spec.VLANID == 0 {
		if vl != nil {
			return delLink(vl)
		}
		return nil
	}
	up, err := lookupLink(l.Uplink)
	if err != nil {
		return err
	}
	if up == nil {
		return fmt.Errorf("uplink interface %q does not exist", l.Uplink)
	}
	if v, ok := vl.(*netlink.Vlan); vl != nil && (!ok || v.VlanId != spec.VLANID || v.ParentIndex != up.Attrs().Index) {
		if err := delLink(vl); err != nil {
			return err
		}
		vl = nil
	}
	if vl == nil {
		v := &netlink.Vlan{
			LinkAttrs: netlink.LinkAttrs{Name: name, ParentIndex: up.Attrs().Index, MTU: spec.MTU, Alias: alias},
			VlanId:    spec.VLANID,
		}
		if vl, err = addOwnedLink(v, alias); err != nil {
			return err
		}
	}
	if vl.Attrs().MTU != spec.MTU {
		if err := netlink.LinkSetMTU(vl, spec.MTU); err != nil {
			return linkErr("set mtu on", name, err)
		}
	}
	if vl.Attrs().MasterIndex != br.Attrs().Index {
		if err := netlink.LinkSetMaster(vl, br); err != nil {
			return linkErr("enslave", name, err)
		}
	}
	if err := netlink.LinkSetUp(vl); err != nil {
		return linkErr("bring up", name, err)
	}
	return nil
}

// Remove deletes the network's VLAN port and bridge and revokes its ACL
// entry. It refuses (ErrBridgeInUse) while any other port -- e.g. a running
// guest's tap -- is still attached, and never deletes a link it does not own
// (ErrForeignLink). Only spec.NetworkID is needed; an absent network is
// already removed.
func (l *Linux) Remove(ctx context.Context, spec Spec) error {
	bridge, err := BridgeName(spec.NetworkID)
	if err != nil {
		return err
	}
	if spec.Bridge != "" && spec.Bridge != bridge {
		return fmt.Errorf("bridge %q does not match network id (want %q)", spec.Bridge, bridge)
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	l.mu.Lock()
	defer l.mu.Unlock()

	alias := ownerAlias(spec.NetworkID)
	vlanName := vlanLinkName(bridge)
	br, err := lookupLink(bridge)
	if err != nil {
		return err
	}
	vl, err := lookupLink(vlanName)
	if err != nil {
		return err
	}
	if br != nil {
		if err := checkOwned(br, "bridge", alias); err != nil {
			return err
		}
	}
	if vl != nil {
		if err := checkOwned(vl, "vlan", alias); err != nil {
			return err
		}
	}
	if br != nil {
		links, err := listLinks()
		if err != nil {
			return err
		}
		var ports []string
		for _, link := range links {
			if a := link.Attrs(); a.MasterIndex == br.Attrs().Index && a.Name != vlanName {
				ports = append(ports, a.Name)
			}
		}
		if len(ports) > 0 {
			sort.Strings(ports)
			return fmt.Errorf("%w: %s has %s", ErrBridgeInUse, bridge, strings.Join(ports, ", "))
		}
	}
	if vl != nil {
		if err := delLink(vl); err != nil {
			return err
		}
	}
	if br != nil {
		if err := delLink(br); err != nil {
			return err
		}
	}
	if l.ACL != nil {
		return l.ACL.Revoke(bridge)
	}
	return nil
}

// ensureGateway makes the gateway (with the network's prefix length) the
// bridge's only global address; kernel-assigned link-local addresses stay.
func ensureGateway(spec Spec, br netlink.Link) error {
	addrs, err := listAddrs(br)
	if err != nil {
		return err
	}
	var want *net.IPNet
	if spec.Gateway.IsValid() {
		want = &net.IPNet{IP: net.IP(spec.Gateway.AsSlice()), Mask: net.CIDRMask(spec.CIDR.Bits(), spec.Gateway.BitLen())}
	}
	have := false
	for i := range addrs {
		a := &addrs[i]
		if a.IP.IsLinkLocalUnicast() {
			continue
		}
		if want != nil && a.IPNet.String() == want.String() {
			have = true
			continue
		}
		if err := netlink.AddrDel(br, a); err != nil {
			return linkErr("remove stale address "+a.IPNet.String()+" from", spec.Bridge, err)
		}
	}
	if want != nil && !have {
		if err := netlink.AddrAdd(br, &netlink.Addr{IPNet: want}); err != nil {
			return linkErr("assign gateway "+want.String()+" to", spec.Bridge, err)
		}
	}
	return nil
}

// checkAddressConflict fails when an interface other than the network's own
// bridge carries an address whose prefix overlaps spec.CIDR.
func checkAddressConflict(spec Spec) error {
	links, err := listLinks()
	if err != nil {
		return err
	}
	names := make(map[int]string, len(links))
	for _, link := range links {
		names[link.Attrs().Index] = link.Attrs().Name
	}
	addrs, err := listAddrs(nil)
	if err != nil {
		return err
	}
	for _, a := range addrs {
		if a.IPNet == nil || names[a.LinkIndex] == spec.Bridge {
			continue
		}
		ip, ok := netip.AddrFromSlice(a.IP)
		if !ok {
			continue
		}
		ones, _ := a.Mask.Size()
		if p := netip.PrefixFrom(ip.Unmap(), ones).Masked(); p.Overlaps(spec.CIDR) {
			return fmt.Errorf("%w: %s has %s", ErrAddressConflict, names[a.LinkIndex], a.IPNet)
		}
	}
	return nil
}

// addOwnedLink creates link (its Alias set to alias) and returns the kernel's
// view of it. Kernels that ignore IFLA_IFALIAS on create get the alias set
// right after; if even that fails the just-created link is deleted rather
// than left behind unowned.
func addOwnedLink(link netlink.Link, alias string) (netlink.Link, error) {
	name := link.Attrs().Name
	if err := netlink.LinkAdd(link); err != nil {
		return nil, linkErr("create "+link.Type(), name, err)
	}
	created, err := lookupLink(name)
	if err != nil {
		return nil, err
	}
	if created == nil {
		return nil, fmt.Errorf("created %s %s but it vanished", link.Type(), name)
	}
	if created.Attrs().Alias != alias {
		if err := netlink.LinkSetAlias(created, alias); err != nil {
			_ = netlink.LinkDel(created)
			return nil, linkErr("set owner alias on", name, err)
		}
		created.Attrs().Alias = alias
	}
	return created, nil
}

func checkOwned(link netlink.Link, kind, alias string) error {
	if a := link.Attrs(); link.Type() != kind || a.Alias != alias {
		return fmt.Errorf("%w: %s (type %s, alias %q)", ErrForeignLink, a.Name, link.Type(), a.Alias)
	}
	return nil
}

func lookupLink(name string) (netlink.Link, error) {
	link, err := netlink.LinkByName(name)
	var notFound netlink.LinkNotFoundError
	if errors.As(err, &notFound) {
		return nil, nil
	}
	if err != nil {
		return nil, linkErr("look up", name, err)
	}
	return link, nil
}

func delLink(link netlink.Link) error {
	if err := netlink.LinkDel(link); err != nil {
		return linkErr("delete", link.Attrs().Name, err)
	}
	return nil
}

// dumpRetries bounds retries of netlink dumps, which newer netlink versions
// report as interrupted (with partial results) when the kernel's table
// changes mid-dump.
const dumpRetries = 3

func listLinks() ([]netlink.Link, error) {
	var err error
	for i := 0; i < dumpRetries; i++ {
		var links []netlink.Link
		if links, err = netlink.LinkList(); err == nil {
			return links, nil
		}
	}
	return nil, fmt.Errorf("list links: %w", err)
}

func listAddrs(link netlink.Link) ([]netlink.Addr, error) {
	var err error
	for i := 0; i < dumpRetries; i++ {
		var addrs []netlink.Addr
		if addrs, err = netlink.AddrList(link, netlink.FAMILY_ALL); err == nil {
			return addrs, nil
		}
	}
	return nil, fmt.Errorf("list addresses: %w", err)
}

func linkErr(op, name string, err error) error {
	if errors.Is(err, syscall.EPERM) {
		return fmt.Errorf("%s %s: %w (network provisioning needs CAP_NET_ADMIN)", op, name, err)
	}
	return fmt.Errorf("%s %s: %w", op, name, err)
}
