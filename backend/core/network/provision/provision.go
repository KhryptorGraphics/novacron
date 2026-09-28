// Package provision creates and removes the host Linux bridges that back the
// NovaCron networks catalog (the /networks API, database table `networks`).
//
// Attachment model: a KVM guest on a catalog network gets one virtio-net NIC
// whose backend is `-netdev bridge,br=<bridge>` (see core/vm kvmNICArgs), so
// qemu-bridge-helper creates the guest's tap and enslaves it to the bridge at
// launch and removes it when qemu exits. This package therefore owns only the
// bridge itself (plus, for a VLAN network, the tagged sub-interface of the
// node's uplink that is enslaved to it) and the qemu-bridge-helper ACL entry
// that permits qemu to attach to it.
//
// Safety: every link this package creates is named from the network id
// (ncbr-<10 hex> / ncvl-<10 hex>) and carries the alias
// "novacron-network:<id>". A link with one of those names but without that
// alias is refused (ErrForeignLink), never modified or deleted, so a
// non-NovaCron bridge is never touched.
package provision

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"net/netip"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/google/uuid"
)

const (
	// MinMTU and MaxMTU bound a network's MTU: 576 is the IPv4 minimum
	// datagram size, 9000 the common jumbo-frame ceiling.
	MinMTU = 576
	MaxMTU = 9000
	// DefaultMTU is the Ethernet MTU used when a create request omits it.
	DefaultMTU = 1500
	// minIPv6MTU is the IPv6 link minimum (RFC 8200 §5).
	minIPv6MTU = 1280

	bridgePrefix = "ncbr-"
	vlanPrefix   = "ncvl-"
	nameSuffix   = 10 // hex chars of the network id; prefix+suffix = 15 = IFNAMSIZ-1
	aliasPrefix  = "novacron-network:"
)

var (
	// ErrForeignLink: a link with a NovaCron-reserved name exists but is not
	// owned by this network (wrong alias or type). It is left untouched.
	ErrForeignLink = errors.New("link exists and is not owned by this NovaCron network")
	// ErrBridgeInUse: the bridge still has ports (e.g. a running guest's tap),
	// so removing it would cut live traffic.
	ErrBridgeInUse = errors.New("bridge still has attached ports")
	// ErrNoUplink: a VLAN network needs an uplink interface to tag onto and
	// none is configured on this node.
	ErrNoUplink = errors.New("vlan networks require an uplink interface on this node (NOVACRON_NETWORK_UPLINK)")
)

// Spec is the host-side description of one catalog network.
type Spec struct {
	NetworkID string       // catalog id (UUID)
	Bridge    string       // BridgeName(NetworkID)
	CIDR      netip.Prefix // masked (no host bits)
	Gateway   netip.Addr   // host address on the bridge; zero value = pure L2 bridge
	VLANID    int          // 0 = untagged; 1..4094 tags onto the node uplink
	MTU       int
}

// Provisioner realises catalog networks on the local node. Ensure and Remove
// are idempotent: Ensure on an already-provisioned network converges it,
// Remove on an absent one succeeds.
type Provisioner interface {
	Ensure(ctx context.Context, spec Spec) error
	Remove(ctx context.Context, spec Spec) error
}

// BridgeName derives the bridge interface name for a network id.
func BridgeName(networkID string) (string, error) {
	suffix, err := idSuffix(networkID)
	if err != nil {
		return "", err
	}
	return bridgePrefix + suffix, nil
}

func vlanLinkName(bridge string) string {
	return vlanPrefix + strings.TrimPrefix(bridge, bridgePrefix)
}

func ownerAlias(networkID string) string {
	return aliasPrefix + strings.ToLower(networkID)
}

func idSuffix(networkID string) (string, error) {
	id, err := uuid.Parse(networkID)
	if err != nil {
		return "", fmt.Errorf("network id %q is not a UUID", networkID)
	}
	return strings.ReplaceAll(id.String(), "-", "")[:nameSuffix], nil
}

// NewSpec parses and validates the catalog fields of a network. gateway may
// be empty; vlanID 0 means untagged.
func NewSpec(networkID, cidr, gateway string, vlanID, mtu int) (Spec, error) {
	bridge, err := BridgeName(networkID)
	if err != nil {
		return Spec{}, err
	}
	prefix, err := netip.ParsePrefix(strings.TrimSpace(cidr))
	if err != nil {
		return Spec{}, fmt.Errorf("cidr %q is not a valid network prefix", cidr)
	}
	spec := Spec{NetworkID: networkID, Bridge: bridge, CIDR: prefix, VLANID: vlanID, MTU: mtu}
	if gw := strings.TrimSpace(gateway); gw != "" {
		addr, err := netip.ParseAddr(gw)
		if err != nil || addr.Zone() != "" {
			return Spec{}, fmt.Errorf("gateway %q is not a valid IP address", gateway)
		}
		spec.Gateway = addr
	}
	return spec, spec.Validate()
}

// Validate checks every invariant Ensure relies on.
func (s Spec) Validate() error {
	bridge, err := BridgeName(s.NetworkID)
	if err != nil {
		return err
	}
	if s.Bridge != bridge {
		return fmt.Errorf("bridge %q does not match network id (want %q)", s.Bridge, bridge)
	}
	p := s.CIDR
	if !p.IsValid() {
		return errors.New("cidr is required")
	}
	if p != p.Masked() {
		return fmt.Errorf("cidr %s has host bits set (did you mean %s?)", p, p.Masked())
	}
	if p.Addr().Is4In6() {
		return fmt.Errorf("cidr %s: use the plain IPv4 form", p)
	}
	if p.Addr().Is4() {
		if p.Bits() < 8 || p.Bits() > 30 {
			return fmt.Errorf("cidr %s: IPv4 prefix length must be between /8 and /30", p)
		}
	} else if p.Bits() < 16 || p.Bits() > 126 {
		return fmt.Errorf("cidr %s: IPv6 prefix length must be between /16 and /126", p)
	}
	for _, special := range reservedPrefixes {
		if p.Overlaps(special) {
			return fmt.Errorf("cidr %s overlaps reserved range %s (loopback, link-local, multicast or unspecified)", p, special)
		}
	}
	if s.Gateway.IsValid() {
		gw := s.Gateway
		if gw.Is4() != p.Addr().Is4() || gw.Is4In6() {
			return fmt.Errorf("gateway %s is not the same address family as cidr %s", gw, p)
		}
		if !p.Contains(gw) {
			return fmt.Errorf("gateway %s is not inside cidr %s", gw, p)
		}
		if gw == p.Addr() {
			return fmt.Errorf("gateway %s is the network address of %s", gw, p)
		}
		if gw.Is4() && gw == lastAddr(p) {
			return fmt.Errorf("gateway %s is the broadcast address of %s", gw, p)
		}
	}
	if s.VLANID != 0 && (s.VLANID < 1 || s.VLANID > 4094) {
		return fmt.Errorf("vlan_id %d must be between 1 and 4094", s.VLANID)
	}
	if s.MTU < MinMTU || s.MTU > MaxMTU {
		return fmt.Errorf("mtu %d must be between %d and %d", s.MTU, MinMTU, MaxMTU)
	}
	if !p.Addr().Is4() && s.MTU < minIPv6MTU {
		return fmt.Errorf("mtu %d is below the IPv6 minimum of %d", s.MTU, minIPv6MTU)
	}
	return nil
}

// lastAddr is the highest address in p (the IPv4 broadcast address).
func lastAddr(p netip.Prefix) netip.Addr {
	b := p.Addr().AsSlice()
	for i := p.Bits(); i < len(b)*8; i++ {
		b[i/8] |= 1 << (7 - uint(i%8))
	}
	a, _ := netip.AddrFromSlice(b)
	return a
}

// reservedPrefixes can never be a bridged guest network: routing them onto a
// bridge would shadow loopback, link-local or multicast handling.
var reservedPrefixes = []netip.Prefix{
	netip.MustParsePrefix("0.0.0.0/8"),
	netip.MustParsePrefix("127.0.0.0/8"),
	netip.MustParsePrefix("169.254.0.0/16"),
	netip.MustParsePrefix("224.0.0.0/3"), // multicast + reserved/broadcast
	netip.MustParsePrefix("::/127"),      // unspecified + loopback
	netip.MustParsePrefix("fe80::/10"),
	netip.MustParsePrefix("ff00::/8"),
}

// BridgeACL maintains the qemu-bridge-helper ACL that lets qemu attach guests
// to NovaCron bridges. qemu-bridge-helper reads <Dir>/bridge.conf and refuses
// any bridge not allowed there; NovaCron keeps its own allow list in
// <Dir>/novacron-bridges.conf and makes bridge.conf include it (the only
// change ever made to bridge.conf is appending that include line).
type BridgeACL struct {
	Dir string
	mu  sync.Mutex
}

const (
	helperConf   = "bridge.conf"
	novacronConf = "novacron-bridges.conf"
)

// Allow permits qemu-bridge-helper to attach to bridge.
func (a *BridgeACL) Allow(bridge string) error {
	return a.update(bridge, true)
}

// Revoke removes bridge from NovaCron's allow list.
func (a *BridgeACL) Revoke(bridge string) error {
	return a.update(bridge, false)
}

func (a *BridgeACL) update(bridge string, allow bool) error {
	if !strings.HasPrefix(bridge, bridgePrefix) {
		return fmt.Errorf("refusing to manage ACL for non-NovaCron bridge %q", bridge)
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	listPath := filepath.Join(a.Dir, novacronConf)
	entries, err := readAllowList(listPath)
	if err != nil {
		return err
	}
	if allow {
		entries[bridge] = struct{}{}
	} else {
		delete(entries, bridge)
	}
	names := make([]string, 0, len(entries))
	for name := range entries {
		names = append(names, name)
	}
	sort.Strings(names)
	var buf bytes.Buffer
	buf.WriteString("# Managed by NovaCron (networks catalog); do not edit.\n")
	for _, name := range names {
		fmt.Fprintf(&buf, "allow %s\n", name)
	}
	if err := os.MkdirAll(a.Dir, 0o755); err != nil {
		return fmt.Errorf("qemu bridge ACL dir: %w", err)
	}
	if err := writeFileAtomic(listPath, buf.Bytes()); err != nil {
		return err
	}
	if allow {
		return ensureInclude(filepath.Join(a.Dir, helperConf), listPath)
	}
	return nil
}

func readAllowList(path string) (map[string]struct{}, error) {
	entries := map[string]struct{}{}
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return entries, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	sc := bufio.NewScanner(bytes.NewReader(data))
	for sc.Scan() {
		fields := strings.Fields(sc.Text())
		if len(fields) == 2 && fields[0] == "allow" && strings.HasPrefix(fields[1], bridgePrefix) {
			entries[fields[1]] = struct{}{}
		}
	}
	return entries, nil
}

// ensureInclude appends "include <target>" to conf unless already present.
func ensureInclude(conf, target string) error {
	line := "include " + target
	data, err := os.ReadFile(conf)
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("read %s: %w", conf, err)
	}
	for _, l := range strings.Split(string(data), "\n") {
		if strings.TrimSpace(l) == line {
			return nil
		}
	}
	if len(data) > 0 && !bytes.HasSuffix(data, []byte("\n")) {
		data = append(data, '\n')
	}
	data = append(data, line+"\n"...)
	return writeFileAtomic(conf, data)
}

func writeFileAtomic(path string, data []byte) error {
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".*")
	if err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return fmt.Errorf("write %s: %w", path, err)
	}
	if err := tmp.Chmod(0o644); err != nil {
		tmp.Close()
		return fmt.Errorf("write %s: %w", path, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	if err := os.Rename(tmp.Name(), path); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	return nil
}
