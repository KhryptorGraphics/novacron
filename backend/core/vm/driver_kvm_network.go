package vm

import (
	"crypto/sha256"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/khryptorgraphics/novacron/backend/core/network/provision"
)

// sysClassNet is where the host's interfaces are exposed (a var for tests).
var sysClassNet = "/sys/class/net"

// kvmNICArgs returns the -netdev/-device pair for a KVM guest's primary NIC.
//
// A VM whose NetworkID is a networks-catalog id (a UUID; see
// core/network/provision) is bridged onto that network's host bridge:
// qemu-bridge-helper creates the guest's tap and enslaves it to the bridge
// (and the tap disappears with qemu). The NIC gets a MAC derived from the VM
// id -- stable across restarts and migrations, and distinct per guest,
// unlike qemu's fixed default which would collide on a shared bridge -- and,
// when the bridge MTU is not the Ethernet default, advertises it to the guest
// (virtio host_mtu) so jumbo/reduced-MTU networks work without guest config.
//
// Anything else (no network, or a legacy non-catalog label) keeps the
// isolated user-mode NIC every VM had before the catalog existed.
func kvmNICArgs(vmID, networkID string) []string {
	bridge, err := provision.BridgeName(networkID)
	if err != nil {
		return []string{"-netdev", "user,id=net0", "-device", "virtio-net-pci,netdev=net0"}
	}
	device := "virtio-net-pci,netdev=net0,mac=" + guestMAC(vmID)
	if mtu := hostLinkMTU(bridge); mtu > 0 && mtu != 1500 {
		device += ",host_mtu=" + strconv.Itoa(mtu)
	}
	return []string{"-netdev", "bridge,id=net0,br=" + bridge, "-device", device}
}

// guestMAC derives a locally administered unicast MAC (qemu's 52:54 prefix)
// from the VM id.
func guestMAC(vmID string) string {
	sum := sha256.Sum256([]byte(vmID))
	return fmt.Sprintf("52:54:%02x:%02x:%02x:%02x", sum[0], sum[1], sum[2], sum[3])
}

// hostLinkMTU reads a host interface's MTU; 0 when it cannot be read (qemu
// then reports the bridge error itself when it tries to attach).
func hostLinkMTU(name string) int {
	b, err := os.ReadFile(filepath.Join(sysClassNet, name, "mtu"))
	if err != nil {
		return 0
	}
	mtu, err := strconv.Atoi(strings.TrimSpace(string(b)))
	if err != nil {
		return 0
	}
	return mtu
}
