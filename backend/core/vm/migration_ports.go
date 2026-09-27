package vm

import (
	"fmt"
	"math/rand/v2"
	"net"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// MigrationPortRangeEnv bounds the TCP ports this node listens on for incoming QEMU
// migration streams and NBD block-migration exports. A fixed, power-of-two-aligned range
// lets host QoS (deploy/p2pnet qos) classify migration traffic with one u32 port mask.
const MigrationPortRangeEnv = "NOVACRON_MIGRATION_PORT_RANGE"

const (
	DefaultMigrationPortMin = 49152
	DefaultMigrationPortMax = 49215
)

var migrationPortRangePattern = regexp.MustCompile(`^\s*(\d+)\s*-\s*(\d+)\s*$`)

// MigrationPortRange returns the inclusive TCP port bounds for incoming migration
// and NBD listeners. An unset or blank NOVACRON_MIGRATION_PORT_RANGE uses the
// libvirt default, 49152-49215.
func MigrationPortRange() (lo, hi int, err error) {
	v := os.Getenv(MigrationPortRangeEnv)
	if strings.TrimSpace(v) == "" {
		return DefaultMigrationPortMin, DefaultMigrationPortMax, nil
	}
	m := migrationPortRangePattern.FindStringSubmatch(v)
	if m == nil {
		return 0, 0, fmt.Errorf("invalid %s %q: want LO-HI with 1024<=LO<=HI<=65535", MigrationPortRangeEnv, v)
	}
	lo, err = strconv.Atoi(m[1])
	if err != nil {
		return 0, 0, fmt.Errorf("invalid %s %q: want LO-HI with 1024<=LO<=HI<=65535", MigrationPortRangeEnv, v)
	}
	hi, err = strconv.Atoi(m[2])
	if err != nil || lo < 1024 || lo > hi || hi > 65535 {
		return 0, 0, fmt.Errorf("invalid %s %q: want LO-HI with 1024<=LO<=HI<=65535", MigrationPortRangeEnv, v)
	}
	return lo, hi, nil
}

// AllocateMigrationPort returns a currently-unused TCP port inside MigrationPortRange.
// The tiny TOCTOU window until QEMU binds is unchanged from the previous ephemeral picker.
func AllocateMigrationPort() (int, error) {
	lo, hi, err := MigrationPortRange()
	if err != nil {
		return 0, err
	}
	n := hi - lo + 1
	start := rand.IntN(n)
	for i := 0; i < n; i++ {
		p := lo + (start+i)%n
		ln, listenErr := net.Listen("tcp", fmt.Sprintf("0.0.0.0:%d", p))
		if listenErr != nil {
			continue
		}
		ln.Close()
		return p, nil
	}
	return 0, fmt.Errorf("no free TCP port in %s %d-%d", MigrationPortRangeEnv, lo, hi)
}
