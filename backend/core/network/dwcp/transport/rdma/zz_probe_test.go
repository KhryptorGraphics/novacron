package rdma

import (
	"fmt"
	"testing"
)

func TestProbeRunnerVerbs(t *testing.T) {
	devs, _ := GetDeviceList()
	msg := fmt.Sprintf("PROBE avail=%v numDevices=%d", CheckAvailability(), len(devs))
	for _, d := range devs {
		msg += fmt.Sprintf(" | dev=%s ports=%d", d.Name, d.NumPorts)
	}
	// Query port + GID directly through the package's own Initialize to get
	// the precise failing stage.
	_, err := Initialize("", 1, false)
	if err != nil {
		msg += " | initErr=" + err.Error()
	}
	t.Fatal(msg)
}
