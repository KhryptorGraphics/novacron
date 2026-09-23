package vm

// PCI passthrough for the KVM driver: validate VMConfig.PCIPassthroughDevices
// (BDF list, max 8) at Create, bind every requested BDF — plus every non-NVMe
// peer in its IOMMU group, all-or-none — to vfio-pci at launch, emit one
// `-device vfio-pci,host=<BDF>` per requested device, release the bindings on
// VM stop, and keep a cross-process in-use ledger (pci-inuse.json, flock'ed)
// so Teardown only ever unbinds what THIS api-server process bound this run.

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"syscall"
	"time"
)

// maxPCIPassthroughDevices caps how many BDFs a single VM may pass through;
// enforced at Create (see normalizePCIPassthroughDevices).
const maxPCIPassthroughDevices = 8

// Sysfs/host layout. Package vars so tests can point them at a copied fake
// tree under t.TempDir() (real sysfs is read-mostly + write-to-bind, neither
// mockable nor safe to poke in a unit test).
var (
	pciSysfsBusDir    = "/sys/bus/pci"
	pciIOMMUGroupsDir = "/sys/kernel/iommu_groups"
	pciKVMDevicePath  = "/dev/kvm"
)

// pciWriteSys performs one sysfs write. A package var so tests can record the
// exact write ordering (unbind-before-bind, group-wide rollback, ...).
var pciWriteSys = func(path, data string) error {
	return os.WriteFile(path, []byte(data), 0)
}

// pciBDFRe matches a canonical PCI bus/device/function address, 0000:01:00.0.
var pciBDFRe = regexp.MustCompile(`^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$`)

// pciClassNVMe is the PCI class code for NVMe storage controllers. Group
// peers with this class are NEVER unbound from the host (they may be the boot
// disk); the all-or-none group rule applies to every other affected device.
const pciClassNVMe = "0x010802"

// gpuPassthroughOverrideEnv lets an operator force the GPU/passthrough
// capability advertisement off (any of 0/false/no/off/disabled).
const gpuPassthroughOverrideEnv = "NOVACRON_GPU_PASSTHROUGH"

// pciBindingRecord is one entry of the host-level pci-inuse.json ledger.
type pciBindingRecord struct {
	BDF            string    `json:"bdf"`
	VMID           string    `json:"vm_id"`
	Owner          string    `json:"owner"` // per-process driver nonce; only the owner may release
	PreviousDriver string    `json:"previous_driver,omitempty"`
	IOMMUGroup     string    `json:"iommu_group"`
	BoundAt        time.Time `json:"bound_at"`
}

// newPCIOwnerNonce returns the per-process token stamped into ledger records.
func newPCIOwnerNonce() string {
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("pid%d-%d", os.Getpid(), time.Now().UnixNano())
	}
	return hex.EncodeToString(b[:])
}

// normalizePCIPassthroughDevices validates and canonicalizes the requested
// BDF list: lowercase, strict BDF format, no duplicates, at most
// maxPCIPassthroughDevices. Called at Create so a misconfigured VM can never
// reach launch.
func normalizePCIPassthroughDevices(devices []string) ([]string, error) {
	if len(devices) == 0 {
		return nil, nil
	}
	if len(devices) > maxPCIPassthroughDevices {
		return nil, fmt.Errorf("pci passthrough: %d devices requested, max is %d", len(devices), maxPCIPassthroughDevices)
	}
	seen := make(map[string]bool, len(devices))
	out := make([]string, 0, len(devices))
	for _, raw := range devices {
		bdf := strings.ToLower(strings.TrimSpace(raw))
		if !pciBDFRe.MatchString(bdf) {
			return nil, fmt.Errorf("pci passthrough: %q is not a valid BDF (want dddd:bb:dd.f, e.g. 0000:01:00.0)", raw)
		}
		if seen[bdf] {
			return nil, fmt.Errorf("pci passthrough: duplicate device %s", bdf)
		}
		seen[bdf] = true
		out = append(out, bdf)
	}
	return out, nil
}

// gpuPassthroughSupported reports the capability advertisement: true unless an
// operator explicitly disabled it via the override env var.
func gpuPassthroughSupported() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(gpuPassthroughOverrideEnv))) {
	case "0", "false", "no", "off", "disabled":
		return false
	}
	return true
}

// pciHostCheck verifies the host can pass PCI devices to a guest: vfio-pci
// driver present, IOMMU groups populated, and on arm64 a usable /dev/kvm
// (VFIO-PCI requires KVM acceleration; TCG cannot serve it).
func pciHostCheck() error {
	if runtime.GOARCH == "arm64" {
		if _, err := os.Stat(pciKVMDevicePath); err != nil {
			return fmt.Errorf("pci passthrough on arm64 requires KVM, but %s is unavailable: %w (load the kvm module / fix host privileges, or drop the PCI device request)", pciKVMDevicePath, err)
		}
	}
	vfioDir := filepath.Join(pciSysfsBusDir, "drivers", "vfio-pci")
	if _, err := os.Stat(vfioDir); err != nil {
		return fmt.Errorf("vfio-pci driver not available at %s: %w (modprobe vfio-pci and enable IOMMU)", vfioDir, err)
	}
	entries, err := os.ReadDir(pciIOMMUGroupsDir)
	if err != nil || len(entries) == 0 {
		return fmt.Errorf("no IOMMU groups under %s (enable IOMMU/SMMU in firmware and kernel cmdline)", pciIOMMUGroupsDir)
	}
	return nil
}

// pciStateFile / pciLockFile locate the host-level ledger next to the VM base
// dir (default /var/lib/novacron/pci-inuse.json).
func (d *KVMDriverEnhanced) pciStateFile() string {
	return filepath.Join(filepath.Dir(d.vmBasePath), "pci-inuse.json")
}
func (d *KVMDriverEnhanced) pciLockFile() string {
	return filepath.Join(filepath.Dir(d.vmBasePath), "pci-inuse.lock")
}

// withPCILedger runs fn under an exclusive flock on the ledger lock file, with
// the current ledger passed in and (when fn returns a non-nil map) atomically
// persisted back. This is what makes the ledger verify between processes.
func (d *KVMDriverEnhanced) withPCILedger(fn func(map[string]pciBindingRecord) (map[string]pciBindingRecord, error)) error {
	lockPath := d.pciLockFile()
	if err := os.MkdirAll(filepath.Dir(lockPath), 0755); err != nil {
		return err
	}
	lf, err := os.OpenFile(lockPath, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return err
	}
	defer lf.Close()
	if err := syscall.Flock(int(lf.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	defer syscall.Flock(int(lf.Fd()), syscall.LOCK_UN)

	records := map[string]pciBindingRecord{}
	if raw, err := os.ReadFile(d.pciStateFile()); err == nil && len(raw) > 0 {
		_ = json.Unmarshal(raw, &records) // corrupt ledger -> start fresh, sysfs is the truth
	}
	out, err := fn(records)
	if err != nil || out == nil {
		return err
	}
	data, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		return err
	}
	tmp := d.pciStateFile() + ".tmp"
	if err := os.WriteFile(tmp, data, 0644); err != nil {
		return err
	}
	return os.Rename(tmp, d.pciStateFile())
}

// bindPCIDevices binds every requested BDF (and every non-NVMe device sharing
// its IOMMU group) to vfio-pci, recording ownership in the ledger. Atomic:
// any failure unbinds/undoes everything this call did — a VM launches with
// its full groups passthrough-ready or not at all. The returned error names
// the offending BDF.
func (d *KVMDriverEnhanced) bindPCIDevices(vmID string, devices []string) error {
	if len(devices) == 0 {
		return nil
	}
	if err := pciHostCheck(); err != nil {
		return err
	}

	// Reserve the BDFs in the ledger first (cross-process double-bind guard);
	// rolled back below on any bind failure.
	if err := d.withPCILedger(func(rec map[string]pciBindingRecord) (map[string]pciBindingRecord, error) {
		for _, bdf := range devices {
			if existing, ok := rec[bdf]; ok {
				return nil, fmt.Errorf("pci passthrough: device %s already bound for VM %s", bdf, existing.VMID)
			}
		}
		for _, bdf := range devices {
			rec[bdf] = pciBindingRecord{BDF: bdf, VMID: vmID, Owner: d.pciOwnerNonce, BoundAt: time.Now()}
		}
		return rec, nil
	}); err != nil {
		return err
	}

	var bound []pciBindingRecord // what THIS call must roll back on failure
	rollback := func(cause error) error {
		for i := len(bound) - 1; i >= 0; i-- {
			b := bound[i]
			_ = pciWriteSys(filepath.Join(pciSysfsBusDir, "drivers", "vfio-pci", "unbind"), b.BDF)
			if b.PreviousDriver != "" {
				_ = pciWriteSys(filepath.Join(pciSysfsBusDir, "drivers", b.PreviousDriver, "bind"), b.BDF)
			}
		}
		_ = d.withPCILedger(func(rec map[string]pciBindingRecord) (map[string]pciBindingRecord, error) {
			for _, bdf := range devices {
				if r, ok := rec[bdf]; ok && r.Owner == d.pciOwnerNonce {
					delete(rec, bdf)
				}
			}
			return rec, nil
		})
		return cause
	}

	for _, bdf := range devices {
		devDir := filepath.Join(pciSysfsBusDir, "devices", bdf)
		if _, err := os.Stat(devDir); err != nil {
			return rollback(fmt.Errorf("pci passthrough: device %s not present: %w", bdf, err))
		}
		// IOMMU group of the device (symlink <dev>/iommu_group -> ../../../kernel/iommu_groups/<N>).
		link, err := os.Readlink(filepath.Join(devDir, "iommu_group"))
		if err != nil {
			return rollback(fmt.Errorf("pci passthrough: device %s has no IOMMU group (IOMMU off?): %w", bdf, err))
		}
		group := filepath.Base(link)
		entries, err := os.ReadDir(filepath.Join(pciIOMMUGroupsDir, group, "devices"))
		if err != nil {
			return rollback(fmt.Errorf("pci passthrough: cannot list IOMMU group %s for %s: %w", group, bdf, err))
		}

		// Bind the WHOLE group (all-or-none) except NVMe-class peers, which are
		// left alone — unbinding a host NVMe controller can take out the boot disk.
		for _, e := range entries {
			peer := e.Name()
			peerDir := filepath.Join(pciSysfsBusDir, "devices", peer)
			if raw, err := os.ReadFile(filepath.Join(peerDir, "class")); err == nil {
				if strings.EqualFold(strings.TrimSpace(string(raw)), pciClassNVMe) {
					continue
				}
			}
			if err := d.bindOnePCIDevice(vmID, peer, group, &bound); err != nil {
				return rollback(fmt.Errorf("pci passthrough: failed to bind %s (group %s, requested by %s): %w", peer, group, bdf, err))
			}
		}
	}
	return nil
}

// bindOnePCIDevice moves one BDF to vfio-pci and appends it to bound (the
// caller's rollback trail).
func (d *KVMDriverEnhanced) bindOnePCIDevice(vmID, bdf, group string, bound *[]pciBindingRecord) error {
	devDir := filepath.Join(pciSysfsBusDir, "devices", bdf)

	prev := ""
	if link, err := os.Readlink(filepath.Join(devDir, "driver")); err == nil {
		prev = filepath.Base(link)
		if prev == "vfio-pci" {
			// Already on vfio-pci: recorded in the ledger by our reservation step;
			// already-bound is a stable state, not a bind we must roll back.
			return nil
		}
	}

	// Teach vfio-pci the vendor/device IDs so it claims the device on bind.
	vendor, _ := os.ReadFile(filepath.Join(devDir, "vendor"))
	device, _ := os.ReadFile(filepath.Join(devDir, "device"))
	ids := strings.TrimSpace(fmt.Sprintf("%s %s", strings.TrimSpace(string(vendor)), strings.TrimSpace(string(device))))
	if ids != "" {
		if err := pciWriteSys(filepath.Join(pciSysfsBusDir, "drivers", "vfio-pci", "new_id"), ids); err != nil &&
			!strings.Contains(strings.ToLower(err.Error()), "exist") {
			return err
		}
	}
	if prev != "" {
		if err := pciWriteSys(filepath.Join(pciSysfsBusDir, "drivers", prev, "unbind"), bdf); err != nil {
			return err
		}
	}
	if err := pciWriteSys(filepath.Join(pciSysfsBusDir, "drivers", "vfio-pci", "bind"), bdf); err != nil {
		return err
	}

	// On the real sysfs the kernel moves the driver symlink; verify the bind
	// actually took. Skipped against a fake tree (which cannot move links).
	if pciSysfsBusDir == "/sys/bus/pci" {
		if link, err := os.Readlink(filepath.Join(devDir, "driver")); err != nil || filepath.Base(link) != "vfio-pci" {
			return fmt.Errorf("post-bind check: driver is not vfio-pci")
		}
	}

	*bound = append(*bound, pciBindingRecord{
		BDF: bdf, VMID: vmID, Owner: d.pciOwnerNonce,
		PreviousDriver: prev, IOMMUGroup: group, BoundAt: time.Now(),
	})
	// Enrich the reservation with driver/group detail.
	return d.withPCILedger(func(rec map[string]pciBindingRecord) (map[string]pciBindingRecord, error) {
		rec[bdf] = (*bound)[len(*bound)-1]
		return rec, nil
	})
}

// releasePCIDevices unbinds every ledger entry for vmID — but ONLY those whose
// Owner is this driver process (an adopted VM bound by a previous api-server
// run is left untouched). Errors are logged, never fatal to VM stop.
func (d *KVMDriverEnhanced) releasePCIDevices(vmID string) {
	var owned []pciBindingRecord
	err := d.withPCILedger(func(rec map[string]pciBindingRecord) (map[string]pciBindingRecord, error) {
		for bdf, r := range rec {
			if r.VMID == vmID && r.Owner == d.pciOwnerNonce {
				owned = append(owned, r)
				delete(rec, bdf)
			}
		}
		return rec, nil
	})
	if err != nil {
		log.Printf("PCI passthrough: cannot open ledger to release VM %s: %v", vmID, err)
		return
	}
	for _, r := range owned {
		if err := pciWriteSys(filepath.Join(pciSysfsBusDir, "drivers", "vfio-pci", "unbind"), r.BDF); err != nil {
			log.Printf("PCI passthrough: unbind %s (VM %s) failed: %v", r.BDF, vmID, err)
			continue
		}
		if r.PreviousDriver != "" && r.PreviousDriver != "vfio-pci" {
			if err := pciWriteSys(filepath.Join(pciSysfsBusDir, "drivers", r.PreviousDriver, "bind"), r.BDF); err != nil {
				log.Printf("PCI passthrough: rebind %s to %s failed: %v", r.BDF, r.PreviousDriver, err)
			}
		}
	}
}
