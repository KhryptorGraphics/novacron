package storage

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// volumeIDPattern constrains volume IDs to a path-safe character set so a
// hostile ID can never escape the driver's base path (path traversal).
var volumeIDPattern = regexp.MustCompile(`^[A-Za-z0-9._-]+$`)

// localDriver is the filesystem-backed StorageDriver registered under the name
// "local". It stores each volume as a sparse file (<basePath>/<id>.img) and
// snapshots under <basePath>/snapshots/<volumeID>/<snapshotID>.img.
type localDriver struct {
	basePath string

	mu          sync.RWMutex
	attachments map[string]string // volumeID -> nodeID
}

// localDefaultBasePath resolves the default storage root: $STORAGE_PATH/local
// when STORAGE_PATH is set, otherwise the canonical /var/lib/novacron path.
func localDefaultBasePath() string {
	if root := os.Getenv("STORAGE_PATH"); root != "" {
		return filepath.Join(root, "local")
	}
	return "/var/lib/novacron/storage/local"
}

// newLocalDriver is the DriverFactory for the "local" driver. It validates the
// config but does not touch the filesystem; Initialize does that.
func newLocalDriver(config map[string]interface{}) (StorageDriver, error) {
	basePath := ""
	if v, ok := config["base_path"].(string); ok && v != "" {
		basePath = v
	} else {
		basePath = localDefaultBasePath()
	}
	return &localDriver{
		basePath:    basePath,
		attachments: make(map[string]string),
	}, nil
}

func init() {
	RegisterDriver("local", newLocalDriver)
}

func (d *localDriver) validateVolumeID(volumeID string) error {
	if !volumeIDPattern.MatchString(volumeID) {
		return fmt.Errorf("invalid volume id %q", volumeID)
	}
	return nil
}

func (d *localDriver) volumePath(volumeID string) string {
	return filepath.Join(d.basePath, volumeID+".img")
}

func (d *localDriver) snapshotsDir(volumeID string) string {
	return filepath.Join(d.basePath, "snapshots", volumeID)
}

func (d *localDriver) snapshotPath(volumeID, snapshotID string) string {
	return filepath.Join(d.snapshotsDir(volumeID), snapshotID+".img")
}

// Initialize creates the base and snapshots directories.
func (d *localDriver) Initialize() error {
	if err := os.MkdirAll(d.basePath, 0o755); err != nil {
		return fmt.Errorf("failed to create storage directory %s: %w", d.basePath, err)
	}
	if err := os.MkdirAll(filepath.Join(d.basePath, "snapshots"), 0o755); err != nil {
		return fmt.Errorf("failed to create snapshots directory: %w", err)
	}
	return nil
}

// Shutdown releases driver resources; the local driver holds none.
func (d *localDriver) Shutdown() error {
	return nil
}

// CreateVolume creates a sparse file-backed volume.
func (d *localDriver) CreateVolume(ctx context.Context, volumeID string, sizeBytes int64) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}
	if sizeBytes < 0 {
		return fmt.Errorf("invalid volume size %d", sizeBytes)
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	path := d.volumePath(volumeID)
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		if os.IsExist(err) {
			return fmt.Errorf("volume %s already exists", volumeID)
		}
		return fmt.Errorf("failed to create volume %s: %w", volumeID, err)
	}
	defer f.Close()

	if err := f.Truncate(sizeBytes); err != nil {
		return fmt.Errorf("failed to size volume %s to %d bytes: %w", volumeID, sizeBytes, err)
	}
	return nil
}

// DeleteVolume removes the volume file and its snapshot directory.
func (d *localDriver) DeleteVolume(ctx context.Context, volumeID string) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	path := d.volumePath(volumeID)
	if _, err := os.Stat(path); err != nil {
		return fmt.Errorf("volume %s does not exist: %w", volumeID, os.ErrNotExist)
	}
	if err := os.Remove(path); err != nil {
		return fmt.Errorf("failed to delete volume %s: %w", volumeID, err)
	}
	if err := os.RemoveAll(d.snapshotsDir(volumeID)); err != nil {
		return fmt.Errorf("failed to remove snapshots for volume %s: %w", volumeID, err)
	}
	delete(d.attachments, volumeID)
	return nil
}

// AttachVolume records the volume-to-node attachment.
func (d *localDriver) AttachVolume(ctx context.Context, volumeID, nodeID string) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if existing, ok := d.attachments[volumeID]; ok && existing != nodeID {
		return fmt.Errorf("volume %s already attached to %s", volumeID, existing)
	}
	d.attachments[volumeID] = nodeID
	return nil
}

// DetachVolume clears the volume-to-node attachment.
func (d *localDriver) DetachVolume(ctx context.Context, volumeID, nodeID string) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	existing, ok := d.attachments[volumeID]
	if !ok {
		return fmt.Errorf("volume %s is not attached", volumeID)
	}
	if existing != nodeID {
		return fmt.Errorf("volume %s is attached to %s, not %s", volumeID, existing, nodeID)
	}
	delete(d.attachments, volumeID)
	return nil
}

// ReadVolume reads size bytes at offset. A read past EOF returns the short
// read without error, matching POSIX read semantics.
func (d *localDriver) ReadVolume(ctx context.Context, volumeID string, offset int64, size int) ([]byte, error) {
	if err := d.validateVolumeID(volumeID); err != nil {
		return nil, err
	}

	d.mu.RLock()
	defer d.mu.RUnlock()

	f, err := os.Open(d.volumePath(volumeID))
	if err != nil {
		return nil, fmt.Errorf("failed to open volume %s: %w", volumeID, err)
	}
	defer f.Close()

	buf := make([]byte, size)
	n, err := f.ReadAt(buf, offset)
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("failed to read volume %s at offset %d: %w", volumeID, offset, err)
	}
	return buf[:n], nil
}

// WriteVolume writes data at offset; a write past the current size grows the
// file.
func (d *localDriver) WriteVolume(ctx context.Context, volumeID string, offset int64, data []byte) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	f, err := os.OpenFile(d.volumePath(volumeID), os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("failed to open volume %s for write: %w", volumeID, err)
	}
	defer f.Close()

	if _, err := f.WriteAt(data, offset); err != nil {
		return fmt.Errorf("failed to write volume %s at offset %d: %w", volumeID, offset, err)
	}
	return nil
}

// GetVolumeInfo returns stat-backed information about a volume.
func (d *localDriver) GetVolumeInfo(ctx context.Context, volumeID string) (*VolumeInfo, error) {
	if err := d.validateVolumeID(volumeID); err != nil {
		return nil, err
	}

	d.mu.RLock()
	defer d.mu.RUnlock()

	path := d.volumePath(volumeID)
	fi, err := os.Stat(path)
	if err != nil {
		return nil, fmt.Errorf("volume %s does not exist: %w", volumeID, err)
	}

	info := &VolumeInfo{
		ID:   volumeID,
		Size: fi.Size(),
	}
	if !fi.ModTime().IsZero() {
		info.UpdatedAt = fi.ModTime()
		info.CreatedAt = fi.ModTime()
	}
	if nodeID, ok := d.attachments[volumeID]; ok {
		info.AttachedToVM = nodeID
	}
	return info, nil
}

// ListVolumes returns the sorted IDs of all volumes in the base path.
func (d *localDriver) ListVolumes(ctx context.Context) ([]string, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	entries, err := os.ReadDir(d.basePath)
	if err != nil {
		return nil, fmt.Errorf("failed to list storage directory %s: %w", d.basePath, err)
	}
	volumes := make([]string, 0, len(entries))
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, ".img") {
			continue
		}
		volumes = append(volumes, name[:len(name)-len(".img")])
	}
	sort.Strings(volumes)
	return volumes, nil
}

// GetCapabilities reports what the local driver supports.
func (d *localDriver) GetCapabilities() DriverCapabilities {
	return DriverCapabilities{
		SupportsSnapshots: true,
		MinVolumeSize:     1 << 20, // 1 MiB
	}
}

// CreateSnapshot copies the volume image to snapshots/<volumeID>/<snapshotID>.img.
func (d *localDriver) CreateSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}
	if err := d.validateVolumeID(snapshotID); err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	return d.copyFileLocked(d.volumePath(volumeID), d.snapshotPath(volumeID, snapshotID), "snapshot")
}

// DeleteSnapshot removes a single snapshot file.
func (d *localDriver) DeleteSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}
	if err := d.validateVolumeID(snapshotID); err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if err := os.Remove(d.snapshotPath(volumeID, snapshotID)); err != nil {
		return fmt.Errorf("failed to delete snapshot %s of volume %s: %w", snapshotID, volumeID, err)
	}
	return nil
}

// RestoreSnapshot copies the snapshot back over the volume file. Restoring is
// refused while the volume is attached, mirroring how a block device cannot be
// rewritten underneath a live consumer.
func (d *localDriver) RestoreSnapshot(ctx context.Context, volumeID, snapshotID string) error {
	if err := d.validateVolumeID(volumeID); err != nil {
		return err
	}
	if err := d.validateVolumeID(snapshotID); err != nil {
		return err
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if _, attached := d.attachments[volumeID]; attached {
		return fmt.Errorf("volume %s is attached; detach before restoring a snapshot", volumeID)
	}
	if err := d.copyFileLocked(d.snapshotPath(volumeID, snapshotID), d.volumePath(volumeID), "restore"); err != nil {
		return err
	}
	return nil
}

// copyFileLocked copies src to dst; the caller holds d.mu.
func (d *localDriver) copyFileLocked(src, dst, action string) error {
	in, err := os.Open(src)
	if err != nil {
		return fmt.Errorf("failed to open %s source %s: %w", action, src, err)
	}
	defer in.Close()

	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("failed to create %s directory: %w", action, err)
	}

	out, err := os.OpenFile(dst, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("failed to open %s target %s: %w", action, dst, err)
	}
	defer out.Close()

	if _, err := io.Copy(out, in); err != nil {
		return fmt.Errorf("failed to copy %s: %w", action, err)
	}
	return nil
}
