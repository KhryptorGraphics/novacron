package migration

import (
	"os"
	"path/filepath"
)

func init() {
	if os.Getenv("NOVACRON_CHECKPOINT_DIR") == "" {
		os.Setenv("NOVACRON_CHECKPOINT_DIR", filepath.Join(os.TempDir(), "novacron-test-checkpoints"))
	}
}
