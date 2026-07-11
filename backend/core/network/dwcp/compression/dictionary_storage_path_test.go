package compression

import (
	"crypto/rand"
	"path/filepath"
	"testing"
	"time"

	"go.uber.org/zap"
)

// dictFilesIn returns the *.dict.json files directly under dir. A missing dir
// yields an empty slice (Glob returns no matches, not an error, for a path that
// does not exist), which is exactly the "nothing was written here" signal the
// assertions below rely on.
func dictFilesIn(dir string) []string {
	matches, _ := filepath.Glob(filepath.Join(dir, "*.dict.json"))
	return matches
}

// TestDeltaEncoder_DictionaryStoragePathOverride is the regression guard for
// novacron-113. It proves that a DeltaEncodingConfig carrying an explicit
// DictionaryStoragePath causes NewDeltaEncoder's DictionaryTrainer to persist
// trained dictionaries THERE (an isolated t.TempDir()), rather than the relative
// default "./compression/dictionaries" that — because go test's CWD is the
// package source dir — resolves to backend/core/network/dwcp/compression/
// compression/dictionaries/ and pollutes the working tree on every training run.
//
// Before the fix DeltaEncodingConfig had no such field and NewDeltaEncoder
// hardcoded DefaultDictionaryTrainingConfig(), so this test could not even be
// written (no override) and fails; after the fix the override is honored and the
// dictionary lands in the temp dir with no doubled-path pollution.
func TestDeltaEncoder_DictionaryStoragePathOverride(t *testing.T) {
	logger, _ := zap.NewDevelopment()

	tmpDir := t.TempDir()

	config := &DeltaEncodingConfig{
		Enabled:               true,
		BaselineInterval:      1 * time.Hour,
		MaxBaselineAge:        2 * time.Hour,
		MaxDeltaChain:         10,
		CompressionLevel:      3,
		EnableDictionary:      true,
		DictionaryStoragePath: tmpDir, // the override under test
		DeltaAlgorithm:        "xor",
		EnableAdaptive:        false,
		EnableBaselineSync:    false,
	}

	encoder, err := NewDeltaEncoder(config, logger)
	if err != nil {
		t.Fatalf("NewDeltaEncoder failed: %v", err)
	}
	defer encoder.Close()

	// Capture the relative default (doubled) path's contents before training so
	// we can assert we created nothing NEW there. A stale dictionary written by
	// a sibling test in this same process would otherwise mask a regression, and
	// pre-existing files must not be counted against this test.
	const defaultDir = "./compression/dictionaries"
	beforeDefault := len(dictFilesIn(defaultDir))

	// Proven-safe training input (mirrors TestDeltaEncoder_Phase1_DictionaryTraining):
	// a uniform repeating pattern at these dimensions hits a real upstream
	// zstd.BuildDict "integer divide by zero"; the mixed zero/pattern/random shape
	// avoids it, so training actually succeeds and saveDictionary writes a file.
	pattern := []byte("VM_MEMORY_PAGE_DATA_PATTERN_REPEATING_")
	for i := range 50 {
		data := make([]byte, 10*1024) // 10KB, above MinSampleSize (1KB)
		switch i % 3 {
		case 0:
			// zero page: leave as-is
		case 1:
			for j := 0; j < len(data); j += len(pattern) {
				copy(data[j:], pattern)
			}
		case 2:
			rand.Read(data)
		}
		encoder.dictionaryTrainer.AddSample("vm-memory", data)
	}

	if err := encoder.TrainDictionaries(); err != nil {
		t.Fatalf("TrainDictionaries failed: %v", err)
	}

	// Primary assertion: the dictionary must have been persisted into the
	// override dir. Empty here means DictionaryStoragePath was ignored.
	got := dictFilesIn(tmpDir)
	if len(got) == 0 {
		t.Fatalf("no *.dict.json written to override StoragePath %q; "+
			"DictionaryStoragePath was not honored by NewDeltaEncoder", tmpDir)
	}
	t.Logf("dictionary files written to override dir %q: %v", tmpDir, got)

	// Secondary guard: training must not have leaked any NEW file into the
	// relative default (doubled) path — that would mean persistence was not
	// fully redirected.
	afterDefault := len(dictFilesIn(defaultDir))
	if afterDefault > beforeDefault {
		t.Errorf("training polluted relative default dir %q (before=%d after=%d files); "+
			"override did not fully redirect dictionary persistence",
			defaultDir, beforeDefault, afterDefault)
	}
}
