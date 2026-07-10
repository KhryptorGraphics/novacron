package dwcp

import (
	"bytes"
	"crypto/rand"
	"strings"
	"testing"

	"github.com/klauspost/compress/zstd"
)

// TestTrainDictionaryProducesLoadableDict proves TrainDictionary produces a
// genuinely usable zstd dictionary, not just a call that returns nil error.
//
// Before this fix (novacron-o0e), TrainDictionary failed unconditionally:
// zstd.BuildDict was called with only Contents set, never History, and
// BuildDict hard-requires len(History) >= 8. That is fixed by passing the
// concatenated samples as History.
//
// A second, independent bug remained after the History fix alone: BuildDict
// embeds BuildDictOptions.ID verbatim into the dictionary bytes, and zstd's
// own dictionary loader (invoked by WithEncoderDict/WithDecoderDicts,
// exercised below) hard-rejects ID 0 with "dictionaries cannot have ID 0".
// Leaving ID unset would make TrainDictionary appear to succeed while
// producing a dictionary that zstd itself refuses to load — this test
// exercises exactly that load path (via compressWithDict, unexported,
// hence this white-box package-internal test) plus a full compress+decode
// round trip through the dictionary, which a train-and-count-only test
// would not catch.
//
// Sample data here uses a mixed zero/pattern/random shape (confirmed via
// standalone reproduction not to trigger the zstd.BuildDict panic covered
// by TestTrainDictionaryRecoversFromBuildDictPanic below) so this test
// exercises the success path cleanly.
//
// Scope note: this proves TrainDictionary's OUTPUT is a well-formed,
// loadable zstd dictionary. It does not prove HDE.CompressMemory/
// CompressDisk + HDE.Decompress round-trip correctly THROUGH a dictionary
// end-to-end — HDE's own decoders (see NewHDE) are never constructed with
// WithDecoderDicts, and CompressMemory never calls compressWithDict at
// all (only CompressDisk does). Those are real, separate gaps, filed
// as a follow-up; MigrationAdapter's receive path sidesteps them entirely
// by forcing EnableDictionary=false (see NewMigrationAdapter).
func TestTrainDictionaryProducesLoadableDict(t *testing.T) {
	hde, err := NewHDE(HDEConfig{
		GlobalLevel:      9,
		EnableDictionary: true,
		DictSize:         1024,
		TrainingSamples:  100,
	})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	samples := make([][]byte, 100)
	for i := range samples {
		s := make([]byte, 4096)
		switch i % 3 {
		case 0:
			// zero page: leave as-is
		case 1:
			for j := range s {
				s[j] = byte(j % 256)
			}
		case 2:
			rand.Read(s)
		}
		samples[i] = s
	}

	dictID := "vm-disk-type-A"
	if err := hde.TrainDictionary(dictID, samples); err != nil {
		t.Fatalf("TrainDictionary failed: %v", err)
	}

	hde.dictMu.RLock()
	dict, exists := hde.dictionaries[dictID]
	hde.dictMu.RUnlock()
	if !exists {
		t.Fatal("trained dictionary not found in hde.dictionaries")
	}
	if len(dict) == 0 {
		t.Fatal("trained dictionary is empty")
	}

	// Prove the dictionary is genuinely LOADABLE by zstd itself: this is
	// exactly the code path (WithEncoderDict) that unconditionally failed
	// with "dictionaries cannot have ID 0" before the ID fix.
	original := make([]byte, 8192)
	rand.Read(original)
	// Overlay the dictionary's own training-sample pattern onto part of
	// the payload so the dictionary has something real to reference.
	copy(original, samples[1])

	compressed, err := hde.compressWithDict(original, dict, CompressionRegional)
	if err != nil {
		t.Fatalf("compressWithDict failed with trained dictionary (ID=0 regression?): %v", err)
	}
	if len(compressed) == 0 {
		t.Fatal("compressWithDict produced empty output")
	}

	// Full round trip THROUGH the dictionary via zstd directly (not via
	// HDE.Decompress, which does not wire dictionaries into its decoders
	// — see scope note above): proves the dictionary bytes are valid for
	// both encode and decode, not just encode.
	decoder, err := zstd.NewReader(nil, zstd.WithDecoderDicts(dict))
	if err != nil {
		t.Fatalf("failed to construct decoder with trained dictionary: %v", err)
	}
	defer decoder.Close()

	decoded, err := decoder.DecodeAll(compressed, nil)
	if err != nil {
		t.Fatalf("dictionary-aware decode failed: %v", err)
	}
	if !bytes.Equal(decoded, original) {
		t.Fatalf("round trip through trained dictionary corrupted data: got %d bytes, want %d bytes matching original", len(decoded), len(original))
	}
}

// TestTrainDictionaryRecoversFromBuildDictPanic proves TrainDictionary
// returns a clean error instead of crashing the process when the
// underlying zstd.BuildDict hits its internal "integer divide by zero"
// bug (klauspost/compress@v1.18.1, confirmed unfixed through v1.18.5:
// zstd/dict.go's literal-table sizing computes avgSize := min(litTotal, …)
// and then litTotal/avgSize, which panics when avgSize is 0).
//
// Confirmed via standalone reproduction that this panic is triggered by
// realistic-looking, not-obviously-invalid training data: highly
// repetitive/patterned samples (this test's smooth byte ramp) and small
// random sample sets both panic; mixed zero/pattern/random data (used by
// TestTrainDictionaryProducesLoadableDict above) does not. Before the
// History fix, TrainDictionary always failed at an earlier check
// (len(History) < 8) and never reached this code path, so the panic was
// latent, not actually exercised — fixing History alone would have turned
// a clean, pre-existing error into a process crash for these inputs.
func TestTrainDictionaryRecoversFromBuildDictPanic(t *testing.T) {
	hde, err := NewHDE(HDEConfig{GlobalLevel: 9, EnableDictionary: true})
	if err != nil {
		t.Fatalf("NewHDE failed: %v", err)
	}
	defer hde.Close()

	// Smooth deterministic byte ramp: confirmed via standalone
	// reproduction to trigger zstd.BuildDict's divide-by-zero panic.
	samples := make([][]byte, 100)
	for i := range samples {
		s := make([]byte, 4096)
		for j := range s {
			s[j] = byte((i + j) % 251)
		}
		samples[i] = s
	}

	err = hde.TrainDictionary("panic-trigger", samples)
	if err == nil {
		t.Fatal("expected TrainDictionary to return an error for panic-triggering input, got nil — either the upstream zstd bug was fixed (safe to leave the recover() in place regardless) or this input no longer reproduces it and should be replaced with one that does")
	}
	if !strings.Contains(err.Error(), "panicked") {
		t.Fatalf("expected a recovered-panic error, got a different error (upstream behavior may have changed): %v", err)
	}
}
