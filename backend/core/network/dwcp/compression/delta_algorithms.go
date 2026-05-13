package compression

import (
	"bytes"
	"fmt"
	"hash/adler32"
)

// DeltaAlgorithm represents a delta encoding algorithm
type DeltaAlgorithm string

const (
	DeltaAlgorithmXOR    DeltaAlgorithm = "xor"
	DeltaAlgorithmRSync  DeltaAlgorithm = "rsync"
	DeltaAlgorithmBSDiff DeltaAlgorithm = "bsdiff"
	DeltaAlgorithmAuto   DeltaAlgorithm = "auto"
)

// DeltaComputer interface for delta computation algorithms
type DeltaComputer interface {
	ComputeDelta(baseline, current []byte) ([]byte, error)
	ApplyDelta(baseline, delta []byte) ([]byte, error)
	Name() string
}

// XORDeltaComputer implements simple XOR-based delta encoding
type XORDeltaComputer struct{}

func (x *XORDeltaComputer) Name() string {
	return "xor"
}

func (x *XORDeltaComputer) ComputeDelta(baseline, current []byte) ([]byte, error) {
	maxLen := len(baseline)
	if len(current) > maxLen {
		maxLen = len(current)
	}

	delta := make([]byte, maxLen)

	// XOR corresponding bytes
	minLen := len(baseline)
	if len(current) < minLen {
		minLen = len(current)
	}

	for i := 0; i < minLen; i++ {
		delta[i] = baseline[i] ^ current[i]
	}

	// If current is longer, append the extra bytes
	if len(current) > len(baseline) {
		copy(delta[len(baseline):], current[len(baseline):])
	} else if len(baseline) > len(current) {
		// If baseline is longer, store original bytes
		for i := len(current); i < len(baseline); i++ {
			delta[i] = baseline[i]
		}
	}

	return delta, nil
}

func (x *XORDeltaComputer) ApplyDelta(baseline, delta []byte) ([]byte, error) {
	maxLen := len(baseline)
	if len(delta) > maxLen {
		maxLen = len(delta)
	}

	reconstructed := make([]byte, maxLen)

	// XOR to reconstruct
	minLen := len(baseline)
	if len(delta) < minLen {
		minLen = len(delta)
	}

	for i := 0; i < minLen; i++ {
		reconstructed[i] = baseline[i] ^ delta[i]
	}

	// Handle size differences
	if len(delta) > len(baseline) {
		copy(reconstructed[len(baseline):], delta[len(baseline):])
	}

	return reconstructed, nil
}

// RSyncDeltaComputer implements rsync-style rolling checksums
type RSyncDeltaComputer struct {
	blockSize int
}

func NewRSyncDeltaComputer(blockSize int) *RSyncDeltaComputer {
	if blockSize <= 0 {
		blockSize = 4096 // Default 4KB blocks
	}
	return &RSyncDeltaComputer{blockSize: blockSize}
}

func (r *RSyncDeltaComputer) Name() string {
	return "rsync"
}

func (r *RSyncDeltaComputer) ComputeDelta(baseline, current []byte) ([]byte, error) {
	// Build block signature map from baseline
	signatures := r.buildSignatures(baseline)

	// Find matching and non-matching blocks in current
	var delta bytes.Buffer

	pos := 0
	currentLen := len(current)

	for pos < currentLen {
		// Try to find a matching block
		matchFound := false
		if pos+r.blockSize <= currentLen {
			block := current[pos : pos+r.blockSize]
			checksum := adler32.Checksum(block)

			if baselinePos, exists := signatures[checksum]; exists {
				// Matching block - write reference
				delta.WriteByte(0xFF) // Marker for block reference
				delta.Write(uint32ToBytes(uint32(baselinePos)))
				pos += r.blockSize
				matchFound = true
			}
		}

		if !matchFound {
			// Non-matching data - write literal
			endPos := pos + r.blockSize
			if endPos > currentLen {
				endPos = currentLen
			}

			literalData := current[pos:endPos]
			delta.WriteByte(0x00) // Marker for literal data
			delta.Write(uint32ToBytes(uint32(len(literalData))))
			delta.Write(literalData)
			pos = endPos
		}
	}

	return delta.Bytes(), nil
}

func (r *RSyncDeltaComputer) ApplyDelta(baseline, delta []byte) ([]byte, error) {
	var result bytes.Buffer
	pos := 0
	deltaLen := len(delta)

	for pos < deltaLen {
		if pos >= deltaLen {
			break
		}

		marker := delta[pos]
		pos++

		if marker == 0xFF {
			// Block reference
			if pos+4 > deltaLen {
				return nil, fmt.Errorf("truncated delta: expected block position")
			}
			blockPos := bytesToUint32(delta[pos : pos+4])
			pos += 4

			// Copy block from baseline
			if int(blockPos)+r.blockSize > len(baseline) {
				return nil, fmt.Errorf("invalid block reference: %d", blockPos)
			}
			result.Write(baseline[blockPos : blockPos+uint32(r.blockSize)])

		} else if marker == 0x00 {
			// Literal data
			if pos+4 > deltaLen {
				return nil, fmt.Errorf("truncated delta: expected literal length")
			}
			literalLen := bytesToUint32(delta[pos : pos+4])
			pos += 4

			if pos+int(literalLen) > deltaLen {
				return nil, fmt.Errorf("truncated delta: expected %d bytes of literal data", literalLen)
			}
			result.Write(delta[pos : pos+int(literalLen)])
			pos += int(literalLen)

		} else {
			return nil, fmt.Errorf("invalid delta marker: 0x%02x", marker)
		}
	}

	return result.Bytes(), nil
}

func (r *RSyncDeltaComputer) buildSignatures(data []byte) map[uint32]uint32 {
	signatures := make(map[uint32]uint32)
	dataLen := len(data)

	for pos := 0; pos+r.blockSize <= dataLen; pos += r.blockSize {
		block := data[pos : pos+r.blockSize]
		checksum := adler32.Checksum(block)
		signatures[checksum] = uint32(pos)
	}

	return signatures
}

// BSDiffDeltaComputer implements bsdiff algorithm for binary diffs
type BSDiffDeltaComputer struct{}

const (
	bsdiffMagic          = "NBD1"
	bsdiffMinCopyLength  = 16
	bsdiffMaxCandidates  = 8
	bsdiffCopyCommand    = 0x01
	bsdiffLiteralCommand = 0x02
)

func (b *BSDiffDeltaComputer) Name() string {
	return "bsdiff"
}

func (b *BSDiffDeltaComputer) ComputeDelta(baseline, current []byte) ([]byte, error) {
	index := buildBSDiffIndex(baseline)

	var delta bytes.Buffer
	delta.WriteString(bsdiffMagic)

	var literal bytes.Buffer
	flushLiteral := func() {
		if literal.Len() == 0 {
			return
		}
		delta.WriteByte(bsdiffLiteralCommand)
		delta.Write(uint32ToBytes(uint32(literal.Len())))
		delta.Write(literal.Bytes())
		literal.Reset()
	}

	for pos := 0; pos < len(current); {
		matchPos, matchLen := longestBSDiffMatch(baseline, current, pos, index)
		if matchLen >= bsdiffMinCopyLength {
			flushLiteral()
			delta.WriteByte(bsdiffCopyCommand)
			delta.Write(uint32ToBytes(uint32(matchPos)))
			delta.Write(uint32ToBytes(uint32(matchLen)))
			pos += matchLen
			continue
		}

		literal.WriteByte(current[pos])
		pos++
	}
	flushLiteral()

	return delta.Bytes(), nil
}

func (b *BSDiffDeltaComputer) ApplyDelta(baseline, delta []byte) ([]byte, error) {
	if len(delta) < len(bsdiffMagic) || string(delta[:len(bsdiffMagic)]) != bsdiffMagic {
		return nil, fmt.Errorf("invalid bsdiff delta header")
	}

	var result bytes.Buffer
	pos := len(bsdiffMagic)
	for pos < len(delta) {
		cmd := delta[pos]
		pos++

		switch cmd {
		case bsdiffCopyCommand:
			if pos+8 > len(delta) {
				return nil, fmt.Errorf("truncated bsdiff copy command")
			}
			baselinePos := int(bytesToUint32(delta[pos : pos+4]))
			pos += 4
			length := int(bytesToUint32(delta[pos : pos+4]))
			pos += 4
			if baselinePos < 0 || length < 0 || baselinePos+length > len(baseline) {
				return nil, fmt.Errorf("invalid bsdiff copy range: pos=%d len=%d", baselinePos, length)
			}
			result.Write(baseline[baselinePos : baselinePos+length])

		case bsdiffLiteralCommand:
			if pos+4 > len(delta) {
				return nil, fmt.Errorf("truncated bsdiff literal command")
			}
			length := int(bytesToUint32(delta[pos : pos+4]))
			pos += 4
			if length < 0 || pos+length > len(delta) {
				return nil, fmt.Errorf("truncated bsdiff literal payload")
			}
			result.Write(delta[pos : pos+length])
			pos += length

		default:
			return nil, fmt.Errorf("invalid bsdiff command: 0x%02x", cmd)
		}
	}

	return result.Bytes(), nil
}

func buildBSDiffIndex(baseline []byte) map[string][]int {
	index := make(map[string][]int)
	if len(baseline) < bsdiffMinCopyLength {
		return index
	}
	for pos := 0; pos+bsdiffMinCopyLength <= len(baseline); pos += bsdiffMinCopyLength {
		key := string(baseline[pos : pos+bsdiffMinCopyLength])
		if len(index[key]) >= bsdiffMaxCandidates {
			continue
		}
		index[key] = append(index[key], pos)
	}
	return index
}

func longestBSDiffMatch(baseline, current []byte, currentPos int, index map[string][]int) (int, int) {
	if currentPos+bsdiffMinCopyLength > len(current) {
		return 0, 0
	}

	key := string(current[currentPos : currentPos+bsdiffMinCopyLength])
	candidates := index[key]
	bestPos := 0
	bestLen := 0
	for _, baselinePos := range candidates {
		matchLen := 0
		for baselinePos+matchLen < len(baseline) &&
			currentPos+matchLen < len(current) &&
			baseline[baselinePos+matchLen] == current[currentPos+matchLen] {
			matchLen++
		}
		if matchLen > bestLen {
			bestPos = baselinePos
			bestLen = matchLen
		}
	}

	return bestPos, bestLen
}

// AdaptiveDeltaComputer selects the best algorithm based on data characteristics
type AdaptiveDeltaComputer struct {
	xor    *XORDeltaComputer
	rsync  *RSyncDeltaComputer
	bsdiff *BSDiffDeltaComputer
}

func NewAdaptiveDeltaComputer() *AdaptiveDeltaComputer {
	return &AdaptiveDeltaComputer{
		xor:    &XORDeltaComputer{},
		rsync:  NewRSyncDeltaComputer(4096),
		bsdiff: &BSDiffDeltaComputer{},
	}
}

func (a *AdaptiveDeltaComputer) Name() string {
	return "auto"
}

func (a *AdaptiveDeltaComputer) ComputeDelta(baseline, current []byte) ([]byte, error) {
	// Select algorithm based on data size and characteristics
	dataSize := len(current)

	// Small data (<10KB): use XOR (fastest)
	if dataSize < 10*1024 {
		return a.xor.ComputeDelta(baseline, current)
	}

	// Medium data (10KB-1MB): use rsync (balanced)
	if dataSize < 1024*1024 {
		return a.rsync.ComputeDelta(baseline, current)
	}

	// Large data (>1MB): use bsdiff (best compression)
	return a.bsdiff.ComputeDelta(baseline, current)
}

func (a *AdaptiveDeltaComputer) ApplyDelta(baseline, delta []byte) ([]byte, error) {
	// Try algorithms in order until one succeeds
	// This is a fallback mechanism since we don't store which algo was used

	// Try bsdiff first (most common for large files)
	if result, err := a.bsdiff.ApplyDelta(baseline, delta); err == nil {
		return result, nil
	}

	// Try rsync
	if result, err := a.rsync.ApplyDelta(baseline, delta); err == nil {
		return result, nil
	}

	// Fall back to XOR
	return a.xor.ApplyDelta(baseline, delta)
}

// Utility functions

func uint32ToBytes(n uint32) []byte {
	return []byte{
		byte(n >> 24),
		byte(n >> 16),
		byte(n >> 8),
		byte(n),
	}
}

func bytesToUint32(b []byte) uint32 {
	return uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
}

// DeltaAlgorithmFactory creates delta computers based on algorithm type
func DeltaAlgorithmFactory(algorithm DeltaAlgorithm) DeltaComputer {
	switch algorithm {
	case DeltaAlgorithmXOR:
		return &XORDeltaComputer{}
	case DeltaAlgorithmRSync:
		return NewRSyncDeltaComputer(4096)
	case DeltaAlgorithmBSDiff:
		return &BSDiffDeltaComputer{}
	case DeltaAlgorithmAuto:
		return NewAdaptiveDeltaComputer()
	default:
		return &XORDeltaComputer{}
	}
}
