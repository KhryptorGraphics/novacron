package consensus

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// RaftStorage is the durable-state contract for a Raft node. currentTerm,
// votedFor, and the log MUST survive a process restart: losing votedFor
// allows a node to grant a second vote in a term it already voted in
// (double-vote, breaking the Raft safety property), and losing committed
// log entries silently discards accepted writes.
//
// Implementations MUST make SaveState and AppendLog durable (e.g. fsync)
// before returning a nil error, so a crash immediately after a successful
// call cannot lose the write.
type RaftStorage interface {
	// SaveState atomically persists currentTerm and votedFor together, so a
	// reader never observes one updated without the other.
	SaveState(currentTerm int64, votedFor string) error

	// AppendLog durably appends entries to the persisted log, in order.
	AppendLog(entries []LogEntry) error

	// TruncateLog discards any persisted entries with Index >= fromIndex
	// (1-based), e.g. on a Raft log conflict or snapshot compaction.
	TruncateLog(fromIndex int64) error

	// Load returns the last persisted currentTerm, votedFor, and full log.
	// Storage with nothing persisted yet returns the zero values and a nil
	// error.
	Load() (currentTerm int64, votedFor string, log []LogEntry, err error)

	// Close releases any underlying resources (e.g. open file handles).
	Close() error
}

// InMemoryRaftStorage is a non-durable RaftStorage used as the default for
// callers that don't need to survive a process restart (existing tests and
// call sites written before persistence existed). It preserves state only
// for the lifetime of the process/object -- passing the SAME instance to a
// freshly constructed RaftNode simulates a restart without process exit.
type InMemoryRaftStorage struct {
	mu          sync.Mutex
	currentTerm int64
	votedFor    string
	log         []LogEntry
}

// NewInMemoryRaftStorage creates a non-durable RaftStorage.
func NewInMemoryRaftStorage() *InMemoryRaftStorage {
	return &InMemoryRaftStorage{}
}

func (s *InMemoryRaftStorage) SaveState(currentTerm int64, votedFor string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.currentTerm = currentTerm
	s.votedFor = votedFor
	return nil
}

func (s *InMemoryRaftStorage) AppendLog(entries []LogEntry) error {
	if len(entries) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.log = append(s.log, entries...)
	return nil
}

func (s *InMemoryRaftStorage) TruncateLog(fromIndex int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if fromIndex < 1 {
		fromIndex = 1
	}
	if fromIndex-1 < int64(len(s.log)) {
		s.log = s.log[:fromIndex-1]
	}
	return nil
}

func (s *InMemoryRaftStorage) Load() (int64, string, []LogEntry, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	logCopy := make([]LogEntry, len(s.log))
	copy(logCopy, s.log)
	return s.currentTerm, s.votedFor, logCopy, nil
}

func (s *InMemoryRaftStorage) Close() error {
	return nil
}

// raftHardState is the on-disk representation of currentTerm and votedFor.
type raftHardState struct {
	CurrentTerm int64  `json:"current_term"`
	VotedFor    string `json:"voted_for"`
}

// FileRaftStorage is a file-backed, fsync'd RaftStorage. currentTerm and
// votedFor live in a single small JSON file written via write-tmp+fsync+
// rename (atomic on POSIX filesystems, so a crash mid-write can never leave
// a torn/partial hard state on disk). The log is an append-only
// newline-delimited JSON file; every AppendLog call fsyncs before
// returning, so an acknowledged append survives a crash.
type FileRaftStorage struct {
	mu        sync.Mutex
	dir       string
	stateFile string
	logFile   string
	logFH     *os.File
}

// NewFileRaftStorage creates (or opens) file-backed Raft storage rooted at
// dir. dir is created if it does not already exist.
func NewFileRaftStorage(dir string) (*FileRaftStorage, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("raft storage: create dir %q: %w", dir, err)
	}

	fs := &FileRaftStorage{
		dir:       dir,
		stateFile: filepath.Join(dir, "raft-state.json"),
		logFile:   filepath.Join(dir, "raft-log.jsonl"),
	}

	fh, err := os.OpenFile(fs.logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, fmt.Errorf("raft storage: open log file %q: %w", fs.logFile, err)
	}
	fs.logFH = fh

	return fs, nil
}

func (fs *FileRaftStorage) SaveState(currentTerm int64, votedFor string) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	data, err := json.Marshal(raftHardState{CurrentTerm: currentTerm, VotedFor: votedFor})
	if err != nil {
		return fmt.Errorf("raft storage: marshal state: %w", err)
	}

	tmp := fs.stateFile + ".tmp"
	if err := writeFileFsync(tmp, data); err != nil {
		return fmt.Errorf("raft storage: write tmp state file: %w", err)
	}
	if err := os.Rename(tmp, fs.stateFile); err != nil {
		return fmt.Errorf("raft storage: rename state file: %w", err)
	}
	fsyncDir(fs.dir)

	return nil
}

func (fs *FileRaftStorage) AppendLog(entries []LogEntry) error {
	if len(entries) == 0 {
		return nil
	}

	fs.mu.Lock()
	defer fs.mu.Unlock()

	var buf bytes.Buffer
	for _, e := range entries {
		data, err := json.Marshal(e)
		if err != nil {
			return fmt.Errorf("raft storage: marshal log entry %d: %w", e.Index, err)
		}
		buf.Write(data)
		buf.WriteByte('\n')
	}

	if _, err := fs.logFH.Write(buf.Bytes()); err != nil {
		return fmt.Errorf("raft storage: write log entries: %w", err)
	}
	if err := fs.logFH.Sync(); err != nil {
		return fmt.Errorf("raft storage: fsync log file: %w", err)
	}

	return nil
}

func (fs *FileRaftStorage) TruncateLog(fromIndex int64) error {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	entries, err := fs.readLogLocked()
	if err != nil {
		return err
	}

	kept := entries[:0]
	for _, e := range entries {
		if e.Index < fromIndex {
			kept = append(kept, e)
		}
	}

	var buf bytes.Buffer
	for _, e := range kept {
		data, mErr := json.Marshal(e)
		if mErr != nil {
			return fmt.Errorf("raft storage: marshal log entry %d: %w", e.Index, mErr)
		}
		buf.Write(data)
		buf.WriteByte('\n')
	}

	tmp := fs.logFile + ".tmp"
	if err := writeFileFsync(tmp, buf.Bytes()); err != nil {
		return fmt.Errorf("raft storage: write tmp log file: %w", err)
	}

	if err := fs.logFH.Close(); err != nil {
		return fmt.Errorf("raft storage: close log file: %w", err)
	}
	if err := os.Rename(tmp, fs.logFile); err != nil {
		return fmt.Errorf("raft storage: rename log file: %w", err)
	}
	fsyncDir(fs.dir)

	fh, err := os.OpenFile(fs.logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("raft storage: reopen log file: %w", err)
	}
	fs.logFH = fh

	return nil
}

// readLogLocked reads and parses the on-disk log file. Callers must hold
// fs.mu.
func (fs *FileRaftStorage) readLogLocked() ([]LogEntry, error) {
	data, err := os.ReadFile(fs.logFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("raft storage: read log file: %w", err)
	}

	var entries []LogEntry
	for _, line := range bytes.Split(data, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if len(line) == 0 {
			continue
		}
		var e LogEntry
		if err := json.Unmarshal(line, &e); err != nil {
			return nil, fmt.Errorf("raft storage: unmarshal log entry: %w", err)
		}
		entries = append(entries, e)
	}

	return entries, nil
}

func (fs *FileRaftStorage) Load() (int64, string, []LogEntry, error) {
	fs.mu.Lock()
	defer fs.mu.Unlock()

	var state raftHardState
	data, err := os.ReadFile(fs.stateFile)
	switch {
	case err == nil:
		if uErr := json.Unmarshal(data, &state); uErr != nil {
			return 0, "", nil, fmt.Errorf("raft storage: unmarshal state: %w", uErr)
		}
	case os.IsNotExist(err):
		// Nothing persisted yet: zero values are correct.
	default:
		return 0, "", nil, fmt.Errorf("raft storage: read state file: %w", err)
	}

	entries, err := fs.readLogLocked()
	if err != nil {
		return 0, "", nil, err
	}

	return state.CurrentTerm, state.VotedFor, entries, nil
}

func (fs *FileRaftStorage) Close() error {
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if fs.logFH == nil {
		return nil
	}
	err := fs.logFH.Close()
	fs.logFH = nil
	return err
}

// writeFileFsync writes data to path (creating/truncating it) and fsyncs
// before returning, so a crash immediately after a successful return cannot
// observe a partially-written file.
func writeFileFsync(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return fmt.Errorf("create %q: %w", path, err)
	}
	if _, err := f.Write(data); err != nil {
		f.Close()
		return fmt.Errorf("write %q: %w", path, err)
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return fmt.Errorf("fsync %q: %w", path, err)
	}
	return f.Close()
}

// fsyncDir best-effort fsyncs a directory so a preceding rename into it is
// durable. Not all platforms support fsync on directories; failures here
// are non-fatal since the rename itself already landed.
func fsyncDir(dir string) {
	d, err := os.Open(dir)
	if err != nil {
		return
	}
	_ = d.Sync()
	_ = d.Close()
}
