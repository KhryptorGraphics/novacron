package websocket

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/sirupsen/logrus"
)

const vmCopyChunkSize = 64 * 1024

type VMGuestFileClient interface {
	FileOpen(ctx context.Context, path, mode string) (int, error)
	FileRead(ctx context.Context, handle, count int) ([]byte, bool, error)
	FileWrite(ctx context.Context, handle int, payload []byte) (int, bool, error)
	FileFlush(ctx context.Context, handle int) error
	FileClose(ctx context.Context, handle int) error
}

type VMGuestFileClientResolver interface {
	ResolveGuestFileClient(ctx context.Context, vmID string) (VMGuestFileClient, error)
}

type VMGuestFileCommitter interface {
	CommitUploadedFile(ctx context.Context, sourcePath, destinationPath string, overwrite bool, mode string) error
}

type VMGuestFileRemover interface {
	RemoveFile(ctx context.Context, path string) error
}

type VMCopyTransferEvent struct {
	VMID        string
	UserID      string
	TenantID    string
	Path        string
	Direction   string
	Bytes       int64
	Checksum    string
	Result      string
	Error       string
	StartedAt   time.Time
	CompletedAt time.Time
}

type VMCopyAuditSink interface {
	RecordVMCopyAudit(ctx context.Context, event VMCopyTransferEvent)
}

type VMCopyProgressSink interface {
	RecordVMCopyProgress(ctx context.Context, event VMCopyTransferEvent)
}

type VMCopyRateLimiter interface {
	Wait(ctx context.Context, tenantID string, bytes int) error
}

type QGAVMCopyService struct {
	resolver     VMGuestFileClientResolver
	auditSink    VMCopyAuditSink
	progressSink VMCopyProgressSink
	rateLimiter  VMCopyRateLimiter
}

type QGAVMCopyServiceOption func(*QGAVMCopyService)

func WithVMCopyAuditSink(sink VMCopyAuditSink) QGAVMCopyServiceOption {
	return func(service *QGAVMCopyService) {
		service.auditSink = sink
	}
}

func WithVMCopyProgressSink(sink VMCopyProgressSink) QGAVMCopyServiceOption {
	return func(service *QGAVMCopyService) {
		service.progressSink = sink
	}
}

func WithVMCopyRateLimiter(limiter VMCopyRateLimiter) QGAVMCopyServiceOption {
	return func(service *QGAVMCopyService) {
		service.rateLimiter = limiter
	}
}

func NewQGAVMCopyService(resolver VMGuestFileClientResolver, options ...QGAVMCopyServiceOption) *QGAVMCopyService {
	service := &QGAVMCopyService{resolver: resolver}
	for _, option := range options {
		if option != nil {
			option(service)
		}
	}
	return service
}

func (s *QGAVMCopyService) HandleVMCopy(ctx context.Context, vmID string, options VMCopyOptions, conn *websocket.Conn) error {
	if s == nil || s.resolver == nil {
		return errors.New("vm copy guest file resolver is required")
	}
	client, err := s.resolver.ResolveGuestFileClient(ctx, vmID)
	if err != nil {
		return fmt.Errorf("resolve guest file client: %w", err)
	}

	switch options.Direction {
	case "upload":
		return s.handleUpload(ctx, vmID, options, client, conn)
	case "download":
		return s.handleDownload(ctx, vmID, options, client, conn)
	default:
		return fmt.Errorf("unsupported vm copy direction %q", options.Direction)
	}
}

func (s *QGAVMCopyService) handleUpload(ctx context.Context, vmID string, options VMCopyOptions, client VMGuestFileClient, conn *websocket.Conn) (err error) {
	startedAt := time.Now()
	var writtenTotal int64
	var auditChecksum string
	defer func() {
		s.recordAudit(ctx, VMCopyTransferEvent{
			VMID:        vmID,
			UserID:      options.UserID,
			TenantID:    options.TenantID,
			Path:        options.Path,
			Direction:   options.Direction,
			Bytes:       writtenTotal,
			Checksum:    auditChecksum,
			Result:      transferResult(err),
			Error:       transferError(err),
			StartedAt:   startedAt,
			CompletedAt: time.Now(),
		})
	}()

	metadata, err := readCopyMetadata(conn)
	if err != nil {
		_ = writeVMIOError(conn, "invalid_metadata", err.Error())
		return err
	}
	if metadata.Path != "" && metadata.Path != options.Path {
		err := fmt.Errorf("metadata path %q does not match requested path %q", metadata.Path, options.Path)
		_ = writeVMIOError(conn, "path_mismatch", err.Error())
		return err
	}

	uploadPath := tempUploadPath(options.Path)
	handle, err := client.FileOpen(ctx, uploadPath, "wb")
	if err != nil {
		_ = writeVMIOError(conn, "open_failed", err.Error())
		return err
	}
	closed := false
	committed := false
	defer func() {
		if !closed {
			_ = client.FileClose(ctx, handle)
		}
		if !committed {
			removeGuestFile(ctx, client, uploadPath)
		}
	}()

	if err := writeVMIOAck(conn, 0); err != nil {
		return err
	}

	hash := sha256.New()
	for {
		messageType, frame, err := conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("read upload frame: %w", err)
		}
		if messageType != websocket.BinaryMessage {
			err := errors.New("upload frames must be binary websocket messages")
			_ = writeVMIOError(conn, "invalid_frame", err.Error())
			return err
		}

		frameType, payload, err := DecodeVMIOFrame(frame)
		if err != nil {
			_ = writeVMIOError(conn, "invalid_frame", err.Error())
			return err
		}

		switch frameType {
		case VMIOFrameCopyData:
			if err := s.waitTransfer(ctx, options.TenantID, len(payload)); err != nil {
				_ = writeVMIOError(conn, "rate_limited", err.Error())
				return err
			}
			written, _, err := client.FileWrite(ctx, handle, payload)
			if err != nil {
				_ = writeVMIOError(conn, "write_failed", err.Error())
				return err
			}
			writtenTotal += int64(written)
			_, _ = hash.Write(payload[:written])
			s.recordProgress(ctx, VMCopyTransferEvent{
				VMID:      vmID,
				UserID:    options.UserID,
				TenantID:  options.TenantID,
				Path:      options.Path,
				Direction: options.Direction,
				Bytes:     writtenTotal,
				StartedAt: startedAt,
			})
		case VMIOFrameCopyEOF:
			var eof VMCopyEOF
			if _, err := DecodeVMIOJSONFrame(frame, &eof); err != nil {
				_ = writeVMIOError(conn, "invalid_eof", err.Error())
				return err
			}
			if eof.Bytes > 0 && eof.Bytes != writtenTotal {
				err := fmt.Errorf("eof byte count %d does not match received bytes %d", eof.Bytes, writtenTotal)
				_ = writeVMIOError(conn, "byte_count_mismatch", err.Error())
				return err
			}
			if metadata.Size > 0 && metadata.Size != writtenTotal {
				err := fmt.Errorf("metadata byte count %d does not match received bytes %d", metadata.Size, writtenTotal)
				_ = writeVMIOError(conn, "byte_count_mismatch", err.Error())
				return err
			}
			expectedChecksum := firstNonEmptyString(eof.SHA256, metadata.SHA256)
			if expectedChecksum != "" {
				actualChecksum := hex.EncodeToString(hash.Sum(nil))
				if !strings.EqualFold(expectedChecksum, actualChecksum) {
					err := fmt.Errorf("sha256 checksum mismatch")
					_ = writeVMIOError(conn, "checksum_mismatch", err.Error())
					return err
				}
				auditChecksum = actualChecksum
			}
			if err := client.FileFlush(ctx, handle); err != nil {
				_ = writeVMIOError(conn, "flush_failed", err.Error())
				return err
			}
			if err := client.FileClose(ctx, handle); err != nil {
				_ = writeVMIOError(conn, "close_failed", err.Error())
				return err
			}
			closed = true

			committer, ok := client.(VMGuestFileCommitter)
			if !ok {
				err := errors.New("guest file client does not support atomic upload commit")
				_ = writeVMIOError(conn, "commit_unsupported", err.Error())
				return err
			}
			mode := firstNonEmptyString(options.Mode, metadata.Mode)
			if err := committer.CommitUploadedFile(ctx, uploadPath, options.Path, options.Overwrite, mode); err != nil {
				_ = writeVMIOError(conn, "commit_failed", err.Error())
				return err
			}
			committed = true
			if err := writeVMIOAck(conn, writtenTotal); err != nil {
				return err
			}
			return nil
		default:
			err := fmt.Errorf("unexpected upload frame type %#x", frameType)
			_ = writeVMIOError(conn, "unexpected_frame", err.Error())
			return err
		}
	}
}

func tempUploadPath(destination string) string {
	dir := path.Dir(destination)
	base := path.Base(destination)
	return path.Join(dir, fmt.Sprintf(".%s.novacron-upload-%d.tmp", base, time.Now().UnixNano()))
}

func removeGuestFile(ctx context.Context, client VMGuestFileClient, path string) {
	if remover, ok := client.(VMGuestFileRemover); ok {
		_ = remover.RemoveFile(ctx, path)
	}
}

func firstNonEmptyString(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func (s *QGAVMCopyService) handleDownload(ctx context.Context, vmID string, options VMCopyOptions, client VMGuestFileClient, conn *websocket.Conn) (err error) {
	startedAt := time.Now()
	var sentTotal int64
	defer func() {
		s.recordAudit(ctx, VMCopyTransferEvent{
			VMID:        vmID,
			UserID:      options.UserID,
			TenantID:    options.TenantID,
			Path:        options.Path,
			Direction:   options.Direction,
			Bytes:       sentTotal,
			Result:      transferResult(err),
			Error:       transferError(err),
			StartedAt:   startedAt,
			CompletedAt: time.Now(),
		})
	}()

	handle, err := client.FileOpen(ctx, options.Path, "rb")
	if err != nil {
		_ = writeVMIOError(conn, "open_failed", err.Error())
		return err
	}
	defer client.FileClose(ctx, handle)

	metadataFrame, err := EncodeVMIOJSONFrame(VMIOFrameCopyMetadata, VMCopyMetadata{Path: options.Path})
	if err != nil {
		return err
	}
	if err := conn.WriteMessage(websocket.BinaryMessage, metadataFrame); err != nil {
		return fmt.Errorf("write download metadata: %w", err)
	}

	for {
		chunk, eof, err := client.FileRead(ctx, handle, vmCopyChunkSize)
		if err != nil {
			_ = writeVMIOError(conn, "read_failed", err.Error())
			return err
		}
		if len(chunk) > 0 {
			if err := s.waitTransfer(ctx, options.TenantID, len(chunk)); err != nil {
				_ = writeVMIOError(conn, "rate_limited", err.Error())
				return err
			}
			if err := conn.WriteMessage(websocket.BinaryMessage, EncodeVMIODataFrame(VMIOFrameCopyData, chunk)); err != nil {
				return fmt.Errorf("write download data: %w", err)
			}
			sentTotal += int64(len(chunk))
			s.recordProgress(ctx, VMCopyTransferEvent{
				VMID:      vmID,
				UserID:    options.UserID,
				TenantID:  options.TenantID,
				Path:      options.Path,
				Direction: options.Direction,
				Bytes:     sentTotal,
				StartedAt: startedAt,
			})
		}
		if eof {
			eofFrame, err := EncodeVMIOJSONFrame(VMIOFrameCopyEOF, VMCopyEOF{Bytes: sentTotal})
			if err != nil {
				return err
			}
			if err := conn.WriteMessage(websocket.BinaryMessage, eofFrame); err != nil {
				return err
			}
			return nil
		}
	}
}

func (s *QGAVMCopyService) waitTransfer(ctx context.Context, tenantID string, bytes int) error {
	if s != nil && s.rateLimiter != nil && bytes > 0 {
		return s.rateLimiter.Wait(ctx, tenantID, bytes)
	}
	return nil
}

func (s *QGAVMCopyService) recordProgress(ctx context.Context, event VMCopyTransferEvent) {
	if s != nil && s.progressSink != nil {
		s.progressSink.RecordVMCopyProgress(ctx, event)
	}
}

func (s *QGAVMCopyService) recordAudit(ctx context.Context, event VMCopyTransferEvent) {
	if s != nil && s.auditSink != nil {
		s.auditSink.RecordVMCopyAudit(ctx, event)
	}
}

func transferResult(err error) string {
	if err != nil {
		return "failure"
	}
	return "success"
}

func transferError(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

type LogrusVMCopyAuditSink struct {
	logger *logrus.Logger
}

func NewLogrusVMCopyAuditSink(logger *logrus.Logger) *LogrusVMCopyAuditSink {
	if logger == nil {
		logger = logrus.New()
	}
	return &LogrusVMCopyAuditSink{logger: logger}
}

func (s *LogrusVMCopyAuditSink) RecordVMCopyAudit(ctx context.Context, event VMCopyTransferEvent) {
	if s == nil || s.logger == nil {
		return
	}
	s.logger.WithFields(logrus.Fields{
		"event":        "vm_copy",
		"vm_id":        event.VMID,
		"user_id":      event.UserID,
		"tenant_id":    event.TenantID,
		"path":         event.Path,
		"direction":    event.Direction,
		"bytes":        event.Bytes,
		"checksum":     event.Checksum,
		"result":       event.Result,
		"error":        event.Error,
		"started_at":   event.StartedAt.UTC().Format(time.RFC3339Nano),
		"completed_at": event.CompletedAt.UTC().Format(time.RFC3339Nano),
	}).Info("vm copy transfer audited")
}

type TenantByteRateLimiter struct {
	bytesPerSecond int64
	mu             sync.Mutex
	nextByTenant   map[string]time.Time
}

func NewTenantByteRateLimiter(bytesPerSecond int64) *TenantByteRateLimiter {
	return &TenantByteRateLimiter{
		bytesPerSecond: bytesPerSecond,
		nextByTenant:   make(map[string]time.Time),
	}
}

func (l *TenantByteRateLimiter) Wait(ctx context.Context, tenantID string, bytes int) error {
	if l == nil || l.bytesPerSecond <= 0 || bytes <= 0 {
		return nil
	}
	if tenantID == "" {
		tenantID = "default"
	}
	wait := l.reserve(tenantID, bytes)
	if wait <= 0 {
		return nil
	}
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (l *TenantByteRateLimiter) reserve(tenantID string, bytes int) time.Duration {
	l.mu.Lock()
	defer l.mu.Unlock()

	now := time.Now()
	next := l.nextByTenant[tenantID]
	if next.Before(now) {
		next = now
	}
	duration := time.Duration(int64(time.Second) * int64(bytes) / l.bytesPerSecond)
	l.nextByTenant[tenantID] = next.Add(duration)
	return next.Sub(now)
}

func readCopyMetadata(conn *websocket.Conn) (VMCopyMetadata, error) {
	messageType, frame, err := conn.ReadMessage()
	if err != nil {
		return VMCopyMetadata{}, fmt.Errorf("read metadata frame: %w", err)
	}
	if messageType != websocket.BinaryMessage {
		return VMCopyMetadata{}, errors.New("metadata frame must be a binary websocket message")
	}

	var metadata VMCopyMetadata
	frameType, err := DecodeVMIOJSONFrame(frame, &metadata)
	if err != nil {
		return VMCopyMetadata{}, err
	}
	if frameType != VMIOFrameCopyMetadata {
		return VMCopyMetadata{}, fmt.Errorf("expected metadata frame %#x, got %#x", VMIOFrameCopyMetadata, frameType)
	}
	return metadata, nil
}

func writeVMIOAck(conn *websocket.Conn, bytes int64) error {
	frame, err := EncodeVMIOJSONFrame(VMIOFrameCopyAck, VMIOAck{Bytes: bytes})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.BinaryMessage, frame)
}

func writeVMIOError(conn *websocket.Conn, code, message string) error {
	frame, err := EncodeVMIOJSONFrame(VMIOFrameCopyError, VMIOError{Code: code, Message: message})
	if err != nil {
		return err
	}
	return conn.WriteMessage(websocket.BinaryMessage, frame)
}
