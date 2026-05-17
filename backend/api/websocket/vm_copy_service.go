package websocket

import (
	"context"
	"errors"
	"fmt"

	"github.com/gorilla/websocket"
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

type QGAVMCopyService struct {
	resolver VMGuestFileClientResolver
}

func NewQGAVMCopyService(resolver VMGuestFileClientResolver) *QGAVMCopyService {
	return &QGAVMCopyService{resolver: resolver}
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
		return s.handleUpload(ctx, options, client, conn)
	case "download":
		return s.handleDownload(ctx, options, client, conn)
	default:
		return fmt.Errorf("unsupported vm copy direction %q", options.Direction)
	}
}

func (s *QGAVMCopyService) handleUpload(ctx context.Context, options VMCopyOptions, client VMGuestFileClient, conn *websocket.Conn) error {
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

	mode := "wb"
	handle, err := client.FileOpen(ctx, options.Path, mode)
	if err != nil {
		_ = writeVMIOError(conn, "open_failed", err.Error())
		return err
	}
	defer client.FileClose(ctx, handle)

	if err := writeVMIOAck(conn, 0); err != nil {
		return err
	}

	var writtenTotal int64
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
			written, _, err := client.FileWrite(ctx, handle, payload)
			if err != nil {
				_ = writeVMIOError(conn, "write_failed", err.Error())
				return err
			}
			writtenTotal += int64(written)
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
			if err := client.FileFlush(ctx, handle); err != nil {
				_ = writeVMIOError(conn, "flush_failed", err.Error())
				return err
			}
			return writeVMIOAck(conn, writtenTotal)
		default:
			err := fmt.Errorf("unexpected upload frame type %#x", frameType)
			_ = writeVMIOError(conn, "unexpected_frame", err.Error())
			return err
		}
	}
}

func (s *QGAVMCopyService) handleDownload(ctx context.Context, options VMCopyOptions, client VMGuestFileClient, conn *websocket.Conn) error {
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

	var sentTotal int64
	for {
		chunk, eof, err := client.FileRead(ctx, handle, vmCopyChunkSize)
		if err != nil {
			_ = writeVMIOError(conn, "read_failed", err.Error())
			return err
		}
		if len(chunk) > 0 {
			if err := conn.WriteMessage(websocket.BinaryMessage, EncodeVMIODataFrame(VMIOFrameCopyData, chunk)); err != nil {
				return fmt.Errorf("write download data: %w", err)
			}
			sentTotal += int64(len(chunk))
		}
		if eof {
			eofFrame, err := EncodeVMIOJSONFrame(VMIOFrameCopyEOF, VMCopyEOF{Bytes: sentTotal})
			if err != nil {
				return err
			}
			return conn.WriteMessage(websocket.BinaryMessage, eofFrame)
		}
	}
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
