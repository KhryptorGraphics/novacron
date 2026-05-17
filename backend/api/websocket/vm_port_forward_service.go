package websocket

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"

	"github.com/gorilla/websocket"
)

const vmPortForwardChunkSize = 32 * 1024

type VMGuestPortDialer interface {
	DialGuestPort(ctx context.Context, vmID string, port int) (net.Conn, error)
}

type VMPortForwardService struct {
	dialer VMGuestPortDialer
}

func NewVMPortForwardService(dialer VMGuestPortDialer) *VMPortForwardService {
	return &VMPortForwardService{dialer: dialer}
}

func (s *VMPortForwardService) HandleVMPortForward(ctx context.Context, vmID string, options VMPortForwardOptions, conn *websocket.Conn) error {
	if s == nil || s.dialer == nil {
		return errors.New("vm port-forward guest dialer is required")
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	session := &vmPortForwardSession{
		ctx:         ctx,
		vmID:        vmID,
		port:        options.Port,
		conn:        conn,
		dialer:      s.dialer,
		connections: make(map[string]net.Conn),
	}
	defer session.closeAll()

	return session.run()
}

type vmPortForwardSession struct {
	ctx         context.Context
	vmID        string
	port        int
	conn        *websocket.Conn
	dialer      VMGuestPortDialer
	writeMu     sync.Mutex
	connMu      sync.Mutex
	connections map[string]net.Conn
}

func (s *vmPortForwardSession) run() error {
	for {
		messageType, frame, err := s.conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("read port-forward frame: %w", err)
		}
		if messageType != websocket.BinaryMessage {
			return s.writeError("", "invalid_frame", "port-forward frames must be binary websocket messages")
		}

		frameType, _, err := DecodeVMIOFrame(frame)
		if err != nil {
			_ = s.writeError("", "invalid_frame", err.Error())
			return err
		}

		switch frameType {
		case VMIOFramePortForwardOpen:
			if err := s.handleOpen(frame); err != nil {
				return err
			}
		case VMIOFramePortForwardData:
			if err := s.handleData(frame); err != nil {
				return err
			}
		case VMIOFramePortForwardClose:
			if err := s.handleClose(frame); err != nil {
				return err
			}
		case VMIOFramePortForwardHeartbeat:
			if err := s.writeFrame(frame); err != nil {
				return err
			}
		default:
			err := fmt.Errorf("unexpected port-forward frame type %#x", frameType)
			_ = s.writeError("", "unexpected_frame", err.Error())
			return err
		}
	}
}

func (s *vmPortForwardSession) handleOpen(frame []byte) error {
	var open VMPortForwardOpen
	frameType, err := DecodeVMIOJSONFrame(frame, &open)
	if err != nil {
		_ = s.writeError("", "invalid_open", err.Error())
		return err
	}
	if frameType != VMIOFramePortForwardOpen {
		return fmt.Errorf("expected open frame %#x, got %#x", VMIOFramePortForwardOpen, frameType)
	}
	if open.ConnectionID == "" {
		err := errors.New("connection id is required")
		_ = s.writeError("", "invalid_connection", err.Error())
		return err
	}
	if open.Port != 0 && open.Port != s.port {
		err := fmt.Errorf("open port %d does not match requested port %d", open.Port, s.port)
		_ = s.writeError(open.ConnectionID, "port_mismatch", err.Error())
		return err
	}

	guestConn, err := s.dialer.DialGuestPort(s.ctx, s.vmID, s.port)
	if err != nil {
		_ = s.writeError(open.ConnectionID, "connect_failed", err.Error())
		return err
	}

	s.connMu.Lock()
	if _, exists := s.connections[open.ConnectionID]; exists {
		s.connMu.Unlock()
		_ = guestConn.Close()
		err := fmt.Errorf("connection %q is already open", open.ConnectionID)
		_ = s.writeError(open.ConnectionID, "duplicate_connection", err.Error())
		return err
	}
	s.connections[open.ConnectionID] = guestConn
	s.connMu.Unlock()

	go s.forwardGuestToWebSocket(open.ConnectionID, guestConn)

	ack, err := EncodeVMIOJSONFrame(VMIOFramePortForwardOpen, VMPortForwardOpen{
		ConnectionID: open.ConnectionID,
		Port:         s.port,
	})
	if err != nil {
		return err
	}
	return s.writeFrame(ack)
}

func (s *vmPortForwardSession) handleData(frame []byte) error {
	connectionID, payload, err := DecodeVMPortForwardDataFrame(frame)
	if err != nil {
		_ = s.writeError("", "invalid_data", err.Error())
		return err
	}

	guestConn := s.connection(connectionID)
	if guestConn == nil {
		err := fmt.Errorf("connection %q is not open", connectionID)
		_ = s.writeError(connectionID, "unknown_connection", err.Error())
		return err
	}

	if len(payload) == 0 {
		return nil
	}
	if _, err := guestConn.Write(payload); err != nil {
		_ = s.writeError(connectionID, "guest_write_failed", err.Error())
		s.closeConnection(connectionID)
		return err
	}
	return nil
}

func (s *vmPortForwardSession) handleClose(frame []byte) error {
	var closeFrame VMPortForwardClose
	frameType, err := DecodeVMIOJSONFrame(frame, &closeFrame)
	if err != nil {
		_ = s.writeError("", "invalid_close", err.Error())
		return err
	}
	if frameType != VMIOFramePortForwardClose {
		return fmt.Errorf("expected close frame %#x, got %#x", VMIOFramePortForwardClose, frameType)
	}
	s.closeConnection(closeFrame.ConnectionID)
	return nil
}

func (s *vmPortForwardSession) forwardGuestToWebSocket(connectionID string, guestConn net.Conn) {
	buffer := make([]byte, vmPortForwardChunkSize)
	for {
		n, err := guestConn.Read(buffer)
		if n > 0 {
			frame, encodeErr := EncodeVMPortForwardDataFrame(connectionID, buffer[:n])
			if encodeErr != nil || s.writeFrame(frame) != nil {
				s.closeConnection(connectionID)
				return
			}
		}
		if err != nil {
			s.closeConnection(connectionID)
			reason := "eof"
			if !errors.Is(err, io.EOF) {
				reason = "guest_read_failed"
			}
			_ = s.writeClose(connectionID, reason)
			return
		}
	}
}

func (s *vmPortForwardSession) connection(connectionID string) net.Conn {
	s.connMu.Lock()
	defer s.connMu.Unlock()
	return s.connections[connectionID]
}

func (s *vmPortForwardSession) closeConnection(connectionID string) {
	s.connMu.Lock()
	guestConn := s.connections[connectionID]
	delete(s.connections, connectionID)
	s.connMu.Unlock()
	if guestConn != nil {
		_ = guestConn.Close()
	}
}

func (s *vmPortForwardSession) closeAll() {
	s.connMu.Lock()
	connections := s.connections
	s.connections = make(map[string]net.Conn)
	s.connMu.Unlock()
	for _, guestConn := range connections {
		_ = guestConn.Close()
	}
}

func (s *vmPortForwardSession) writeFrame(frame []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(websocket.BinaryMessage, frame)
}

func (s *vmPortForwardSession) writeError(connectionID, code, message string) error {
	frame, err := EncodeVMIOJSONFrame(VMIOFramePortForwardError, VMIOError{
		ConnectionID: connectionID,
		Code:         code,
		Message:      message,
	})
	if err != nil {
		return err
	}
	return s.writeFrame(frame)
}

func (s *vmPortForwardSession) writeClose(connectionID, reason string) error {
	frame, err := EncodeVMIOJSONFrame(VMIOFramePortForwardClose, VMPortForwardClose{
		ConnectionID: connectionID,
		Reason:       reason,
	})
	if err != nil {
		return err
	}
	return s.writeFrame(frame)
}
