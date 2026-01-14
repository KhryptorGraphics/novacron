package network

import (
	"encoding/binary"
	"fmt"
	"io"
	"time"
)

// Message types
const (
	// Control messages
	TypeHandshake         = 0x01
	TypeHandshakeResponse = 0x02
	TypePing              = 0x03
	TypePong              = 0x04
	TypeDisconnect        = 0x05

	// Node communication
	TypeNodeState     = 0x10
	TypeNodeStateReq  = 0x11
	TypeNodeStateResp = 0x12
	TypeResourceState = 0x13
	TypeTaskDistribute = 0x14

	// VM operations
	TypeVMCreate     = 0x20
	TypeVMCreateResp = 0x21
	TypeVMStart      = 0x22
	TypeVMStartResp  = 0x23
	TypeVMStop       = 0x24
	TypeVMStopResp   = 0x25
	TypeVMDelete     = 0x26
	TypeVMDeleteResp = 0x27
	TypeVMState      = 0x28
	TypeVMMetrics    = 0x29

	// Data transfer
	TypeDataStream      = 0x30
	TypeDataStreamAck   = 0x31
	TypeDataStreamClose = 0x32
	TypeBulkTransfer    = 0x33
	TypeBulkTransferAck = 0x34

	// Migration
	TypeMigrationInit   = 0x40
	TypeMigrationData   = 0x41
	TypeMigrationFinish = 0x42
	TypeMigrationAbort  = 0x43
)

// Protocol version
const (
	ProtocolVersion = 1
)

// Header flags
const (
	FlagCompressed = 1 << 0
	FlagEncrypted  = 1 << 1
	FlagUrgent     = 1 << 2
	FlagReliable   = 1 << 3
	FlagAck        = 1 << 4
)

// Message header structure
// Fixed size: 16 bytes
type MessageHeader struct {
	Version       uint8  // Protocol version
	Type          uint8  // Message type
	Flags         uint16 // Flags for compression, encryption, etc.
	SequenceID    uint32 // Message sequence ID for reliability
	PayloadLength uint32 // Length of the payload
	Timestamp     int64  // Unix timestamp in milliseconds
}

// Message represents a protocol message
type Message struct {
	Header  MessageHeader
	Payload []byte
}

// SerializeHeader serializes the message header to binary
func SerializeHeader(header MessageHeader) []byte {
	buf := make([]byte, 16)
	buf[0] = header.Version
	buf[1] = header.Type
	binary.BigEndian.PutUint16(buf[2:4], header.Flags)
	binary.BigEndian.PutUint32(buf[4:8], header.SequenceID)
	binary.BigEndian.PutUint32(buf[8:12], header.PayloadLength)
	binary.BigEndian.PutUint64(buf[12:], uint64(header.Timestamp))
	return buf
}

// DeserializeHeader deserializes a message header from binary
func DeserializeHeader(data []byte) (MessageHeader, error) {
	if len(data) < 16 {
		return MessageHeader{}, fmt.Errorf("header data too short: %d bytes", len(data))
	}

	return MessageHeader{
		Version:       data[0],
		Type:          data[1],
		Flags:         binary.BigEndian.Uint16(data[2:4]),
		SequenceID:    binary.BigEndian.Uint32(data[4:8]),
		PayloadLength: binary.BigEndian.Uint32(data[8:12]),
		Timestamp:     int64(binary.BigEndian.Uint64(data[12:])),
	}, nil
}

// Serialize serializes a message to binary
func (m *Message) Serialize() []byte {
	header := SerializeHeader(m.Header)
	result := make([]byte, len(header)+len(m.Payload))
	copy(result, header)
	copy(result[len(header):], m.Payload)
	return result
}

// Deserialize deserializes a message from binary
func Deserialize(data []byte) (*Message, error) {
	if len(data) < 16 {
		return nil, fmt.Errorf("data too short for message: %d bytes", len(data))
	}

	header, err := DeserializeHeader(data[:16])
	if err != nil {
		return nil, err
	}

	if uint32(len(data)-16) != header.PayloadLength {
		return nil, fmt.Errorf("payload length mismatch: expected %d, got %d", header.PayloadLength, len(data)-16)
	}

	return &Message{
		Header:  header,
		Payload: data[16:],
	}, nil
}

// NewMessage creates a new message
func NewMessage(msgType uint8, payload []byte, flags uint16, sequenceID uint32) *Message {
	return &Message{
		Header: MessageHeader{
			Version:       ProtocolVersion,
			Type:          msgType,
			Flags:         flags,
			SequenceID:    sequenceID,
			PayloadLength: uint32(len(payload)),
			Timestamp:     time.Now().UnixMilli(),
		},
		Payload: payload,
	}
}

// ReadMessage reads a message from a reader
func ReadMessage(r io.Reader) (*Message, error) {
	// Read header
	headerBuf := make([]byte, 16)
	_, err := io.ReadFull(r, headerBuf)
	if err != nil {
		return nil, err
	}

	header, err := DeserializeHeader(headerBuf)
	if err != nil {
		return nil, err
	}

	// Read payload
	payload := make([]byte, header.PayloadLength)
	_, err = io.ReadFull(r, payload)
	if err != nil {
		return nil, err
	}

	return &Message{
		Header:  header,
		Payload: payload,
	}, nil
}

// WriteMessage writes a message to a writer
func WriteMessage(w io.Writer, msg *Message) error {
	data := msg.Serialize()
	_, err := w.Write(data)
	return err
}

// IsFlag checks if a flag is set
func (h *MessageHeader) IsFlag(flag uint16) bool {
	return (h.Flags & flag) != 0
}

// SetFlag sets a flag
func (h *MessageHeader) SetFlag(flag uint16) {
	h.Flags |= flag
}

// ClearFlag clears a flag
func (h *MessageHeader) ClearFlag(flag uint16) {
	h.Flags &= ^flag
}
