package vm

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"time"
)

const defaultQGACommandTimeout = 30 * time.Second

type QGAClient struct {
	socketPath string
	timeout    time.Duration
}

type qgaRequest struct {
	Execute   string                 `json:"execute"`
	Arguments map[string]interface{} `json:"arguments,omitempty"`
}

type qgaResponse struct {
	Return json.RawMessage `json:"return,omitempty"`
	Error  *qgaError       `json:"error,omitempty"`
}

type qgaError struct {
	Class string `json:"class"`
	Desc  string `json:"desc"`
}

type qgaFileReadResult struct {
	Count int    `json:"count"`
	Buf64 string `json:"buf-b64"`
	EOF   bool   `json:"eof"`
}

type qgaFileWriteResult struct {
	Count int  `json:"count"`
	EOF   bool `json:"eof"`
}

func NewQGAClient(socketPath string) *QGAClient {
	return &QGAClient{
		socketPath: socketPath,
		timeout:    defaultQGACommandTimeout,
	}
}

func (c *QGAClient) Execute(ctx context.Context, command string, arguments map[string]interface{}, result interface{}) error {
	if c == nil || c.socketPath == "" {
		return errors.New("qga socket path is required")
	}
	if command == "" {
		return errors.New("qga command is required")
	}

	dialer := net.Dialer{Timeout: c.timeout}
	conn, err := dialer.DialContext(ctx, "unix", c.socketPath)
	if err != nil {
		return fmt.Errorf("connect qga socket: %w", err)
	}
	defer conn.Close()

	deadline, ok := ctx.Deadline()
	if !ok {
		deadline = time.Now().Add(c.timeout)
	}
	_ = conn.SetDeadline(deadline)

	request := qgaRequest{Execute: command, Arguments: arguments}
	if err := json.NewEncoder(conn).Encode(request); err != nil {
		return fmt.Errorf("send qga command: %w", err)
	}

	var response qgaResponse
	if err := json.NewDecoder(bufio.NewReader(conn)).Decode(&response); err != nil {
		return fmt.Errorf("read qga response: %w", err)
	}
	if response.Error != nil {
		return fmt.Errorf("qga command %s failed: %s: %s", command, response.Error.Class, response.Error.Desc)
	}
	if result == nil || len(response.Return) == 0 {
		return nil
	}
	if err := json.Unmarshal(response.Return, result); err != nil {
		return fmt.Errorf("decode qga response: %w", err)
	}
	return nil
}

func (c *QGAClient) FileOpen(ctx context.Context, path, mode string) (int, error) {
	args := map[string]interface{}{"path": path}
	if mode != "" {
		args["mode"] = mode
	}

	var handle int
	if err := c.Execute(ctx, "guest-file-open", args, &handle); err != nil {
		return 0, err
	}
	return handle, nil
}

func (c *QGAClient) FileClose(ctx context.Context, handle int) error {
	return c.Execute(ctx, "guest-file-close", map[string]interface{}{
		"handle": handle,
	}, nil)
}

func (c *QGAClient) FileRead(ctx context.Context, handle, count int) ([]byte, bool, error) {
	args := map[string]interface{}{"handle": handle}
	if count > 0 {
		args["count"] = count
	}

	var result qgaFileReadResult
	if err := c.Execute(ctx, "guest-file-read", args, &result); err != nil {
		return nil, false, err
	}

	decoded, err := base64.StdEncoding.DecodeString(result.Buf64)
	if err != nil {
		return nil, false, fmt.Errorf("decode qga file read payload: %w", err)
	}
	return decoded, result.EOF, nil
}

func (c *QGAClient) FileWrite(ctx context.Context, handle int, payload []byte) (int, bool, error) {
	var result qgaFileWriteResult
	if err := c.Execute(ctx, "guest-file-write", map[string]interface{}{
		"handle":  handle,
		"buf-b64": base64.StdEncoding.EncodeToString(payload),
		"count":   len(payload),
	}, &result); err != nil {
		return 0, false, err
	}
	return result.Count, result.EOF, nil
}

func (c *QGAClient) FileFlush(ctx context.Context, handle int) error {
	return c.Execute(ctx, "guest-file-flush", map[string]interface{}{
		"handle": handle,
	}, nil)
}
