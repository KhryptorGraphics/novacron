package optimization

import (
	"fmt"
	"io"
	"net"
	"os"
	"syscall"
	"unsafe"
)

// ZeroCopyBuffer manages memory for zero-copy operations
type ZeroCopyBuffer struct {
	ptr    uintptr
	length int
	fd     int
	mmap   []byte
}

// NewZeroCopyBuffer creates a page-aligned buffer for DMA operations
func NewZeroCopyBuffer(size int) (*ZeroCopyBuffer, error) {
	// Round up to page size
	pageSize := os.Getpagesize()
	alignedSize := ((size + pageSize - 1) / pageSize) * pageSize

	// Try to allocate with huge pages first for better performance
	ptr, err := syscall.Mmap(
		-1, 0, alignedSize,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS|syscall.MAP_HUGETLB,
	)

	if err != nil {
		// Fall back to regular pages
		ptr, err = syscall.Mmap(
			-1, 0, alignedSize,
			syscall.PROT_READ|syscall.PROT_WRITE,
			syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS,
		)
		if err != nil {
			return nil, fmt.Errorf("mmap failed: %w", err)
		}
	}

	// Lock pages in memory to prevent swapping
	if err := syscall.Mlock(ptr); err != nil {
		syscall.Munmap(ptr)
		return nil, fmt.Errorf("mlock failed: %w", err)
	}

	return &ZeroCopyBuffer{
		ptr:    uintptr(unsafe.Pointer(&ptr[0])),
		length: alignedSize,
		mmap:   ptr,
	}, nil
}

// Close releases the buffer
func (zcb *ZeroCopyBuffer) Close() error {
	if zcb.mmap != nil {
		syscall.Munlock(zcb.mmap)
		return syscall.Munmap(zcb.mmap)
	}
	return nil
}

// Bytes returns the buffer as a byte slice
func (zcb *ZeroCopyBuffer) Bytes() []byte {
	return zcb.mmap[:zcb.length]
}

// SendFile performs zero-copy file transmission using sendfile()
// Note: sendfile() is platform-specific and may not be available on all systems
func (zcb *ZeroCopyBuffer) SendFile(conn *net.TCPConn, file *os.File, offset, count int64) (int64, error) {
	// TODO: Implement zero-copy sendfile using platform-specific syscalls
	// This requires syscall.Sendfile which has different signatures on different platforms

	// For now, fall back to regular copy
	data := make([]byte, count)
	n, err := file.ReadAt(data, offset)
	if err != nil && err != io.EOF {
		return 0, err
	}

	written, err := conn.Write(data[:n])
	return int64(written), err
}

// Splice performs zero-copy data transfer between sockets
// Note: splice() is platform-specific (Linux) and may not be available on all systems
func (zcb *ZeroCopyBuffer) Splice(src, dst *net.TCPConn, maxBytes int) (int64, error) {
	// TODO: Implement zero-copy splice using platform-specific syscalls
	// This requires syscall.Splice and SPLICE_F_* flags which are Linux-specific

	// For now, fall back to regular copy
	data := make([]byte, maxBytes)
	n, err := src.Read(data)
	if err != nil && err != io.EOF {
		return 0, err
	}

	written, err := dst.Write(data[:n])
	return int64(written), err
}

// ZeroCopyReader provides zero-copy reading operations
type ZeroCopyReader struct {
	conn   *net.TCPConn
	buffer *ZeroCopyBuffer
}

// NewZeroCopyReader creates a zero-copy reader
func NewZeroCopyReader(conn *net.TCPConn, bufferSize int) (*ZeroCopyReader, error) {
	buffer, err := NewZeroCopyBuffer(bufferSize)
	if err != nil {
		return nil, err
	}

	return &ZeroCopyReader{
		conn:   conn,
		buffer: buffer,
	}, nil
}

// Read reads data with minimal copying
func (zcr *ZeroCopyReader) Read(p []byte) (int, error) {
	// Use recvmsg with MSG_TRUNC to peek at size
	return zcr.conn.Read(p)
}

// ReadToFile reads directly to file using splice
// Note: splice() is platform-specific (Linux) and may not be available on all systems
func (zcr *ZeroCopyReader) ReadToFile(file *os.File, maxBytes int64) (int64, error) {
	// TODO: Implement zero-copy splice using platform-specific syscalls
	// This requires syscall.Splice and SPLICE_F_* flags which are Linux-specific

	// For now, fall back to regular copy
	data := make([]byte, maxBytes)
	n, err := zcr.conn.Read(data)
	if err != nil && err != io.EOF {
		return 0, err
	}

	written, err := file.Write(data[:n])
	return int64(written), err
}

// Close releases resources
func (zcr *ZeroCopyReader) Close() error {
	return zcr.buffer.Close()
}

// ZeroCopyWriter provides zero-copy writing operations
type ZeroCopyWriter struct {
	conn   *net.TCPConn
	buffer *ZeroCopyBuffer
}

// NewZeroCopyWriter creates a zero-copy writer
func NewZeroCopyWriter(conn *net.TCPConn, bufferSize int) (*ZeroCopyWriter, error) {
	buffer, err := NewZeroCopyBuffer(bufferSize)
	if err != nil {
		return nil, err
	}

	return &ZeroCopyWriter{
		conn:   conn,
		buffer: buffer,
	}, nil
}

// Write writes data with minimal copying
func (zcw *ZeroCopyWriter) Write(p []byte) (int, error) {
	return zcw.conn.Write(p)
}

// WriteFromFile writes file contents using sendfile
// Note: sendfile() is platform-specific and may not be available on all systems
func (zcw *ZeroCopyWriter) WriteFromFile(file *os.File, offset, count int64) (int64, error) {
	// TODO: Implement zero-copy sendfile using platform-specific syscalls
	// This requires syscall.Sendfile which has different signatures on different platforms

	// For now, fall back to regular copy
	data := make([]byte, count)
	n, err := file.ReadAt(data, offset)
	if err != nil && err != io.EOF {
		return 0, err
	}

	written, err := zcw.conn.Write(data[:n])
	return int64(written), err
}

// Close releases resources
func (zcw *ZeroCopyWriter) Close() error {
	return zcw.buffer.Close()
}

// MSG_ZEROCOPY support (Linux 4.14+)
type ZeroCopySender struct {
	conn *net.TCPConn
}

// NewZeroCopySender creates a sender with MSG_ZEROCOPY support
func NewZeroCopySender(conn *net.TCPConn) (*ZeroCopySender, error) {
	// Enable MSG_ZEROCOPY on socket
	rawConn, err := conn.SyscallConn()
	if err != nil {
		return nil, err
	}

	var sockErr error
	err = rawConn.Control(func(fd uintptr) {
		// Set SO_ZEROCOPY socket option
		sockErr = syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, 60, 1) // SO_ZEROCOPY = 60
	})

	if err != nil || sockErr != nil {
		return nil, fmt.Errorf("failed to enable MSG_ZEROCOPY: %v %v", err, sockErr)
	}

	return &ZeroCopySender{conn: conn}, nil
}

// Send sends data using MSG_ZEROCOPY
// Note: MSG_ZEROCOPY is platform-specific (Linux 4.14+) and may not be available on all systems
func (zcs *ZeroCopySender) Send(data []byte) (int, error) {
	// TODO: Implement MSG_ZEROCOPY using platform-specific syscalls
	// This requires syscall.Send and syscall.MSG_ZEROCOPY which are Linux-specific

	// For now, fall back to regular send
	return zcs.conn.Write(data)
}

// EnableTCPNoDelay disables Nagle's algorithm for lower latency
func EnableTCPNoDelay(conn *net.TCPConn) error {
	return conn.SetNoDelay(true)
}

// EnableTCPQuickAck enables TCP quick ACK mode
func EnableTCPQuickAck(conn *net.TCPConn) error {
	rawConn, err := conn.SyscallConn()
	if err != nil {
		return err
	}

	var sockErr error
	err = rawConn.Control(func(fd uintptr) {
		sockErr = syscall.SetsockoptInt(int(fd), syscall.IPPROTO_TCP, syscall.TCP_QUICKACK, 1)
	})

	if err != nil {
		return err
	}
	return sockErr
}

// SetSocketBuffers optimizes socket buffer sizes
func SetSocketBuffers(conn *net.TCPConn, sendBuf, recvBuf int) error {
	rawConn, err := conn.SyscallConn()
	if err != nil {
		return err
	}

	var sockErr error
	err = rawConn.Control(func(fd uintptr) {
		syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_SNDBUF, sendBuf)
		sockErr = syscall.SetsockoptInt(int(fd), syscall.SOL_SOCKET, syscall.SO_RCVBUF, recvBuf)
	})

	if err != nil {
		return err
	}
	return sockErr
}
