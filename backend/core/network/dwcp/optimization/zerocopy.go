package optimization

import (
	"fmt"
	"net"
	"os"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
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
	return sendFileToTCP(conn, file, offset, count)
}

// Splice performs zero-copy data transfer between sockets
// Note: splice() is platform-specific (Linux) and may not be available on all systems
func (zcb *ZeroCopyBuffer) Splice(src, dst *net.TCPConn, maxBytes int) (int64, error) {
	return spliceTCPToTCP(src, dst, maxBytes)
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
	return spliceTCPToFile(zcr.conn, file, maxBytes)
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
	return sendFileToTCP(zcw.conn, file, offset, count)
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
	if len(data) == 0 {
		return 0, nil
	}

	rawConn, err := zcs.conn.SyscallConn()
	if err != nil {
		return 0, err
	}

	var written int
	var sendErr error
	err = rawConn.Write(func(fd uintptr) bool {
		written, sendErr = unix.SendmsgN(int(fd), data, nil, nil, unix.MSG_ZEROCOPY)
		return sendErr != unix.EAGAIN && sendErr != unix.EINTR
	})
	if err != nil {
		return written, err
	}
	return written, sendErr
}

func sendFileToTCP(conn *net.TCPConn, file *os.File, offset, count int64) (int64, error) {
	if count < 0 {
		return 0, fmt.Errorf("count must be >= 0")
	}
	if count == 0 {
		return 0, nil
	}

	outFile, err := conn.File()
	if err != nil {
		return 0, err
	}
	defer outFile.Close()

	inFD := int(file.Fd())
	outFD := int(outFile.Fd())
	currentOffset := offset
	var total int64
	for total < count {
		chunk := count - total
		if chunk > int64(^uint(0)>>1) {
			chunk = int64(^uint(0) >> 1)
		}
		n, err := unix.Sendfile(outFD, inFD, &currentOffset, int(chunk))
		if n > 0 {
			total += int64(n)
		}
		if err == unix.EINTR || err == unix.EAGAIN {
			continue
		}
		if err != nil {
			return total, err
		}
		if n == 0 {
			break
		}
	}
	return total, nil
}

func spliceTCPToTCP(src, dst *net.TCPConn, maxBytes int) (int64, error) {
	if maxBytes < 0 {
		return 0, fmt.Errorf("maxBytes must be >= 0")
	}
	if maxBytes == 0 {
		return 0, nil
	}

	srcFile, err := src.File()
	if err != nil {
		return 0, err
	}
	defer srcFile.Close()
	dstFile, err := dst.File()
	if err != nil {
		return 0, err
	}
	defer dstFile.Close()

	return spliceFDToFD(int(srcFile.Fd()), int(dstFile.Fd()), int64(maxBytes))
}

func spliceTCPToFile(src *net.TCPConn, file *os.File, maxBytes int64) (int64, error) {
	if maxBytes < 0 {
		return 0, fmt.Errorf("maxBytes must be >= 0")
	}
	if maxBytes == 0 {
		return 0, nil
	}

	srcFile, err := src.File()
	if err != nil {
		return 0, err
	}
	defer srcFile.Close()

	return spliceFDToFD(int(srcFile.Fd()), int(file.Fd()), maxBytes)
}

func spliceFDToFD(srcFD, dstFD int, maxBytes int64) (int64, error) {
	pipeFDs := []int{0, 0}
	if err := unix.Pipe2(pipeFDs, unix.O_CLOEXEC); err != nil {
		return 0, err
	}
	defer unix.Close(pipeFDs[0])
	defer unix.Close(pipeFDs[1])

	var total int64
	for total < maxBytes {
		chunk := maxBytes - total
		if chunk > 1<<20 {
			chunk = 1 << 20
		}

		n, err := unix.Splice(srcFD, nil, pipeFDs[1], nil, int(chunk), unix.SPLICE_F_MOVE)
		if n > 0 {
			written, writeErr := unix.Splice(pipeFDs[0], nil, dstFD, nil, int(n), unix.SPLICE_F_MOVE)
			total += written
			if writeErr == unix.EINTR || writeErr == unix.EAGAIN {
				continue
			}
			if writeErr != nil {
				return total, writeErr
			}
			if written < n {
				break
			}
		}
		if err == unix.EINTR || err == unix.EAGAIN {
			continue
		}
		if err != nil {
			return total, err
		}
		if n == 0 {
			break
		}
	}
	return total, nil
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
