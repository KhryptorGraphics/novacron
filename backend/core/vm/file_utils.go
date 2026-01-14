package vm

import (
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
)

// copyFileSimple copies a file from src to dst using simple read/write
func copyFileSimple(src, dst string) error {
	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	// Read source file
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}

	// Write destination file
	return os.WriteFile(dst, data, 0644)
}

// copyFileWithProgress copies a file from src to dst using streaming I/O and returns bytes copied
func copyFileWithProgress(src, dst string) (int64, error) {
	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return 0, err
	}

	sourceFile, err := os.Open(src)
	if err != nil {
		return 0, err
	}
	defer sourceFile.Close()

	destFile, err := os.Create(dst)
	if err != nil {
		return 0, err
	}
	defer destFile.Close()

	return io.Copy(destFile, sourceFile)
}

// copyQemuImage copies a QEMU image file using qemu-img convert
func copyQemuImage(src, dst string) error {
	// Create destination directory if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	// Use qemu-img convert for QCOW2 files to ensure proper copying
	cmd := exec.Command("qemu-img", "convert", "-f", "qcow2", "-O", "qcow2", src, dst)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to copy QEMU image: %w, output: %s", err, string(output))
	}
	return nil
}
