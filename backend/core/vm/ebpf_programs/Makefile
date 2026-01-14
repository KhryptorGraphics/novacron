# Makefile for building eBPF programs for VM migration optimization

CLANG ?= clang
LLVM_STRIP ?= llvm-strip
BPFTOOL ?= bpftool

# Architecture detection
ARCH := $(shell uname -m | sed 's/x86_64/x86/' | sed 's/aarch64/arm64/')

# Kernel headers
KERNEL_VERSION ?= $(shell uname -r)
LINUX_HEADERS ?= /usr/src/linux-headers-$(KERNEL_VERSION)
LIBBPF_HEADERS ?= /usr/include

# Compiler flags
INCLUDES := -I$(LINUX_HEADERS)/include \
            -I$(LINUX_HEADERS)/include/uapi \
            -I$(LINUX_HEADERS)/arch/$(ARCH)/include \
            -I$(LINUX_HEADERS)/arch/$(ARCH)/include/generated \
            -I$(LIBBPF_HEADERS)

CLANG_FLAGS := -g -O2 -target bpf -D__TARGET_ARCH_$(ARCH) $(INCLUDES)

# Source and object files
BPF_SOURCES := page_tracker.bpf.c
BPF_OBJECTS := $(BPF_SOURCES:.c=.o)

.PHONY: all clean install check

all: $(BPF_OBJECTS)

# Build eBPF object files
%.bpf.o: %.c
	@echo "Building eBPF program: $<"
	$(CLANG) $(CLANG_FLAGS) -c $< -o $@
	$(LLVM_STRIP) -g $@
	@echo "Successfully built: $@"

# Check if required tools are available
check:
	@echo "Checking build environment..."
	@which $(CLANG) > /dev/null || (echo "Error: clang not found. Install with: apt-get install clang" && exit 1)
	@which $(LLVM_STRIP) > /dev/null || (echo "Error: llvm-strip not found. Install with: apt-get install llvm" && exit 1)
	@test -d $(LINUX_HEADERS) || (echo "Error: Kernel headers not found at $(LINUX_HEADERS). Install with: apt-get install linux-headers-$(KERNEL_VERSION)" && exit 1)
	@echo "Build environment OK"

# Install object files (copy to parent directory for embedding)
install: all
	@echo "Installing eBPF objects..."
	@cp $(BPF_OBJECTS) ../
	@echo "Installed: $(BPF_OBJECTS)"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -f *.o ../*.bpf.o
	@echo "Clean complete"

# Verify built objects
verify: all
	@echo "Verifying eBPF objects..."
	@for obj in $(BPF_OBJECTS); do \
		echo "Verifying $$obj..."; \
		file $$obj | grep -q "eBPF" || (echo "Error: $$obj is not a valid eBPF object" && exit 1); \
	done
	@echo "All objects verified successfully"

# Display build information
info:
	@echo "eBPF Build Configuration:"
	@echo "  Architecture: $(ARCH)"
	@echo "  Kernel Version: $(KERNEL_VERSION)"
	@echo "  Kernel Headers: $(LINUX_HEADERS)"
	@echo "  Clang: $(shell which $(CLANG))"
	@echo "  LLVM Strip: $(shell which $(LLVM_STRIP))"
	@echo "  Sources: $(BPF_SOURCES)"
	@echo "  Objects: $(BPF_OBJECTS)"
