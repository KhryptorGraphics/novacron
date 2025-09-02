#!/bin/bash

# K6 Installation Script for NovaCron Load Testing
# Supports multiple installation methods and platforms

set -euo pipefail

# Configuration
K6_VERSION="v0.47.0"
INSTALL_DIR="/usr/local/bin"
TEMP_DIR="/tmp/k6-install"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect platform
detect_platform() {
    local os
    local arch
    
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    arch=$(uname -m)
    
    case $arch in
        x86_64) arch="amd64" ;;
        arm64|aarch64) arch="arm64" ;;
        *) 
            log_error "Unsupported architecture: $arch"
            exit 1
            ;;
    esac
    
    case $os in
        linux) os="linux" ;;
        darwin) os="darwin" ;;
        *) 
            log_error "Unsupported operating system: $os"
            exit 1
            ;;
    esac
    
    echo "${os}-${arch}"
}

# Check if k6 is already installed
check_existing_installation() {
    if command -v k6 &> /dev/null; then
        local current_version
        current_version=$(k6 version 2>&1 | grep -o 'v[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)
        
        if [ "$current_version" == "$K6_VERSION" ]; then
            log_success "k6 $K6_VERSION is already installed"
            k6 version
            return 0
        else
            log_warning "k6 $current_version is installed, but $K6_VERSION is required"
            log_info "Proceeding with installation of $K6_VERSION"
        fi
    fi
    
    return 1
}

# Install k6 from GitHub releases
install_from_github() {
    local platform
    platform=$(detect_platform)
    
    log_info "Installing k6 $K6_VERSION for $platform..."
    
    # Create temp directory
    mkdir -p "$TEMP_DIR"
    cd "$TEMP_DIR"
    
    # Download k6 binary
    local download_url="https://github.com/grafana/k6/releases/download/${K6_VERSION}/k6-${K6_VERSION}-${platform}.tar.gz"
    
    log_info "Downloading from: $download_url"
    
    if curl -L -f -o "k6.tar.gz" "$download_url"; then
        log_success "Download completed"
    else
        log_error "Failed to download k6"
        return 1
    fi
    
    # Extract and install
    tar -xzf k6.tar.gz
    
    local extracted_dir
    extracted_dir=$(find . -name "k6-${K6_VERSION}-${platform}" -type d | head -1)
    
    if [ -z "$extracted_dir" ]; then
        log_error "Failed to find extracted k6 directory"
        return 1
    fi
    
    # Install k6 binary
    if [ -w "$INSTALL_DIR" ]; then
        cp "$extracted_dir/k6" "$INSTALL_DIR/"
        chmod +x "$INSTALL_DIR/k6"
        log_success "k6 installed to $INSTALL_DIR/k6"
    else
        log_info "Installing with sudo to $INSTALL_DIR"
        sudo cp "$extracted_dir/k6" "$INSTALL_DIR/"
        sudo chmod +x "$INSTALL_DIR/k6"
        log_success "k6 installed to $INSTALL_DIR/k6"
    fi
    
    # Cleanup
    cd /
    rm -rf "$TEMP_DIR"
    
    return 0
}

# Install using package manager
install_with_package_manager() {
    log_info "Attempting installation with package manager..."
    
    if command -v brew &> /dev/null; then
        log_info "Installing k6 with Homebrew..."
        brew install k6
        return $?
    elif command -v apt-get &> /dev/null; then
        log_info "Installing k6 with apt..."
        sudo gpg -k
        sudo gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D57D77C6C491D6AC1D69
        echo "deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main" | sudo tee /etc/apt/sources.list.d/k6.list
        sudo apt-get update
        sudo apt-get install k6
        return $?
    elif command -v yum &> /dev/null; then
        log_info "Installing k6 with yum..."
        sudo dnf install https://dl.k6.io/rpm/repo.rpm
        sudo dnf install k6
        return $?
    fi
    
    return 1
}

# Verify installation
verify_installation() {
    log_info "Verifying k6 installation..."
    
    if command -v k6 &> /dev/null; then
        log_success "k6 is available in PATH"
        k6 version
        
        # Test basic functionality
        log_info "Testing k6 functionality..."
        
        if k6 run --vus 1 --duration 1s - <<< 'export default function() { console.log("k6 test successful"); }' &> /dev/null; then
            log_success "k6 functionality test passed"
            return 0
        else
            log_error "k6 functionality test failed"
            return 1
        fi
    else
        log_error "k6 installation failed - not found in PATH"
        return 1
    fi
}

# Main installation function
install_k6() {
    log_info "Starting k6 installation for NovaCron load testing"
    
    # Check if already installed with correct version
    if check_existing_installation; then
        return 0
    fi
    
    # Try package manager first (faster and handles dependencies)
    if install_with_package_manager; then
        log_success "k6 installed via package manager"
    elif install_from_github; then
        log_success "k6 installed from GitHub releases"
    else
        log_error "All installation methods failed"
        exit 1
    fi
    
    # Verify installation
    if verify_installation; then
        log_success "k6 installation completed successfully"
    else
        log_error "k6 installation verification failed"
        exit 1
    fi
}

# Show usage information
show_usage() {
    cat << EOF
K6 Installation Script for NovaCron Load Testing

Usage: $0 [OPTIONS]

OPTIONS:
    --version VERSION    Install specific k6 version (default: $K6_VERSION)
    --install-dir DIR    Installation directory (default: $INSTALL_DIR)
    --force             Force reinstallation even if already installed
    --package-manager   Use package manager only (no GitHub fallback)
    --github-only       Use GitHub releases only (no package manager)
    --help              Show this help message

EXAMPLES:
    # Install latest supported version
    $0
    
    # Install specific version
    $0 --version v0.46.0
    
    # Force reinstallation
    $0 --force
    
    # Use only package manager
    $0 --package-manager

SUPPORTED PLATFORMS:
    - Linux (amd64, arm64)
    - macOS (amd64, arm64)
    - Package managers: brew, apt, yum/dnf

EOF
}

# Parse command line arguments
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --version)
                K6_VERSION="$2"
                shift 2
                ;;
            --install-dir)
                INSTALL_DIR="$2"
                shift 2
                ;;
            --force)
                FORCE_INSTALL="true"
                shift
                ;;
            --package-manager)
                PACKAGE_MANAGER_ONLY="true"
                shift
                ;;
            --github-only)
                GITHUB_ONLY="true"
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
}

# Main execution
main() {
    parse_arguments "$@"
    
    log_info "K6 Installation Configuration:"
    log_info "  Version: $K6_VERSION"
    log_info "  Install Directory: $INSTALL_DIR"
    log_info "  Platform: $(detect_platform)"
    
    # Skip existing installation check if force install
    if [ "${FORCE_INSTALL:-false}" == "true" ]; then
        log_info "Force installation requested, skipping version check"
    elif check_existing_installation; then
        exit 0
    fi
    
    # Install based on method preference
    if [ "${PACKAGE_MANAGER_ONLY:-false}" == "true" ]; then
        if install_with_package_manager; then
            log_success "Package manager installation completed"
        else
            log_error "Package manager installation failed"
            exit 1
        fi
    elif [ "${GITHUB_ONLY:-false}" == "true" ]; then
        if install_from_github; then
            log_success "GitHub installation completed"
        else
            log_error "GitHub installation failed"
            exit 1
        fi
    else
        install_k6
    fi
    
    verify_installation
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi