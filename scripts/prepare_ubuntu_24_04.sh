#!/bin/bash
# Script to download and prepare Ubuntu 24.04 LTS image for NovaCron

set -e

# Configuration
UBUNTU_VERSION="24.04"
UBUNTU_CODENAME="noble"
IMAGE_NAME="noble-server-cloudimg-amd64.img"
IMAGE_URL="https://cloud-images.ubuntu.com/${UBUNTU_CODENAME}/current/${IMAGE_NAME}"
STORAGE_DIR="/var/lib/novacron/images"
QCOW2_NAME="ubuntu-${UBUNTU_VERSION}-server-cloudimg-amd64.qcow2"
FINAL_PATH="${STORAGE_DIR}/${QCOW2_NAME}"

# Create storage directory if it doesn't exist
mkdir -p "${STORAGE_DIR}"

echo "Preparing Ubuntu ${UBUNTU_VERSION} LTS (${UBUNTU_CODENAME}) image for NovaCron..."

# Check if image already exists
if [ -f "${FINAL_PATH}" ]; then
    echo "Image already exists at ${FINAL_PATH}"
    echo "To force a re-download, delete the existing image first."
    exit 0
fi

# Download the image
echo "Downloading Ubuntu ${UBUNTU_VERSION} cloud image..."
wget -O "${STORAGE_DIR}/${IMAGE_NAME}" "${IMAGE_URL}"

# Convert to qcow2 format if needed
if [[ "${IMAGE_NAME}" != *qcow2 ]]; then
    echo "Converting image to qcow2 format..."
    qemu-img convert -f qcow2 -O qcow2 "${STORAGE_DIR}/${IMAGE_NAME}" "${FINAL_PATH}"
    rm "${STORAGE_DIR}/${IMAGE_NAME}"
else
    mv "${STORAGE_DIR}/${IMAGE_NAME}" "${FINAL_PATH}"
fi

# Resize the image to 20GB
echo "Resizing image to 20GB..."
qemu-img resize "${FINAL_PATH}" 20G

# Set appropriate permissions
echo "Setting permissions..."
chmod 644 "${FINAL_PATH}"

echo "Image preparation complete!"
echo "Ubuntu ${UBUNTU_VERSION} LTS image is available at: ${FINAL_PATH}"
echo ""
echo "To use this image with NovaCron, update your configuration to point to this image path."
echo "Example configuration:"
echo "  Image path: ${FINAL_PATH}"
echo "  OS: Ubuntu"
echo "  Version: ${UBUNTU_VERSION}"
echo "  Codename: ${UBUNTU_CODENAME^}"
echo ""
echo "For KVM VMs, you can use this image as a backing file for new VMs."
echo "For container-based VMs, you can use this as a rootfs source."

exit 0
