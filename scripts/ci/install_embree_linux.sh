#!/usr/bin/env bash

# Download Embree's official prebuilt Linux release into /embree inside the
# manylinux container for the wheel build

set -euo pipefail

EMBREE_VERSION="${EMBREE_VERSION:-4.4.1}"
DEST="/embree"
ASSET="embree-${EMBREE_VERSION}.x86_64.linux.tar.gz"
URL="https://github.com/RenderKit/embree/releases/download/v${EMBREE_VERSION}/${ASSET}"

echo "Installing Embree ${EMBREE_VERSION} from ${URL}"
mkdir -p "${DEST}"
curl -fL -o /tmp/embree.tar.gz "${URL}"
tar -xzf /tmp/embree.tar.gz -C "${DEST}"
echo "Embree installed:"
ls "${DEST}/lib64" | head
