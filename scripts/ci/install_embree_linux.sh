#!/usr/bin/env bash
# Download Embree's official prebuilt Linux release into /embree, inside the
# manylinux container, for the wheel build. The tarball bundles libtbb too, so
# vendoring /embree/lib64 via auditwheel captures the full runtime.
#
# Pinned version — bump deliberately and re-test. Confirm the asset name at
# https://github.com/RenderKit/embree/releases before changing.
set -euo pipefail

EMBREE_VERSION="${EMBREE_VERSION:-4.4.1}"
DEST="/embree"
ASSET="embree-${EMBREE_VERSION}.x86_64.linux.tar.gz"
URL="https://github.com/RenderKit/embree/releases/download/v${EMBREE_VERSION}/${ASSET}"

echo "Installing Embree ${EMBREE_VERSION} from ${URL}"
mkdir -p "${DEST}"
curl -fL -o /tmp/embree.tar.gz "${URL}"
# The Linux tarball has no wrapping directory — bin/, lib64/, include/ sit at
# the archive root — so extract straight into ${DEST} (no --strip-components,
# which would peel off the lib64/ prefix and scatter the .so files).
tar -xzf /tmp/embree.tar.gz -C "${DEST}"
echo "Embree installed:"
ls "${DEST}/lib64" | head
