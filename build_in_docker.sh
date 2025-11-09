#!/bin/bash
# Build NetCDF Driver in Docker Container
# This bypasses macOS Bazel issues

set -e

echo "=== Building NetCDF Driver in Docker ==="
echo

# Step 1: Build Docker image
echo "Step 1: Building Docker image..."
docker build -f Dockerfile.netcdf -t netcdf-driver . || {
    echo "⚠ Docker build failed, trying to use existing image..."
}

# Step 2: Run compilation in container
echo
echo "Step 2: Compiling driver in Linux container..."
docker run --rm -v "$(pwd)/ts-netcdf:/workspace/ts-netcdf" netcdf-driver bash -c "
    cd /workspace/ts-netcdf
    echo '=== Bazel Build in Linux Container ==='
    bazel build //tensorstore/driver/netcdf:netcdf_driver_full 2>&1 | tee /tmp/build.log

    if [ \$? -eq 0 ]; then
        echo
        echo '✓✓✓ BUILD SUCCESSFUL ✓✓✓'
        echo
        echo 'Build artifacts:'
        find bazel-bin/tensorstore/driver/netcdf -name '*.a' -o -name '*.so' 2>/dev/null || echo 'Looking for artifacts...'
    else
        echo
        echo 'Build had errors. Checking for specific issues...'
        grep -A3 'error:' /tmp/build.log | head -30 || echo 'See full log above'
        exit 1
    fi
"

echo
echo "=== Build Complete ==="
