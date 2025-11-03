#!/bin/bash

# Build script for NetCDF write examples
# This script compiles all NetCDF write examples and tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}NetCDF Write Examples Build Script${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Check if netcdf is installed
if ! command -v nc-config &> /dev/null; then
    echo -e "${RED}Error: NetCDF library not found${NC}"
    echo "Please install NetCDF:"
    echo "  macOS: brew install netcdf"
    echo "  Ubuntu: sudo apt-get install libnetcdf-dev"
    exit 1
fi

# Get NetCDF paths
NETCDF_PREFIX=$(nc-config --prefix)
NETCDF_INCLUDE="$NETCDF_PREFIX/include"
NETCDF_LIB="$NETCDF_PREFIX/lib"

echo -e "${YELLOW}NetCDF found at: $NETCDF_PREFIX${NC}"
echo ""

# Common compiler flags
CXXFLAGS="-std=c++17 -I. -I./ts-netcdf -I$NETCDF_INCLUDE"
LDFLAGS="-L$NETCDF_LIB -lnetcdf"

# Create output directory
mkdir -p build

# Build comprehensive write test
echo -e "${YELLOW}Building comprehensive_write_test...${NC}"
g++ $CXXFLAGS \
    experiments/netcdf_week10/comprehensive_write_test.cc \
    ts-netcdf/tensorstore/driver/netcdf/minidriver.cc \
    $LDFLAGS \
    -o build/comprehensive_write_test

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ comprehensive_write_test built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build comprehensive_write_test${NC}"
    exit 1
fi

# Build write demo
echo -e "${YELLOW}Building write_demo...${NC}"
g++ $CXXFLAGS \
    experiments/netcdf_week10/write_demo.cc \
    experiments/netcdf_week10/ts_netcdf_io.cc \
    $LDFLAGS \
    -o build/write_demo

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ write_demo built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build write_demo${NC}"
    exit 1
fi

# Build read demo
echo -e "${YELLOW}Building read_demo...${NC}"
g++ $CXXFLAGS \
    experiments/netcdf_week10/read_demo.cc \
    experiments/netcdf_week10/ts_netcdf_io.cc \
    $LDFLAGS \
    -o build/read_demo

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ read_demo built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build read_demo${NC}"
    exit 1
fi

# Build roundtrip test
echo -e "${YELLOW}Building roundtrip_test...${NC}"
g++ $CXXFLAGS \
    experiments/netcdf_week10/roundtrip_test.cc \
    experiments/netcdf_week10/ts_netcdf_io.cc \
    $LDFLAGS \
    -o build/roundtrip_test

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ roundtrip_test built successfully${NC}"
else
    echo -e "${RED}✗ Failed to build roundtrip_test${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}All examples built successfully!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo "Binaries are in the 'build' directory:"
echo "  - build/comprehensive_write_test"
echo "  - build/write_demo"
echo "  - build/read_demo"
echo "  - build/roundtrip_test"
echo ""
echo "Run tests with:"
echo "  ./build/comprehensive_write_test"
echo "  ./build/roundtrip_test"
echo ""
echo "Run demos with:"
echo "  ./build/write_demo [output.nc]"
echo "  ./build/read_demo"
echo ""
