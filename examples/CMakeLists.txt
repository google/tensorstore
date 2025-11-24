# Standalone CMakeLists.txt for read_structured_zarr3 example
#
# Build instructions:
#   mkdir -p /home/ubuntu/source/tensorstore/examples/build
#   cd /home/ubuntu/source/tensorstore/examples/build
#   cmake ..
#   make
#
# Run:
#   ./read_structured_zarr3 --zarr_path=/home/ubuntu/source/tensorstore/filt_mig.mdio/headers

cmake_minimum_required(VERSION 3.24)
project(read_structured_zarr3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the tensorstore build directory
set(TENSORSTORE_BUILD_DIR "/home/ubuntu/source/tensorstore/build" CACHE PATH "Path to tensorstore build directory")
set(TENSORSTORE_SOURCE_DIR "/home/ubuntu/source/tensorstore" CACHE PATH "Path to tensorstore source directory")
set(DEPS_DIR "${TENSORSTORE_BUILD_DIR}/_deps")

# Include paths (matching what tensorstore tests use)
include_directories(
    ${TENSORSTORE_SOURCE_DIR}
    ${DEPS_DIR}/absl-src
    ${DEPS_DIR}/re2-src
    ${DEPS_DIR}/riegeli-src
)

include_directories(SYSTEM
    ${DEPS_DIR}/half-build/include
    ${DEPS_DIR}/half-src/include
    ${DEPS_DIR}/nlohmann_json-build/include
    ${DEPS_DIR}/nlohmann_json-src/include
    ${TENSORSTORE_BUILD_DIR}
)

# Compiler flags
add_compile_options(
    -fPIE
    -Wno-deprecated-declarations
    -Wno-sign-compare
    -Wno-unused-but-set-parameter
    -Wno-maybe-uninitialized
    -Wno-sequence-point
    -Wno-unknown-warning-option
    -Wno-stringop-overflow
    -fsized-deallocation
)

# Find all the static libraries we need from the tensorstore build
file(GLOB TENSORSTORE_LIBS "${TENSORSTORE_BUILD_DIR}/libtensorstore*.a")
file(GLOB_RECURSE ABSEIL_LIBS "${DEPS_DIR}/absl-build/absl/*.a")
file(GLOB_RECURSE RIEGELI_LIBS "${DEPS_DIR}/riegeli-build/*.a")

# Additional dependency libraries - corrected paths
file(GLOB_RECURSE BLOSC_LIBS "${DEPS_DIR}/blosc-build/*.a")
file(GLOB_RECURSE ZSTD_LIBS "${DEPS_DIR}/zstd-build/*.a")
file(GLOB_RECURSE RE2_LIBS "${DEPS_DIR}/re2-build/*.a")
file(GLOB_RECURSE SNAPPY_LIBS "${DEPS_DIR}/snappy-build/*.a")
file(GLOB_RECURSE BROTLI_LIBS "${DEPS_DIR}/brotli-build/*.a")
file(GLOB_RECURSE LZ4_LIBS "${DEPS_DIR}/lz4-build/*.a")
file(GLOB_RECURSE ZLIB_LIBS "${DEPS_DIR}/zlib-build/*.a")
file(GLOB_RECURSE PROTOBUF_LIBS "${DEPS_DIR}/protobuf-build/*.a")
file(GLOB_RECURSE GRPC_LIBS "${DEPS_DIR}/grpc-build/*.a")
file(GLOB_RECURSE CARES_LIBS "${DEPS_DIR}/c-ares-build/*.a")
file(GLOB_RECURSE SSL_LIBS "${DEPS_DIR}/boringssl-build/ssl/*.a")
file(GLOB_RECURSE CRYPTO_LIBS "${DEPS_DIR}/boringssl-build/crypto/*.a")
file(GLOB_RECURSE LIBLZMA_LIBS "${DEPS_DIR}/liblzma-build/*.a")
file(GLOB_RECURSE BZIP2_LIBS "${DEPS_DIR}/bzip2-build/*.a")
file(GLOB_RECURSE JPEG_LIBS "${DEPS_DIR}/jpeg-build/*.a")
file(GLOB_RECURSE PNG_LIBS "${DEPS_DIR}/png-build/*.a")
file(GLOB_RECURSE TIFF_LIBS "${DEPS_DIR}/tiff-build/*.a")
file(GLOB_RECURSE AVIF_LIBS "${DEPS_DIR}/avif-build/*.a")
file(GLOB_RECURSE AOM_LIBS "${DEPS_DIR}/aom-build/*.a")
file(GLOB_RECURSE WEBP_LIBS "${DEPS_DIR}/webp-build/*.a")
file(GLOB_RECURSE CURL_LIBS "${DEPS_DIR}/curl-build/*.a")

# Create executable
add_executable(read_structured_zarr3 read_structured_zarr3.cc)

# Link libraries - use whole-archive for libraries that use static registration
# These include drivers, codecs, kvstores, and context resource providers
target_link_libraries(read_structured_zarr3 PRIVATE
    # Force inclusion of libraries with static registrations
    -Wl,--whole-archive
    
    # Context resource providers
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_internal_data_copy_concurrency_resource.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_internal_file_io_concurrency_resource.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_internal_cache_cache_pool_resource.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_internal_concurrency_resource.a
    
    # Zarr3 driver and codecs
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_driver.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_blosc.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_bytes.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_crc32c.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_gzip.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_transpose.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_zstd.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_sharding_indexed.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_driver_zarr3_codec_codec_chain_spec.a
    
    # File kvstore and its resource providers
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_kvstore_file.a
    ${TENSORSTORE_BUILD_DIR}/libtensorstore_kvstore_file_file_resource.a
    
    -Wl,--no-whole-archive
    
    -Wl,--start-group
    
    # Tensorstore libs
    ${TENSORSTORE_LIBS}
    
    # Riegeli
    ${RIEGELI_LIBS}
    
    # Abseil
    ${ABSEIL_LIBS}
    
    # Compression libs
    ${BLOSC_LIBS}
    ${ZSTD_LIBS}
    ${LZ4_LIBS}
    ${SNAPPY_LIBS}
    ${BROTLI_LIBS}
    ${ZLIB_LIBS}
    ${LIBLZMA_LIBS}
    ${BZIP2_LIBS}
    
    # Regex
    ${RE2_LIBS}
    
    # Protocol buffers and gRPC
    ${PROTOBUF_LIBS}
    ${GRPC_LIBS}
    ${CARES_LIBS}
    
    # SSL/TLS
    ${SSL_LIBS}
    ${CRYPTO_LIBS}
    
    # Image libraries  
    ${JPEG_LIBS}
    ${PNG_LIBS}
    ${TIFF_LIBS}
    ${AVIF_LIBS}
    ${AOM_LIBS}
    ${WEBP_LIBS}
    
    # HTTP
    ${CURL_LIBS}
    
    -Wl,--end-group
    
    # System libraries
    pthread
    dl
    m
    rt
)
