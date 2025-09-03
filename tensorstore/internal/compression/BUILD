load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

filegroup(
    name = "testdata",
    srcs = [
        "testdata/data.zip",
    ],
)

tensorstore_cc_library(
    name = "blosc",
    srcs = ["blosc.cc"],
    hdrs = ["blosc.h"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:string_view",
        "@abseil-cpp//absl/types:optional",
        "@org_blosc_cblosc//:blosc",
        "@riegeli//riegeli/base:types",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_library(
    name = "blosc_compressor",
    srcs = ["blosc_compressor.cc"],
    hdrs = ["blosc_compressor.h"],
    deps = [
        ":blosc",
        ":json_specified_compressor",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@org_blosc_cblosc//:blosc",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "blosc_test",
    size = "small",
    srcs = ["blosc_test.cc"],
    deps = [
        ":blosc",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_library(
    name = "bzip2_compressor",
    srcs = ["bzip2_compressor.cc"],
    hdrs = ["bzip2_compressor.h"],
    deps = [
        ":json_specified_compressor",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/bzip2:bzip2_reader",
        "@riegeli//riegeli/bzip2:bzip2_writer",
    ],
)

tensorstore_cc_library(
    name = "cord_stream_manager",
    hdrs = ["cord_stream_manager.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "json_specified_compressor",
    srcs = ["json_specified_compressor.cc"],
    hdrs = ["json_specified_compressor.h"],
    deps = [
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json_registry",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:write",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_library(
    name = "neuroglancer_compressed_segmentation",
    srcs = ["neuroglancer_compressed_segmentation.cc"],
    hdrs = ["neuroglancer_compressed_segmentation.h"],
    deps = [
        "//tensorstore/util:endian",
        "@abseil-cpp//absl/container:flat_hash_map",
    ],
)

tensorstore_cc_test(
    name = "neuroglancer_compressed_segmentation_test",
    size = "small",
    srcs = ["neuroglancer_compressed_segmentation_test.cc"],
    deps = [
        ":neuroglancer_compressed_segmentation",
        "//tensorstore/util:endian",
        "@abseil-cpp//absl/random",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "xz_compressor",
    srcs = ["xz_compressor.cc"],
    hdrs = ["xz_compressor.h"],
    deps = [
        ":json_specified_compressor",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/xz:xz_reader",
        "@riegeli//riegeli/xz:xz_writer",
        "@xz//:lzma",
    ],
)

tensorstore_cc_test(
    name = "xz_compressor_test",
    size = "small",
    srcs = ["xz_compressor_test.cc"],
    deps = [
        ":xz_compressor",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:cord_test_helpers",
        "@googletest//:gtest_main",
        "@xz//:lzma",
    ],
)

tensorstore_cc_library(
    name = "zlib",
    srcs = ["zlib.cc"],
    hdrs = ["zlib.h"],
    deps = [
        ":cord_stream_manager",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@zlib",
    ],
)

tensorstore_cc_library(
    name = "zlib_compressor",
    srcs = ["zlib_compressor.cc"],
    hdrs = ["zlib_compressor.h"],
    deps = [
        ":json_specified_compressor",
        ":zlib",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/zlib:zlib_reader",
        "@riegeli//riegeli/zlib:zlib_writer",
    ],
)

tensorstore_cc_test(
    name = "zlib_test",
    size = "small",
    srcs = ["zlib_test.cc"],
    deps = [
        ":zlib",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:cord_test_helpers",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "zstd_compressor",
    srcs = ["zstd_compressor.cc"],
    hdrs = ["zstd_compressor.h"],
    deps = [
        ":json_specified_compressor",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
        "@riegeli//riegeli/zstd:zstd_reader",
        "@riegeli//riegeli/zstd:zstd_writer",
    ],
)

tensorstore_cc_library(
    name = "zip_details",
    srcs = ["zip_details.cc"],
    hdrs = ["zip_details.h"],
    deps = [
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/riegeli:find",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@riegeli//riegeli/bytes:limiting_reader",
        "@riegeli//riegeli/bytes:prefix_limiting_reader",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bzip2:bzip2_reader",
        "@riegeli//riegeli/endian:endian_reading",
        "@riegeli//riegeli/xz:xz_reader",
        "@riegeli//riegeli/zlib:zlib_reader",
        "@riegeli//riegeli/zstd:zstd_reader",
    ],
)

tensorstore_cc_test(
    name = "zip_details_test",
    srcs = ["zip_details_test.cc"],
    args = [
        "--tensorstore_test_data=$(location :testdata)",
    ],
    data = [":testdata"],
    deps = [
        ":zip_details",
        "//tensorstore/internal/riegeli:find",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:read_all",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:string_reader",
    ],
)
