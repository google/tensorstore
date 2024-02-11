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
        "@com_google_absl//absl/status",
        "@org_blosc_cblosc//:blosc",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/base:chain",
        "@com_google_riegeli//riegeli/bytes:chain_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:read_all",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:write",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_test(
    name = "blosc_test",
    size = "small",
    srcs = ["blosc_test.cc"],
    deps = [
        ":blosc",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_library(
    name = "bzip2_compressor",
    srcs = ["bzip2_compressor.cc"],
    hdrs = ["bzip2_compressor.h"],
    deps = [
        ":json_specified_compressor",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/bzip2:bzip2_reader",
        "@com_google_riegeli//riegeli/bzip2:bzip2_writer",
    ],
)

tensorstore_cc_library(
    name = "cord_stream_manager",
    hdrs = ["cord_stream_manager.h"],
    visibility = ["//visibility:private"],
    deps = [
        "@com_google_absl//absl/strings:cord",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:read_all",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:write",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_library(
    name = "neuroglancer_compressed_segmentation",
    srcs = ["neuroglancer_compressed_segmentation.cc"],
    hdrs = ["neuroglancer_compressed_segmentation.h"],
    deps = [
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tensorstore_cc_test(
    name = "neuroglancer_compressed_segmentation_test",
    size = "small",
    srcs = ["neuroglancer_compressed_segmentation_test.cc"],
    deps = [
        ":neuroglancer_compressed_segmentation",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "xz_compressor",
    srcs = ["xz_compressor.cc"],
    hdrs = ["xz_compressor.h"],
    deps = [
        ":json_specified_compressor",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/xz:xz_reader",
        "@com_google_riegeli//riegeli/xz:xz_writer",
        "@org_tukaani_xz//:liblzma",
    ],
)

tensorstore_cc_test(
    name = "xz_compressor_test",
    size = "small",
    srcs = ["xz_compressor_test.cc"],
    deps = [
        ":xz_compressor",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:cord_test_helpers",
        "@com_google_googletest//:gtest_main",
        "@org_tukaani_xz//:liblzma",
    ],
)

tensorstore_cc_library(
    name = "zlib",
    srcs = ["zlib.cc"],
    hdrs = ["zlib.h"],
    deps = [
        ":cord_stream_manager",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@net_zlib//:zlib",
    ],
)

tensorstore_cc_library(
    name = "zlib_compressor",
    srcs = ["zlib_compressor.cc"],
    hdrs = ["zlib_compressor.h"],
    deps = [
        ":json_specified_compressor",
        ":zlib",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/zlib:zlib_reader",
        "@com_google_riegeli//riegeli/zlib:zlib_writer",
    ],
)

tensorstore_cc_test(
    name = "zlib_test",
    size = "small",
    srcs = ["zlib_test.cc"],
    deps = [
        ":zlib",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:cord_test_helpers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "zstd_compressor",
    srcs = ["zstd_compressor.cc"],
    hdrs = ["zstd_compressor.h"],
    deps = [
        ":json_specified_compressor",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/zstd:zstd_reader",
        "@com_google_riegeli//riegeli/zstd:zstd_writer",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_riegeli//riegeli/bytes:limiting_reader",
        "@com_google_riegeli//riegeli/bytes:prefix_limiting_reader",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bzip2:bzip2_reader",
        "@com_google_riegeli//riegeli/endian:endian_reading",
        "@com_google_riegeli//riegeli/xz:xz_reader",
        "@com_google_riegeli//riegeli/zlib:zlib_reader",
        "@com_google_riegeli//riegeli/zstd:zstd_reader",
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
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@com_google_riegeli//riegeli/bytes:read_all",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:string_reader",
    ],
)
