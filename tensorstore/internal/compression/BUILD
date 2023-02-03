load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "blosc",
    srcs = ["blosc.cc"],
    hdrs = ["blosc.h"],
    deps = [
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_library(
    name = "blosc_compressor",
    hdrs = ["blosc_compressor.h"],
    deps = [
        ":blosc",
        ":json_specified_compressor",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_test(
    name = "blosc_test",
    size = "small",
    srcs = ["blosc_test.cc"],
    deps = [
        ":blosc",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:cord_test_helpers",
        "@com_google_googletest//:gtest_main",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_library(
    name = "bzip2",
    srcs = ["bzip2.cc"],
    hdrs = ["bzip2.h"],
    deps = [
        ":cord_stream_manager",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@org_sourceware_bzip2//:bzip2",
    ],
)

tensorstore_cc_library(
    name = "bzip2_compressor",
    hdrs = ["bzip2_compressor.h"],
    deps = [
        ":bzip2",
        ":json_specified_compressor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
    ],
)

tensorstore_cc_test(
    name = "bzip2_test",
    size = "small",
    srcs = ["bzip2_test.cc"],
    deps = [
        ":bzip2",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:cord_test_helpers",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "lzma",
    srcs = ["lzma.cc"],
    hdrs = ["lzma.h"],
    deps = [
        ":cord_stream_manager",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@org_tukaani_xz//:liblzma",
    ],
)

tensorstore_cc_test(
    name = "lzma_test",
    size = "small",
    srcs = ["lzma_test.cc"],
    deps = [
        ":lzma",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:cord_test_helpers",
        "@com_google_googletest//:gtest_main",
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
    hdrs = ["xz_compressor.h"],
    deps = [
        ":json_specified_compressor",
        ":lzma",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
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
    hdrs = ["zlib_compressor.h"],
    deps = [
        ":json_specified_compressor",
        ":zlib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
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
