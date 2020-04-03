load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

tensorstore_cc_library(
    name = "blosc",
    srcs = ["blosc.cc"],
    hdrs = ["blosc.h"],
    deps = [
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_test(
    name = "blosc_test",
    size = "small",
    srcs = ["blosc_test.cc"],
    deps = [
        ":blosc",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_library(
    name = "zlib",
    srcs = ["zlib.cc"],
    hdrs = ["zlib.h"],
    deps = [
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings",
        "@net_zlib//:zlib",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "bzip2",
    srcs = ["bzip2.cc"],
    hdrs = ["bzip2.h"],
    deps = [
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings",
        "@org_sourceware_bzip2//:bzip2",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "lzma",
    srcs = ["lzma.cc"],
    hdrs = ["lzma.h"],
    deps = [
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "jpeg",
    srcs = ["jpeg.cc"],
    hdrs = ["jpeg.h"],
    deps = [
        "//tensorstore/util:function_view",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/strings",
        "@jpeg",
    ],
)

tensorstore_cc_test(
    name = "jpeg_test",
    size = "small",
    srcs = ["jpeg_test.cc"],
    deps = [
        ":jpeg",
        "//tensorstore:array",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
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
    name = "json_specified_compressor",
    srcs = ["json_specified_compressor.cc"],
    hdrs = ["json_specified_compressor.h"],
    deps = [
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json_registry",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "zlib_compressor",
    hdrs = ["zlib_compressor.h"],
    deps = [
        ":json_specified_compressor",
        ":zlib",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "bzip2_compressor",
    hdrs = ["bzip2_compressor.h"],
    deps = [
        ":bzip2",
        ":json_specified_compressor",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "blosc_compressor",
    hdrs = ["blosc_compressor.h"],
    deps = [
        ":blosc",
        ":json_specified_compressor",
        "//tensorstore/internal:json",
        "//tensorstore/util:quote_string",
        "@com_google_absl//absl/strings",
        "@org_blosc_cblosc//:blosc",
    ],
)

tensorstore_cc_library(
    name = "xz_compressor",
    hdrs = ["xz_compressor.h"],
    deps = [
        ":json_specified_compressor",
        ":lzma",
        "@com_google_absl//absl/strings",
    ],
)
