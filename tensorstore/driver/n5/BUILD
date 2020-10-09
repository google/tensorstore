# N5 TensorStore driver

load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "blosc_compressor",
    srcs = ["blosc_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal:json",
        "//tensorstore/internal/compression:blosc_compressor",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "blosc_compressor_test",
    size = "small",
    srcs = ["blosc_compressor_test.cc"],
    deps = [
        ":blosc_compressor",
        ":compressor",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "bzip2_compressor",
    srcs = ["bzip2_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal:json",
        "//tensorstore/internal/compression:bzip2_compressor",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "bzip2_compressor_test",
    size = "small",
    srcs = ["bzip2_compressor_test.cc"],
    deps = [
        ":bzip2_compressor",
        ":compressor",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "compressor",
    srcs = ["compressor.cc"],
    hdrs = [
        "compressor.h",
        "compressor_registry.h",
    ],
    deps = [
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/compression:json_specified_compressor",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":n5",
        "//tensorstore:open",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "golden_file_test",
    size = "small",
    srcs = ["golden_file_test.cc"],
    args = [
        "--tensorstore_test_data_dir",
        package_name() + "/testdata",
    ],
    data = [":testdata"],
    deps = [
        "xz_compressor",
        ":blosc_compressor",
        ":bzip2_compressor",
        ":gzip_compressor",
        ":n5",
        "//tensorstore:array",
        "//tensorstore:array_testutil",
        "//tensorstore:context",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore/internal:path",
        "//tensorstore/kvstore/file",
        "//tensorstore/util:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "gzip_compressor",
    srcs = ["gzip_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal:json",
        "//tensorstore/internal/compression:zlib_compressor",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "gzip_compressor_test",
    size = "small",
    srcs = ["gzip_compressor_test.cc"],
    deps = [
        ":compressor",
        ":gzip_compressor",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        ":compressor",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:strided_layout",
        "//tensorstore/internal:container_to_shared",
        "//tensorstore/internal:data_type_endian_conversion",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:json",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "metadata_test",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata",
        "//tensorstore:array",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "n5",
    srcs = ["driver.cc"],
    deps = [
        ":metadata",
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/internal:chunk_cache",
        "//tensorstore/internal:json",
        "//tensorstore/internal:path",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:future",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = 1,
)

filegroup(
    name = "testdata",
    srcs = glob(
        include = [
            "testdata/**",
        ],
        exclude = ["testdata/*.py"],
    ),
)

tensorstore_cc_library(
    name = "xz_compressor",
    srcs = ["xz_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal:json",
        "//tensorstore/internal/compression:xz_compressor",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "xz_compressor_test",
    size = "small",
    srcs = ["xz_compressor_test.cc"],
    deps = [
        ":compressor",
        ":metadata",
        ":xz_compressor",
        "//tensorstore:array",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)
