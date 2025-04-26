load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    deps = [
        ":metadata",
        "//tensorstore:array",
        "//tensorstore/driver",
        "//tensorstore:chunk_layout",
        "//tensorstore/driver:chunk_cache_driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:kvs_backed_chunk_cache",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/tiff:tiff_key_value_store",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        ":compressor",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:schema",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/riegeli:array_endian_codec",
        "//tensorstore/kvstore/tiff:tiff_details",
        "//tensorstore/kvstore/tiff:tiff_dir_cache",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "metadata_test",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":compressor",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:codec_spec",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/internal/riegeli:array_endian_codec",
        "//tensorstore/kvstore/tiff:tiff_details",
        "//tensorstore/kvstore/tiff:tiff_dir_cache",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
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
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal/compression:json_specified_compressor",
        "//tensorstore/kvstore/tiff:tiff_details",
    ],
)
