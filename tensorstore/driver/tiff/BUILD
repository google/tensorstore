load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])


tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    deps = [
        ":metadata",
        "//tensorstore:index",
        "//tensorstore:data_type",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/tiff:tiff_details",
        "@com_github_nlohmann_json//:json",
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
        "//tensorstore:index",
        "//tensorstore:schema",
        "//tensorstore:data_type",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/kvstore/tiff:tiff_details",
        "//tensorstore/kvstore/tiff:tiff_dir_cache",
        "@com_google_absl//absl/log:absl_log",
        "@com_github_nlohmann_json//:json",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "metadata_test",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:codec_spec",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/kvstore/tiff:tiff_details",
        "//tensorstore/kvstore/tiff:tiff_dir_cache",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)
