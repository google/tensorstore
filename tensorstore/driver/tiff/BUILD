load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    deps = [
        ":compressor",
        ":metadata",
        ":zlib_compressor",
        ":zstd_compressor",
        "//tensorstore:array",
        "//tensorstore:chunk_layout",
        "//tensorstore:index",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk_cache_driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:kvs_backed_chunk_cache",
        "//tensorstore/internal/json_binding:staleness_bound",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore/tiff:tiff_key_value_store",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
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
        ":zlib_compressor",
        ":zstd_compressor",
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
        "//tensorstore/util/garbage_collection",
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
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":driver",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:codec_spec",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/internal/riegeli:array_endian_codec",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/kvstore/tiff:tiff_details",
        "//tensorstore/kvstore/tiff:tiff_dir_cache",
        "//tensorstore/kvstore/tiff:tiff_test_util",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
    ],
)

tensorstore_cc_test(
    name = "golden_file_test",
    size = "small",
    srcs = ["golden_file_test.cc"],
    args = [
        "--tensorstore_test_data_dir=" +
        package_name() + "/testdata",
    ],
    data = [":testdata"],
    deps = [
        ":driver",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:context",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore/internal:path",
        "//tensorstore/kvstore/file",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_googletest//:gtest_main",
    ],
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
    name = "zstd_compressor",
    srcs = ["zstd_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:zstd_compressor",
        "//tensorstore/internal/json_binding",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "zlib_compressor",
    srcs = ["zlib_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:zlib_compressor",
        "//tensorstore/internal/json_binding",
    ],
    alwayslink = 1,
)
