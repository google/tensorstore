# Neuroglancer precomputed format TensorStore driver

load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

DOCTEST_SOURCES = glob([
    "**/*.rst",
    "**/*.yml",
])

doctest_test(
    name = "doctest_test",
    srcs = DOCTEST_SOURCES,
)

filegroup(
    name = "doc_sources",
    srcs = DOCTEST_SOURCES,
)

tensorstore_cc_library(
    name = "chunk_encoding",
    srcs = ["chunk_encoding.cc"],
    hdrs = ["chunk_encoding.h"],
    deps = [
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/internal:data_type_endian_conversion",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal/compression:neuroglancer_compressed_segmentation",
        "//tensorstore/internal/image",
        "//tensorstore/internal/image:jpeg",
        "//tensorstore/util:endian",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
    ],
)

tensorstore_cc_test(
    name = "chunk_encoding_test",
    size = "small",
    srcs = ["chunk_encoding_test.cc"],
    deps = [
        ":chunk_encoding",
        ":metadata",
        "//tensorstore:array",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":neuroglancer_precomputed",
        "//tensorstore:open",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/driver/zarr",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/internal:test_util",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/image",
        "//tensorstore/internal/image:jpeg",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        "//tensorstore:box",
        "//tensorstore:codec_spec",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open_mode",
        "//tensorstore:schema",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:bit_operations",
        "//tensorstore/internal:json_metadata_matching",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:data_type",
        "//tensorstore/kvstore/neuroglancer_uint64_sharded:uint64_sharded",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "metadata_test",
    size = "small",
    srcs = ["metadata_test.cc"],
    deps = [
        ":metadata",
        "//tensorstore:box",
        "//tensorstore:codec_spec",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "neuroglancer_precomputed",
    srcs = ["driver.cc"],
    deps = [
        ":chunk_encoding",
        ":metadata",
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:path",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/neuroglancer_uint64_sharded",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:division",
        "//tensorstore/util:future",
        "@com_google_absl//absl/status",
    ],
    alwayslink = 1,
)
