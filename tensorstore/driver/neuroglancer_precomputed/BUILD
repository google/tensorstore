# Neuroglancer precomputed format TensorStore driver

load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "chunk_encoding",
    srcs = ["chunk_encoding.cc"],
    hdrs = ["chunk_encoding.h"],
    deps = [
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/internal:container_to_shared",
        "//tensorstore/internal:data_type_endian_conversion",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal/compression:jpeg",
        "//tensorstore/internal/compression:neuroglancer_compressed_segmentation",
        "//tensorstore/util:endian",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_google_absl//absl/algorithm:container",
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
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/internal/compression:jpeg",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        ":uint64_sharded",
        "//tensorstore:box",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/internal:bit_operations",
        "//tensorstore/internal:json",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span_json",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/algorithm:container",
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
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "murmurhash3",
    srcs = ["murmurhash3.cc"],
    hdrs = ["murmurhash3.h"],
)

tensorstore_cc_test(
    name = "murmurhash3_test",
    size = "small",
    srcs = ["murmurhash3_test.cc"],
    deps = [
        ":murmurhash3",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "neuroglancer_precomputed",
    srcs = ["driver.cc"],
    deps = [
        ":chunk_encoding",
        ":metadata",
        ":uint64_sharded_key_value_store",
        "//tensorstore",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/internal:cache_key",
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

tensorstore_cc_library(
    name = "uint64_sharded",
    srcs = ["uint64_sharded.cc"],
    hdrs = ["uint64_sharded.h"],
    deps = [
        ":murmurhash3",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:json",
        "//tensorstore/internal:path",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:division",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tensorstore_cc_library(
    name = "uint64_sharded_decoder",
    srcs = ["uint64_sharded_decoder.cc"],
    hdrs = ["uint64_sharded_decoder.h"],
    deps = [
        ":uint64_sharded",
        "//tensorstore/internal:cord_util",
        "//tensorstore/internal/compression:zlib",
        "//tensorstore/util:endian",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "uint64_sharded_decoder_test",
    size = "small",
    srcs = ["uint64_sharded_decoder_test.cc"],
    deps = [
        ":uint64_sharded",
        ":uint64_sharded_decoder",
        ":uint64_sharded_encoder",
        "//tensorstore/internal/compression:zlib",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "uint64_sharded_encoder",
    srcs = ["uint64_sharded_encoder.cc"],
    hdrs = ["uint64_sharded_encoder.h"],
    deps = [
        ":uint64_sharded",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal/compression:zlib",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:endian",
        "//tensorstore/util:function_view",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
    ],
)

tensorstore_cc_test(
    name = "uint64_sharded_encoder_test",
    size = "small",
    srcs = ["uint64_sharded_encoder_test.cc"],
    deps = [
        ":uint64_sharded",
        ":uint64_sharded_encoder",
        "//tensorstore/internal/compression:zlib",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "uint64_sharded_key_value_store",
    srcs = ["uint64_sharded_key_value_store.cc"],
    hdrs = ["uint64_sharded_key_value_store.h"],
    deps = [
        ":uint64_sharded",
        ":uint64_sharded_decoder",
        ":uint64_sharded_encoder",
        "//tensorstore/internal:aggregate_writeback_cache",
        "//tensorstore/internal:async_cache",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:estimate_heap_usage",
        "//tensorstore/internal:kvs_backed_cache",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal/compression:zlib",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:endian",
    ],
)

tensorstore_cc_test(
    name = "uint64_sharded_key_value_store_test",
    size = "small",
    srcs = ["uint64_sharded_key_value_store_test.cc"],
    deps = [
        ":uint64_sharded_key_value_store",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:kvs_backed_cache_testutil",
        "//tensorstore/internal:thread_pool",
        "//tensorstore/internal/compression:zlib",
        "//tensorstore/kvstore:generation_testutil",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:executor",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "uint64_sharded_test",
    size = "small",
    srcs = ["uint64_sharded_test.cc"],
    deps = [
        ":uint64_sharded",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)
