#Neuroglancer precomputed sharded format kvstore driver

load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_library",
    "tensorstore_cc_test",
)
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
    name = "uint64_sharded",
    srcs = ["uint64_sharded.cc"],
    hdrs = ["uint64_sharded.h"],
    deps = [
        ":murmurhash3",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:path",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/util:division",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
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
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/algorithm:container",
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
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
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
    name = "neuroglancer_uint64_sharded",
    srcs = ["neuroglancer_uint64_sharded.cc"],
    hdrs = ["neuroglancer_uint64_sharded.h"],
    deps = [
        ":uint64_sharded",
        ":uint64_sharded_decoder",
        ":uint64_sharded_encoder",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:result_sender",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/base:endian",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "neuroglancer_uint64_sharded_test",
    size = "small",
    srcs = ["neuroglancer_uint64_sharded_test.cc"],
    deps = [
        ":neuroglancer_uint64_sharded",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:thread_pool",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:kvs_backed_cache_testutil",
        "//tensorstore/internal/compression:zlib",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation_testutil",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_util",
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
