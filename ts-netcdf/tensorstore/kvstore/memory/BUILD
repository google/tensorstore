# In-memory KeyValueStore driver

load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_cc_library(
    name = "memory",
    srcs = ["memory_key_value_store.cc"],
    hdrs = ["memory_key_value_store.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore:transaction",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:btree",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "memory_key_value_store_test",
    size = "small",
    srcs = ["memory_key_value_store_test.cc"],
    deps = [
        ":memory",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)
