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
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:sender",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
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
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/cache_key",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
