# In-memory KeyValueStore driver

load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "memory",
    srcs = ["memory_key_value_store.cc"],
    hdrs = ["memory_key_value_store.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal:json",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:execution",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:sender",
        "//tensorstore/util:status",
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
        "//tensorstore/internal:cache_key",
        "//tensorstore/kvstore:key_value_store_testutil",
        "//tensorstore/util:execution",
        "//tensorstore/util:future",
        "//tensorstore/util:sender",
        "//tensorstore/util:sender_testutil",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
