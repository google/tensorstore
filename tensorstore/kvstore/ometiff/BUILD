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
    name = "ometiff",
    srcs = [
        "ometiff_key_value_store.cc",
        "ometiff_spec.cc",
    ],
    hdrs = [
        "ometiff_key_value_store.h",
        "ometiff_spec.h",
    ],
    deps = [
        "//tensorstore:chunk_layout",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:kvs_read_streambuf",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:data_type",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@libtiff//:tiff",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "ometiff_key_value_store_test",
    size = "small",
    srcs = ["ometiff_key_value_store_test.cc"],
    deps = [
        ":ometiff",
        "//tensorstore:context",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/cache_key",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
