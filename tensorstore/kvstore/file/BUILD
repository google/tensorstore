# Filesystem-backed KeyValueStore driver

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
    name = "file",
    srcs = [
        "file_key_value_store.cc",
    ],
    deps = [
        ":file_resource",
        ":util",
        "//tensorstore:batch",
        "//tensorstore:context",
        "//tensorstore/internal:file_io_concurrency_resource",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/os:error_code",
        "//tensorstore/internal/os:file_lister",
        "//tensorstore/internal/os:file_lock",
        "//tensorstore/internal/os:file_util",
        "//tensorstore/internal/os:unique_handle",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:common_metrics",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "file_key_value_store_test",
    size = "large",
    srcs = ["file_key_value_store_test.cc"],
    deps = [
        ":file",
        "//tensorstore:context",
        "//tensorstore/internal:file_io_concurrency_resource",
        "//tensorstore/internal/os:filesystem",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "file_resource",
    srcs = ["file_resource.cc"],
    hdrs = ["file_resource.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorstore/kvstore:key_range",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorstore/kvstore:key_range",
        "@com_google_googletest//:gtest_main",
    ],
)
