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
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/os:error_code",
        "//tensorstore/internal/os:file_descriptor",
        "//tensorstore/internal/os:file_lister",
        "//tensorstore/internal/os:file_lock",
        "//tensorstore/internal/os:file_util",
        "//tensorstore/internal/os:memory_region",
        "//tensorstore/internal/os:unique_handle",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:common_metrics",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:division",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
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
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/os:filesystem",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/internal/testing:on_windows",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
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
        "@abseil-cpp//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "util",
    srcs = ["util.cc"],
    hdrs = ["util.h"],
    deps = [
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal:path",
        "//tensorstore/internal/os:file_descriptor",
        "//tensorstore/internal/os:file_util",
        "//tensorstore/internal/os:hugepages",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:division",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_test(
    name = "util_test",
    size = "small",
    srcs = ["util_test.cc"],
    deps = [
        ":util",
        "//tensorstore/internal/testing:on_windows",
        "//tensorstore/kvstore:key_range",
        "@googletest//:gtest_main",
    ],
)
