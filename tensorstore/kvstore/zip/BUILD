load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//tensorstore:internal_packages"])

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
    name = "zip",
    srcs = ["zip_key_value_store.cc"],
    deps = [
        ":zip_dir_cache",
        "//tensorstore:context",
        "//tensorstore:transaction",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:auto_detect",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:common_metrics",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/serialization",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@riegeli//riegeli/bytes:cord_reader",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "zip_key_value_store_test",
    srcs = ["zip_key_value_store_test.cc"],
    args = [
        "--tensorstore_test_data=$(location //tensorstore/internal/compression:testdata)",
    ],
    data = ["//tensorstore/internal/compression:testdata"],
    deps = [
        ":zip",  # build_cleaner: keep
        "//tensorstore:context",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:read_all",
    ],
)

tensorstore_cc_library(
    name = "zip_dir_cache",
    srcs = ["zip_dir_cache.cc"],
    hdrs = ["zip_dir_cache.h"],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/compression:zip_details",
        "//tensorstore/internal/estimate_heap_usage",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
    ],
)

tensorstore_cc_test(
    name = "zip_dir_cache_test",
    srcs = ["zip_dir_cache_test.cc"],
    args = [
        "--tensorstore_test_data=$(location //tensorstore/internal/compression:testdata)",
    ],
    data = ["//tensorstore/internal/compression:testdata"],
    deps = [
        ":zip_dir_cache",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:executor",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:fd_reader",
        "@riegeli//riegeli/bytes:read_all",
    ],
)
