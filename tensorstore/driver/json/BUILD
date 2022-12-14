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
    name = "json",
    srcs = ["driver.cc"],
    deps = [
        ":json_change_map",
        "//tensorstore/driver",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:json_pointer",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:async_initialized_cache_mixin",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:staleness_bound",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:sender_util",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "json_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":json",
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/driver/n5",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "json_change_map",
    srcs = ["json_change_map.cc"],
    hdrs = ["json_change_map.h"],
    deps = [
        "//tensorstore/internal:json_pointer",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:btree",
    ],
)

tensorstore_cc_test(
    name = "json_change_map_test",
    size = "small",
    srcs = ["json_change_map_test.cc"],
    deps = [
        ":json_change_map",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
