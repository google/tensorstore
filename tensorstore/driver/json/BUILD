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
        "//tensorstore:chunk_layout",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:rank",
        "//tensorstore:schema",
        "//tensorstore:staleness_bound",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:arena",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json_pointer",
        "//tensorstore/internal:lock_collection",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:nditerable",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal:unowned_to_shared",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:async_initialized_cache_mixin",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json:same",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:staleness_bound",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:sender_util",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "json_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":json",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:context",
        "//tensorstore:open",
        "//tensorstore:spec",
        "//tensorstore:transaction",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/driver/n5",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
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
