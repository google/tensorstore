load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

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
    name = "json",
    srcs = ["driver.cc"],
    deps = [
        ":json_change_map",
        "//tensorstore/driver",
        "//tensorstore/internal:async_cache",
        "//tensorstore/internal:async_initialized_cache_mixin",
        "//tensorstore/internal:cache_pool_resource",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_pointer",
        "//tensorstore/internal:kvs_backed_cache",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal:staleness_bound_json_binder",
        "//tensorstore/util:sender",
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
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json",
        "//tensorstore/internal:parse_json_matches",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/kvstore:key_value_store_testutil",
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
