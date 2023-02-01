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
    name = "stack",
    srcs = ["driver.cc"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:open_mode",
        "//tensorstore:resize_options",
        "//tensorstore:schema",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:grid_partition",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:irregular_grid",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:staleness_bound",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:iterate",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

tensorstore_cc_test(
    name = "stack_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":stack",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:array_testutil",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:progress",
        "//tensorstore:strided_layout",
        "//tensorstore/driver/array",  # build_cleaner: keep
        "//tensorstore/driver/n5",  # build_cleaner: keep
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/kvstore/memory",  # build_cleaner: keep
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "integration_test",
    srcs = ["integration_test.cc"],
    deps = [
        ":stack",  # build_cleaner: keep
        "//tensorstore",
        "//tensorstore:array_testutil",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:progress",
        "//tensorstore/driver/image/png",  # build_cleaner: keep
        "//tensorstore/driver/neuroglancer_precomputed",  # build_cleaner: keep
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore/memory",  # build_cleaner: keep
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)
