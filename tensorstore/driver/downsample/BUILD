load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")
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
    name = "downsample",
    srcs = ["downsample.cc"],
    hdrs = ["downsample.h"],
    deps = [
        ":downsample_array",
        ":downsample_method_json_binder",
        ":downsample_nditerable",
        ":downsample_util",
        ":grid_occupancy_map",
        "//tensorstore:downsample_method",
        "//tensorstore:spec",
        "//tensorstore/driver",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal/json_binding",
        "//tensorstore/serialization",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:sender_util",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

tensorstore_cc_library(
    name = "downsample_array",
    srcs = ["downsample_array.cc"],
    hdrs = ["downsample_array.h"],
    deps = [
        ":downsample_nditerable",
        ":downsample_util",
        "//tensorstore:array",
        "//tensorstore:downsample_method",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:arena",
        "//tensorstore/internal:nditerable",
        "//tensorstore/internal:nditerable_array",
        "//tensorstore/internal:nditerable_copy",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "downsample_array_test",
    size = "small",
    srcs = ["downsample_array_test.cc"],
    deps = [
        ":downsample_array",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "downsample_nditerable",
    srcs = ["downsample_nditerable.cc"],
    hdrs = ["downsample_nditerable.h"],
    deps = [
        "//tensorstore:box",
        "//tensorstore:data_type",
        "//tensorstore:downsample_method",
        "//tensorstore:index",
        "//tensorstore/internal:arena",
        "//tensorstore/internal:nditerable",
        "//tensorstore/internal:nditerable_buffer_management",
        "//tensorstore/internal:nditerable_util",
        "//tensorstore/internal:unique_with_intrusive_allocator",
        "//tensorstore/util:extents",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_binary(
    name = "downsample_benchmark_test",
    testonly = 1,
    srcs = ["downsample_benchmark_test.cc"],
    tags = ["benchmark"],
    deps = [
        ":downsample_array",
        ":downsample_nditerable",
        ":downsample_util",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/internal:data_type_random_generator",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/random",
        "@com_google_benchmark//:benchmark",
        "@com_google_benchmark//:benchmark_main",
    ],
)

tensorstore_cc_test(
    name = "downsample_test",
    size = "small",
    srcs = ["downsample_test.cc"],
    deps = [
        ":downsample",
        "//tensorstore:context",
        "//tensorstore:downsample",
        "//tensorstore:open",
        "//tensorstore:spec",
        "//tensorstore/driver",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/driver/array",
        "//tensorstore/driver/cast",
        "//tensorstore/driver/n5",
        "//tensorstore/driver/zarr",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution:sender_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "downsample_util",
    srcs = ["downsample_util.cc"],
    hdrs = ["downsample_util.h"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:downsample_method",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "downsample_util_test",
    size = "small",
    srcs = ["downsample_util_test.cc"],
    deps = [
        ":downsample_array",
        ":downsample_util",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:data_type_random_generator",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "downsample_method_json_binder",
    hdrs = ["downsample_method_json_binder.h"],
    deps = [
        "//tensorstore:downsample_method",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
    ],
)

tensorstore_cc_library(
    name = "grid_occupancy_map",
    srcs = ["grid_occupancy_map.cc"],
    hdrs = ["grid_occupancy_map.h"],
    deps = [
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
    ],
)

tensorstore_cc_test(
    name = "grid_occupancy_map_test",
    size = "small",
    srcs = ["grid_occupancy_map_test.cc"],
    deps = [
        ":grid_occupancy_map",
        "//tensorstore:box",
        "//tensorstore:index",
        "@com_google_googletest//:gtest_main",
    ],
)
