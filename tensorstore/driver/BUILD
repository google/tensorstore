load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_multi_test")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

DRIVERS = [
    "array",
    "cast",
    "downsample",
    "json",
    "n5",
    "neuroglancer_precomputed",
    "zarr",
]

DOCTEST_SOURCES = glob([
    "**/*.rst",
    "**/*.yml",
])

doctest_multi_test(
    name = "doctest_tests",
    srcs = DOCTEST_SOURCES,
)

filegroup(
    name = "doc_sources",
    srcs = DOCTEST_SOURCES + [
        "//tensorstore/driver/" + driver + ":doc_sources"
        for driver in DRIVERS
    ],
)

tensorstore_cc_library(
    name = "all_drivers",
    deps = [
        "//tensorstore/driver/" + driver
        for driver in DRIVERS
    ],
)

tensorstore_cc_library(
    name = "chunk",
    hdrs = ["chunk.h"],
    deps = [
        "//tensorstore:data_type",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:arena",
        "//tensorstore/internal:element_copy_function",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:lock_collection",
        "//tensorstore/internal:nditerable",
        "//tensorstore/internal:poly",
        "//tensorstore/util:future",
        "//tensorstore/util:sender",
        "//tensorstore/util:status",
    ],
)

tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    hdrs = [
        "driver.h",
        "registry.h",
    ],
    deps = [
        ":chunk",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:open_mode",
        "//tensorstore:progress",
        "//tensorstore:read_write_options",
        "//tensorstore:resize_options",
        "//tensorstore:schema",
        "//tensorstore:static_cast",
        "//tensorstore:transaction",
        "//tensorstore/index_space:alignment",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:json",
        "//tensorstore/index_space:transform_broadcastable_array",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:arena",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:data_type_json_binder",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal:lock_collection",
        "//tensorstore/internal:nditerable_copy",
        "//tensorstore/internal:nditerable_data_type_conversion",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:sender",
        "//tensorstore/util:status",
    ],
)

tensorstore_cc_test(
    name = "driver_test",
    size = "small",
    srcs = ["driver_test.cc"],
    deps = [
        ":driver",
        "//tensorstore:array",
        "//tensorstore:context",
        "//tensorstore:index",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:executor",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "driver_testutil",
    testonly = 1,
    srcs = ["driver_testutil.cc"],
    hdrs = ["driver_testutil.h"],
    deps = [
        ":chunk",
        ":driver",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:json_serialization_options",
        "//tensorstore:open",
        "//tensorstore:transaction",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:data_type_random_generator",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:nditerable_transformed_array",
        "//tensorstore/internal:queue_testutil",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:sync_flow_sender",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_googletest//:gtest",
    ],
)

bool_flag(
    name = "kvs_backed_chunk_driver_debug",
    build_setting_default = False,
)

config_setting(
    name = "kvs_backed_chunk_driver_debug_setting",
    flag_values = {
        ":kvs_backed_chunk_driver_debug": "True",
    },
    visibility = ["//visibility:private"],
)

# To enable debug logging, specify:
# bazel build --//tensorstore/driver:kvs_backed_chunk_driver_debug
tensorstore_cc_library(
    name = "kvs_backed_chunk_driver",
    srcs = [
        "kvs_backed_chunk_driver.cc",
    ],
    hdrs = [
        "kvs_backed_chunk_driver.h",
        "kvs_backed_chunk_driver_impl.h",
    ],
    local_defines = select({
        ":kvs_backed_chunk_driver_debug_setting": ["TENSORSTORE_KVS_DRIVER_DEBUG"],
        "//conditions:default": [],
    }),
    deps = [
        ":driver",
        "//tensorstore",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:open_mode",
        "//tensorstore:spec",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:box_difference",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:open_mode_spec",
        "//tensorstore/internal:staleness_bound_json_binder",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal:unowned_to_shared",
        "//tensorstore/internal/cache:aggregate_writeback_cache",
        "//tensorstore/internal/cache:async_cache",
        "//tensorstore/internal/cache:async_initialized_cache_mixin",
        "//tensorstore/internal/cache:cache_key",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/internal/cache:kvs_backed_cache",
        "//tensorstore/kvstore:key_value_store",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:bit_vec",
        "//tensorstore/util:future",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tensorstore_cc_test(
    name = "kvs_backed_chunk_driver_test",
    size = "small",
    srcs = ["kvs_backed_chunk_driver_test.cc"],
    deps = [
        ":kvs_backed_chunk_driver",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
