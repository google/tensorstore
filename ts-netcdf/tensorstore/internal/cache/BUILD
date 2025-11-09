load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

bool_flag(
    name = "refcount_debug",
    build_setting_default = False,
)

bool_flag(
    name = "async_cache_debug",
    build_setting_default = False,
)

bool_flag(
    name = "chunk_cache_debug",
    build_setting_default = False,
)

config_setting(
    name = "refcount_debug_setting",
    flag_values = {
        ":refcount_debug": "True",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "async_cache_debug_setting",
    flag_values = {
        ":async_cache_debug": "True",
    },
    visibility = ["//visibility:private"],
)

config_setting(
    name = "chunk_cache_debug_setting",
    flag_values = {
        ":chunk_cache_debug": "True",
    },
    visibility = ["//visibility:private"],
)

# To enable debug logging, specify:
# bazel build --//tensorstore/internal/cache:async_cache_debug
tensorstore_cc_library(
    name = "async_cache",
    srcs = ["async_cache.cc"],
    hdrs = ["async_cache.h"],
    defines = select({
        ":async_cache_debug_setting": ["TENSORSTORE_ASYNC_CACHE_DEBUG"],
        "//conditions:default": [],
    }),
    deps = [
        ":cache",
        "//tensorstore:batch",
        "//tensorstore:transaction",
        "//tensorstore/internal:compare",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal/container:intrusive_linked_list",
        "//tensorstore/internal/container:intrusive_red_black_tree",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution:any_receiver",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "async_cache_test",
    size = "small",
    srcs = ["async_cache_test.cc"],
    deps = [
        ":async_cache",
        ":cache",
        "//tensorstore:transaction",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/testing:concurrent",
        "//tensorstore/internal/testing:queue_testutil",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "async_initialized_cache_mixin",
    hdrs = ["async_initialized_cache_mixin.h"],
    deps = [
        ":cache",
        "//tensorstore/util:future",
    ],
)

tensorstore_cc_library(
    name = "kvs_backed_cache",
    srcs = ["kvs_backed_cache.cc"],
    hdrs = ["kvs_backed_cache.h"],
    defines = select({
        ":async_cache_debug_setting": ["TENSORSTORE_ASYNC_CACHE_DEBUG"],
        "//conditions:default": [],
    }),
    deps = [
        ":async_cache",
        "//tensorstore:transaction",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:future",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:future_sender",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "kvs_backed_cache_test",
    size = "small",
    srcs = ["kvs_backed_cache_test.cc"],
    deps = [
        ":cache",
        ":kvs_backed_cache_testutil",
        "//tensorstore:transaction",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "kvs_backed_cache_testutil",
    testonly = True,
    srcs = ["kvs_backed_cache_testutil.cc"],
    hdrs = ["kvs_backed_cache_testutil.h"],
    deps = [
        ":async_cache",
        ":cache",
        ":kvs_backed_cache",
        "//tensorstore:transaction",
        "//tensorstore/internal/testing:dynamic",
        "//tensorstore/internal/testing:random_seed",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest",
        "@re2",
    ],
)

tensorstore_cc_library(
    name = "kvs_backed_chunk_cache",
    srcs = ["kvs_backed_chunk_cache.cc"],
    hdrs = ["kvs_backed_chunk_cache.h"],
    deps = [
        ":async_cache",
        ":cache",
        ":chunk_cache",
        ":kvs_backed_cache",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/internal:memory",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/tracing",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "aggregate_writeback_cache",
    hdrs = ["aggregate_writeback_cache.h"],
    deps = [
        ":async_cache",
        ":cache",
        "//tensorstore/internal/estimate_heap_usage",
    ],
)

tensorstore_cc_library(
    name = "cache",
    srcs = [
        "cache.cc",
        "cache_impl.h",
    ],
    hdrs = [
        "cache.h",
        "cache_pool_limits.h",
    ],
    defines = select({
        ":refcount_debug_setting": ["TENSORSTORE_CACHE_REFCOUNT_DEBUG"],
        "//conditions:default": [],
    }),
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal/container:heterogeneous_container",
        "//tensorstore/internal/container:intrusive_linked_list",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "cache_pool_resource",
    srcs = ["cache_pool_resource.cc"],
    hdrs = ["cache_pool_resource.h"],
    deps = [
        ":cache",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@nlohmann_json//:json",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "cache_pool_resource_test",
    size = "small",
    srcs = ["cache_pool_resource_test.cc"],
    deps = [
        ":cache",
        ":cache_pool_resource",
        "//tensorstore:context",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "cache_test",
    size = "small",
    timeout = "moderate",
    srcs = [
        "cache_test.cc",
    ],
    deps = [
        ":cache",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal/testing:concurrent",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "chunk_cache",
    srcs = ["chunk_cache.cc"],
    hdrs = ["chunk_cache.h"],
    local_defines = select({
        ":chunk_cache_debug_setting": ["TENSORSTORE_INTERNAL_CHUNK_CACHE_DEBUG"],
        "//conditions:default": [],
    }),
    deps = [
        ":async_cache",
        ":cache",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:read_write_options",
        "//tensorstore:transaction",
        "//tensorstore/driver:chunk",
        "//tensorstore/driver:chunk_receiver_utils",
        "//tensorstore/driver:read_request",
        "//tensorstore/driver:write_request",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:arena",
        "//tensorstore/internal:async_write_array",
        "//tensorstore/internal:chunk_grid_specification",
        "//tensorstore/internal:grid_partition",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:lock_collection",
        "//tensorstore/internal:memory",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:nditerable",
        "//tensorstore/internal:regular_grid",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_binary(
    name = "chunk_cache_benchmark_test",
    testonly = 1,
    srcs = ["chunk_cache_benchmark_test.cc"],
    tags = ["benchmark"],
    deps = [
        ":async_cache",
        ":cache",
        ":chunk_cache",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:progress",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk_cache_driver",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:async_write_array",
        "//tensorstore/internal:chunk_grid_specification",
        "//tensorstore/internal:element_copy_function",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:memory",
        "//tensorstore/internal/thread:thread_pool",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@google_benchmark//:benchmark_main",
    ],
)

tensorstore_cc_test(
    name = "chunk_cache_test",
    size = "small",
    srcs = ["chunk_cache_test.cc"],
    deps = [
        ":async_cache",
        ":cache",
        ":chunk_cache",
        ":kvs_backed_cache",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:context",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open_mode",
        "//tensorstore:progress",
        "//tensorstore:read_write_options",
        "//tensorstore:staleness_bound",
        "//tensorstore:strided_layout",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk",
        "//tensorstore/driver:chunk_cache_driver",
        "//tensorstore/driver:driver_testutil",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal:async_write_array",
        "//tensorstore/internal:chunk_grid_specification",
        "//tensorstore/internal:element_copy_function",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:memory",
        "//tensorstore/internal/riegeli:array_endian_codec",
        "//tensorstore/internal/testing:queue_testutil",
        "//tensorstore/internal/thread:thread_pool",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:mock_kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:endian",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@riegeli//riegeli/bytes:cord_reader",
        "@riegeli//riegeli/bytes:cord_writer",
    ],
)
