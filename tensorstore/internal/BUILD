load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

# To debug using a forced block size of 1, specify:
# bazel build --//tensorstore/internal:nditerable_test_unit_block_size
#
# This will significantly harm performance but is useful for exposing bugs in
# NDIterable implementations
# (e.g. https://github.com/google/tensorstore/issues/31).
bool_flag(
    name = "nditerable_test_unit_block_size",
    build_setting_default = False,
)

config_setting(
    name = "nditerable_test_unit_block_size_setting",
    flag_values = {
        ":nditerable_test_unit_block_size": "True",
    },
)

bool_flag(
    name = "grid_storage_statistics_debug",
    build_setting_default = False,
)

config_setting(
    name = "grid_storage_statistics_debug_setting",
    flag_values = {
        ":grid_storage_statistics_debug": "True",
    },
)

NDITERABLE_TEST_UNIT_BLOCK_SIZE_DEFINES = select({
    "nditerable_test_unit_block_size_setting": ["TENSORSTORE_INTERNAL_NDITERABLE_TEST_UNIT_BLOCK_SIZE"],
    "//conditions:default": [],
})

tensorstore_cc_library(
    name = "arena",
    hdrs = ["arena.h"],
    deps = [
        ":integer_overflow",
        "//tensorstore/internal/meta:exception_macros",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "arena_test",
    size = "small",
    srcs = ["arena_test.cc"],
    deps = [
        ":arena",
        "//tensorstore/util:span",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "ascii_set",
    hdrs = ["ascii_set.h"],
)

tensorstore_cc_library(
    name = "global_initializer",
    hdrs = ["global_initializer.h"],
    deps = ["//tensorstore/internal/preprocessor:cat"],
)

tensorstore_cc_test(
    name = "global_initializer_test",
    size = "small",
    srcs = ["global_initializer_test.cc"],
    deps = [
        ":global_initializer",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "box_difference",
    srcs = ["box_difference.cc"],
    hdrs = ["box_difference.h"],
    deps = [
        ":integer_overflow",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
    ],
)

tensorstore_cc_test(
    name = "box_difference_test",
    size = "small",
    srcs = ["box_difference_test.cc"],
    deps = [
        ":box_difference",
        "//tensorstore:box",
        "//tensorstore:index",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "chunk_grid_specification",
    srcs = ["chunk_grid_specification.cc"],
    hdrs = ["chunk_grid_specification.h"],
    deps = [
        ":async_write_array",
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore/internal/cache",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/container:inlined_vector",
    ],
)

tensorstore_cc_library(
    name = "concurrency_resource",
    srcs = ["concurrency_resource.cc"],
    hdrs = [
        "concurrency_resource.h",
        "concurrency_resource_provider.h",
    ],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/thread:thread_pool",
        "//tensorstore/util:executor",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "container_to_shared",
    hdrs = ["container_to_shared.h"],
)

tensorstore_cc_test(
    name = "container_to_shared_test",
    size = "small",
    srcs = ["container_to_shared_test.cc"],
    deps = [
        ":container_to_shared",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "context_binding",
    hdrs = [
        "context_binding.h",
        "context_binding_vector.h",
    ],
    deps = [
        "//tensorstore:context",
        "//tensorstore/util:status",
        "//tensorstore/util/apply_members",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "cord_util",
    srcs = ["cord_util.cc"],
    hdrs = ["cord_util.h"],
    deps = [
        "//tensorstore/util:span",
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "data_copy_concurrency_resource",
    srcs = ["data_copy_concurrency_resource.cc"],
    hdrs = ["data_copy_concurrency_resource.h"],
    deps = [
        ":concurrency_resource",
        "//tensorstore:context",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "data_type_endian_conversion",
    srcs = ["data_type_endian_conversion.cc"],
    hdrs = ["data_type_endian_conversion.h"],
    deps = [
        ":elementwise_function",
        ":unaligned_data_type_functions",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:strided_layout",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:endian",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_test(
    name = "data_type_endian_conversion_test",
    size = "small",
    srcs = ["data_type_endian_conversion_test.cc"],
    deps = [
        ":data_type_endian_conversion",
        ":flat_cord_builder",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/util:endian",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:cord_test_helpers",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "data_type_random_generator",
    testonly = True,
    srcs = ["data_type_random_generator.cc"],
    hdrs = ["data_type_random_generator.h"],
    deps = [
        ":elementwise_function",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/random:bit_gen_ref",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "unaligned_data_type_functions",
    srcs = ["unaligned_data_type_functions.cc"],
    hdrs = ["unaligned_data_type_functions.h"],
    deps = [
        ":elementwise_function",
        ":endian_elementwise_conversion",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/serialization",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "driver_kind_registry",
    srcs = ["driver_kind_registry.cc"],
    hdrs = ["driver_kind_registry.h"],
    deps = [
        "//tensorstore/util:span",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "element_copy_function",
    hdrs = ["element_copy_function.h"],
    deps = [
        ":elementwise_function",
        "//tensorstore:index",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "elementwise_function",
    hdrs = ["elementwise_function.h"],
    deps = [
        ":integer_overflow",
        "//tensorstore:index",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/internal/meta:void_wrapper",
        "//tensorstore/util:byte_strided_pointer",
        "@abseil-cpp//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "elementwise_function_test",
    size = "small",
    srcs = ["elementwise_function_test.cc"],
    deps = [
        ":elementwise_function",
        ":integer_overflow",
        "//tensorstore:index",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "endian_elementwise_conversion",
    hdrs = ["endian_elementwise_conversion.h"],
    deps = [
        ":elementwise_function",
        "//tensorstore:index",
        "//tensorstore/internal/riegeli:delimited",
        "//tensorstore/internal/riegeli:json_input",
        "//tensorstore/internal/riegeli:json_output",
        "//tensorstore/util:endian",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:utf8_string",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
        "@riegeli//riegeli/bytes:reader",
        "@riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_library(
    name = "env",
    srcs = ["env.cc"],
    hdrs = ["env.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:marshalling",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "env_test",
    size = "small",
    srcs = ["env_test.cc"],
    deps = [
        ":env",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "file_io_concurrency_resource",
    srcs = ["file_io_concurrency_resource.cc"],
    hdrs = ["file_io_concurrency_resource.h"],
    deps = [
        ":concurrency_resource",
        "//tensorstore:context",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "flat_cord_builder",
    hdrs = ["flat_cord_builder.h"],
    deps = [
        "//tensorstore/internal/os:memory_region",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "gdb_scripting",
    hdrs = ["gdb_scripting.h"],
    deps = ["@abseil-cpp//absl/base:core_headers"],
)

tensorstore_cc_library(
    name = "grid_chunk_key_ranges",
    srcs = ["grid_chunk_key_ranges.cc"],
    hdrs = ["grid_chunk_key_ranges.h"],
    deps = [
        ":grid_partition",
        ":grid_partition_impl",
        ":lexicographical_grid_index_key",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "grid_chunk_key_ranges_test",
    size = "small",
    srcs = ["grid_chunk_key_ranges_test.cc"],
    deps = [
        ":grid_chunk_key_ranges",
        ":grid_chunk_key_ranges_base10",
        ":grid_partition",
        ":grid_partition_impl",
        ":regular_grid",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:division",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "grid_chunk_key_ranges_base10_test",
    size = "small",
    srcs = ["grid_chunk_key_ranges_base10_test.cc"],
    deps = [
        ":grid_chunk_key_ranges_base10",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "grid_partition",
    srcs = [
        "grid_partition.cc",
        "grid_partition_iterator.cc",
    ],
    hdrs = [
        "grid_partition.h",
        "grid_partition_iterator.h",
    ],
    deps = [
        ":grid_partition_impl",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:output_index_method",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "grid_partition_impl",
    srcs = ["grid_partition_impl.cc"],
    hdrs = [
        "grid_partition_impl.h",
    ],
    deps = [
        ":integer_overflow",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:output_index_method",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:iterate",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "grid_partition_impl_test",
    size = "small",
    srcs = ["grid_partition_impl_test.cc"],
    deps = [
        ":grid_partition_impl",
        ":irregular_grid",
        ":regular_grid",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "grid_partition_test",
    size = "small",
    srcs = ["grid_partition_test.cc"],
    deps = [
        ":grid_partition",
        ":grid_partition_impl",
        ":irregular_grid",
        ":regular_grid",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "storage_statistics",
    srcs = ["storage_statistics.cc"],
    hdrs = ["storage_statistics.h"],
    deps = [
        ":intrusive_ptr",
        "//tensorstore:array_storage_statistics",
        "//tensorstore/util:future",
    ],
)

tensorstore_cc_library(
    name = "grid_chunk_key_ranges_base10",
    srcs = ["grid_chunk_key_ranges_base10.cc"],
    hdrs = ["grid_chunk_key_ranges_base10.h"],
    deps = [
        ":lexicographical_grid_index_key",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "grid_storage_statistics",
    srcs = ["grid_storage_statistics.cc"],
    hdrs = ["grid_storage_statistics.h"],
    local_defines = select({
        ":grid_storage_statistics_debug_setting": ["TENSORSTORE_INTERNAL_GRID_STORAGE_STATISTICS_DEBUG"],
        "//conditions:default": [],
    }),
    deps = [
        ":grid_chunk_key_ranges",
        ":grid_chunk_key_ranges_base10",
        ":grid_partition_impl",
        ":integer_overflow",
        ":intrusive_ptr",
        ":lexicographical_grid_index_key",
        ":regular_grid",
        ":storage_statistics",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:division",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_library(
    name = "integer_overflow",
    hdrs = ["integer_overflow.h"],
    deps = ["@abseil-cpp//absl/base:core_headers"],
)

tensorstore_cc_test(
    name = "integer_overflow_test",
    size = "small",
    srcs = ["integer_overflow_test.cc"],
    deps = [
        ":integer_overflow",
        "//tensorstore:index",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "intrusive_ptr",
    hdrs = ["intrusive_ptr.h"],
    deps = [
        ":memory",
        "//tensorstore/internal/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "intrusive_ptr_test",
    size = "small",
    srcs = ["intrusive_ptr_test.cc"],
    deps = [
        ":intrusive_ptr",
        ":memory",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "irregular_grid",
    srcs = ["irregular_grid.cc"],
    hdrs = ["irregular_grid.h"],
    deps = [
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/container:inlined_vector",
    ],
)

tensorstore_cc_test(
    name = "irregular_grid_test",
    srcs = ["irregular_grid_test.cc"],
    deps = [
        ":grid_partition",
        ":grid_partition_impl",
        ":irregular_grid",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:span",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "json_pointer",
    srcs = ["json_pointer.cc"],
    hdrs = ["json_pointer.h"],
    deps = [
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "json_pointer_test",
    size = "small",
    srcs = ["json_pointer_test.cc"],
    deps = [
        ":json_pointer",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "json_registry",
    srcs = [
        "json_registry_impl.cc",
        "json_registry_impl.h",
    ],
    hdrs = [
        "json_registry.h",
        "json_registry_fwd.h",
    ],
    deps = [
        ":intrusive_ptr",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal/container:heterogeneous_container",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/poly",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "json_registry_test",
    size = "small",
    srcs = ["json_registry_test.cc"],
    deps = [
        ":intrusive_ptr",
        ":json_registry",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "lock_collection",
    srcs = ["lock_collection.cc"],
    hdrs = ["lock_collection.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "lock_collection_test",
    size = "small",
    srcs = ["lock_collection_test.cc"],
    deps = [
        ":lock_collection",
        "//tensorstore/internal/testing:concurrent",
        "@abseil-cpp//absl/algorithm:container",
        "@abseil-cpp//absl/synchronization",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "masked_array",
    srcs = ["masked_array.cc"],
    hdrs = ["masked_array.h"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":integer_overflow",
        ":memory",
        ":nditerable",
        ":nditerable_buffer_management",
        ":nditerable_transformed_array",
        ":unowned_to_shared",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "masked_array_test",
    size = "small",
    srcs = ["masked_array_test.cc"],
    deps = [
        ":element_copy_function",
        ":elementwise_function",
        ":masked_array",
        ":masked_array_testutil",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "async_write_array",
    srcs = ["async_write_array.cc"],
    hdrs = ["async_write_array.h"],
    deps = [
        ":arena",
        ":integer_overflow",
        ":masked_array",
        ":memory",
        ":nditerable",
        ":nditerable_transformed_array",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:output_index_method",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:extents",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "async_write_array_test",
    size = "small",
    srcs = ["async_write_array_test.cc"],
    deps = [
        ":arena",
        ":async_write_array",
        ":nditerable",
        ":nditerable_array",
        ":nditerable_copy",
        "//tensorstore:array",
        "//tensorstore:array_testutil",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/internal/testing:random_seed",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "lexicographical_grid_index_key",
    hdrs = ["lexicographical_grid_index_key.h"],
    deps = [
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_library(
    name = "masked_array_testutil",
    testonly = 1,
    srcs = ["masked_array_testutil.cc"],
    hdrs = ["masked_array_testutil.h"],
    deps = [
        ":arena",
        ":element_copy_function",
        ":masked_array",
        ":nditerable",
        ":nditerable_copy",
        ":nditerable_elementwise_input_transform",
        ":nditerable_transformed_array",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "memory",
    hdrs = ["memory.h"],
)

tensorstore_cc_library(
    name = "mutex",
    hdrs = ["mutex.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "nditerable",
    srcs = ["nditerable.cc"],
    hdrs = ["nditerable.h"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":unique_with_intrusive_allocator",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "nditerable_array",
    srcs = ["nditerable_array.cc"],
    hdrs = ["nditerable_array.h"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":integer_overflow",
        ":nditerable",
        ":nditerable_array_util",
        ":nditerable_util",
        ":unique_with_intrusive_allocator",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_array_test",
    size = "small",
    srcs = ["nditerable_array_test.cc"],
    local_defines = NDITERABLE_TEST_UNIT_BLOCK_SIZE_DEFINES,
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_array",
        ":nditerable_buffer_management",
        ":nditerable_util",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "nditerable_array_util",
    hdrs = ["nditerable_array_util.h"],
    deps = [
        ":integer_overflow",
        ":nditerable",
        "//tensorstore:index",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_library(
    name = "nditerable_buffer_management",
    hdrs = ["nditerable_buffer_management.h"],
    deps = [
        ":arena",
        ":element_copy_function",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_util",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/util:division",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "nditerable_copy",
    srcs = ["nditerable_copy.cc"],
    hdrs = ["nditerable_copy.h"],
    deps = [
        ":arena",
        ":element_copy_function",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_buffer_management",
        ":nditerable_util",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_copy_test",
    size = "small",
    srcs = ["nditerable_copy_test.cc"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_copy",
        ":nditerable_elementwise_input_transform",
        ":nditerable_elementwise_output_transform",
        ":nditerable_transformed_array",
        ":nditerable_util",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "nditerable_copy_benchmark_test",
    size = "small",
    srcs = ["nditerable_copy_benchmark_test.cc"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable_array",
        ":nditerable_copy",
        ":nditerable_transformed_array",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@google_benchmark//:benchmark_main",
    ],
)

tensorstore_cc_library(
    name = "nditerable_data_type_conversion",
    srcs = ["nditerable_data_type_conversion.cc"],
    hdrs = ["nditerable_data_type_conversion.h"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_elementwise_input_transform",
        ":nditerable_elementwise_output_transform",
        ":nditerable_util",
        ":unique_with_intrusive_allocator",
        "//tensorstore:data_type",
        "//tensorstore:index",
    ],
)

tensorstore_cc_test(
    name = "nditerable_data_type_conversion_test",
    size = "small",
    srcs = ["nditerable_data_type_conversion_test.cc"],
    deps = [
        ":arena",
        ":nditerable",
        ":nditerable_array",
        ":nditerable_copy",
        ":nditerable_data_type_conversion",
        ":nditerable_transformed_array",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "nditerable_elementwise_input_transform",
    srcs = ["nditerable_elementwise_input_transform.cc"],
    hdrs = ["nditerable_elementwise_input_transform.h"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_buffer_management",
        ":nditerable_util",
        ":unique_with_intrusive_allocator",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_elementwise_input_transform_test",
    size = "small",
    srcs = ["nditerable_elementwise_input_transform_test.cc"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable_copy",
        ":nditerable_elementwise_input_transform",
        ":nditerable_transformed_array",
        ":nditerable_util",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "nditerable_elementwise_output_transform",
    srcs = ["nditerable_elementwise_output_transform.cc"],
    hdrs = ["nditerable_elementwise_output_transform.h"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_buffer_management",
        ":nditerable_util",
        ":unique_with_intrusive_allocator",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_elementwise_output_transform_test",
    size = "small",
    srcs = ["nditerable_elementwise_output_transform_test.cc"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable_copy",
        ":nditerable_elementwise_output_transform",
        ":nditerable_transformed_array",
        ":nditerable_util",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "nditerable_transformed_array",
    srcs = ["nditerable_transformed_array.cc"],
    hdrs = ["nditerable_transformed_array.h"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":integer_overflow",
        ":nditerable",
        ":nditerable_array",
        ":nditerable_array_util",
        ":nditerable_util",
        ":unique_with_intrusive_allocator",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_transformed_array_test",
    size = "small",
    srcs = ["nditerable_transformed_array_test.cc"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":nditerable",
        ":nditerable_buffer_management",
        ":nditerable_transformed_array",
        "//tensorstore:array",
        "//tensorstore:array_testutil",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "nditerable_util",
    srcs = ["nditerable_util.cc"],
    hdrs = ["nditerable_util.h"],
    local_defines = NDITERABLE_TEST_UNIT_BLOCK_SIZE_DEFINES,
    deps = [
        ":arena",
        ":elementwise_function",
        ":integer_overflow",
        ":nditerable",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_util_test",
    size = "small",
    srcs = ["nditerable_util_test.cc"],
    local_defines = NDITERABLE_TEST_UNIT_BLOCK_SIZE_DEFINES,
    deps = [
        ":nditerable_util",
        "//tensorstore:index",
        "//tensorstore/util:span",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "path",
    srcs = ["path.cc"],
    hdrs = ["path.h"],
    deps = [
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "path_test",
    size = "small",
    srcs = ["path_test.cc"],
    deps = [
        ":path",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "regular_grid",
    hdrs = ["regular_grid.h"],
    deps = [
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/util:division",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_library(
    name = "retry",
    srcs = ["retry.cc"],
    hdrs = ["retry.h"],
    deps = [
        "@abseil-cpp//absl/random",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "retry_test",
    size = "small",
    srcs = ["retry_test.cc"],
    deps = [
        ":retry",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "retries_context_resource",
    hdrs = ["retries_context_resource.h"],
    deps = [
        ":retry",
        "//tensorstore:context",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = ["@abseil-cpp//absl/base:config"],
)

tensorstore_cc_test(
    name = "source_location_test",
    size = "small",
    srcs = ["source_location_test.cc"],
    deps = [
        ":source_location",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "string_like",
    hdrs = ["string_like.h"],
    deps = [
        "//tensorstore/util:span",
        "@abseil-cpp//absl/base:core_headers",
    ],
)

tensorstore_cc_test(
    name = "string_like_test",
    size = "small",
    srcs = ["string_like_test.cc"],
    deps = [
        ":string_like",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "tagged_ptr",
    hdrs = ["tagged_ptr.h"],
)

tensorstore_cc_test(
    name = "tagged_ptr_test",
    size = "small",
    srcs = ["tagged_ptr_test.cc"],
    deps = [
        ":intrusive_ptr",
        ":memory",
        ":tagged_ptr",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "unique_with_intrusive_allocator",
    hdrs = ["unique_with_intrusive_allocator.h"],
)

tensorstore_cc_test(
    name = "unique_with_intrusive_allocator_test",
    size = "small",
    srcs = ["unique_with_intrusive_allocator_test.cc"],
    deps = [
        ":arena",
        ":unique_with_intrusive_allocator",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "unowned_to_shared",
    hdrs = ["unowned_to_shared.h"],
)

tensorstore_cc_library(
    name = "utf8",
    srcs = ["utf8.cc"],
    hdrs = ["utf8.h"],
)

tensorstore_cc_test(
    name = "utf8_test",
    size = "small",
    srcs = ["utf8_test.cc"],
    deps = [
        ":utf8",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "open_mode_spec",
    srcs = ["open_mode_spec.cc"],
    hdrs = ["open_mode_spec.h"],
    deps = [
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
    ],
)

tensorstore_cc_library(
    name = "dimension_labels",
    srcs = ["dimension_labels.cc"],
    hdrs = ["dimension_labels.h"],
    deps = [
        "//tensorstore:rank",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/container:fixed_array",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "dimension_labels_test",
    size = "small",
    srcs = ["dimension_labels_test.cc"],
    deps = [
        ":dimension_labels",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "json_metadata_matching",
    srcs = ["json_metadata_matching.cc"],
    hdrs = ["json_metadata_matching.h"],
    deps = [
        "//tensorstore/internal/json:same",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "multi_barrier",
    srcs = ["multi_barrier.cc"],
    hdrs = ["multi_barrier.h"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "multi_barrier_test",
    srcs = ["multi_barrier_test.cc"],
    deps = [
        ":multi_barrier",
        "//tensorstore/internal/thread",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "ref_counted_string",
    srcs = ["ref_counted_string.cc"],
    hdrs = ["ref_counted_string.h"],
)

tensorstore_cc_test(
    name = "ref_counted_string_test",
    size = "small",
    srcs = ["ref_counted_string_test.cc"],
    deps = [
        ":ref_counted_string",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "uri_utils",
    srcs = ["uri_utils.cc"],
    hdrs = ["uri_utils.h"],
    deps = [
        ":ascii_set",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "uri_utils_test",
    srcs = ["uri_utils_test.cc"],
    deps = [
        ":ascii_set",
        ":uri_utils",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "compare",
    hdrs = ["compare.h"],
    deps = ["@abseil-cpp//absl/types:compare"],
)

tensorstore_cc_test(
    name = "regular_grid_test",
    srcs = ["regular_grid_test.cc"],
    deps = [
        ":regular_grid",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "grid_partition_iterator_test",
    srcs = ["grid_partition_iterator_test.cc"],
    deps = [
        ":grid_partition",
        ":grid_partition_impl",
        ":regular_grid",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)
