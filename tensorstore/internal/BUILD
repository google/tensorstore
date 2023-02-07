load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(
    default_visibility = ["//tensorstore:internal_packages"],
)

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
    visibility = ["//tensorstore:internal_packages"],
)

NDITERABLE_TEST_UNIT_BLOCK_SIZE_DEFINES = select({
    "nditerable_test_unit_block_size_setting": ["TENSORSTORE_INTERNAL_NDITERABLE_TEST_UNIT_BLOCK_SIZE"],
    "//conditions:default": [],
})

tensorstore_cc_library(
    name = "arena",
    hdrs = ["arena.h"],
    deps = [
        ":exception_macros",
        ":integer_overflow",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "attributes",
    hdrs = ["attributes.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tensorstore_cc_library(
    name = "bit_operations",
    hdrs = ["bit_operations.h"],
)

tensorstore_cc_test(
    name = "bit_operations_test",
    size = "small",
    srcs = ["bit_operations_test.cc"],
    deps = [
        ":bit_operations",
        "@com_google_googletest//:gtest_main",
    ],
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
        "@com_google_googletest//:gtest_main",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "compressed_pair",
    hdrs = ["compressed_pair.h"],
)

tensorstore_cc_test(
    name = "compressed_pair_test",
    size = "small",
    srcs = ["compressed_pair_test.cc"],
    deps = [
        ":compressed_pair",
        "@com_google_googletest//:gtest_main",
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
        ":thread_pool",
        ":type_traits",
        "//tensorstore:context",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:executor",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base",
    ],
)

tensorstore_cc_library(
    name = "concurrent_testutil",
    testonly = 1,
    srcs = ["concurrent_testutil.cc"],
    hdrs = ["concurrent_testutil.h"],
    deps = [
        ":multi_barrier",
        ":thread",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

tensorstore_cc_test(
    name = "concurrent_testutil_test",
    srcs = ["concurrent_testutil_test.cc"],
    deps = [
        ":concurrent_testutil",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "cord_util",
    srcs = ["cord_util.cc"],
    hdrs = ["cord_util.h"],
    deps = [
        "//tensorstore/util:span",
        "@com_google_absl//absl/strings:cord",
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
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
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
        "//tensorstore:strided_layout",
        "//tensorstore/util:endian",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:cord_test_helpers",
        "@com_google_googletest//:gtest_main",
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
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
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
        "//tensorstore/util:endian",
        "//tensorstore/util:status",
        "//tensorstore/util:utf8_string",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "decoded_matches",
    testonly = 1,
    srcs = ["decoded_matches.cc"],
    hdrs = ["decoded_matches.h"],
    deps = [
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "decoded_matches_test",
    size = "small",
    srcs = ["decoded_matches_test.cc"],
    deps = [
        ":decoded_matches",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "elementwise_function",
    hdrs = ["elementwise_function.h"],
    deps = [
        ":integer_overflow",
        ":type_traits",
        ":void_wrapper",
        "//tensorstore:index",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "endian_elementwise_conversion",
    hdrs = ["endian_elementwise_conversion.h"],
    deps = [
        ":elementwise_function",
        ":riegeli_json_input",
        ":riegeli_json_output",
        "//tensorstore:index",
        "//tensorstore/serialization:riegeli_delimited",
        "//tensorstore/util:endian",
        "//tensorstore/util:status",
        "//tensorstore/util:utf8_string",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "env",
    srcs = ["env.cc"],
    hdrs = ["env.h"],
)

tensorstore_cc_test(
    name = "env_test",
    size = "small",
    srcs = ["env_test.cc"],
    deps = [
        ":env",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "exception_macros",
    hdrs = ["exception_macros.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
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
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
    ],
)

tensorstore_cc_library(
    name = "gdb_scripting",
    hdrs = ["gdb_scripting.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tensorstore_cc_library(
    name = "grid_partition",
    srcs = ["grid_partition.cc"],
    hdrs = [
        "grid_partition.h",
    ],
    deps = [
        ":grid_partition_impl",
        ":intrusive_ptr",
        ":memory",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:division",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
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
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:division",
        "//tensorstore/util:iterate",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "grid_partition_impl_test",
    size = "small",
    srcs = ["grid_partition_impl_test.cc"],
    deps = [
        ":grid_partition_impl",
        ":irregular_grid",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "grid_partition_test",
    size = "small",
    srcs = ["grid_partition_test.cc"],
    deps = [
        ":grid_partition",
        ":irregular_grid",
        ":memory",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "half_gtest",
    testonly = 1,
    hdrs = ["half_gtest.h"],
    deps = ["@net_sourceforge_half//:half"],
)

tensorstore_cc_library(
    name = "init_tensorstore",
    srcs = ["init_tensorstore.cc"],
    hdrs = ["init_tensorstore.h"],
    deps = ["@com_google_absl//absl/flags:parse"],
)

tensorstore_cc_library(
    name = "heterogeneous_container",
    hdrs = ["heterogeneous_container.h"],
    deps = ["@com_google_absl//absl/container:flat_hash_set"],
)

tensorstore_cc_test(
    name = "heterogeneous_container_test",
    size = "small",
    srcs = ["heterogeneous_container_test.cc"],
    deps = [
        ":heterogeneous_container",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "integer_overflow",
    hdrs = ["integer_overflow.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tensorstore_cc_test(
    name = "integer_overflow_test",
    size = "small",
    srcs = ["integer_overflow_test.cc"],
    deps = [
        ":integer_overflow",
        "//tensorstore:index",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "integer_types",
    hdrs = ["integer_types.h"],
)

tensorstore_cc_library(
    name = "intrusive_linked_list",
    hdrs = ["intrusive_linked_list.h"],
)

tensorstore_cc_test(
    name = "intrusive_linked_list_test",
    size = "small",
    srcs = ["intrusive_linked_list_test.cc"],
    deps = [
        ":intrusive_linked_list",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "intrusive_ptr",
    hdrs = ["intrusive_ptr.h"],
    deps = [
        ":memory",
        ":type_traits",
    ],
)

tensorstore_cc_test(
    name = "intrusive_ptr_test",
    size = "small",
    srcs = ["intrusive_ptr_test.cc"],
    deps = [
        ":intrusive_ptr",
        ":memory",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "intrusive_red_black_tree",
    srcs = ["intrusive_red_black_tree.cc"],
    hdrs = ["intrusive_red_black_tree.h"],
    deps = [":tagged_ptr"],
)

tensorstore_cc_test(
    name = "intrusive_red_black_tree_test",
    size = "small",
    srcs = ["intrusive_red_black_tree_test.cc"],
    deps = [
        ":intrusive_red_black_tree",
        "@com_google_absl//absl/random",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/container:inlined_vector",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "json_fwd",
    hdrs = ["json_fwd.h"],
)

tensorstore_cc_library(
    name = "json_gtest",
    testonly = 1,
    srcs = ["json_gtest.cc"],
    hdrs = ["json_gtest.h"],
    deps = [
        "//tensorstore/internal/json",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest",
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
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "json_pointer_test",
    size = "small",
    srcs = ["json_pointer_test.cc"],
    deps = [
        ":json_gtest",
        ":json_pointer",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
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
        ":heterogeneous_container",
        ":intrusive_ptr",
        ":json_fwd",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/poly",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "json_registry_test",
    size = "small",
    srcs = ["json_registry_test.cc"],
    deps = [
        ":intrusive_ptr",
        ":json_gtest",
        ":json_registry",
        ":no_destructor",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "lock_collection",
    srcs = ["lock_collection.cc"],
    hdrs = ["lock_collection.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "lock_collection_test",
    size = "small",
    srcs = ["lock_collection_test.cc"],
    deps = [
        ":concurrent_testutil",
        ":lock_collection",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
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
        ":nditerable_transformed_array",
        ":nditerable_util",
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
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
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
        ":memory",
        ":meta",
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
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "async_write_array",
    srcs = ["async_write_array.cc"],
    hdrs = ["async_write_array.h"],
    deps = [
        ":masked_array",
        ":nditerable",
        ":nditerable_transformed_array",
        ":unowned_to_shared",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "async_write_array_test",
    size = "small",
    srcs = ["async_write_array_test.cc"],
    deps = [
        ":arena",
        ":async_write_array",
        ":nditerable_array",
        ":nditerable_copy",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:strided_layout",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
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
        ":memory",
        ":nditerable",
        ":nditerable_copy",
        ":nditerable_elementwise_input_transform",
        ":nditerable_transformed_array",
        ":nditerable_util",
        ":unowned_to_shared",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "memory",
    hdrs = ["memory.h"],
    deps = [":exception_macros"],
)

tensorstore_cc_library(
    name = "meta",
    hdrs = ["meta.h"],
)

tensorstore_cc_test(
    name = "meta_test",
    size = "small",
    srcs = ["meta_test.cc"],
    deps = [
        ":meta",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "multi_vector",
    hdrs = [
        "multi_vector.h",
        "multi_vector_impl.h",
    ],
    deps = [
        ":gdb_scripting",
        ":meta",
        ":type_traits",
        "//tensorstore:rank",
        "//tensorstore/util:division",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "multi_vector_test",
    size = "small",
    srcs = ["multi_vector_test.cc"],
    deps = [
        ":multi_vector",
        "//tensorstore:rank",
        "//tensorstore/util:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "multi_vector_view",
    hdrs = ["multi_vector_view.h"],
    deps = [
        ":gdb_scripting",
        ":meta",
        ":type_traits",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "multi_vector_view_test",
    size = "small",
    srcs = ["multi_vector_view_test.cc"],
    deps = [
        ":multi_vector_view",
        "//tensorstore:rank",
        "//tensorstore/util:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "mutex",
    hdrs = ["mutex.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
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
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
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
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
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
        "@com_google_googletest//:gtest_main",
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
        ":elementwise_function",
        ":nditerable",
        ":nditerable_util",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/util:division",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
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
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_copy_test",
    size = "small",
    srcs = ["nditerable_copy_test.cc"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":memory",
        ":meta",
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
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
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
        ":memory",
        ":nditerable",
        ":nditerable_array",
        ":nditerable_copy",
        ":nditerable_data_type_conversion",
        ":nditerable_transformed_array",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/status",
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
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "nditerable_transformed_array_test",
    size = "small",
    srcs = ["nditerable_transformed_array_test.cc"],
    deps = [
        ":arena",
        ":elementwise_function",
        ":memory",
        ":meta",
        ":nditerable",
        ":nditerable_buffer_management",
        ":nditerable_transformed_array",
        ":nditerable_util",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:inlined_vector",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "no_destructor",
    hdrs = ["no_destructor.h"],
)

tensorstore_cc_test(
    name = "no_destructor_test",
    size = "small",
    srcs = ["no_destructor_test.cc"],
    deps = [
        ":no_destructor",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "non_compile_bypass",
    testonly = True,
    hdrs = ["non_compile_bypass.h"],
)

tensorstore_cc_library(
    name = "os_error_code",
    srcs = ["os_error_code.cc"],
    hdrs = ["os_error_code.h"],
    tags = ["msvc"],
    deps = [
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "parse_json_matches",
    testonly = 1,
    srcs = ["parse_json_matches.cc"],
    hdrs = ["parse_json_matches.h"],
    deps = [
        ":json_gtest",
        "//tensorstore/internal/json_binding",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "parse_json_matches_test",
    size = "small",
    srcs = ["parse_json_matches_test.cc"],
    deps = [
        ":parse_json_matches",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "path",
    srcs = [
        "path.cc",
    ],
    hdrs = [
        "path.h",
    ],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "path_test",
    size = "small",
    srcs = ["path_test.cc"],
    deps = [
        ":path",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "queue_testutil",
    testonly = 1,
    hdrs = ["queue_testutil.h"],
    deps = [
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "retry",
    srcs = [
        "retry.cc",
    ],
    hdrs = [
        "retry.h",
    ],
    deps = [
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "retry_test",
    size = "small",
    srcs = ["retry_test.cc"],
    deps = [
        ":retry",
        "//tensorstore/util:status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "retries_context_resource",
    hdrs = ["retries_context_resource.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
    ],
)

tensorstore_cc_library(
    name = "riegeli_json_input",
    srcs = ["riegeli_json_input.cc"],
    hdrs = ["riegeli_json_input.h"],
    deps = [
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_riegeli//riegeli/bytes:reader",
    ],
)

tensorstore_cc_library(
    name = "riegeli_json_output",
    srcs = ["riegeli_json_output.cc"],
    hdrs = ["riegeli_json_output.h"],
    deps = [
        ":unowned_to_shared",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_test(
    name = "riegeli_json_test",
    size = "small",
    srcs = ["riegeli_json_test.cc"],
    deps = [
        ":json_gtest",
        ":riegeli_json_input",
        ":riegeli_json_output",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@com_google_riegeli//riegeli/bytes:string_writer",
    ],
)

tensorstore_cc_library(
    name = "source_location",
    hdrs = ["source_location.h"],
    deps = ["@com_google_absl//absl/base:config"],
)

tensorstore_cc_test(
    name = "source_location_test",
    size = "small",
    srcs = ["source_location_test.cc"],
    deps = [
        ":source_location",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "string_like",
    hdrs = ["string_like.h"],
    deps = [
        "//tensorstore/util:span",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tensorstore_cc_test(
    name = "string_like_test",
    size = "small",
    srcs = ["string_like_test.cc"],
    deps = [
        ":string_like",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "subprocess",
    testonly = True,
    srcs = ["subprocess.cc"],
    hdrs = ["subprocess.h"],
    tags = ["msvc"],
    deps = [
        ":os_error_code",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "subprocess_test",
    srcs = ["subprocess_test.cc"],
    tags = ["msvc"],
    deps = [
        ":subprocess",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "test_util",
    testonly = True,
    srcs = [
        "test_util.cc",
    ],
    hdrs = [
        "test_util.h",
    ],
    deps = [
        ":env",
        ":os_error_code",
        ":path",
        ":source_location",
        "//tensorstore/kvstore/file:file_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "thread",
    srcs = ["thread.cc"],
    hdrs = ["thread.h"],
    deps = ["@com_google_absl//absl/log:absl_check"],
)

tensorstore_cc_test(
    name = "thread_test",
    srcs = ["thread_test.cc"],
    deps = [
        ":thread",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "thread_pool",
    srcs = ["thread_pool.cc"],
    hdrs = ["thread_pool.h"],
    deps = [
        ":intrusive_ptr",
        ":mutex",
        ":no_destructor",
        ":thread",
        "//tensorstore/internal/metrics",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "thread_pool_test",
    size = "small",
    deps = [
        ":thread_pool",
        ":thread_pool_test_lib",
    ],
)

tensorstore_cc_library(
    name = "thread_pool_test_lib",
    testonly = True,
    srcs = ["thread_pool_test.cc"],
    deps = [
        "//tensorstore/util:executor",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "type_traits",
    hdrs = ["type_traits.h"],
    deps = ["//tensorstore:index"],
)

tensorstore_cc_test(
    name = "type_traits_test",
    size = "small",
    srcs = ["type_traits_test.cc"],
    deps = [
        ":type_traits",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_googletest//:gtest_main",  # buildcleaner: keep
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
        "@com_google_googletest//:gtest_main",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "void_wrapper",
    hdrs = ["void_wrapper.h"],
    deps = ["@com_google_absl//absl/status"],
)

tensorstore_cc_test(
    name = "void_wrapper_test",
    size = "small",
    srcs = ["void_wrapper_test.cc"],
    deps = [
        ":type_traits",
        ":void_wrapper",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/util:iterate",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "dimension_labels_test",
    size = "small",
    srcs = ["dimension_labels_test.cc"],
    deps = [
        ":dimension_labels",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "integer_range",
    hdrs = ["integer_range.h"],
)

tensorstore_cc_library(
    name = "json_metadata_matching",
    srcs = ["json_metadata_matching.cc"],
    hdrs = ["json_metadata_matching.h"],
    deps = [
        "//tensorstore/internal/json",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "schedule_at",
    srcs = ["schedule_at.cc"],
    hdrs = ["schedule_at.h"],
    deps = [
        ":attributes",
        ":no_destructor",
        ":type_traits",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/poly",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "schedule_at_test",
    srcs = ["schedule_at_test.cc"],
    deps = [
        ":schedule_at",
        "//tensorstore/util:executor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "multi_barrier",
    srcs = ["multi_barrier.cc"],
    hdrs = ["multi_barrier.h"],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "multi_barrier_test",
    srcs = ["multi_barrier_test.cc"],
    deps = [
        ":multi_barrier",
        ":thread",
        "@com_google_googletest//:gtest_main",
    ],
)
