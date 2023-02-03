load("//bazel:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")
load("//bazel:non_compile.bzl", "cc_with_non_compile_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "add_new_dims_op",
    srcs = ["internal/add_new_dims_op.cc"],
    hdrs = ["internal/add_new_dims_op.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore:rank",
        "//tensorstore/internal:string_like",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "add_new_dims_op_test",
    size = "small",
    srcs = ["add_new_dims_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "alignment",
    srcs = ["alignment.cc"],
    hdrs = ["alignment.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":index_transform",
        "//tensorstore:index",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "alignment_test",
    size = "small",
    srcs = ["alignment_test.cc"],
    deps = [
        ":alignment",
        ":index_transform",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "compose_transforms_test",
    size = "small",
    srcs = ["compose_transforms_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "deep_copy_transform_rep_ptr_test",
    size = "small",
    srcs = ["deep_copy_transform_rep_ptr_test.cc"],
    deps = [
        ":index_transform",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "diagonal_op",
    srcs = ["internal/diagonal_op.cc"],
    hdrs = ["internal/diagonal_op.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore:rank",
        "//tensorstore/internal:string_like",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "diagonal_op_test",
    size = "small",
    srcs = ["diagonal_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dim_expression",
    srcs = ["internal/dim_expression_helper.h"],
    hdrs = ["dim_expression.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":add_new_dims_op",
        ":diagonal_op",
        ":dimension_index_buffer",
        ":dimension_selection",
        ":index_array_slice_op",
        ":index_transform",
        ":index_vector_or_scalar",
        ":interval_slice_op",
        ":label_op",
        ":mark_explicit_op",
        ":single_index_slice_op",
        ":translate_op",
        ":transpose_op",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/internal:meta",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
    ],
)

cc_with_non_compile_test(
    name = "dim_expression_nc_test",
    srcs = ["dim_expression_nc_test.cc"],
    nc_test_shard_count = 1,
    deps = [
        ":dim_expression",
        ":index_transform",
        "//tensorstore:index",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "dim_expression_test",
    srcs = ["dim_expression_test.cc"],
    deps = [
        ":dim_expression",
        ":index_transform",
        ":transformed_array",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dim_expression_testutil",
    testonly = 1,
    hdrs = ["internal/dim_expression_testutil.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "dimension_identifier",
    srcs = ["dimension_identifier.cc"],
    hdrs = ["dimension_identifier.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":dimension_index_buffer",
        "//tensorstore:index",
        "//tensorstore/util:division",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "dimension_identifier_test",
    size = "small",
    srcs = ["dimension_identifier_test.cc"],
    deps = [
        ":dimension_identifier",
        "//tensorstore:index",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dimension_index_buffer",
    hdrs = ["dimension_index_buffer.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:iterate",
        "@com_google_absl//absl/container:inlined_vector",
    ],
)

tensorstore_cc_library(
    name = "dimension_selection",
    srcs = ["internal/dimension_selection.cc"],
    hdrs = ["internal/dimension_selection.h"],
    deps = [
        ":dimension_identifier",
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore:index",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "dimension_selection_test",
    size = "small",
    srcs = ["dimension_selection_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "get_output_range_test",
    size = "small",
    srcs = ["get_output_range_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "identity_transform_test",
    size = "small",
    srcs = ["identity_transform_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index_array_slice_op",
    srcs = ["internal/index_array_slice_op.cc"],
    hdrs = ["internal/index_array_slice_op.h"],
    deps = [
        ":dimension_identifier",
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:rank",
        "//tensorstore/internal:meta",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "index_array_slice_op_test",
    size = "small",
    srcs = ["index_array_slice_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index_transform",
    srcs = [
        "index_transform.cc",
        "index_transform_builder.cc",
        "internal/compose_transforms.cc",
        "internal/identity_transform.cc",
        "internal/inverse_transform.cc",
        "internal/iterate.cc",
        "internal/propagate_bounds.cc",
        "internal/transform_array.cc",
        "internal/transform_rep.cc",
        "json.cc",
    ],
    hdrs = [
        "index_domain.h",
        "index_domain_builder.h",
        "index_transform.h",
        "index_transform_builder.h",
        "internal/compose_transforms.h",
        "internal/deep_copy_transform_rep_ptr.h",
        "internal/identity_transform.h",
        "internal/inverse_transform.h",
        "internal/iterate_impl.h",
        "internal/propagate_bounds.h",
        "internal/transform_array.h",
        "internal/transform_rep.h",
        "internal/transform_rep_impl.h",
        "json.h",
        "output_index_map.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":dimension_identifier",
        ":output_index_method",
        ":transform_array_constraints",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:json_serialization_options",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/internal:dimension_labels",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:gdb_scripting",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:string_like",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json:array",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:array",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:division",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "index_transform_builder_test",
    size = "small",
    srcs = ["index_transform_builder_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "index_transform_test",
    size = "small",
    srcs = ["index_transform_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index_transform_testutil",
    testonly = True,
    srcs = ["index_transform_testutil.cc"],
    hdrs = ["index_transform_testutil.h"],
    deps = [
        ":dim_expression",
        ":index_transform",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
    ],
)

tensorstore_cc_library(
    name = "index_vector_or_scalar",
    srcs = ["index_vector_or_scalar.cc"],
    hdrs = ["index_vector_or_scalar.h"],
    deps = [
        "//tensorstore:index",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "index_vector_or_scalar_test",
    size = "small",
    srcs = ["index_vector_or_scalar_test.cc"],
    deps = [
        ":index_vector_or_scalar",
        "//tensorstore:index",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "interval_slice_op",
    srcs = ["internal/interval_slice_op.cc"],
    hdrs = ["internal/interval_slice_op.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        ":index_vector_or_scalar",
        "//tensorstore:rank",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:meta",
        "//tensorstore/util:division",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "interval_slice_op_test",
    size = "small",
    srcs = ["interval_slice_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore:box",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "inverse_transform_test",
    size = "small",
    srcs = ["inverse_transform_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_testutil",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_binary(
    name = "iterate_benchmark_test",
    testonly = 1,
    srcs = ["iterate_benchmark_test.cc"],
    tags = ["benchmark"],
    deps = [
        ":dim_expression",
        ":transformed_array",
        "//tensorstore:index",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_benchmark//:benchmark",
        "@com_google_benchmark//:benchmark_main",
    ],
)

cc_with_non_compile_test(
    name = "iterate_nc_test",
    srcs = ["iterate_nc_test.cc"],
    nc_test_shard_count = 1,
    deps = [
        ":transformed_array",
        "//tensorstore:array",
    ],
)

tensorstore_cc_test(
    name = "iterate_test",
    size = "small",
    srcs = ["iterate_test.cc"],
    deps = [
        ":dim_expression",
        ":index_transform",
        ":transformed_array",
        "//tensorstore:index_interval",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "json_test",
    size = "small",
    srcs = ["json_test.cc"],
    deps = [
        ":dim_expression",
        ":index_transform",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "label_op",
    srcs = ["internal/label_op.cc"],
    hdrs = ["internal/label_op.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore:rank",
        "//tensorstore/internal:dimension_labels",
        "//tensorstore/internal:meta",
        "//tensorstore/internal:string_like",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
    ],
)

tensorstore_cc_test(
    name = "label_op_test",
    size = "small",
    srcs = ["label_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "mark_explicit_op",
    srcs = ["internal/mark_explicit_op.cc"],
    hdrs = ["internal/mark_explicit_op.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore:rank",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "mark_explicit_op_test",
    size = "small",
    srcs = ["mark_explicit_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:index",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "move_to_op_test",
    size = "small",
    srcs = ["move_to_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "numpy_indexing_spec",
    srcs = ["internal/numpy_indexing_spec.cc"],
    hdrs = ["internal/numpy_indexing_spec.h"],
    deps = [
        ":dimension_identifier",
        ":dimension_index_buffer",
        ":dimension_selection",
        ":index_transform",
        ":index_vector_or_scalar",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/internal:container_to_shared",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
    ],
)

tensorstore_cc_test(
    name = "output_index_map_test",
    size = "small",
    srcs = ["output_index_map_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:strided_layout",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "output_index_method",
    srcs = ["output_index_method.cc"],
    hdrs = ["output_index_method.h"],
    visibility = ["//visibility:public"],
)

tensorstore_cc_test(
    name = "propagate_bounds_test",
    size = "small",
    srcs = ["propagate_bounds_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "propagate_input_domain_resize_to_output_test",
    size = "small",
    srcs = ["propagate_input_domain_resize_to_output_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "single_index_slice_op",
    srcs = ["internal/single_index_slice_op.cc"],
    hdrs = ["internal/single_index_slice_op.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        ":index_vector_or_scalar",
        "//tensorstore:rank",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:meta",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "single_index_slice_op_test",
    size = "small",
    srcs = ["single_index_slice_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "slice_by_box_test",
    size = "small",
    srcs = ["slice_by_box_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "slice_by_index_domain_test",
    size = "small",
    srcs = ["slice_by_index_domain_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "transform_array_constraints",
    hdrs = ["transform_array_constraints.h"],
    visibility = ["//visibility:public"],
    deps = ["//tensorstore/util:iterate"],
)

tensorstore_cc_test(
    name = "transform_array_constraints_test",
    size = "small",
    srcs = ["transform_array_constraints_test.cc"],
    deps = [
        ":transform_array_constraints",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "transform_array_test",
    size = "small",
    srcs = ["transform_array_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "transform_rep_test",
    size = "small",
    srcs = ["transform_rep_test.cc"],
    deps = [
        ":index_transform",
        "//tensorstore/internal:concurrent_testutil",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "transformed_array",
    srcs = ["transformed_array.cc"],
    hdrs = ["transformed_array.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":index_transform",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/internal:element_copy_function",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "transformed_array_test",
    size = "small",
    srcs = ["transformed_array_test.cc"],
    deps = [
        ":dim_expression",
        ":index_transform",
        ":transformed_array",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "translate_op",
    srcs = ["internal/translate_op.cc"],
    hdrs = ["internal/translate_op.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        ":index_vector_or_scalar",
        "//tensorstore:rank",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:meta",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "translate_op_test",
    size = "small",
    srcs = ["translate_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "transpose_op",
    srcs = ["internal/transpose_op.cc"],
    hdrs = ["internal/transpose_op.h"],
    deps = [
        ":dimension_identifier",
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore:rank",
        "//tensorstore/internal:meta",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "transpose_op_test",
    size = "small",
    srcs = ["transpose_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "transpose_to_op_test",
    size = "small",
    srcs = ["transpose_to_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "transform_broadcastable_array",
    srcs = ["transform_broadcastable_array.cc"],
    hdrs = ["transform_broadcastable_array.h"],
    deps = [
        ":index_transform",
        ":transformed_array",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore:index_interval",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "transform_broadcastable_array_test",
    size = "small",
    srcs = ["transform_broadcastable_array_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_testutil",
        ":transform_broadcastable_array",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "dimension_units",
    srcs = ["dimension_units.cc"],
    hdrs = ["dimension_units.h"],
    deps = [
        ":index_transform",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:unit",
    ],
)

tensorstore_cc_test(
    name = "dimension_units_test",
    srcs = ["dimension_units_test.cc"],
    deps = [
        ":dimension_units",
        ":index_transform",
        ":index_transform_testutil",
        "//tensorstore/internal:test_util",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:unit",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_googletest//:gtest_main",
    ],
)
