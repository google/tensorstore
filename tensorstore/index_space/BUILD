load("//tensorstore:tensorstore.bzl", "tensorstore_cc_binary", "tensorstore_cc_library", "tensorstore_cc_test")
load("//tensorstore:non_compile.bzl", "cc_with_non_compile_test")

package(default_visibility = ["//visibility:private"])

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
        ":index_transform_builder",
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
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "alignment_test",
    size = "small",
    srcs = ["alignment_test.cc"],
    deps = [
        ":alignment",
        ":index_transform",
        ":index_transform_builder",
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
        ":index_transform_builder",
        "//tensorstore:array",
        "//tensorstore/util:bit_vec",
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
        ":internal/deep_copy_transform_rep_ptr",
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
        ":index_transform_builder",
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
        "@com_google_absl//absl/meta:type_traits",
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

tensorstore_cc_library(
    name = "dim_expression_testutil",
    testonly = 1,
    hdrs = ["internal/dim_expression_testutil.h"],
    deps = [
        ":dimension_index_buffer",
        ":index_transform",
        "//tensorstore/util:status_testutil",
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
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:variant",
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
        ":index_transform_builder",
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
        ":index_transform_builder",
        "//tensorstore:array",
        "//tensorstore/util:bit_vec",
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
        ":index_transform_builder",
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
        ":index_transform_builder",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index_transform",
    srcs = [
        "index_transform.cc",
        "internal/compose_transforms.cc",
        "internal/identity_transform.cc",
        "internal/inverse_transform.cc",
        "internal/iterate.cc",
        "internal/propagate_bounds.cc",
        "internal/transform_array.cc",
        "internal/transform_rep.cc",
    ],
    hdrs = [
        "index_transform.h",
        "internal/compose_transforms.h",
        "internal/identity_transform.h",
        "internal/inverse_transform.h",
        "internal/iterate_impl.h",
        "internal/propagate_bounds.h",
        "internal/transform_array.h",
        "internal/transform_rep.h",
        "internal/transform_rep_impl.h",
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
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:memory",
        "//tensorstore/internal:string_like",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:bit_vec",
        "//tensorstore/util:constant_bit_vector",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:division",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_library(
    name = "index_transform_builder",
    srcs = ["index_transform_builder.cc"],
    hdrs = [
        "index_domain_builder.h",
        "index_transform_builder.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":index_transform",
        ":internal/deep_copy_transform_rep_ptr",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:span",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "index_transform_builder_test",
    size = "small",
    srcs = ["index_transform_builder_test.cc"],
    deps = [
        ":index_transform_builder",
        "//tensorstore:array",
        "//tensorstore:box",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index_transform_spec",
    srcs = ["index_transform_spec.cc"],
    hdrs = ["index_transform_spec.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":index_transform",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/internal:attributes",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
    ],
)

tensorstore_cc_test(
    name = "index_transform_spec_test",
    size = "small",
    srcs = ["index_transform_spec_test.cc"],
    deps = [
        ":dim_expression",
        ":index_transform",
        ":index_transform_builder",
        ":index_transform_spec",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "index_transform_test",
    size = "small",
    srcs = ["index_transform_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_builder",
        "//tensorstore:array",
        "//tensorstore/util:bit_vec",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
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
    name = "internal/deep_copy_transform_rep_ptr",
    hdrs = ["internal/deep_copy_transform_rep_ptr.h"],
    deps = [
        ":index_transform",
        "@com_google_absl//absl/utility",
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
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "interval_slice_op_test",
    size = "small",
    srcs = ["interval_slice_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform_builder",
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
        ":index_transform_builder",
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
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
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
        ":index_transform_builder",
        ":transformed_array",
        "//tensorstore:index_interval",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "json",
    srcs = ["json.cc"],
    hdrs = ["json.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":index_transform",
        ":index_transform_builder",
        ":index_transform_spec",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_array",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "json_test",
    size = "small",
    srcs = ["json_test.cc"],
    deps = [
        ":dim_expression",
        ":index_transform_builder",
        ":json",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_gtest",
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
        "//tensorstore/internal:meta",
        "//tensorstore/internal:string_like",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "label_op_test",
    size = "small",
    srcs = ["label_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform_builder",
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
    ],
)

tensorstore_cc_test(
    name = "mark_explicit_op_test",
    size = "small",
    srcs = ["mark_explicit_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform_builder",
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
        ":index_transform_builder",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "output_index_map_test",
    size = "small",
    srcs = ["output_index_map_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_builder",
        "//tensorstore:array",
        "//tensorstore:strided_layout",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "output_index_method",
    hdrs = ["output_index_method.h"],
    visibility = ["//visibility:public"],
)

tensorstore_cc_test(
    name = "propagate_bounds_test",
    size = "small",
    srcs = ["propagate_bounds_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_builder",
        "//tensorstore/util:bit_vec",
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
        ":index_transform_builder",
        "//tensorstore:array",
        "//tensorstore/util:bit_vec",
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
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "single_index_slice_op_test",
    size = "small",
    srcs = ["single_index_slice_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform_builder",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "slice_by_index_domain_test",
    size = "small",
    srcs = ["slice_by_index_domain_test.cc"],
    deps = [
        ":index_transform",
        ":index_transform_builder",
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
        ":index_transform_builder",
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
        ":index_transform_builder",
        "//tensorstore/internal:concurrent_testutil",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "transformed_array",
    srcs = [
        "internal/transformed_array_impl.h",
        "transformed_array.cc",
    ],
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
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
    ],
)

tensorstore_cc_test(
    name = "transformed_array_test",
    size = "small",
    srcs = ["transformed_array_test.cc"],
    deps = [
        ":dim_expression",
        ":index_transform_builder",
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
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "translate_op_test",
    size = "small",
    srcs = ["translate_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform_builder",
        "//tensorstore/util:status",
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
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "transpose_op_test",
    size = "small",
    srcs = ["transpose_op_test.cc"],
    deps = [
        ":dim_expression",
        ":dim_expression_testutil",
        ":index_transform_builder",
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
        ":index_transform_builder",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@com_google_googletest//:gtest_main",
    ],
)
