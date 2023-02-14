load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//bazel:non_compile.bzl", "cc_with_non_compile_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

tensorstore_cc_library(
    name = "array",
    srcs = [
        "array.cc",
        "make_array.inc",
    ],
    hdrs = ["array.h"],
    deps = [
        ":box",
        ":data_type",
        ":rank",
        ":strided_layout",
        "//tensorstore/internal:element_copy_function",
        "//tensorstore/internal:meta",
        "//tensorstore/internal:unaligned_data_type_functions",
        "//tensorstore/serialization",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:element_traits",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_riegeli//riegeli/varint:varint_reading",
        "@com_google_riegeli//riegeli/varint:varint_writing",
    ],
)

cc_with_non_compile_test(
    name = "array_nc_test",
    srcs = ["array_nc_test.cc"],
    nc_test_shard_count = 1,
    deps = [
        ":array",
        ":index",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "array_test",
    size = "small",
    srcs = ["array_test.cc"],
    deps = [
        ":array",
        ":box",
        ":index",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/internal:data_type_random_generator",
        "//tensorstore/internal:test_util",
        "//tensorstore/serialization",
        "//tensorstore/serialization:batch",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "array_testutil",
    testonly = 1,
    srcs = ["array_testutil_matches_array.inc"],
    hdrs = ["array_testutil.h"],
    deps = [
        ":array",
        "//tensorstore/util:iterate_over_index_range",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_test(
    name = "array_testutil_test",
    size = "small",
    srcs = ["array_testutil_test.cc"],
    deps = [
        ":array_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "box",
    srcs = ["box.cc"],
    hdrs = ["box.h"],
    deps = [
        ":index",
        ":index_interval",
        ":rank",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:gdb_scripting",
        "//tensorstore/internal:multi_vector",
        "//tensorstore/internal:multi_vector_view",
        "//tensorstore/serialization",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:extents",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "box_test",
    size = "small",
    srcs = ["box_test.cc"],
    deps = [
        ":box",
        ":rank",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "cast",
    hdrs = ["cast.h"],
    deps = [
        ":data_type",
        ":tensorstore",
        "//tensorstore/driver/cast",
    ],
)

tensorstore_cc_library(
    name = "chunk_layout",
    srcs = ["chunk_layout.cc"],
    hdrs = ["chunk_layout.h"],
    deps = [
        ":box",
        ":contiguous_layout",
        ":index",
        ":index_interval",
        ":json_serialization_options",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:integer_range",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:division",
        "//tensorstore/util:maybe_hard_constraint",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/garbage_collection",
    ],
)

tensorstore_cc_test(
    name = "chunk_layout_test",
    size = "small",
    srcs = ["chunk_layout_test.cc"],
    deps = [
        ":array",
        ":box",
        ":chunk_layout",
        ":index_interval",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:test_util",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:division",
        "//tensorstore/util:iterate_over_index_range",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "virtual_chunked",
    srcs = ["//tensorstore/driver/virtual_chunked:virtual_chunked.cc"],
    hdrs = ["virtual_chunked.h"],
    deps = [
        ":array",
        ":box",
        ":context",
        ":staleness_bound",
        ":tensorstore",
        ":transaction",
        "//tensorstore/driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal/cache:cache_pool_resource",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/kvstore:generation",
        "//tensorstore/serialization",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/serialization:function",
        "//tensorstore/util:executor",
        "//tensorstore/util:option",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "container_kind",
    hdrs = ["container_kind.h"],
)

tensorstore_cc_library(
    name = "context",
    srcs = [
        "context.cc",
        "context_impl_base.h",
    ],
    hdrs = [
        "context.h",
        "context_impl.h",
        "context_resource_provider.h",
    ],
    deps = [
        ":json_serialization_options",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:heterogeneous_container",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "context_test",
    size = "small",
    srcs = [
        "context_test.cc",
    ],
    deps = [
        ":context",
        ":json_serialization_options",
        "//tensorstore/internal:concurrent_testutil",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "contiguous_layout",
    srcs = ["contiguous_layout.cc"],
    hdrs = ["contiguous_layout.h"],
    deps = [
        ":index",
        "//tensorstore/util:span",
    ],
)

tensorstore_cc_test(
    name = "contiguous_layout_test",
    size = "small",
    srcs = ["contiguous_layout_test.cc"],
    deps = [
        ":contiguous_layout",
        ":index",
        "//tensorstore/util:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "data_type",
    srcs = ["data_type.cc"],
    hdrs = [
        "data_type.h",
        "data_type_conversion.h",
    ],
    deps = [
        ":static_cast",
        "//tensorstore/internal:bit_operations",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:integer_types",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:memory",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal:utf8",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/internal/preprocessor:defer",
        "//tensorstore/internal/preprocessor:expand",
        "//tensorstore/serialization",
        "//tensorstore/util:bfloat16",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:division",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:utf8_string",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/status",
        "@net_sourceforge_half//:half",
    ],
)

tensorstore_cc_test(
    name = "data_type_conversion_test",
    size = "small",
    srcs = ["data_type_conversion_test.cc"],
    deps = [
        ":data_type",
        ":index",
        "//tensorstore/internal:element_copy_function",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:half_gtest",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "data_type_test",
    size = "small",
    srcs = ["data_type_test.cc"],
    deps = [
        ":data_type",
        ":index",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index",
    hdrs = ["index.h"],
)

tensorstore_cc_library(
    name = "index_interval",
    srcs = ["index_interval.cc"],
    hdrs = ["index_interval.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":container_kind",
        ":index",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/serialization",
        "//tensorstore/util:division",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "index_interval_test",
    size = "small",
    srcs = ["index_interval_test.cc"],
    deps = [
        ":index_interval",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/hash:hash_testing",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "index_space",
    hdrs = ["index_space.h"],
    deps = [
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
    ],
)

tensorstore_cc_library(
    name = "json_serialization_options",
    hdrs = ["json_serialization_options.h"],
    deps = [
        ":data_type",
        ":index",
        ":json_serialization_options_base",
        ":rank",
    ],
)

tensorstore_cc_library(
    name = "json_serialization_options_base",
    hdrs = ["json_serialization_options_base.h"],
    deps = [
        ":index",
        ":rank",
    ],
)

tensorstore_cc_library(
    name = "open",
    hdrs = ["open.h"],
    deps = [
        ":context",
        ":open_mode",
        ":open_options",
        ":rank",
        ":spec",
        ":tensorstore",
        "//tensorstore/driver",
        "//tensorstore/util:status",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "open_mode",
    srcs = ["open_mode.cc"],
    hdrs = ["open_mode.h"],
    deps = ["@com_google_absl//absl/status"],
)

tensorstore_cc_test(
    name = "open_mode_test",
    size = "small",
    srcs = ["open_mode_test.cc"],
    deps = [
        ":open_mode",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "open_options",
    hdrs = ["open_options.h"],
    deps = [
        ":context",
        ":open_mode",
        ":schema",
        ":staleness_bound",
        ":transaction",
        "//tensorstore/kvstore",
        "//tensorstore/util:option",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "progress",
    srcs = ["progress.cc"],
    hdrs = ["progress.h"],
    deps = [
        ":index",
        "//tensorstore/internal/poly",
        "//tensorstore/util:future",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "progress_test",
    size = "small",
    srcs = ["progress_test.cc"],
    deps = [
        ":progress",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "rank",
    srcs = ["rank.cc"],
    hdrs = ["rank.h"],
    deps = [
        ":index",
        ":static_cast",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
    ],
)

tensorstore_cc_test(
    name = "rank_test",
    size = "small",
    srcs = ["rank_test.cc"],
    deps = [
        ":rank",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "read_write_options",
    hdrs = ["read_write_options.h"],
    deps = [
        ":contiguous_layout",
        ":progress",
        "//tensorstore/index_space:alignment",
    ],
)

tensorstore_cc_library(
    name = "resize_options",
    srcs = ["resize_options.cc"],
    hdrs = ["resize_options.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tensorstore_cc_test(
    name = "resize_options_test",
    size = "small",
    srcs = ["resize_options_test.cc"],
    deps = [
        ":resize_options",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "spec",
    srcs = [
        "spec.cc",
        "spec_impl.h",
    ],
    hdrs = ["spec.h"],
    deps = [
        ":context",
        ":data_type",
        ":index",
        ":rank",
        ":schema",
        "//tensorstore/driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/serialization",
        "//tensorstore/util:result",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:nlohmann_json",
    ],
)

tensorstore_cc_test(
    name = "spec_test",
    size = "small",
    srcs = ["spec_test.cc"],
    deps = [
        ":spec",
        "//tensorstore/driver/array",
        "//tensorstore/driver/zarr",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "staleness_bound",
    hdrs = ["staleness_bound.h"],
    deps = [
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "static_cast",
    srcs = ["static_cast.cc"],
    hdrs = ["static_cast.h"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "static_cast_test",
    size = "small",
    srcs = ["static_cast_test.cc"],
    deps = [
        ":static_cast",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "strided_layout",
    srcs = ["strided_layout.cc"],
    hdrs = ["strided_layout.h"],
    deps = [
        ":box",
        ":container_kind",
        ":contiguous_layout",
        ":rank",
        "//tensorstore/internal:gdb_scripting",
        "//tensorstore/internal:multi_vector",
        "//tensorstore/internal:multi_vector_view",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:extents",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
    ],
)

tensorstore_cc_test(
    name = "strided_layout_test",
    size = "small",
    srcs = ["strided_layout_test.cc"],
    deps = [
        ":index",
        ":strided_layout",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "tensorstore",
    srcs = [
        "tensorstore.cc",
        "tensorstore_impl.h",
    ],
    hdrs = ["tensorstore.h"],
    deps = [
        ":chunk_layout",
        ":data_type",
        ":index",
        ":open_mode",
        ":progress",
        ":rank",
        ":read_write_options",
        ":resize_options",
        ":spec",
        "//tensorstore/driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/serialization",
        "//tensorstore/util:future",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "transaction",
    srcs = ["transaction.cc"],
    hdrs = [
        "transaction.h",
        "transaction_impl.h",
    ],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:intrusive_red_black_tree",
        "//tensorstore/internal:mutex",
        "//tensorstore/serialization",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "transaction_test",
    size = "small",
    srcs = ["transaction_test.cc"],
    deps = [
        ":transaction",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "all_drivers",
    deps = [
        "//tensorstore/driver:all_drivers",
        "//tensorstore/kvstore:all_drivers",
    ],
)

# Package group with visibility to internal-only targets.
package_group(
    name = "internal_packages",
    packages = [
        "//...",
        "//python/tensorstore/...",
        "//tensorstore/...",
    ],
)

tensorstore_cc_library(
    name = "downsample_method",
    srcs = ["downsample_method.cc"],
    hdrs = ["downsample_method.h"],
)

tensorstore_cc_library(
    name = "downsample",
    hdrs = ["downsample.h"],
    deps = [
        ":downsample_method",
        ":rank",
        ":spec",
        ":tensorstore",
        "//tensorstore/driver/downsample",
        "//tensorstore/internal:type_traits",
    ],
)

tensorstore_cc_library(
    name = "codec_spec",
    srcs = ["codec_spec.cc"],
    hdrs = [
        "codec_spec.h",
        "codec_spec_registry.h",
    ],
    deps = [
        ":json_serialization_options",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util/garbage_collection",
    ],
)

tensorstore_cc_test(
    name = "codec_spec_test",
    size = "small",
    srcs = ["codec_spec_test.cc"],
    deps = [
        ":codec_spec",
        "//tensorstore/driver/zarr",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "schema",
    srcs = ["schema.cc"],
    hdrs = ["schema.h"],
    deps = [
        ":array",
        ":chunk_layout",
        ":codec_spec",
        ":index",
        ":json_serialization_options",
        ":rank",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transform_broadcastable_array",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json:array",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:array",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:data_type",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/internal/json_binding:unit",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:division",
        "//tensorstore/util:status",
        "//tensorstore/util:unit",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_test(
    name = "schema_test",
    size = "small",
    srcs = ["schema_test.cc"],
    deps = [
        ":schema",
        "//tensorstore/driver/n5:metadata",
        "//tensorstore/driver/zarr:spec",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_testutil",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:test_util",
        "//tensorstore/internal/json_binding:gtest",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/random:bit_gen_ref",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "tensorstore_serialization_test",
    size = "small",
    srcs = ["tensorstore_serialization_test.cc"],
    deps = [
        ":context",
        ":open",
        ":spec",
        ":tensorstore",
        "//tensorstore/driver/array",
        "//tensorstore/driver/n5",
        "//tensorstore/internal:test_util",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/memory",
        "//tensorstore/serialization",
        "//tensorstore/serialization:test_util",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)
