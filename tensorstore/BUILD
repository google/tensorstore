load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//tensorstore:non_compile.bzl", "cc_with_non_compile_test")

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
        "//tensorstore/internal:unowned_to_shared",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:element_traits",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
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
        ":index",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
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
        "//tensorstore/internal:gdb_scripting",
        "//tensorstore/internal:multi_vector",
        "//tensorstore/internal:multi_vector_view",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:extents",
        "//tensorstore/util:span",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "box_test",
    size = "small",
    srcs = ["box_test.cc"],
    deps = [
        ":box",
        ":rank",
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
        "//tensorstore/internal:cache_key",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:json",
        "//tensorstore/internal:logging",
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/strings",
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
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_gtest",
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
        "@com_google_absl//absl/base:core_headers",
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
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/internal:exception_macros",
        "//tensorstore/internal:integer_types",
        "//tensorstore/internal:json",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:memory",
        "//tensorstore/internal:preprocessor",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal:utf8",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:division",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:utf8_string",
        "@com_github_nlohmann_json//:nlohmann_json",
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
        "//tensorstore/util:division",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "index_interval_test",
    size = "small",
    srcs = ["index_interval_test.cc"],
    deps = [
        ":index_interval",
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
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/index_space:transformed_array",
    ],
)

tensorstore_cc_library(
    name = "json_serialization_options",
    hdrs = ["json_serialization_options.h"],
    deps = [
        ":index",
        ":rank",
    ],
)

tensorstore_cc_library(
    name = "open",
    srcs = ["open.cc"],
    hdrs = ["open.h"],
    deps = [
        ":context",
        ":open_mode",
        ":rank",
        ":spec",
        ":tensorstore",
        "//tensorstore/driver",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "open_mode",
    srcs = ["open_mode.cc"],
    hdrs = ["open_mode.h"],
    deps = [":staleness_bound"],
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
    name = "progress",
    srcs = ["progress.cc"],
    hdrs = ["progress.h"],
    deps = [
        ":index",
        "//tensorstore/internal:poly",
        "//tensorstore/util:future",
    ],
)

tensorstore_cc_test(
    name = "progress_test",
    size = "small",
    srcs = ["progress_test.cc"],
    deps = [
        ":progress",
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
        "//tensorstore/util:assert_macros",
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
        ":data_type",
        ":index",
        ":rank",
        "//tensorstore/driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_spec",
        "//tensorstore/index_space:json",
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:data_type_json_binder",
        "//tensorstore/internal:json",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_github_nlohmann_json//:nlohmann_json",
    ],
)

tensorstore_cc_library(
    name = "spec_request_options",
    hdrs = ["spec_request_options.h"],
    deps = [
        ":open_mode",
        ":staleness_bound",
    ],
)

tensorstore_cc_test(
    name = "spec_test",
    size = "small",
    srcs = ["spec_test.cc"],
    deps = [
        ":spec",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/internal:json_gtest",
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
    deps = ["@com_google_absl//absl/time"],
)

tensorstore_cc_library(
    name = "static_cast",
    srcs = ["static_cast.cc"],
    hdrs = ["static_cast.h"],
    deps = [
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
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
        "//tensorstore/internal:multi_vector",
        "//tensorstore/internal:multi_vector_view",
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:assert_macros",
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
        "//tensorstore/util:future",
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
        "//tensorstore/util:function_view",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
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
