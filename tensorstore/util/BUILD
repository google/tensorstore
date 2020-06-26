load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//tensorstore:non_compile.bzl", "cc_with_non_compile_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "assert_macros",
    hdrs = ["assert_macros.h"],
    deps = [
        "//tensorstore/internal:log_message",
        "//tensorstore/internal:source_location",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tensorstore_cc_test(
    name = "assert_macros_test",
    size = "small",
    srcs = [
        "assert_macros_test.cc",
    ],
    deps = [
        ":assert_macros",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "bit_span",
    hdrs = ["bit_span.h"],
    deps = [
        ":span",
        "//tensorstore/internal:attributes",
    ],
)

tensorstore_cc_test(
    name = "bit_span_test",
    size = "small",
    srcs = ["bit_span_test.cc"],
    deps = [
        ":bit_span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "bit_vec",
    srcs = [
        "bit_vec_impl.cc",
        "bit_vec_impl.h",
    ],
    hdrs = ["bit_vec.h"],
    deps = [
        ":bit_span",
        ":span",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "bit_vec_test",
    size = "small",
    srcs = ["bit_vec_test.cc"],
    deps = [
        ":bit_span",
        ":bit_vec",
        ":str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "byte_strided_pointer",
    hdrs = ["byte_strided_pointer.h"],
    deps = [
        ":element_traits",
        "//tensorstore/internal:integer_overflow",
    ],
)

tensorstore_cc_test(
    name = "byte_strided_pointer_test",
    size = "small",
    srcs = ["byte_strided_pointer_test.cc"],
    deps = [
        ":byte_strided_pointer",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "collecting_sender",
    hdrs = ["collecting_sender.h"],
    deps = [":execution"],
)

tensorstore_cc_test(
    name = "collecting_sender_test",
    size = "small",
    srcs = ["collecting_sender_test.cc"],
    deps = [
        ":collecting_sender",
        ":execution",
        ":sender",
        ":sender_testutil",
        ":span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "constant_bit_vector",
    hdrs = ["constant_bit_vector.h"],
    deps = [
        ":bit_span",
        ":constant_vector",
    ],
)

tensorstore_cc_test(
    name = "constant_bit_vector_test",
    size = "small",
    srcs = ["constant_bit_vector_test.cc"],
    deps = [
        ":bit_span",
        ":constant_bit_vector",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "constant_vector",
    srcs = ["constant_vector.cc"],
    hdrs = ["constant_vector.h"],
    deps = [
        ":span",
        "//tensorstore:index",
        "//tensorstore:rank",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/utility",
    ],
)

tensorstore_cc_test(
    name = "constant_vector_test",
    size = "small",
    srcs = [
        "constant_vector_test.cc",
    ],
    deps = [
        ":constant_vector",
        ":span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "default_iteration_result",
    hdrs = ["default_iteration_result.h"],
)

tensorstore_cc_test(
    name = "default_iteration_result_test",
    size = "small",
    srcs = ["default_iteration_result_test.cc"],
    deps = [
        ":default_iteration_result",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "division",
    hdrs = [
        "division.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
    ],
)

tensorstore_cc_test(
    name = "division_test",
    size = "small",
    srcs = [
        "division_test.cc",
    ],
    deps = [
        ":division",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "element_pointer",
    srcs = ["element_pointer.cc"],
    hdrs = ["element_pointer.h"],
    deps = [
        ":byte_strided_pointer",
        ":element_traits",
        ":str_cat",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:static_cast",
        "//tensorstore/internal:compressed_pair",
        "//tensorstore/internal:memory",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal:unowned_to_shared",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "element_pointer_test",
    size = "small",
    srcs = ["element_pointer_test.cc"],
    deps = [
        ":element_pointer",
        ":result",
        ":status",
        ":status_testutil",
        "//tensorstore:data_type",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "element_traits",
    hdrs = ["element_traits.h"],
)

tensorstore_cc_test(
    name = "element_traits_test",
    size = "small",
    srcs = ["element_traits_test.cc"],
    deps = [
        ":element_traits",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "endian",
    hdrs = ["endian.h"],
    deps = [
        "//tensorstore:index",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:endian",
    ],
)

tensorstore_cc_library(
    name = "execution",
    hdrs = ["execution.h"],
    deps = [
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tensorstore_cc_library(
    name = "executor",
    hdrs = ["executor.h"],
    deps = [
        "//tensorstore/internal:attributes",
        "//tensorstore/internal:poly",
        "//tensorstore/internal:type_traits",
    ],
)

tensorstore_cc_test(
    name = "executor_test",
    size = "small",
    srcs = ["executor_test.cc"],
    deps = [
        ":executor",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "extents",
    hdrs = ["extents.h"],
    deps = [
        ":span",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/internal:integer_overflow",
        "//tensorstore/internal:meta",
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
    ],
)

tensorstore_cc_test(
    name = "extents_test",
    size = "small",
    srcs = ["extents_test.cc"],
    deps = [
        ":extents",
        ":span",
        "//tensorstore:index",
        "//tensorstore:rank",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "function_view",
    hdrs = ["function_view.h"],
    deps = ["//tensorstore/internal:type_traits"],
)

tensorstore_cc_test(
    name = "function_view_test",
    size = "small",
    srcs = ["function_view_test.cc"],
    deps = [
        ":assert_macros",
        ":function_view",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "future",
    srcs = [
        "future.cc",
    ],
    hdrs = [
        "future.h",
        "future_impl.h",
    ],
    deps = [
        ":assert_macros",
        ":execution",
        ":executor",
        ":result",
        ":status",
        "//tensorstore/internal:intrusive_linked_list",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:meta",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal:void_wrapper",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/utility",
    ],
)

tensorstore_cc_test(
    name = "future_sender_test",
    size = "small",
    srcs = ["future_sender_test.cc"],
    deps = [
        ":execution",
        ":future",
        ":result",
        ":sender",
        ":sender_testutil",
        ":status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "future_test",
    size = "small",
    srcs = [
        "future_test.cc",
    ],
    deps = [
        ":executor",
        ":future",
        ":result",
        ":status",
        ":status_testutil",
        ":str_cat",
        "//tensorstore/internal:concurrent_testutil",
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "iterate",
    srcs = ["iterate.cc"],
    hdrs = [
        "internal/iterate.h",
        "internal/iterate_impl.h",
        "iterate.h",
    ],
    deps = [
        ":byte_strided_pointer",
        ":span",
        ":status",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "//tensorstore/internal:elementwise_function",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/utility",
    ],
)

tensorstore_cc_library(
    name = "iterate_over_index_range",
    hdrs = ["iterate_over_index_range.h"],
    deps = [
        ":constant_vector",
        ":default_iteration_result",
        ":iterate",
        ":span",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "//tensorstore/internal:void_wrapper",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_cc_test(
    name = "iterate_over_index_range_test",
    size = "small",
    srcs = ["iterate_over_index_range_test.cc"],
    deps = [
        ":iterate_over_index_range",
        ":span",
        "//tensorstore:box",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "iterate_test",
    size = "small",
    srcs = ["iterate_test.cc"],
    deps = [
        ":iterate",
        ":span",
        ":str_cat",
        "//tensorstore:contiguous_layout",
        "//tensorstore:index",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "quote_string",
    srcs = ["quote_string.cc"],
    hdrs = ["quote_string.h"],
    deps = ["@com_google_absl//absl/strings"],
)

tensorstore_cc_test(
    name = "quote_string_test",
    size = "small",
    srcs = ["quote_string_test.cc"],
    deps = [
        ":quote_string",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "result",
    hdrs = [
        "result.h",
        "result_impl.h",
    ],
    deps = [
        ":assert_macros",
        ":execution",
        ":status",
        "//tensorstore/internal:preprocessor",
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/utility",
    ],
)

cc_with_non_compile_test(
    name = "result_nc_test",
    srcs = ["result_nc_test.cc"],
    nc_test_shard_count = 1,
    deps = [
        ":result",
        ":status",
    ],
)

tensorstore_cc_test(
    name = "result_sender_test",
    size = "small",
    srcs = ["result_sender_test.cc"],
    deps = [
        ":execution",
        ":result",
        ":sender",
        ":sender_testutil",
        ":status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "result_test",
    size = "small",
    srcs = [
        "result_test.cc",
    ],
    deps = [
        ":result",
        ":status",
        ":status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "sender",
    hdrs = ["sender.h"],
    deps = [
        ":execution",
        ":executor",
        "//tensorstore/internal:poly",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/utility",
    ],
)

tensorstore_cc_test(
    name = "sender_test",
    size = "small",
    srcs = ["sender_test.cc"],
    deps = [
        ":execution",
        ":executor",
        ":sender",
        ":sender_testutil",
        "//tensorstore/internal:poly",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "sender_testutil",
    testonly = 1,
    hdrs = ["sender_testutil.h"],
    deps = [
        ":sender",
        ":str_cat",
        ":sync_flow_sender",
        "//tensorstore/internal:mutex",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "span",
    hdrs = ["span.h"],
    deps = [
        "//tensorstore/internal:gdb_scripting",
        "//tensorstore/internal:log_message",
        "//tensorstore/internal:source_location",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tensorstore_cc_library(
    name = "span_json",
    hdrs = ["span_json.h"],
    deps = [
        ":span",
        "@com_github_nlohmann_json//:nlohmann_json",
    ],
)

tensorstore_cc_test(
    name = "span_json_test",
    size = "small",
    srcs = ["span_json_test.cc"],
    deps = [
        ":span",
        ":span_json",
        "//tensorstore/internal:json_gtest",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "span_test",
    size = "small",
    srcs = ["span_test.cc"],
    deps = [
        ":span",
        ":str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "status",
    srcs = [
        "status.cc",
    ],
    hdrs = [
        "status.h",
    ],
    deps = [
        ":assert_macros",
        ":str_cat",
        "//tensorstore/internal:preprocessor",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "status_test",
    size = "small",
    srcs = [
        "status_test.cc",
    ],
    deps = [
        ":status",
        ":str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "status_testutil",
    testonly = 1,
    srcs = ["status_testutil.cc"],
    hdrs = ["status_testutil.h"],
    deps = [
        ":result",
        ":status",
        "@com_google_googletest//:gtest",
    ],
)

tensorstore_cc_library(
    name = "sync_flow_sender",
    hdrs = ["sync_flow_sender.h"],
    deps = [":execution"],
)

tensorstore_cc_test(
    name = "sync_flow_sender_test",
    size = "small",
    srcs = ["sync_flow_sender_test.cc"],
    deps = [
        ":execution",
        ":sender_testutil",
        ":sync_flow_sender",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "str_cat",
    hdrs = ["str_cat.h"],
    deps = [
        ":span",
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "str_cat_test",
    size = "small",
    srcs = ["str_cat_test.cc"],
    deps = [
        ":str_cat",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "utf8_string",
    hdrs = ["utf8_string.h"],
)
