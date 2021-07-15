load(":pybind11.bzl", "pybind11_cc_library", "pybind11_cc_test", "pybind11_py_extension")
load(":pytest.bzl", "tensorstore_pytest_test")
load(":pytype.bzl", "pytype_strict_binary")
load("//docs:doctest.bzl", "doctest_multi_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files([
    "LICENSE",
    "bazel_pytest_main.py",
])

doctest_multi_test(
    name = "doctest_tests",
    srcs = glob([
        "*.py",
        "*.cc",
    ]),
)

pybind11_py_extension(
    name = "_tensorstore",
    srcs = ["tensorstore.cc"],
    imports = [".."],
    deps = [
        ":chunk_layout",
        ":context",
        ":data_type",
        ":downsample",
        ":future",
        ":index_space",
        ":numpy",
        ":spec",
        ":tensorstore_class",
        ":transaction",
        ":write_futures",
        "//tensorstore:all_drivers",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    textual_hdrs = ["numpy.h"],
    deps = [
        "@local_config_python//:python_headers",  # buildcleaner: keep
        "@pypa_numpy//:headers",
    ],
)

py_library(
    name = "tensorstore",
    srcs = ["__init__.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":_tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

pytype_strict_binary(
    name = "shell",
    srcs = ["shell.py"],
    python_version = "PY3",
    tags = ["manual"],
    deps = [
        ":tensorstore",
        "@pypa_absl_py//:absl_py",
        "@pypa_ipython//:ipython",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "dim_test",
    size = "small",
    srcs = ["tests/dim_test.py"],
    deps = [
        ":tensorstore",
    ],
)

tensorstore_pytest_test(
    name = "index_domain_test",
    size = "small",
    srcs = ["tests/index_domain_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "dim_expression_test",
    size = "small",
    srcs = ["tests/dim_expression_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "indexing_test",
    size = "small",
    srcs = ["tests/indexing_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "index_transform_test",
    size = "small",
    srcs = ["tests/index_transform_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "downsample_test",
    size = "small",
    srcs = ["tests/downsample_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

pybind11_cc_library(
    name = "subscript_method",
    hdrs = ["subscript_method.h"],
    deps = [
        "//tensorstore/internal:type_traits",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "json_type_caster",
    srcs = ["json_type_caster.cc"],
    hdrs = ["json_type_caster.h"],
    deps = [
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/status",
    ],
)

pybind11_cc_library(
    name = "intrusive_ptr_holder",
    hdrs = ["intrusive_ptr_holder.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "index",
    srcs = ["index.cc"],
    hdrs = ["index.h"],
    deps = [
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:numpy_indexing_spec",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "dim_expression",
    srcs = ["dim_expression.cc"],
    hdrs = ["dim_expression.h"],
    deps = [
        ":index",
        ":numpy_indexing_spec",
        ":subscript_method",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/strings",
    ],
)

pybind11_cc_library(
    name = "index_space",
    srcs = ["index_space.cc"],
    hdrs = ["index_space.h"],
    deps = [
        ":array_type_caster",
        ":dim_expression",
        ":index",
        ":json_type_caster",
        ":numpy",
        ":numpy_indexing_spec",
        ":result_type_caster",
        ":status",
        ":subscript_method",
        "//tensorstore:array",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/index_space:json",
        "//tensorstore/index_space:output_index_method",
        "//tensorstore/internal:json",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/hash",
    ],
)

pybind11_cc_library(
    name = "result_type_caster",
    hdrs = ["result_type_caster.h"],
    deps = [
        ":status",
        "//tensorstore/util:result",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "array_type_caster",
    srcs = ["array_type_caster.cc"],
    hdrs = ["array_type_caster.h"],
    deps = [
        ":data_type",
        ":json_type_caster",
        ":numpy",
        "//tensorstore:array",
        "//tensorstore:container_kind",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:static_cast",
        "//tensorstore:strided_layout",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "spec",
    srcs = ["spec.cc"],
    hdrs = ["spec.h"],
    deps = [
        ":array_type_caster",
        ":data_type",
        ":index",
        ":index_space",
        ":intrusive_ptr_holder",
        ":json_type_caster",
        ":keyword_arguments",
        ":result_type_caster",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options",
        "//tensorstore:rank",
        "//tensorstore:schema",
        "//tensorstore:spec",
        "//tensorstore/driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal:json_pprint_python",
        "//tensorstore/internal:preprocessor",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "data_type",
    srcs = [
        "bfloat16.cc",
        "data_type.cc",
    ],
    hdrs = [
        "bfloat16.h",
        "data_type.h",
    ],
    deps = [
        ":json_type_caster",
        ":numpy",
        "//tensorstore:data_type",
        "//tensorstore/util:bfloat16",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/strings",
    ],
)

pybind11_cc_library(
    name = "tensorstore_class",
    srcs = ["tensorstore_class.cc"],
    hdrs = ["tensorstore_class.h"],
    deps = [
        ":array_type_caster",
        ":context",
        ":data_type",
        ":future",
        ":index_space",
        ":json_type_caster",
        ":keyword_arguments",
        ":result_type_caster",
        ":spec",
        ":transaction",
        ":write_futures",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:cast",
        "//tensorstore:context",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:progress",
        "//tensorstore:rank",
        "//tensorstore:resize_options",
        "//tensorstore:spec",
        "//tensorstore:strided_layout",
        "//tensorstore/driver/array",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_pprint_python",
        "//tensorstore/internal:preprocessor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = ["context.h"],
    deps = [
        ":intrusive_ptr_holder",
        ":json_type_caster",
        ":result_type_caster",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:json_pprint_python",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "future",
    srcs = ["future.cc"],
    hdrs = ["future.h"],
    deps = [
        ":result_type_caster",
        ":status",
        "//tensorstore/internal:intrusive_linked_list",
        "//tensorstore/internal:logging",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/functional:function_ref",
    ],
)

pybind11_cc_library(
    name = "write_futures",
    srcs = ["write_futures.cc"],
    hdrs = ["write_futures.h"],
    deps = [
        ":future",
        "//tensorstore:progress",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "numpy_indexing_spec",
    srcs = ["numpy_indexing_spec.cc"],
    hdrs = ["numpy_indexing_spec.h"],
    deps = [
        ":array_type_caster",
        ":data_type",
        ":index",
        ":numpy",
        ":result_type_caster",
        ":status",
        ":subscript_method",
        "//tensorstore:array",
        "//tensorstore:container_kind",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:static_cast",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:numpy_indexing_spec",
        "//tensorstore/util:assert_macros",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_pytest_test(
    name = "context_test",
    size = "small",
    srcs = ["tests/context_test.py"],
    deps = [
        ":tensorstore",
    ],
)

tensorstore_pytest_test(
    name = "spec_test",
    size = "small",
    srcs = ["tests/spec_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "data_type_test",
    size = "small",
    srcs = ["tests/data_type_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "tensorstore_test",
    size = "small",
    srcs = ["tests/tensorstore_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "transaction_test",
    size = "small",
    srcs = ["tests/transaction_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "future_test",
    size = "small",
    srcs = ["tests/future_test.py"],
    deps = [":tensorstore"],
)

tensorstore_pytest_test(
    name = "chunk_layout_test",
    size = "small",
    srcs = ["tests/chunk_layout_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

pybind11_cc_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        ":index",
        "//tensorstore:index",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_pytest_test(
    name = "bfloat16_test",
    size = "small",
    srcs = ["tests/bfloat16_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

pybind11_cc_library(
    name = "transaction",
    srcs = ["transaction.cc"],
    hdrs = ["transaction.h"],
    deps = [
        ":future",
        "//tensorstore:transaction",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "downsample",
    srcs = ["downsample.cc"],
    hdrs = ["downsample.h"],
    deps = [
        ":index",
        ":result_type_caster",
        ":spec",
        ":status",
        "//tensorstore:downsample",
        "//tensorstore:downsample_method",
        "//tensorstore:spec",
        "//tensorstore/driver/downsample:downsample_method_json_binder",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "chunk_layout",
    srcs = [
        "chunk_layout.cc",
        "chunk_layout_keyword_arguments.h",
    ],
    hdrs = ["chunk_layout.h"],
    deps = [
        ":array_type_caster",
        ":index",
        ":index_space",
        ":json_type_caster",
        ":keyword_arguments",
        "//tensorstore:chunk_layout",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/internal:json_pprint_python",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/status",
    ],
)

pybind11_cc_library(
    name = "keyword_arguments",
    hdrs = ["keyword_arguments.h"],
    deps = [
        ":status",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/strings",
    ],
)

pybind11_cc_library(
    name = "keyword_arguments_test",
    testonly = True,
    srcs = ["keyword_arguments_test.cc"],
    deps = [
        ":keyword_arguments",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)
