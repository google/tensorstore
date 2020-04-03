load(":pybind11.bzl", "pybind11_cc_library", "pybind11_cc_test", "pybind11_py_extension")
load(":pytest.bzl", "tensorstore_pytest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files([
    "LICENSE",
])

pybind11_py_extension(
    name = "_tensorstore",
    srcs = ["tensorstore.cc"],
    imports = [".."],
    deps = [
        ":context",
        ":data_type",
        ":future",
        ":index_space",
        ":spec",
        ":tensorstore_class",
        ":write_futures",
        "//tensorstore/driver/n5",
        "//tensorstore/driver/n5:blosc_compressor",
        "//tensorstore/driver/n5:bzip2_compressor",
        "//tensorstore/driver/n5:gzip_compressor",
        "//tensorstore/driver/n5:xz_compressor",
        "//tensorstore/driver/neuroglancer_precomputed",
        "//tensorstore/driver/zarr",
        "//tensorstore/driver/zarr:blosc_compressor",
        "//tensorstore/driver/zarr:bzip2_compressor",
        "//tensorstore/driver/zarr:zlib_compressor",
        "//tensorstore/kvstore/file",
        "//tensorstore/kvstore/gcs",
        "//tensorstore/kvstore/memory",
        "@com_github_pybind_pybind11//:pybind11",
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

py_binary(
    name = "shell",
    srcs = ["shell.py"],
    python_version = "PY3",
    deps = [
        ":tensorstore",
        "@pypa_absl_py//:absl_py",
        "@pypa_ipython//:ipython",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "index_interval_test",
    size = "small",
    srcs = ["index_interval_test.py"],
    deps = [
        ":tensorstore",
    ],
)

tensorstore_pytest_test(
    name = "index_domain_test",
    size = "small",
    srcs = ["index_domain_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "dim_expression_test",
    size = "small",
    srcs = ["dim_expression_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "indexing_test",
    size = "small",
    srcs = ["indexing_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "index_transform_test",
    size = "small",
    srcs = ["index_transform_test.py"],
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
        "//tensorstore/util:to_string",
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
    name = "index",
    srcs = ["index.cc"],
    hdrs = ["index.h"],
    deps = [
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/strings",
    ],
)

pybind11_cc_library(
    name = "dim_expression",
    srcs = ["dim_expression.cc"],
    hdrs = ["dim_expression.h"],
    deps = [
        ":index",
        ":indexing_spec",
        ":subscript_method",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
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
        ":indexing_spec",
        ":json_type_caster",
        ":result_type_caster",
        ":status",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/index_space:json",
        "//tensorstore/index_space:output_index_method",
        "//tensorstore/internal:json",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:to_string",
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
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/internal:elementwise_function",
        "//tensorstore/util:iterate",
        "//tensorstore/util:to_string",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/strings",
    ],
)

pybind11_cc_library(
    name = "spec",
    srcs = ["spec.cc"],
    hdrs = ["spec.h"],
    deps = [
        ":data_type",
        ":index_space",
        ":json_type_caster",
        "//tensorstore:spec",
        "//tensorstore/internal:json_pprint_python",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "data_type",
    srcs = ["data_type.cc"],
    hdrs = ["data_type.h"],
    deps = [
        ":json_type_caster",
        "//tensorstore:data_type",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:to_string",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/hash",
    ],
)

pybind11_cc_library(
    name = "tensorstore_class",
    srcs = ["tensorstore_class.cc"],
    hdrs = ["tensorstore_class.h"],
    deps = [
        ":array_type_caster",
        ":data_type",
        ":future",
        ":index_space",
        ":spec",
        ":write_futures",
        "//tensorstore",
        "//tensorstore:cast",
        "//tensorstore:open",
        "//tensorstore/driver/array",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = ["context.h"],
    deps = [
        ":json_type_caster",
        ":result_type_caster",
        "//tensorstore:context",
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
        "//tensorstore/internal:logging",
        "//tensorstore/util:future",
        "@com_github_pybind_pybind11//:pybind11",
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
    name = "indexing_spec",
    srcs = ["indexing_spec.cc"],
    hdrs = ["indexing_spec.h"],
    deps = [
        ":array_type_caster",
        ":index",
        ":result_type_caster",
        ":status",
        ":subscript_method",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:index_transform_builder",
        "//tensorstore/internal:container_to_shared",
        "//tensorstore/util:constant_vector",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

tensorstore_pytest_test(
    name = "spec_test",
    size = "small",
    srcs = ["spec_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "data_type_test",
    size = "small",
    srcs = ["data_type_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "tensorstore_test",
    size = "small",
    srcs = ["tensorstore_test.py"],
    deps = [
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "future_test",
    size = "small",
    srcs = ["future_test.py"],
    deps = [":tensorstore"],
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
