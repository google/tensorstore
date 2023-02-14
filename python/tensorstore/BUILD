load(
    "//bazel:pybind11.bzl",
    "pybind11_cc_googletest_test",
    "pybind11_cc_library",
    "pybind11_py_extension",
)
load("//bazel:pytest.bzl", "tensorstore_pytest_test")
load("//bazel:pytype.bzl", "pytype_strict_binary")
load("//bazel:tensorstore.bzl", "tensorstore_cc_library")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

exports_files([
    "LICENSE",
    "bazel_pytest_main.py",
    "cc_test_driver.cc",
    "cc_test_driver_main.py",
])

doctest_test(
    name = "doctest_test",
    srcs = glob([
        "*.py",
        "*.cc",
    ]),
)

pybind11_py_extension(
    name = "_tensorstore",
    srcs = ["tensorstore_module.cc"],
    imports = [".."],
    deps = [
        ":chunk_layout",
        ":context",
        ":data_type",
        ":dim_expression",
        ":downsample",
        ":future",
        ":garbage_collection",
        ":gil_safe",
        ":index_space",
        ":kvstore",
        ":metrics",
        ":numpy",
        ":python_imports",
        ":serialization",
        ":spec",
        ":tensorstore_class",
        ":tensorstore_module_components",
        ":transaction",
        ":unit",
        ":virtual_chunked",
        ":write_futures",
        "//tensorstore:all_drivers",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "tensorstore_module_components",
    srcs = ["tensorstore_module_components.cc"],
    hdrs = ["tensorstore_module_components.h"],
    deps = [
        "//tensorstore/internal:no_destructor",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "numpy",
    srcs = ["numpy.cc"],
    textual_hdrs = ["numpy.h"],
    deps = [
        "@com_github_pybind_pybind11//:pybind11",
        "@pypa_numpy//:headers",
    ],
)

py_library(
    name = "tensorstore",
    srcs = ["__init__.py"],
    srcs_version = "PY3ONLY",
    visibility = ["//visibility:public"],
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

py_library(
    name = "conftest",
    testonly = True,
    srcs = ["tests/conftest.py"],
    srcs_version = "PY3ONLY",
    deps = ["@pypa_pytest//:pytest"],
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
        ":python_imports",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_boringssl//:crypto",
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
    name = "homogeneous_tuple",
    hdrs = ["homogeneous_tuple.h"],
    deps = [
        ":type_name_override",
        "//tensorstore/util:span",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "sequence_parameter",
    hdrs = ["sequence_parameter.h"],
    deps = [
        ":type_name_override",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "index",
    srcs = ["index.cc"],
    hdrs = ["index.h"],
    deps = [
        ":sequence_parameter",
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
        ":sequence_parameter",
        ":subscript_method",
        ":tensorstore_module_components",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
    ],
    alwayslink = True,
)

tensorstore_pytest_test(
    name = "exit_test",
    size = "small",
    srcs = ["tests/exit_test.py"],
    deps = [
        ":tensorstore",
    ],
)

pybind11_cc_library(
    name = "index_space",
    srcs = ["index_space.cc"],
    hdrs = ["index_space.h"],
    deps = [
        ":array_type_caster",
        ":dim_expression",
        ":gil_safe",
        ":homogeneous_tuple",
        ":index",
        ":json_type_caster",
        ":numpy",
        ":numpy_indexing_spec",
        ":python_imports",
        ":result_type_caster",
        ":sequence_parameter",
        ":serialization",
        ":status",
        ":subscript_method",
        ":tensorstore_module_components",
        "//tensorstore:array",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:output_index_method",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:executor",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/base:core_headers",
    ],
    alwayslink = True,
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
        ":gil_safe",
        ":json_type_caster",
        ":numpy",
        ":type_name_override",
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
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/status",
    ],
)

pybind11_cc_library(
    name = "spec",
    srcs = ["spec.cc"],
    hdrs = ["spec.h"],
    deps = [
        ":array_type_caster",
        ":context",
        ":data_type",
        ":garbage_collection",
        ":homogeneous_tuple",
        ":index",
        ":index_space",
        ":intrusive_ptr_holder",
        ":json_type_caster",
        ":keyword_arguments",
        ":kvstore",
        ":result_type_caster",
        ":sequence_parameter",
        ":serialization",
        ":tensorstore_module_components",
        ":unit",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options",
        "//tensorstore:rank",
        "//tensorstore:schema",
        "//tensorstore:spec",
        "//tensorstore/driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/util:executor",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
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
        ":serialization",
        ":tensorstore_module_components",
        "//tensorstore:data_type",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:bfloat16",
        "//tensorstore/util:executor",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/hash",
    ],
    alwayslink = True,
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
        ":garbage_collection",
        ":gil_safe",
        ":homogeneous_tuple",
        ":index",
        ":index_space",
        ":json_type_caster",
        ":keyword_arguments",
        ":kvstore",
        ":result_type_caster",
        ":serialization",
        ":spec",
        ":tensorstore_module_components",
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
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "context",
    srcs = ["context.cc"],
    hdrs = ["context.h"],
    deps = [
        ":intrusive_ptr_holder",
        ":json_type_caster",
        ":result_type_caster",
        ":serialization",
        ":tensorstore_module_components",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "gil_safe",
    srcs = ["gil_safe.cc"],
    hdrs = ["gil_safe.h"],
    deps = [
        ":python_imports",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util/garbage_collection",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/synchronization",
    ],
)

pybind11_cc_library(
    name = "garbage_collection",
    srcs = ["garbage_collection.cc"],
    hdrs = ["garbage_collection.h"],
    deps = [
        ":define_heap_type",
        ":gil_safe",
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/util/garbage_collection",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/container:flat_hash_set",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "future",
    srcs = ["future.cc"],
    hdrs = ["future.h"],
    deps = [
        ":define_heap_type",
        ":garbage_collection",
        ":gil_safe",
        ":python_imports",
        ":python_value_or_exception",
        ":result_type_caster",
        ":status",
        ":tensorstore_module_components",
        ":type_name_override",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_linked_list",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/functional:function_ref",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "python_value_or_exception",
    srcs = ["python_value_or_exception.cc"],
    hdrs = ["python_value_or_exception.h"],
    deps = [
        ":garbage_collection",
        ":gil_safe",
        ":status",
        "//tensorstore/util:result",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "python_imports",
    srcs = ["python_imports.cc"],
    hdrs = ["python_imports.h"],
    deps = ["@com_github_pybind_pybind11//:pybind11"],
)

pybind11_cc_library(
    name = "write_futures",
    srcs = ["write_futures.cc"],
    hdrs = ["write_futures.h"],
    deps = [
        ":future",
        ":tensorstore_module_components",
        "//tensorstore:progress",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
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
        ":type_name_override",
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
        "//tensorstore/util:bit_span",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/base:core_headers",
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
    deps = [
        ":conftest",
        ":tensorstore",
    ],
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

pybind11_cc_googletest_test(
    name = "index_test",
    size = "small",
    srcs = ["index_test.cc"],
    deps = [
        ":index",
        "//tensorstore:index",
        "@com_google_googletest//:gtest",
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
    name = "define_heap_type",
    srcs = ["define_heap_type.cc"],
    hdrs = ["define_heap_type.h"],
    deps = [
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "transaction",
    srcs = ["transaction.cc"],
    hdrs = ["transaction.h"],
    deps = [
        ":future",
        ":tensorstore_module_components",
        "//tensorstore:transaction",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
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
        ":tensorstore_class",
        ":tensorstore_module_components",
        "//tensorstore:downsample",
        "//tensorstore:downsample_method",
        "//tensorstore:spec",
        "//tensorstore/driver/downsample:downsample_method_json_binder",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
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
        ":homogeneous_tuple",
        ":index",
        ":index_space",
        ":json_type_caster",
        ":keyword_arguments",
        ":sequence_parameter",
        ":tensorstore_module_components",
        "//tensorstore:chunk_layout",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/status",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "keyword_arguments",
    hdrs = ["keyword_arguments.h"],
    deps = [
        ":status",
        ":type_name_override",
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

pybind11_cc_library(
    name = "unit",
    srcs = ["unit.cc"],
    hdrs = ["unit.h"],
    deps = [
        ":json_type_caster",
        ":result_type_caster",
        ":serialization",
        ":status",
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json_binding:unit",
        "//tensorstore/util:executor",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:unit",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "type_name_override",
    hdrs = ["type_name_override.h"],
    deps = [
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "serialization",
    srcs = ["serialization.cc"],
    hdrs = ["serialization.h"],
    deps = [
        ":garbage_collection",
        ":gil_safe",
        ":result_type_caster",
        ":status",
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:unowned_to_shared",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:str_cat",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:string_reader",
        "@local_config_python//:python_headers",  # build_cleaner: keep
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "kvstore",
    srcs = ["kvstore.cc"],
    hdrs = ["kvstore.h"],
    deps = [
        ":context",
        ":future",
        ":garbage_collection",
        ":json_type_caster",
        ":keyword_arguments",
        ":result_type_caster",
        ":serialization",
        ":status",
        ":tensorstore_module_components",
        ":time",
        ":transaction",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/kvstore",
        "//tensorstore/util:executor",
        "//tensorstore/util/garbage_collection",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

tensorstore_pytest_test(
    name = "kvstore_test",
    size = "small",
    srcs = ["tests/kvstore_test.py"],
    deps = [
        ":tensorstore",
    ],
)

tensorstore_cc_library(
    name = "time",
    srcs = ["time.cc"],
    hdrs = ["time.h"],
    deps = ["@com_google_absl//absl/time"],
)

pybind11_cc_library(
    name = "virtual_chunked",
    srcs = ["virtual_chunked.cc"],
    deps = [
        ":array_type_caster",
        ":future",
        ":gil_safe",
        ":result_type_caster",
        ":serialization",
        ":spec",
        ":status",
        ":tensorstore_class",
        ":tensorstore_module_components",
        ":time",
        ":transaction",
        ":type_name_override",
        "//tensorstore:virtual_chunked",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
        "@com_google_absl//absl/base:core_headers",
    ],
    alwayslink = True,
)

tensorstore_pytest_test(
    name = "virtual_chunked_test",
    size = "small",
    srcs = ["tests/virtual_chunked_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_cloudpickle//:cloudpickle",
        "@pypa_numpy//:numpy",
    ],
)

pybind11_cc_library(
    name = "metrics",
    srcs = ["metrics.cc"],
    deps = [
        ":json_type_caster",
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

tensorstore_pytest_test(
    name = "metrics_test",
    size = "small",
    srcs = ["tests/metrics_test.py"],
    deps = [
        ":tensorstore",
    ],
)
