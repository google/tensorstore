load("@rules_python//python:py_library.bzl", "py_library")
load(
    "//bazel:pybind11.bzl",
    "pybind11_cc_googletest_test",
    "pybind11_cc_library",
    "pybind11_py_extension",
)
load("//bazel:pytest.bzl", "tensorstore_pytest_test")
load("//bazel:pytype.bzl", "pytype_strict_binary", "pytype_strict_library", "pytype_strict_test")
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

PYI_FILES = glob(["**/*.pyi"])

doctest_test(
    name = "doctest_test",
    srcs = glob([
        "*.py",
        "*.cc",
    ]),
    tags = [
        "skip-windows",
    ],
)

pybind11_py_extension(
    name = "_tensorstore",
    srcs = ["tensorstore_module.cc"],
    imports = [".."],
    deps = [
        ":batch",
        ":cast",
        ":chunk_layout",
        ":context",
        ":data_type",
        ":dim_expression",
        ":downsample",
        ":experimental",
        ":future",
        ":garbage_collection",
        ":gil_safe",
        ":index_space",
        ":kvstore",
        ":numpy",
        ":ocdbt",
        ":python_imports",
        ":serialization",
        ":spec",
        ":stack",
        ":tensorstore_class",
        ":tensorstore_module_components",
        ":transaction",
        ":unit",
        ":virtual_chunked",
        ":write_futures",
        "//tensorstore:all_drivers",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/debugging:leak_check",
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/log:initialize",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

py_library(
    name = "_tensorstore_with_python_deps",
    deps = [
        ":_tensorstore",
        "@pypa_ml_dtypes//:ml_dtypes",
        "@pypa_numpy//:numpy",
    ],
)

pybind11_cc_library(
    name = "tensorstore_module_components",
    srcs = ["tensorstore_module_components.cc"],
    hdrs = ["tensorstore_module_components.h"],
    deps = [
        "//tensorstore/util:executor",
        "@abseil-cpp//absl/base:no_destructor",
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

pytype_strict_library(
    name = "core",
    srcs = ["__init__.py"],
    pytype_srcs = PYI_FILES,
    visibility = ["//visibility:public"],
    deps = [
        ":_tensorstore_with_python_deps",
    ],
)

py_library(
    name = "tensorstore",
    visibility = ["//visibility:public"],
    deps = [":core"],
)

pytype_strict_binary(
    name = "shell",
    srcs = ["shell.py"],
    tags = ["manual"],
    deps = [
        ":tensorstore",
        "@pypa_absl_py//:absl_py",
        "@pypa_ipython//:ipython",
        "@pypa_numpy//:numpy",
    ],
)

# On windows conftest.py adds DLL import paths necessary for pytest to run.
py_library(
    name = "conftest",
    testonly = True,
    srcs = ["tests/conftest.py"],
    deps = ["@pypa_pytest//:pytest"],
)

tensorstore_pytest_test(
    name = "dim_test",
    size = "small",
    srcs = ["tests/dim_test.py"],
    tags = ["cpu:2"],
    deps = [
        ":conftest",
        ":tensorstore",
    ],
)

tensorstore_pytest_test(
    name = "index_domain_test",
    size = "small",
    srcs = ["tests/index_domain_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "dim_expression_test",
    size = "small",
    srcs = ["tests/dim_expression_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "indexing_test",
    size = "small",
    srcs = ["tests/indexing_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "index_transform_test",
    size = "small",
    srcs = ["tests/index_transform_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "downsample_test",
    size = "small",
    srcs = ["tests/downsample_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "stack_test",
    size = "small",
    srcs = ["tests/stack_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

pybind11_cc_library(
    name = "subscript_method",
    hdrs = ["subscript_method.h"],
    deps = [
        "//tensorstore/internal/meta:type_traits",
        "@abseil-cpp//absl/strings:str_format",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "json_type_caster",
    srcs = ["json_type_caster.cc"],
    hdrs = ["json_type_caster.h"],
    deps = [
        ":critical_section",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
    ],
)

pybind11_cc_library(
    name = "status",
    srcs = ["status.cc"],
    hdrs = ["status.h"],
    deps = [
        ":python_imports",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@boringssl//:crypto",
        "@com_github_pybind_pybind11//:pybind11",
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
        "//tensorstore/index_space:index_vector_or_scalar",
        "//tensorstore/index_space:numpy_indexing_spec",
        "//tensorstore/util:span",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "dim_expression",
    srcs = ["dim_expression.cc"],
    hdrs = ["dim_expression.h"],
    deps = [
        ":critical_section",
        ":index",
        ":numpy_indexing_spec",
        ":sequence_parameter",
        ":serialization",
        ":subscript_method",
        ":tensorstore_module_components",
        ":typed_slice",
        "//tensorstore:index",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:numpy_indexing_spec",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

tensorstore_pytest_test(
    name = "exit_test",
    size = "small",
    srcs = ["tests/exit_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
    ],
)

pybind11_cc_library(
    name = "index_space",
    srcs = ["index_space.cc"],
    hdrs = ["index_space.h"],
    deps = [
        ":array_type_caster",
        ":critical_section",
        ":dim_expression",
        ":gil_safe",
        ":homogeneous_tuple",
        ":index",
        ":json_type_caster",
        ":locking_type_casters",
        ":numpy",
        ":numpy_indexing_spec",
        ":python_imports",
        ":result_type_caster",
        ":sequence_parameter",
        ":serialization",
        ":status",
        ":subscript_method",
        ":tensorstore_module_components",
        ":typed_slice",
        "//tensorstore:array",
        "//tensorstore:container_kind",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:dimension_identifier",
        "//tensorstore/index_space:dimension_index_buffer",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:numpy_indexing_spec",
        "//tensorstore/index_space:output_index_method",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:bit_span",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:executor",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/strings:str_format",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
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
    name = "typed_slice",
    hdrs = ["typed_slice.h"],
    deps = [
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
        "@abseil-cpp//absl/strings:str_format",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "spec",
    srcs = ["spec.cc"],
    hdrs = ["spec.h"],
    deps = [
        ":array_type_caster",
        ":context",
        ":critical_section",
        ":data_type",
        ":define_heap_type",
        ":garbage_collection",
        ":homogeneous_tuple",
        ":index_space",
        ":intrusive_ptr_holder",
        ":json_type_caster",
        ":keyword_arguments",
        ":kvstore",
        ":locking_type_casters",
        ":result_type_caster",
        ":sequence_parameter",
        ":serialization",
        ":status",
        ":tensorstore_module_components",
        ":unit",
        "//tensorstore:array",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:rank",
        "//tensorstore:schema",
        "//tensorstore:spec",
        "//tensorstore:staleness_bound",
        "//tensorstore/driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/kvstore",
        "//tensorstore/util:executor",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:unit",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "data_type",
    srcs = ["data_type.cc"],
    hdrs = ["data_type.h"],
    deps = [
        ":json_type_caster",
        ":numpy",
        ":serialization",
        ":tensorstore_module_components",
        "//tensorstore:data_type",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "//tensorstore/util:quote_string",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/strings:str_format",
        "@com_github_pybind_pybind11//:pybind11",
        "@pypa_numpy//:headers",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "tensorstore_class",
    srcs = ["tensorstore_class.cc"],
    hdrs = ["tensorstore_class.h"],
    deps = [
        ":array_type_caster",
        ":batch",
        ":context",
        ":critical_section",
        ":data_type",
        ":define_heap_type",
        ":future",
        ":garbage_collection",
        ":gil_safe",
        ":homogeneous_tuple",
        ":index",
        ":index_space",
        ":json_type_caster",
        ":keyword_arguments",
        ":kvstore",
        ":locking_type_casters",
        ":result_type_caster",
        ":sequence_parameter",
        ":serialization",
        ":spec",
        ":status",
        ":tensorstore_module_components",
        ":transaction",
        ":unit",
        ":write_futures",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:batch",
        "//tensorstore:cast",
        "//tensorstore:codec_spec",
        "//tensorstore:context",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:open",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:progress",
        "//tensorstore:rank",
        "//tensorstore:read_write_options",
        "//tensorstore:resize_options",
        "//tensorstore:schema",
        "//tensorstore:spec",
        "//tensorstore:strided_layout",
        "//tensorstore:transaction",
        "//tensorstore/driver/array",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/internal/json_binding",
        "//tensorstore/kvstore",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:unit",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
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
        ":status",
        ":tensorstore_module_components",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
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
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "critical_section",
    hdrs = [
        "critical_section.h",
        "with_handle.h",
    ],
    deps = ["@com_github_pybind_pybind11//:pybind11"],
)

pybind11_cc_library(
    name = "locking_type_casters",
    hdrs = ["locking_type_casters.h"],
    deps = [
        ":critical_section",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:batch",
        "//tensorstore:chunk_layout",
        "//tensorstore:index_interval",
        "//tensorstore:schema",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

pybind11_cc_library(
    name = "garbage_collection",
    srcs = ["garbage_collection.cc"],
    hdrs = ["garbage_collection.h"],
    deps = [
        ":critical_section",
        ":define_heap_type",
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:tagged_ptr",
        "//tensorstore/util:executor",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "future",
    srcs = ["future.cc"],
    hdrs = ["future.h"],
    deps = [
        ":critical_section",
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
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/container:intrusive_linked_list",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@com_github_pybind_pybind11//:pybind11",
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
        ":define_heap_type",
        ":future",
        ":garbage_collection",
        ":tensorstore_module_components",
        ":type_name_override",
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
        ":critical_section",
        ":data_type",
        ":index",
        ":numpy",
        ":result_type_caster",
        ":status",
        ":subscript_method",
        ":type_name_override",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore:static_cast",
        "//tensorstore/index_space:numpy_indexing_spec",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:iterate",
        "//tensorstore/util:span",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/meta:type_traits",
        "@abseil-cpp//absl/status",
        "@com_github_pybind_pybind11//:pybind11",
    ],
)

tensorstore_pytest_test(
    name = "context_test",
    size = "small",
    srcs = ["tests/context_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
    ],
)

tensorstore_pytest_test(
    name = "spec_test",
    size = "small",
    srcs = ["tests/spec_test.py"],
    tags = ["cpu:2"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "data_type_test",
    size = "small",
    srcs = ["tests/data_type_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "tensorstore_test",
    size = "small",
    srcs = ["tests/tensorstore_test.py"],
    tags = ["cpu:2"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "transaction_test",
    size = "small",
    srcs = ["tests/transaction_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "batch_test",
    size = "small",
    srcs = ["tests/batch_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "future_test",
    size = "medium",
    srcs = ["tests/future_test.py"],
    tags = ["cpu:4"],
    deps = [
        ":conftest",
        ":tensorstore",
    ],
)

tensorstore_pytest_test(
    name = "chunk_layout_test",
    size = "small",
    srcs = ["tests/chunk_layout_test.py"],
    tags = ["cpu:2"],
    deps = [
        ":conftest",
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
        "@googletest//:gtest",
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
        ":result_type_caster",
        ":tensorstore_module_components",
        "//tensorstore:transaction",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "batch",
    srcs = ["batch.cc"],
    hdrs = ["batch.h"],
    deps = [
        ":critical_section",
        ":locking_type_casters",
        ":tensorstore_module_components",
        "//tensorstore:batch",
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
        ":sequence_parameter",
        ":spec",
        ":status",
        ":tensorstore_class",
        ":tensorstore_module_components",
        "//tensorstore",
        "//tensorstore:downsample",
        "//tensorstore:downsample_method",
        "//tensorstore:index",
        "//tensorstore:spec",
        "//tensorstore/driver/downsample:downsample_method_json_binder",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "stack",
    srcs = ["stack.cc"],
    deps = [
        ":dim_expression",
        ":index",
        ":keyword_arguments",
        ":result_type_caster",
        ":sequence_parameter",
        ":spec",
        ":tensorstore_class",
        ":tensorstore_module_components",
        "//tensorstore",
        "//tensorstore:index",
        "//tensorstore:spec",
        "//tensorstore:stack",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/util:executor",
        "@com_github_pybind_pybind11//:pybind11",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "cast",
    srcs = ["cast.cc"],
    deps = [
        ":data_type",
        ":result_type_caster",
        ":spec",
        ":tensorstore_class",
        ":tensorstore_module_components",
        "//tensorstore",
        "//tensorstore:cast",
        "//tensorstore:spec",
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
        ":critical_section",
        ":homogeneous_tuple",
        ":index",
        ":index_space",
        ":json_type_caster",
        ":keyword_arguments",
        ":locking_type_casters",
        ":result_type_caster",
        ":sequence_parameter",
        ":serialization",
        ":status",
        ":tensorstore_module_components",
        "//tensorstore:chunk_layout",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:rank",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/util:executor",
        "//tensorstore/util:maybe_hard_constraint",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
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
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@com_github_pybind_pybind11//:pybind11",
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
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json_binding:bindable",
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
        ":critical_section",
        ":garbage_collection",
        ":gil_safe",
        ":result_type_caster",
        ":status",
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal:unowned_to_shared",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@com_github_pybind_pybind11//:pybind11",
        "@riegeli//riegeli/bytes:cord_writer",
        "@riegeli//riegeli/bytes:string_reader",
        "@riegeli//riegeli/bytes:writer",
        "@rules_python//python/cc:current_py_cc_headers",  # build_cleaner: keep
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "kvstore",
    srcs = ["kvstore.cc"],
    hdrs = ["kvstore.h"],
    deps = [
        ":batch",
        ":context",
        ":critical_section",
        ":define_heap_type",
        ":future",
        ":garbage_collection",
        ":json_type_caster",
        ":keyword_arguments",
        ":locking_type_casters",
        ":result_type_caster",
        ":serialization",
        ":status",
        ":tensorstore_module_components",
        ":time",
        ":transaction",
        ":type_name_override",
        ":typed_slice",
        "//tensorstore:batch",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:transaction",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/json:pprint_python",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

tensorstore_pytest_test(
    name = "kvstore_test",
    size = "medium",
    srcs = ["tests/kvstore_test.py"],
    tags = ["cpu:2"],
    deps = [
        ":conftest",
        ":tensorstore",
    ],
)

tensorstore_cc_library(
    name = "time",
    srcs = ["time.cc"],
    hdrs = ["time.h"],
    deps = ["@abseil-cpp//absl/time"],
)

pybind11_cc_library(
    name = "ocdbt",
    srcs = ["ocdbt.cc"],
    deps = [
        ":context",
        ":future",
        ":json_type_caster",
        ":kvstore",
        ":result_type_caster",
        ":status",
        ":tensorstore_module_components",
        "//tensorstore:context",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/kvstore/ocdbt:dump_util",
        "//tensorstore/kvstore/ocdbt/distributed:coordinator_server",
        "//tensorstore/kvstore/ocdbt/format:dump",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util/garbage_collection:json",
        "@abseil-cpp//absl/strings:cord",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

pybind11_cc_library(
    name = "virtual_chunked",
    srcs = ["virtual_chunked.cc"],
    deps = [
        ":array_type_caster",
        ":context",
        ":data_type",
        ":future",
        ":garbage_collection",
        ":gil_safe",
        ":keyword_arguments",
        ":locking_type_casters",
        ":result_type_caster",
        ":serialization",
        ":spec",
        ":status",
        ":tensorstore_class",
        ":tensorstore_module_components",
        ":time",
        ":transaction",
        ":type_name_override",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:batch",
        "//tensorstore:container_kind",
        "//tensorstore:context",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore:transaction",
        "//tensorstore:virtual_chunked",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@com_github_pybind_pybind11//:pybind11",
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
    name = "experimental",
    srcs = ["experimental.cc"],
    deps = [
        ":future",
        ":json_type_caster",
        ":tensorstore_module_components",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:prometheus",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/strings:cord",
        "@com_github_pybind_pybind11//:pybind11",
        "@nlohmann_json//:json",
    ],
    alwayslink = True,
)

tensorstore_pytest_test(
    name = "experimental_test",
    size = "small",
    srcs = ["tests/experimental_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
    ],
)

tensorstore_pytest_test(
    name = "custom_dtypes_test",
    size = "small",
    srcs = ["tests/custom_dtypes_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
        "@pypa_ml_dtypes//:ml_dtypes",
        "@pypa_numpy//:numpy",
    ],
)

tensorstore_pytest_test(
    name = "fork_test",
    size = "small",
    srcs = ["tests/fork_test.py"],
    deps = [
        ":conftest",
        ":tensorstore",
    ],
)

pytype_strict_test(
    name = "generate_type_stubs",
    size = "small",
    srcs = ["generate_type_stubs.py"],
    args = ["--strip-generic-slice"],
    data = ["__init__.py"] + PYI_FILES,
    main = "generate_type_stubs.py",
    deps = [
        ":_tensorstore_with_python_deps",  # build_cleaner: keep
        "@pypa_black//:black",
        "@pypa_pybind11_stubgen//:pybind11_stubgen",
    ],
)
