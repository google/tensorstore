load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:pytest.bzl", "tensorstore_pytest_test")
load("//bazel:pytype.bzl", "pytype_strict_binary", "pytype_strict_test")
load("//bazel:tensorstore.bzl", "tensorstore_cc_library")
load("//docs:defs.bzl", "cc_preprocessed_output")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

# To exclude the Python API documentation from the generation documentation,
# specify:
#
# bazel build --//docs:exclude_python_api
#
# This significantly speeds up the documentation build since the Python
# extension module (which has a large dependency tree) does not have to be
# built.
bool_flag(
    name = "exclude_python_api",
    build_setting_default = False,
)

config_setting(
    name = "exclude_python_api_setting",
    flag_values = {
        ":exclude_python_api": "True",
    },
    visibility = ["//visibility:private"],
)

filegroup(
    name = "doc_sources",
    srcs = [
        "_templates/logo.svg",
        "conf.py",
    ] + glob([
        "**/*.rst",
        "**/*.yml",
        "intersphinx_inv/**",
    ]) + [
        "//tensorstore/driver:doc_sources",
        "//tensorstore/kvstore:doc_sources",
        "//third_party:doc_sources",
    ],
)

pytype_strict_binary(
    name = "generate_logo",
    srcs = ["generate_logo.py"],
    python_version = "PY3",
    deps = [
        "@pypa_numpy//:numpy",
    ],
)

genrule(
    name = "generate_logo_rule",
    outs = ["_templates/logo.svg"],
    cmd = "./$(location :generate_logo) $@",
    tools = [
        ":generate_logo",
    ],
)

pytype_strict_test(
    name = "build_docs",
    timeout = "long",
    srcs = ["build_docs.py"],
    args = [
        # Treat warnings as errors
        "-W",
        # Keep going after the first warning.
        "--keep-going",
    ],
    data = [
        "cpp_api.json",
        ":doc_sources",
        "@pypa_clang_format//:clang-format_binary",
    ] + glob(
        ["cached_external_resources/**"],
        allow_empty = True,
    ),
    env = {
        "TENSORSTORE_SPECIAL_CPU_USER_LIMITS": "forge-00=4",
        "SPHINX_CLANG_FORMAT": "$(location @pypa_clang_format//:clang-format_binary)",
    },
    python_version = "PY3",
    tags = [
        "cpu:4",
        "manual",
        "nosan",
        "optonly",
    ],
    deps = [
        "//docs/tensorstore_sphinx_ext:doctest",
        "@pypa_libclang//:libclang",  # buildcleaner: keep
        "@pypa_pyyaml//:pyyaml",  # buildcleaner: keep
        "@pypa_sphinx//:sphinx",
        "@pypa_sphinx_immaterial//:sphinx_immaterial",
    ] + select({
        ":exclude_python_api_setting": [],
        "//conditions:default": ["//python/tensorstore:core"],
    }),
)

pytype_strict_binary(
    name = "update_doctests",
    testonly = True,
    srcs = ["doctest_test.py"],
    args = ["--"],
    main = "doctest_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//docs/tensorstore_sphinx_ext:json_pprint",
        "//python/tensorstore",
        "@pypa_absl_py//:absl_py",
        "@pypa_numpy//:numpy",
        "@pypa_yapf//:yapf",
    ],
)

tensorstore_pytest_test(
    name = "_doctest_test",
    size = "medium",
    srcs = [
        "conftest.py",
        "doctest_test.py",
    ],
    tags = [
        "manual",
    ],
    tests = ["doctest_test.py"],
    deps = [
        "//docs/tensorstore_sphinx_ext:json_pprint",
        "//python/tensorstore",
        "@pypa_numpy//:numpy",
        "@pypa_yapf//:yapf",
    ],
)

doctest_test(
    name = "doctest_test",
    srcs = glob(["python/**/*.rst"]),
)

tensorstore_cc_library(
    name = "cpp_api_include",
    testonly = True,
    srcs = ["cpp_api_include.cc"],
    copts = [
        # Generated preprocessed output rather than object file.
        "-E",
        # Retain comments.
        "-C",
        # GCC/clang flag to output macro definitions.
        "-dD",
    ],
    features = [
        "-use_header_modules",
        "-layering_check",
    ],
    linkstatic = True,
    local_defines = [
        "TENSORSTORE_CPP_DOC_GENERATION",
    ],
    tags = ["manual"],
    deps = [
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:cast",
        "//tensorstore:data_type",
        "//tensorstore:downsample",
        "//tensorstore:open",
        "//tensorstore:rank",
        "//tensorstore/index_space:alignment",
        "//tensorstore/index_space:dim_expression",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transformed_array",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:element_traits",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
    ],
)

# Define a special constraint_setting that will be matched in order to select a
# hermetic clang toolchain for preprocessing the C++ API headers used for the
# API documentation.  This is used when the default toolchain is not clang.
#
# The C++ API documentation is extracted using libclang. libstdc++ (typically
# used on Linux) is compatible with clang but only when preprocessed with clang.
# If it is first preprocessed by GCC, then the resultant output contains
# GCC-specific builtins, etc. and is not compatible with libclang.
constraint_setting(
    name = "docs_toolchain_setting",
    visibility = ["//visibility:public"],
)

constraint_value(
    name = "docs_toolchain_value",
    constraint_setting = ":docs_toolchain_setting",
    visibility = ["//visibility:public"],
)

platform(
    name = "docs_toolchain_platform",
    constraint_values = [":docs_toolchain_value"],
    parents = ["@platforms//host"],
    visibility = ["//visibility:public"],
)

cc_preprocessed_output(
    name = "cpp_api_preprocessed.cc",
    testonly = True,
    cpp_compiler_constraint = select({
        "//:compiler_clang": "@platforms//host",
        "//conditions:default": ":docs_toolchain_platform",
    }),
    flags_output = "compiler_flags.json",
    tags = ["manual"],
    target = ":cpp_api_include",
)

pytype_strict_binary(
    name = "generate_cpp_api",
    srcs = ["generate_cpp_api.py"],
    deps = [
        "@pypa_libclang//:libclang",  # buildcleaner: keep
        "@pypa_sphinx_immaterial//:sphinx_immaterial",
    ],
)

pytype_strict_binary(
    name = "cpp_api_shell",
    testonly = True,
    srcs = ["generate_cpp_api.py"],
    args = [
        "--source=$(location :cpp_api_preprocessed.cc)",
        "--flags-file=$(location :compiler_flags.json)",
        "--interactive",
    ],
    data = [
        ":compiler_flags.json",
        ":cpp_api_preprocessed.cc",
    ],
    main = "generate_cpp_api.py",
    tags = ["manual"],
    deps = [
        "@pypa_libclang//:libclang",  # buildcleaner: keep
        "@pypa_sphinx_immaterial//:sphinx_immaterial",
    ],
)

genrule(
    name = "genrule_cpp_api.json",
    testonly = True,
    srcs = [
        ":cpp_api_preprocessed.cc",
        ":compiler_flags.json",
    ],
    outs = ["cpp_api.json"],
    cmd = ("$(location :generate_cpp_api) " +
           "--source=$(location :cpp_api_preprocessed.cc) " +
           "--flags-file=$(location :compiler_flags.json) " +
           "--output=$@"),
    tags = ["manual"],
    tools = [
        ":generate_cpp_api",
    ],
)
