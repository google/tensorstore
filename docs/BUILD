load("//python/tensorstore:pytype.bzl", "pytype_strict_binary", "pytype_strict_test")
load("//python/tensorstore:pytest.bzl", "tensorstore_pytest_test")
load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//docs:doctest.bzl", "doctest_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = [
        "conf.py",
        "_templates/logo.svg",
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
    exec_tools = [
        ":generate_logo",
    ],
)

bool_flag(
    name = "no_minify",
    build_setting_default = False,
)

config_setting(
    name = "no_minify_setting",
    flag_values = {
        ":no_minify": "True",
    },
    visibility = ["//docs:__subpackages__"],
)

pytype_strict_test(
    name = "build_docs",
    srcs = ["build_docs.py"],
    args = select({
        ":no_minify_setting": ["--no-minify"],
        "//conditions:default": [],
    }) + [
        # Treat warnings as errors
        "-W",
    ],
    data = [":doc_sources"],
    python_version = "PY3",
    tags = [
        "manual",
        "nosan",
        "optonly",
    ],
    deps = [
        "//docs/tensorstore_sphinx_ext:autodoc",
        "//docs/tensorstore_sphinx_ext:autosummary",
        "//docs/tensorstore_sphinx_ext:doctest",
        "//docs/tensorstore_sphinx_ext:jsonschema_sphinx",
        "//docs/tensorstore_sphinx_material",
        "//python/tensorstore",
        "@pypa_htmlmin//:htmlmin",
        "@pypa_sphinx//:sphinx",
    ],
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
    tags = ["manual"],
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
