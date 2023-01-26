load("//bazel:pytype.bzl", "pytype_strict_binary", "pytype_strict_test")
load("//bazel:pytest.bzl", "tensorstore_pytest_test")
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
    data = [":doc_sources"] + glob(
        ["cached_external_resources/**"],
        allow_empty = True,
    ),
    env = {
        "TENSORSTORE_SPECIAL_CPU_USER_LIMITS": "forge-00=4",
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
        "//python/tensorstore",
        "@pypa_pyyaml//:pyyaml",  # buildcleaner: keep
        "@pypa_sphinx//:sphinx",
        "@pypa_sphinx_immaterial//:sphinx_immaterial",
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
