load("//python/tensorstore:pytype.bzl", "pytype_strict_binary", "pytype_strict_test")
load("//python/tensorstore:pytest.bzl", "tensorstore_pytest_test")

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = [
        "conf.py",
    ] + glob([
        "_templates/**",
        "_static/**",
        "**/*.rst",
        "**/*.yml",
        "intersphinx_inv/**",
    ]),
    data = [
        "//tensorstore/driver:doc_sources",
        "//tensorstore/kvstore:doc_sources",
        "//third_party:doc_sources",
    ],
)

filegroup(
    name = "doctest_sources",
    testonly = True,
    srcs = glob(["python/**/*.rst"]) + [
        "//python/tensorstore:doctest_sources",
    ],
)

pytype_strict_test(
    name = "build_docs",
    srcs = ["build_docs.py"],
    data = [
        ":doc_sources",
    ],
    python_version = "PY3",
    tags = ["manual"],
    deps = [
        "//docs/tensorstore_sphinx_ext:autosummary",
        "//docs/tensorstore_sphinx_ext:jsonschema_sphinx",
        "//python/tensorstore",
        "@pypa_sphinx//:sphinx",
        "@pypa_sphinx_rtd_theme//:sphinx_rtd_theme",
    ],
)

pytype_strict_binary(
    name = "update_doctests",
    testonly = True,
    srcs = ["doctest_test.py"],
    main = "doctest_test.py",
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//python/tensorstore",
        "@pypa_numpy//:numpy",
        "@pypa_yapf//:yapf",
    ],
)

tensorstore_pytest_test(
    name = "doctest_test",
    size = "medium",
    srcs = [
        "conftest.py",
        "doctest_test.py",
    ],
    args = [
        "--doctests",
        "$(locations :doctest_sources)",
    ],
    data = [
        ":doctest_sources",
    ],
    env = {
        # Ensure that even if the test is running on GCE, no attempt is made to
        # use GCE credentials.
        "GCE_METADATA_ROOT": "metadata.google.internal.invalid",
    },
    deps = [
        "//python/tensorstore",
        "@pypa_numpy//:numpy",
        "@pypa_yapf//:yapf",
    ],
)
