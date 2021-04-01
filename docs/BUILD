load("//python/tensorstore:pytype.bzl", "pytype_strict_test")

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

pytype_strict_test(
    name = "build_docs",
    srcs = ["build_docs.py"],
    data = [
        ":doc_sources",
    ],
    python_version = "PY3",
    tags = ["manual"],
    deps = [
        "//python/tensorstore",
        "//tensorstore/docs/tensorstore_sphinx_ext:autosummary",
        "//tensorstore/docs/tensorstore_sphinx_ext:jsonschema_sphinx",
        "@pypa_sphinx//:sphinx",
        "@pypa_sphinx_rtd_theme//:sphinx_rtd_theme",
    ],
)

py_binary(
    name = "update_doctests",
    srcs = ["update_doctests.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//python/tensorstore",
        "@pypa_numpy//:numpy",
    ],
)
