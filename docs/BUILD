package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

py_library(
    name = "json_pprint",
    srcs = ["json_pprint.py"],
)

py_library(
    name = "tensorstore_jsonschema_sphinx",
    srcs = ["tensorstore_jsonschema_sphinx.py"],
    deps = [
        ":json_pprint",
        "@pypa_docutils//:docutils",
        "@pypa_jsonpointer//:jsonpointer",
        "@pypa_jsonschema//:jsonschema",
        "@pypa_pyyaml//:pyyaml",
        "@pypa_sphinx//:sphinx",
    ],
)

py_library(
    name = "tensorstore_autosummary",
    srcs = ["tensorstore_autosummary.py"],
    deps = [
        "@pypa_astor//:astor",
        "@pypa_docutils//:docutils",
        "@pypa_sphinx//:sphinx",
    ],
)

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

py_test(
    name = "build_docs",
    srcs = ["build_docs.py"],
    data = [
        ":doc_sources",
    ],
    python_version = "PY3",
    tags = ["manual"],
    deps = [
        ":tensorstore_autosummary",
        ":tensorstore_jsonschema_sphinx",
        "//python/tensorstore",
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
    ],
)
