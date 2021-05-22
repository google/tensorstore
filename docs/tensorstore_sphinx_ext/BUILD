load("//python/tensorstore:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = ["//docs:__subpackages__"],
)

licenses(["notice"])

pytype_strict_library(
    name = "json_pprint",
    srcs = ["json_pprint.py"],
)

pytype_strict_library(
    name = "sphinx_utils",
    srcs = ["sphinx_utils.py"],
    deps = [
        "@pypa_docutils//:docutils",
        "@pypa_sphinx//:sphinx",
    ],
)

pytype_strict_library(
    name = "jsonschema_sphinx",
    srcs = ["jsonschema_sphinx.py"],
    deps = [
        ":json_pprint",
        ":sphinx_utils",
        "@pypa_docutils//:docutils",
        "@pypa_jsonschema//:jsonschema",
        "@pypa_pyyaml//:pyyaml",
        "@pypa_sphinx//:sphinx",
    ],
)

pytype_strict_library(
    name = "autosummary",
    srcs = ["autosummary.py"],
    deps = [
        ":sphinx_utils",
        "@pypa_docutils//:docutils",
        "@pypa_sphinx//:sphinx",
    ],
)

pytype_strict_library(
    name = "autodoc",
    srcs = ["autodoc.py"],
    deps = [
        "@pypa_docutils//:docutils",
        "@pypa_sphinx//:sphinx",
    ],
)

pytype_strict_library(
    name = "mathjax",
    srcs = ["mathjax.py"],
    deps = [
        "@pypa_sphinx//:sphinx",
    ],
)
