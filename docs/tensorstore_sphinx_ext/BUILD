load("//bazel:pytype.bzl", "pytype_strict_library")

package(
    default_visibility = ["//docs:__subpackages__"],
)

licenses(["notice"])

pytype_strict_library(
    name = "__init__",
    srcs = ["__init__.py"],
)

pytype_strict_library(
    name = "json_pprint",
    srcs = ["json_pprint.py"],
    deps = [":__init__"],
)

pytype_strict_library(
    name = "doctest",
    srcs = ["doctest.py"],
    deps = [
        ":__init__",
        "@pypa_docutils//:docutils",
        "@pypa_sphinx//:sphinx",
        "@pypa_sphinx_immaterial//:sphinx_immaterial",
    ],
)
