package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/workspace.bzl",
        "**/system.BUILD.bazel",
    ]),
)

# Expose files named *.template and **.h. This is used by the org_aomedia_aom build and
# the c-ares build.
filegroup(
    name = "templates",
    srcs = glob([
        "**/*.c",
        "**/*.asm",
        "**/*.h",
    ]),
)
