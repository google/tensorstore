load("@bazel_skylib//rules:common_settings.bzl", "string_flag")
load("//bazel:tag_release.bzl", "tag_release")
load("//bazel:tensorstore.bzl", "tensorstore_cc_library")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

string_flag(
    name = "version",
    build_setting_default = "0.1.dev",
)

tag_release(
    name = "generate_version_h",
    out = "version.h",
    label = ":version",
)

tensorstore_cc_library(
    name = "version_h",
    hdrs = ["version.h"],
)
