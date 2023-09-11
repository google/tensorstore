load("//bazel:tensorstore.bzl", "tensorstore_cc_library")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "name_configuration_json_binder",
    hdrs = ["name_configuration_json_binder.h"],
    deps = [
        "//tensorstore/internal/json_binding",
        "@com_google_absl//absl/status",
    ],
)
