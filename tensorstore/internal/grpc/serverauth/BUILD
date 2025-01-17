load("//bazel:tensorstore.bzl", "tensorstore_cc_library")

package(default_visibility = ["//tensorstore:internal_packages"])

tensorstore_cc_library(
    name = "strategy",
    hdrs = ["strategy.h"],
    deps = ["@com_github_grpc_grpc//:grpc++"],
)

tensorstore_cc_library(
    name = "default_strategy",
    srcs = ["default_strategy.cc"],
    hdrs = ["default_strategy.h"],
    deps = [
        ":strategy",
        "@com_github_grpc_grpc//:grpc++",
    ],
)
