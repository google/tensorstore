load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_cc_library(
    name = "gcs",
    deps = [
        "//tensorstore/kvstore/gcs_http",
    ],
    alwayslink = True,
)

tensorstore_cc_library(
    name = "validate",
    srcs = [
        "validate.cc",
    ],
    hdrs = ["validate.h"],
    deps = [
        "//tensorstore/internal:utf8",
        "//tensorstore/kvstore:generation",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "validate_test",
    size = "small",
    srcs = ["validate_test.cc"],
    deps = [
        ":validate",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "gcs_resource",
    srcs = ["gcs_resource.cc"],
    hdrs = ["gcs_resource.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal:retries_context_resource",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "gcs_testbench",
    testonly = 1,
    srcs = ["gcs_testbench.cc"],
    hdrs = ["gcs_testbench.h"],
    deps = [
        "//tensorstore/internal/grpc:utils",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/os:subprocess",
        "//tensorstore/proto:parse_text_proto_or_die",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googleapis//google/storage/v2:storage_cc_grpc",
        "@com_google_googleapis//google/storage/v2:storage_cc_proto",
    ],
)
