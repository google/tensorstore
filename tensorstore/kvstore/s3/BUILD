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
    name = "s3",
    srcs = ["s3_key_value_store.cc"],
    hdrs = [],
    deps = [
        "@com_google_absl//absl/strings",
        "//tensorstore/internal/http",
    ],
    alwayslink = 1,
)


tensorstore_cc_library(
    name = "s3_request_builder",
    srcs = [
        "s3_request_builder.cc",
    ],
    hdrs = [
        "s3_request_builder.h",
    ],
    deps = [
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "//tensorstore/internal/http",
        "//tensorstore/internal:ascii_utils",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal:path",
        ":validate"
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "s3_request_builder_test",
    size = "small",
    srcs = ["s3_request_builder_test.cc"],
    deps = [
        ":s3_request_builder",
        "@com_google_googletest//:gtest_main",
        "//tensorstore/util:status_testutil",
    ],
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
        "@com_google_absl//absl/strings",
        "@com_google_re2//:re2",
    ],
)

tensorstore_cc_test(
    name = "validate_test",
    size = "small",
    srcs = [
        "validate_test.cc",
    ],
    deps = [
        ":validate",
        "@com_google_googletest//:gtest_main",
    ],
)
