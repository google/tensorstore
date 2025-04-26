load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = ["common.h"],
    deps = [
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/aws:aws_api",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/aws:http_mocking",
        "//tensorstore/internal/aws:string_view",
        "@abseil-cpp//absl/strings",
        "@aws_c_auth",
        "@aws_c_common",
    ],
)

tensorstore_cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "default_credential_provider_test",
    size = "small",
    srcs = ["default_credential_provider_test.cc"],
    # args = ["--tensorstore_verbose_logging=aws=2"],
    deps = [
        ":common",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/aws:http_mocking",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "environmental_credential_provider_test",
    size = "small",
    srcs = ["environment_credential_provider_test.cc"],
    # args = ["--tensorstore_verbose_logging=aws"],
    deps = [
        ":common",
        "//tensorstore/internal:env",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "file_credential_provider_test",
    size = "small",
    srcs = ["file_credential_provider_test.cc"],
    # args = ["--tensorstore_verbose_logging=aws=2"],
    deps = [
        ":common",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/aws:http_mocking",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "imds_credential_provider_test",
    size = "small",
    srcs = ["imds_credential_provider_test.cc"],
    # args = ["--tensorstore_verbose_logging=aws"],
    deps = [
        ":common",
        ":test_utils",
        "//tensorstore/internal:env",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/aws:http_mocking",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)
