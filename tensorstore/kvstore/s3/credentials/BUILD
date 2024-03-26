load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "aws_credentials",
    hdrs = ["aws_credentials.h"],
    deps = [
        "//tensorstore/util:result",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "environment_credential_provider",
    srcs = ["environment_credential_provider.cc"],
    hdrs = ["environment_credential_provider.h"],
    deps = [
        ":aws_credentials",
        "//tensorstore/internal:env",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "file_credential_provider",
    srcs = ["file_credential_provider.cc"],
    hdrs = ["file_credential_provider.h"],
    deps = [
        ":aws_credentials",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/util:result",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@com_google_riegeli//riegeli/lines:line_reading",
    ],
)

tensorstore_cc_library(
    name = "ec2_credential_provider",
    srcs = ["ec2_credential_provider.cc"],
    hdrs = ["ec2_credential_provider.h"],
    deps = [
        ":aws_credentials",
        "//tensorstore/internal:env",
        "//tensorstore/internal/http",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "default_credential_provider",
    srcs = ["default_credential_provider.cc"],
    hdrs = ["default_credential_provider.h"],
    deps = [
        ":aws_credentials",
        ":ec2_credential_provider",
        ":environment_credential_provider",
        ":file_credential_provider",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/util:result",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    deps = [
        "//tensorstore/internal/http",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "default_credential_provider_test",
    size = "small",
    srcs = ["default_credential_provider_test.cc"],
    deps = [
        ":default_credential_provider",
        ":test_utils",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "environmental_credential_provider_test",
    size = "small",
    srcs = ["environment_credential_provider_test.cc"],
    deps = [
        ":environment_credential_provider",
        "//tensorstore/internal:env",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "file_credential_provider_test",
    size = "small",
    srcs = ["file_credential_provider_test.cc"],
    deps = [
        ":file_credential_provider",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "ec2_credential_provider_test",
    size = "small",
    srcs = ["ec2_credential_provider_test.cc"],
    deps = [
        ":ec2_credential_provider",
        ":test_utils",
        "//tensorstore/internal:env",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
