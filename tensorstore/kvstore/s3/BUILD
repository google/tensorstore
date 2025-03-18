# Placeholder: load py_binary
load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "*.rst",
        "*.yml",
    ]),
)

tensorstore_cc_library(
    name = "s3",
    srcs = ["s3_key_value_store.cc"],
    deps = [
        ":aws_credentials",
        ":aws_credentials_resource",
        ":aws_credentials_spec",
        ":s3_endpoint",
        ":s3_metadata",
        ":s3_request_builder",
        ":s3_resource",
        ":s3_uri_utils",
        ":use_conditional_write",
        ":validate",
        "//tensorstore:context",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/rate_limiter",
        "//tensorstore/internal/thread:schedule_at",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:common_metrics",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/http:byte_range_util",
        "//tensorstore/serialization",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@tinyxml2",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "aws_credentials_resource",
    srcs = ["aws_credentials_resource.cc"],
    hdrs = ["aws_credentials_resource.h"],
    deps = [
        ":aws_credentials_spec",
        ":s3_resource",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/cache_key:json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/serialization",
        "//tensorstore/util:result",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "aws_credentials_resource_test",
    size = "small",
    srcs = ["aws_credentials_resource_test.cc"],
    deps = [
        ":aws_credentials_resource",
        "//tensorstore:context",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "s3_metadata",
    srcs = [
        "s3_metadata.cc",
    ],
    hdrs = [
        "s3_metadata.h",
    ],
    deps = [
        "//tensorstore/internal:source_location",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:no_destructor",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_re2//:re2",
        "@tinyxml2",
    ],
)

tensorstore_cc_test(
    name = "s3_metadata_test",
    size = "small",
    srcs = [
        "s3_metadata_test.cc",
    ],
    deps = [
        ":s3",
        ":s3_metadata",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
        "@tinyxml2",
    ],
)

tensorstore_cc_library(
    name = "s3_uri_utils",
    hdrs = ["s3_uri_utils.h"],
    deps = ["//tensorstore/internal:uri_utils"],
)

tensorstore_cc_library(
    name = "s3_resource",
    srcs = ["s3_resource.cc"],
    hdrs = ["s3_resource.h"],
    deps = [
        "//tensorstore:context",
        "//tensorstore/internal:env",
        "//tensorstore/internal:retries_context_resource",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/rate_limiter",
        "//tensorstore/internal/rate_limiter:admission_queue",
        "//tensorstore/internal/rate_limiter:scaling_rate_limiter",
        "//tensorstore/util:result",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/time",
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
        ":aws_credentials",
        ":s3_uri_utils",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore:byte_range",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_boringssl//:crypto",
    ],
)

tensorstore_cc_test(
    name = "s3_request_builder_test",
    size = "small",
    srcs = ["s3_request_builder_test.cc"],
    deps = [
        ":aws_credentials",
        ":s3_request_builder",
        "//tensorstore/internal/http",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
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

tensorstore_cc_library(
    name = "s3_endpoint",
    srcs = ["s3_endpoint.cc"],
    hdrs = ["s3_endpoint.h"],
    deps = [
        ":validate",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/http",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
    ],
)

tensorstore_cc_test(
    name = "s3_endpoint_test",
    size = "small",
    srcs = ["s3_endpoint_test.cc"],
    deps = [
        ":s3_endpoint",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)

py_binary(
    name = "moto_server",
    testonly = 1,
    srcs = ["moto_server.py"],
    tags = [
        "manual",
        "notap",
        "skip-cmake",
    ],
    deps = ["@pypa_moto//:moto"],
)

tensorstore_cc_test(
    name = "s3_key_value_store_test",
    size = "small",
    srcs = ["s3_key_value_store_test.cc"],
    deps = [
        ":s3",
        "//tensorstore:context",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "localstack_test",
    size = "medium",
    srcs = ["localstack_test.cc"],
    args = [
        "--localstack_binary=$(location :moto_server)",
        "--binary_mode=moto",
    ],
    data = [":moto_server"],
    flaky = 1,  # Spawning the test process can be flaky.
    tags = [
        "cpu:2",
        "requires-net:loopback",
        "skip-cmake",
    ],
    deps = [
        ":aws_credentials",
        ":s3",
        ":s3_request_builder",
        ":use_conditional_write",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:env",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/curl:curl_transport",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/os:subprocess",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/s3/credentials:common",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "use_conditional_write",
    srcs = ["use_conditional_write.cc"],
    hdrs = ["use_conditional_write.h"],
    deps = [
        "//tensorstore/internal:env",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/strings",
        "@com_google_re2//:re2",
    ],
)

tensorstore_cc_test(
    name = "use_conditional_write_test",
    srcs = ["use_conditional_write_test.cc"],
    deps = [
        ":use_conditional_write",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "aws_api",
    srcs = [
        "aws_api.cc",
    ],
    hdrs = ["aws_api.h"],
    deps = [
        "//tensorstore/internal/log:verbose_flag",
        "@aws_c_auth",
        "@aws_c_common",
        "@aws_c_io",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:absl_log",
    ],
)

tensorstore_cc_test(
    name = "aws_api_test",
    size = "small",
    srcs = ["aws_api_test.cc"],
    # args = ["--tensorstore_verbose_logging=aws"],
    deps = [
        ":aws_api",
        "@aws_c_common",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "aws_credentials",
    srcs = ["aws_credentials.cc"],
    hdrs = ["aws_credentials.h"],
    deps = [
        ":aws_api",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:future",
        "@aws_c_auth",
        "@aws_c_common",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "aws_credentials_test",
    srcs = ["aws_credentials_test.cc"],
    args = ["--tensorstore_verbose_logging=aws"],
    deps = [
        ":aws_credentials",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "aws_credentials_spec",
    srcs = ["aws_credentials_spec.cc"],
    hdrs = ["aws_credentials_spec.h"],
    deps = [
        ":aws_credentials",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore/s3/credentials:common",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@aws_c_common",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "aws_credentials_spec_test",
    srcs = ["aws_credentials_spec_test.cc"],
    deps = [
        ":aws_credentials_spec",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "aws_http_mocking",
    srcs = ["aws_http_mocking.cc"],
    hdrs = ["aws_http_mocking.h"],
    deps = [
        ":aws_api",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:http_header",
        "//tensorstore/internal/log:verbose_flag",
        "@aws_c_auth",
        "@aws_c_common",
        "@aws_c_http",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/debugging:leak_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
    ],
)
