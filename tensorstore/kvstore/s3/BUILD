load("@rules_python//python:py_binary.bzl", "py_binary")
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
        ":aws_credentials_resource",
        ":aws_credentials_spec",
        ":s3_endpoint",
        ":s3_metadata",
        ":s3_request_builder",
        ":s3_resource",
        ":s3_uri_utils",
        ":validate",
        "//tensorstore:context",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/aws:aws_api",
        "//tensorstore/internal/aws:aws_credentials",
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
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
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
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
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
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@re2",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/base",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/time",
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
        ":s3_uri_utils",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore:byte_range",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@boringssl//:crypto",
    ],
)

tensorstore_cc_test(
    name = "s3_request_builder_test",
    size = "small",
    srcs = ["s3_request_builder_test.cc"],
    deps = [
        ":s3_request_builder",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/http",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/strings",
        "@re2",
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
        "@googletest//:gtest_main",
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
        "//tensorstore/internal/http:http_header",
        "//tensorstore/util:future",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@re2",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "localstack_test",
    size = "large",
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
        ":s3",
        ":s3_endpoint",
        ":s3_request_builder",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:env",
        "//tensorstore/internal:global_initializer",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/aws/credentials:common",
        "//tensorstore/internal/curl:curl_transport",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/os:subprocess",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "aws_credentials_spec",
    srcs = ["aws_credentials_spec.cc"],
    hdrs = ["aws_credentials_spec.h"],
    deps = [
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/aws:aws_api",
        "//tensorstore/internal/aws:aws_credentials",
        "//tensorstore/internal/aws/credentials:common",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@aws_c_common",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "aws_credentials_spec_test",
    srcs = ["aws_credentials_spec_test.cc"],
    deps = [
        ":aws_credentials_spec",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
        "@nlohmann_json//:json",
    ],
)
