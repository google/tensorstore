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
    srcs = [
        "s3_key_value_store.cc",
        "s3_metadata.cc",
    ],
    hdrs = ["s3_metadata.h"],
    deps = [
        ":s3_endpoint",
        ":s3_request_builder",
        ":s3_resource",
        ":s3_uri_utils",
        ":validate",
        "//tensorstore:context",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/rate_limiter",
        "//tensorstore/internal/thread:schedule_at",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/gcs:validate",
        "//tensorstore/kvstore/http:byte_range_util",
        "//tensorstore/kvstore/s3/credentials:aws_credentials",
        "//tensorstore/kvstore/s3/credentials:default_credential_provider",
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
        "@com_google_absl//absl/container:btree",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_re2//:re2",
        "@tinyxml2",
    ],
    alwayslink = 1,
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
        "//tensorstore/internal/rate_limiter",
        "//tensorstore/internal/rate_limiter:admission_queue",
        "//tensorstore/internal/rate_limiter:scaling_rate_limiter",
        "//tensorstore/util:result",
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:marshalling",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "s3_metadata_test",
    size = "small",
    srcs = [
        "s3_metadata_test.cc",
    ],
    deps = [
        ":s3",
        "@com_google_googletest//:gtest_main",
        "@tinyxml2",
    ],
)

tensorstore_cc_test(
    name = "s3_key_value_store_test",
    size = "small",
    srcs = ["s3_key_value_store_test.cc"],
    deps = [
        ":s3",
        "//tensorstore:context",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
    ],
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
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore/s3/credentials:aws_credentials",
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
        ":s3_request_builder",
        "//tensorstore/internal/http",
        "//tensorstore/kvstore/s3/credentials:aws_credentials",
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

tensorstore_cc_test(
    name = "localstack_test",
    size = "small",
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
        ":s3_request_builder",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:env",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/os:subprocess",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:batch_util",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/s3/credentials:aws_credentials",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_github_nlohmann_json//:nlohmann_json",
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
        "//tensorstore/internal/http:mock_http_transport",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
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
