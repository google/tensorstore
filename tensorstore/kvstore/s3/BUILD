load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
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

# To enable debug checks, specify:
# bazel build --//tensorstore/kvstore/s3:debug
bool_flag(
    name = "debug",
    build_setting_default = False,
)

config_setting(
    name = "debug_setting",
    flag_values = {":debug": "True"},
    visibility = ["//visibility:private"],
)

DEBUG_LOCAL_DEFINES = select({
    "//conditions:default": [],
    ":debug_setting": [
        "TENSORSTORE_INTERNAL_S3_LOG_REQUESTS",
        "TENSORSTORE_INTERNAL_S3_LOG_RESPONSES",
        "TENSORSTORE_INTERNAL_S3_LOG_AWS4",
    ],
})

tensorstore_cc_library(
    name = "s3",
    srcs = [
        "s3_key_value_store.cc",
        "s3_metadata.cc",
    ],
    hdrs = ["s3_metadata.h"],
    local_defines = DEBUG_LOCAL_DEFINES,
    deps = [
        ":aws_credential_provider",
        ":s3_request_builder",
        ":s3_resource",
        ":s3_uri_utils",
        ":validate",
        "//tensorstore:context",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:retry",
        "//tensorstore/internal:schedule_at",
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore/gcs:validate",
        "//tensorstore/kvstore/gcs_http:rate_limiter",
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
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_re2//:re2",
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
        "//tensorstore/kvstore/gcs_http:admission_queue",
        "//tensorstore/kvstore/gcs_http:rate_limiter",
        "//tensorstore/kvstore/gcs_http:scaling_rate_limiter",
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
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:generation_testutil",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util:str_cat",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
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
    local_defines = DEBUG_LOCAL_DEFINES,
    deps = [
        ":aws_credential_provider",
        ":s3_uri_utils",
        "//tensorstore/internal:uri_utils",
        "//tensorstore/internal/digest:sha256",
        "//tensorstore/internal/http",
        "//tensorstore/kvstore:byte_range",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_boringssl//:crypto",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "s3_request_builder_test",
    size = "small",
    srcs = ["s3_request_builder_test.cc"],
    deps = [
        ":aws_credential_provider",
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
    name = "aws_credential_provider",
    srcs = ["aws_credential_provider.cc"],
    hdrs = ["aws_credential_provider.h"],
    deps = [
        "//tensorstore/internal:env",
        "//tensorstore/internal:no_destructor",
        "//tensorstore/internal:path",
        "//tensorstore/internal/http",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "aws_credential_provider_test",
    srcs = ["aws_credential_provider_test.cc"],
    deps = [
        ":aws_credential_provider",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal:test_util",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "localstack_test",
    srcs = ["localstack_test.cc"],
    tags = [
        "manual",
        "notap",  # TODO: Enable before linking by default.
        "skip-cmake",  # localstack is python, which will not work in bazel_to_cmake.
    ],
    deps = [
        ":aws_credential_provider",
        ":s3_request_builder",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:env",
        "//tensorstore/internal:json_gtest",
        "//tensorstore/internal:subprocess",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:curl_transport",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)
