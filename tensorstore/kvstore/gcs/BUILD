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
        "//tensorstore/internal:source_location",
        "//tensorstore/internal:utf8",
        "//tensorstore/internal/http",
        "//tensorstore/internal/json",
        "//tensorstore/kvstore:generation",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/strings:str_format",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "validate_test",
    size = "small",
    srcs = ["validate_test.cc"],
    deps = [
        ":validate",
        "@googletest//:gtest_main",
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
        "//tensorstore/util:result",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "gcs_testbench",
    testonly = 1,
    srcs = ["gcs_testbench.cc"],
    hdrs = ["gcs_testbench.h"],
    deps = [
        "//tensorstore/internal/curl:curl_transport",
        "//tensorstore/internal/grpc:utils",
        "//tensorstore/internal/http",
        "//tensorstore/internal/http:default_transport",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/internal/os:subprocess",
        "//tensorstore/proto:parse_text_proto_or_die",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
        "@googleapis//google/storage/v2:storage_cc_grpc",
        "@googleapis//google/storage/v2:storage_cc_proto",
        "@grpc//:grpc++",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_library(
    name = "exp_credentials_resource",
    srcs = ["exp_credentials_resource.cc"],
    hdrs = ["exp_credentials_resource.h"],
    deps = [
        ":exp_credentials_spec",
        "//tensorstore:context",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/cache_key:json",
        "//tensorstore/internal/json_binding",
        "//tensorstore/serialization",
        "//tensorstore/util:result",
    ],
    alwayslink = 1,
)

GCS_CREDENTIALS_DEPS = []

GCS_CREDENTIALS_DEFINES = []

tensorstore_cc_library(
    name = "exp_credentials_spec",
    srcs = ["exp_credentials_spec.cc"],
    hdrs = ["exp_credentials_spec.h"],
    defines = GCS_CREDENTIALS_DEFINES,
    deps = GCS_CREDENTIALS_DEPS + [
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/grpc/clientauth:authentication_strategy",
        "//tensorstore/internal/grpc/clientauth:call_authentication",
        "//tensorstore/internal/grpc/clientauth:channel_authentication",
        "//tensorstore/internal/grpc/clientauth:impersonate_service_account",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/internal/os:file_util",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/status",
        "@nlohmann_json//:json",
    ],
)

tensorstore_cc_test(
    name = "exp_credentials_spec_test",
    srcs = ["exp_credentials_spec_test.cc"],
    deps = [
        ":exp_credentials_spec",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal/testing:json_gtest",
        "//tensorstore/util:status_testutil",
        "@googletest//:gtest_main",
        "@grpc//:grpc++",
        "@nlohmann_json//:json",
    ],
)
