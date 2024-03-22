load("//bazel:cc_grpc_library.bzl", "cc_grpc_library")
load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_binary",
    "tensorstore_cc_library",
    "tensorstore_cc_proto_library",
    "tensorstore_cc_test",
    "tensorstore_proto_library",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

filegroup(
    name = "doc_sources",
    srcs = glob([
        "**/*.rst",
        "**/*.yml",
    ]),
)

tensorstore_proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    deps = [
        "@com_google_googleapis//google/rpc:code_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)

tensorstore_proto_library(
    name = "kvstore_proto",
    srcs = ["kvstore.proto"],
    deps = [
        ":common_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)

tensorstore_cc_proto_library(
    name = "common_cc_proto",
    deps = [":common_proto"],
)

tensorstore_cc_proto_library(
    name = "kvstore_cc_proto",
    deps = [":kvstore_proto"],
)

cc_grpc_library(
    name = "kvstore_cc_grpc",
    srcs = [":kvstore_proto"],
    service_namespace = "grpc_gen",
    deps = [":kvstore_cc_proto"],
)

tensorstore_cc_library(
    name = "common",
    srcs = ["common.cc"],
    hdrs = [
        "common.h",
        "handler_template.h",
    ],
    deps = [
        ":common_cc_proto",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/grpc:utils",
        "//tensorstore/kvstore:generation",
        "//tensorstore/proto:encode_time",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "tsgrpc",
    srcs = ["tsgrpc.cc"],
    deps = [
        ":common",
        ":common_cc_proto",
        ":kvstore_cc_grpc",
        ":kvstore_cc_proto",
        "//tensorstore:context",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:data_copy_concurrency_resource",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/grpc:client_credentials",
        "//tensorstore/internal/grpc:utils",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/proto:encode_time",
        "//tensorstore/proto:proto_util",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/garbage_collection",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "tsgrpc_test",
    srcs = ["tsgrpc_test.cc"],
    tags = ["cpu:2"],
    deps = [
        ":kvstore_cc_grpc",
        ":kvstore_cc_proto",
        ":mock_kvstore_service",
        ":tsgrpc",
        "//tensorstore/internal/grpc:grpc_mock",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/proto:parse_text_proto_or_die",
        "//tensorstore/proto:protobuf_matchers",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "kvstore_server",
    srcs = ["kvstore_server.cc"],
    hdrs = ["kvstore_server.h"],
    deps = [
        ":common",
        ":common_cc_proto",
        ":kvstore_cc_grpc",
        ":kvstore_cc_proto",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:path",
        "//tensorstore/internal/grpc:server_credentials",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/proto:encode_time",
        "//tensorstore/proto:proto_util",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "kvstore_server_test",
    srcs = ["kvstore_server_test.cc"],
    tags = ["cpu:2"],
    deps = [
        ":kvstore_server",
        ":tsgrpc",
        "//tensorstore:context",
        "//tensorstore/internal/http:transport_test_utils",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_matchers",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/kvstore/memory",  # build_cleaner: keep
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "mock_kvstore_service",
    testonly = 1,
    hdrs = [
        "mock_kvstore_service.h",
    ],
    deps = [
        ":kvstore_cc_grpc",
        ":kvstore_cc_proto",
        "//tensorstore/internal/grpc:grpc_mock",
        "@com_github_grpc_grpc//:grpc++",
    ],
)

###############################

tensorstore_cc_binary(
    name = "kvstore_server_main",
    testonly = 1,
    srcs = ["kvstore_server_main.cc"],
    deps = [
        ":kvstore_server",
        "//tensorstore:context",
        "//tensorstore/internal/grpc:client_credentials",
        "//tensorstore/internal/grpc:server_credentials",
        "//tensorstore/kvstore:all_drivers",  # build_cleaner: keep
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)
