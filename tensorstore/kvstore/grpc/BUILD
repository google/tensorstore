load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_library",
    "tensorstore_cc_proto_library",
    "tensorstore_cc_test",
    "tensorstore_proto_library",
)
load("//bazel:cc_grpc_library.bzl", "cc_grpc_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_proto_library(
    name = "common_proto",
    srcs = ["common.proto"],
    deps = [
        "@com_google_protobuf//:timestamp_proto",
        "@local_proto_mirror//google/rpc:code_proto",
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
        "@com_google_protobuf//:protobuf",
    ],
)

tensorstore_cc_library(
    name = "grpc_kvstore",
    srcs = ["grpc_kvstore.cc"],
    hdrs = ["grpc_kvstore.h"],
    deps = [
        ":common",
        ":common_cc_proto",
        ":kvstore_cc_grpc",
        ":kvstore_cc_proto",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore/internal:concurrency_resource",
        "//tensorstore/internal:context_binding",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/grpc:client_credentials",
        "//tensorstore/internal/grpc:utils",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:absl_time",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/proto:encode_time",
        "//tensorstore/serialization:absl_time",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "grpc_kvstore_test",
    srcs = ["grpc_kvstore_test.cc"],
    deps = [
        ":grpc_kvstore",
        ":kvstore_cc_grpc",
        ":kvstore_cc_proto",
        "//tensorstore/internal:concurrent_testutil",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/proto:parse_text_proto_or_die",
        "//tensorstore/proto:protobuf_matchers",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_sender",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_github_grpc_grpc//:grpc++_test",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "kvstore_service",
    srcs = ["kvstore_service.cc"],
    hdrs = ["kvstore_service.h"],
    deps = [
        ":common",
        ":common_cc_proto",
        ":kvstore_cc_grpc",
        ":kvstore_cc_proto",
        "//tensorstore/internal:init_tensorstore",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:type_traits",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:all_drivers",  # build_cleaner: keep
        "//tensorstore/kvstore:byte_range",
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/proto:encode_time",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:any_sender",
        "@com_github_grpc_grpc//:grpc++",  # build_cleaner: keep
        "@com_github_grpc_grpc//:grpc++_public_hdrs",  # build_cleaner: keep
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_protobuf//:protobuf",
    ],
)

tensorstore_cc_test(
    name = "kvstore_service_test",
    srcs = ["kvstore_service_test.cc"],
    deps = [
        ":grpc_kvstore",
        ":kvstore_cc_grpc",
        ":kvstore_service",  # build_cleaner: keep
        "//tensorstore:context",
        "//tensorstore/internal:concurrent_testutil",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:test_util",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:all_drivers",  # build_cleaner: keep
        "//tensorstore/kvstore:generation",
        "//tensorstore/kvstore:key_range",
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "//tensorstore/util/execution",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/execution:any_sender",
        "//tensorstore/util/execution:sender_testutil",
        "@com_github_grpc_grpc//:grpc",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_github_nlohmann_json//:nlohmann_json",
        "@com_google_absl//absl/log:die_if_null",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

###############################

cc_binary(
    name = "kvstore_service_main",
    testonly = 1,
    srcs = ["kvstore_service_main.cc"],
    deps = [
        ":common",
        ":kvstore_service",
        "//tensorstore/internal:init_tensorstore",
        "//tensorstore/internal:json_fwd",
        "//tensorstore/kvstore",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:result",
        "@com_github_grpc_grpc//:grpc++",
        "@com_github_grpc_grpc//:grpc++_public_hdrs",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "kvstore_test",
    testonly = 1,
    srcs = ["kvstore_test.cc"],
    deps = [
        ":grpc_kvstore",  # build_cleaner: keep
        "//tensorstore/internal:init_tensorstore",
        "//tensorstore/kvstore",
        "//tensorstore/kvstore:all_drivers",  # build_cleaner: keep
        "//tensorstore/kvstore:test_util",
        "//tensorstore/util:json_absl_flag",
        "@com_google_absl//absl/log:absl_log",
    ],
)
