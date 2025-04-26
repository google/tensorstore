load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "execution",
    hdrs = ["execution.h"],
    deps = [
        "//tensorstore/internal/meta:type_traits",
        "@abseil-cpp//absl/base:core_headers",
    ],
)

tensorstore_cc_library(
    name = "any_receiver",
    hdrs = ["any_receiver.h"],
    deps = [
        ":execution",
        ":sender",
        "//tensorstore/internal/poly",
        "@abseil-cpp//absl/base:core_headers",
    ],
)

tensorstore_cc_test(
    name = "any_receiver_test",
    srcs = ["any_receiver_test.cc"],
    deps = [
        ":any_receiver",
        ":execution",
        ":sender",
        ":sender_testutil",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "any_sender",
    hdrs = ["any_sender.h"],
    deps = [
        ":any_receiver",
        ":execution",
        ":sender",
        "//tensorstore/internal/poly",
        "@abseil-cpp//absl/base:core_headers",
    ],
)

tensorstore_cc_test(
    name = "any_sender_test",
    srcs = ["any_sender_test.cc"],
    deps = [
        ":any_sender",
        ":execution",
        ":sender",
        ":sender_testutil",
        "//tensorstore/util:executor",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "sender",
    hdrs = ["sender.h"],
    deps = [":execution"],
)

tensorstore_cc_test(
    name = "sender_test",
    size = "small",
    srcs = ["sender_test.cc"],
    deps = [
        ":execution",
        ":sender",
        ":sender_testutil",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:executor",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "sender_util",
    hdrs = ["sender_util.h"],
    deps = [":execution"],
)

tensorstore_cc_library(
    name = "sender_testutil",
    testonly = 1,
    hdrs = ["sender_testutil.h"],
    deps = [
        ":any_receiver",
        ":execution",
        ":sync_flow_sender",
        "//tensorstore/internal/meta:type_traits",
        "//tensorstore/util:str_cat",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_library(
    name = "collecting_sender",
    hdrs = ["collecting_sender.h"],
    deps = [
        ":execution",
    ],
)

tensorstore_cc_test(
    name = "collecting_sender_test",
    size = "small",
    srcs = ["collecting_sender_test.cc"],
    deps = [
        ":collecting_sender",
        ":execution",
        ":sender",
        ":sender_testutil",
        ":sender_util",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "sync_flow_sender",
    hdrs = ["sync_flow_sender.h"],
    deps = [
        ":execution",
        "@abseil-cpp//absl/synchronization",
    ],
)

tensorstore_cc_test(
    name = "sync_flow_sender_test",
    size = "small",
    srcs = ["sync_flow_sender_test.cc"],
    deps = [
        ":execution",
        ":sender_testutil",
        ":sync_flow_sender",
        "//tensorstore/internal/thread",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "sender_util_test",
    srcs = ["sender_util_test.cc"],
    deps = [
        ":any_receiver",
        ":any_sender",
        ":execution",
        ":sender",
        ":sender_testutil",
        ":sender_util",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "future_sender",
    hdrs = ["future_sender.h"],
    deps = [
        ":execution",
        "//tensorstore/util:future",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "future_sender_test",
    srcs = ["future_sender_test.cc"],
    deps = [
        ":any_receiver",
        ":any_sender",
        ":execution",
        ":future_sender",
        ":sender",
        ":sender_testutil",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "result_sender",
    hdrs = ["result_sender.h"],
    deps = [
        ":execution",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "result_sender_test",
    srcs = ["result_sender_test.cc"],
    deps = [
        ":any_receiver",
        ":any_sender",
        ":execution",
        ":result_sender",
        ":sender_testutil",
        "//tensorstore/util:result",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "future_collecting_receiver",
    hdrs = ["future_collecting_receiver.h"],
    deps = [
        ":execution",
        "//tensorstore/util:future",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_test(
    name = "future_collecting_receiver_test",
    size = "small",
    srcs = ["future_collecting_receiver_test.cc"],
    deps = [
        ":execution",
        ":future_collecting_receiver",
        ":sender_testutil",
        ":sender_util",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "flow_sender_operation_state",
    hdrs = ["flow_sender_operation_state.h"],
    visibility = [
        "//tensorstore:internal_packages",
    ],
    deps = [
        ":any_receiver",
        ":execution",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/status",
    ],
)
