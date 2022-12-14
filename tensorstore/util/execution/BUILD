load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "execution",
    hdrs = ["execution.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tensorstore_cc_library(
    name = "any_receiver",
    hdrs = ["any_receiver.h"],
    deps = [
        ":execution",
        ":sender",
        "//tensorstore/internal/poly",
        "@com_google_absl//absl/base:core_headers",
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
        "@com_google_googletest//:gtest_main",
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
        "@com_google_absl//absl/base:core_headers",
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
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/util:executor",
        "@com_google_googletest//:gtest_main",
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
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
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
        ":sender_testutil",
        ":sender_util",
        "//tensorstore/util:span",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "sync_flow_sender",
    hdrs = ["sync_flow_sender.h"],
    deps = [
        ":execution",
        "@com_google_absl//absl/synchronization",
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
        "//tensorstore/internal:mutex",
        "//tensorstore/internal:thread",
        "@com_google_googletest//:gtest_main",
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
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "future_sender",
    hdrs = ["future_sender.h"],
    deps = [
        ":execution",
        "//tensorstore/util:future",
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "result_sender",
    hdrs = ["result_sender.h"],
    deps = [
        ":execution",
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
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
        "@com_google_absl//absl/status",
        "@com_google_googletest//:gtest_main",
    ],
)
