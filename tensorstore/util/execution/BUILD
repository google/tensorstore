load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "execution",
    hdrs = ["execution.h"],
    deps = ["@com_google_absl//absl/base:core_headers"],
)

tensorstore_cc_library(
    name = "sender",
    hdrs = ["sender.h"],
    deps = [
        ":execution",
        "//tensorstore/internal/poly",
        "@com_google_absl//absl/base:core_headers",
    ],
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
        ":execution",
        ":sender",
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
        "//tensorstore/internal:mutex",
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
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "sender_util_test",
    srcs = ["sender_util_test.cc"],
    deps = [
        ":execution",
        ":sender_testutil",
        ":sender_util",
        "//tensorstore/util:executor",
        "@com_google_googletest//:gtest_main",
    ],
)
