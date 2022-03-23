load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_library(
    name = "execution",
    hdrs = ["execution.h"],
    deps = [
        "//tensorstore/internal:type_traits",
        "@com_google_absl//absl/base:core_headers",
    ],
)

tensorstore_cc_library(
    name = "sender",
    hdrs = ["sender.h"],
    deps = [
        ":execution",
        "//tensorstore/internal/poly",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/utility",
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
    name = "sender_testutil",
    testonly = 1,
    hdrs = ["sender_testutil.h"],
    deps = [
        ":sender",
        "//tensorstore/internal:mutex",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:sync_flow_sender",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)
