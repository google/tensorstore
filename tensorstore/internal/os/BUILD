load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "error_code",
    srcs = ["error_code.cc"],
    hdrs = ["error_code.h"],
    tags = ["msvc"],
    deps = [
        "//tensorstore/internal:source_location",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "subprocess",
    testonly = True,
    srcs = select({
        "@platforms//os:windows": ["subprocess_win.cc"],
        "//conditions:default": ["subprocess_posix.cc"],
    }),
    hdrs = ["subprocess.h"],
    tags = ["msvc"],
    deps = [
        ":error_code",
        ":wstring",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "subprocess_test",
    srcs = ["subprocess_test.cc"],
    tags = [
        "cpu:2",
        "msvc",
    ],
    deps = [
        ":subprocess",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@com_google_riegeli//riegeli/bytes:fd_reader",
        "@com_google_riegeli//riegeli/bytes:read_all",
    ],
)

cc_library(
    name = "filesystem",
    testonly = True,
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        ":error_code",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "wstring",
    srcs = ["wstring.cc"],
    hdrs = ["wstring.h"],
    tags = ["msvc"],
    deps = [
        ":error_code",
        "@com_google_absl//absl/status",
    ],
)
