load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

tensorstore_cc_library(
    name = "cwd",
    srcs = ["cwd.cc"],
    hdrs = ["cwd.h"],
    deps = [
        ":error_code",
        ":include_windows",
        ":wstring",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "error_code",
    srcs = select({
        "@platforms//os:windows": ["error_code_win.cc"],
        "//conditions:default": ["error_code_posix.cc"],
    }),
    hdrs = ["error_code.h"],
    tags = ["msvc"],
    deps = [
        ":include_windows",
        "//tensorstore/internal:source_location",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "filesystem",
    testonly = True,
    srcs = ["filesystem.cc"],
    hdrs = ["filesystem.h"],
    deps = [
        ":file_lister",
        ":file_util",
        "//tensorstore/internal:env",
        "//tensorstore/util:status",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "file_util",
    srcs = select({
        "@platforms//os:windows": [
            "file_util_win.cc",
        ],
        "//conditions:default": [
            "file_util_posix.cc",
        ],
    }),
    hdrs = ["file_util.h"],
    deps = [
        ":error_code",
        ":include_windows",
        ":potentially_blocking_region",
        ":unique_handle",
        ":wstring",  # build_cleaner: keep
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/internal/tracing",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/base:config",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_library(
    name = "file_lock",
    srcs = ["file_lock.cc"],
    hdrs = ["file_lock.h"],
    deps = [
        ":file_util",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
    ],
)

tensorstore_cc_test(
    name = "file_lock_test",
    srcs = ["file_lock_test.cc"],
    deps = [
        ":file_lock",
        ":file_util",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "file_util_test",
    srcs = ["file_util_test.cc"],
    deps = [
        ":file_util",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "file_lister",
    srcs = select({
        "@platforms//os:windows": [
            "file_lister_win.cc",
        ],
        "//conditions:default": [
            "file_lister_posix.cc",
        ],
    }),
    hdrs = ["file_lister.h"],
    deps = [
        ":error_code",
        ":file_util",
        ":include_windows",
        ":potentially_blocking_region",
        ":wstring",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:status",
        "@com_google_absl//absl/functional:function_ref",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "file_lister_test",
    srcs = ["file_lister_test.cc"],
    deps = [
        ":file_lister",
        ":file_util",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/log:absl_log",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
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
        ":include_windows",
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

tensorstore_cc_library(
    name = "potentially_blocking_region",
    hdrs = [
        "potentially_blocking_region.h",
    ],
)

tensorstore_cc_library(
    name = "unique_handle",
    hdrs = ["unique_handle.h"],
)

tensorstore_cc_library(
    name = "include_windows",
    hdrs = ["include_windows.h"],
)

tensorstore_cc_library(
    name = "wstring",
    srcs = ["wstring.cc"],
    hdrs = ["wstring.h"],
    tags = ["msvc"],
    deps = [
        ":error_code",
        ":include_windows",
        "@com_google_absl//absl/status",
    ],
)

tensorstore_cc_library(
    name = "get_bios_info",
    srcs = select({
        "@platforms//os:windows": [
            "get_bios_info_win.cc",
        ],
        "//conditions:default": [
            "get_bios_info_linux.cc",
        ],
    }),
    hdrs = ["get_bios_info.h"],
    deps = [
        ":error_code",
        ":potentially_blocking_region",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
