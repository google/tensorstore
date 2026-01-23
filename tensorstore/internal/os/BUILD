load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")

package(default_visibility = ["//tensorstore:internal_packages"])

licenses(["notice"])

_LINKOPTS = select({
    "@rules_cc//cc/compiler:mingw-gcc": [
        "-DEFAULTLIB:kernel32.lib",
        "-lkernel32",
    ],
    "@rules_cc//cc/compiler:msvc-cl": [
        "-DEFAULTLIB:kernel32.lib",
    ],
    "//conditions:default": [],
})

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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "error_code",
    srcs = select({
        "@platforms//os:windows": ["error_code_win.cc"],
        "//conditions:default": ["error_code_posix.cc"],
    }),
    hdrs = ["error_code.h"],
    linkopts = _LINKOPTS,
    tags = ["msvc"],
    deps = [
        ":include_windows",
        "//tensorstore/internal:source_location",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
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
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "file_descriptor",
    srcs = ["file_descriptor.cc"],
    hdrs = ["file_descriptor.h"],
    linkopts = _LINKOPTS,
    deps = [
        ":include_windows",
        ":unique_handle",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/tracing",
        "@abseil-cpp//absl/base:core_headers",
    ],
)

tensorstore_cc_library(
    name = "file_util",
    srcs = [
        "file_util.cc",
    ] + select({
        "@platforms//os:windows": [
            "file_info_win.cc",
            "file_util_win.cc",
        ],
        "//conditions:default": [
            "file_info_posix.cc",
            "file_util_posix.cc",
        ],
    }),
    hdrs = [
        "file_info.h",
        "file_util.h",
    ],
    linkopts = _LINKOPTS,
    deps = [
        ":error_code",
        ":file_descriptor",
        ":include_windows",
        ":memory_region",
        ":potentially_blocking_region",
        ":wstring",  # build_cleaner: keep
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/internal/tracing",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:config",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:inlined_vector",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_library(
    name = "file_lock",
    srcs = ["file_lock.cc"],
    hdrs = ["file_lock.h"],
    deps = [
        ":file_descriptor",
        ":file_util",
        "//tensorstore/internal/metrics",
        "//tensorstore/internal/metrics:metadata",
        "//tensorstore/util:quote_string",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
    ],
)

tensorstore_cc_test(
    name = "file_lock_test",
    srcs = ["file_lock_test.cc"],
    deps = [
        ":file_descriptor",
        ":file_lock",
        ":file_util",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "file_util_test",
    srcs = ["file_util_test.cc"],
    deps = [
        ":file_util",
        ":memory_region",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:cord",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_test(
    name = "file_info_test",
    srcs = ["file_info_test.cc"],
    deps = [
        ":file_util",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest_main",
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
        "@abseil-cpp//absl/functional:function_ref",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
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
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@googletest//:gtest_main",
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
        ":file_util",
        ":include_windows",
        ":wstring",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
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
        ":file_util",
        ":subprocess",
        "//tensorstore/internal:env",
        "//tensorstore/internal:path",
        "//tensorstore/internal/testing:scoped_directory",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@googletest//:gtest",
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
        "@abseil-cpp//absl/status",
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
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_library(
    name = "memory_region",
    srcs = ["memory_region.cc"],
    hdrs = ["memory_region.h"],
    deps = [
        "//tensorstore/util:division",
        "//tensorstore/util:result",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings:cord",
    ],
)

tensorstore_cc_test(
    name = "memory_region_test",
    srcs = ["memory_region_test.cc"],
    deps = [
        ":memory_region",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "fork_detection",
    srcs = ["fork_detection.cc"],
    hdrs = ["fork_detection.h"],
    deps = ["@abseil-cpp//absl/base"],
)

tensorstore_cc_test(
    name = "fork_detection_test",
    srcs = ["fork_detection_test.cc"],
    deps = [
        ":fork_detection",
        "@googletest//:gtest_main",
    ],
)

tensorstore_cc_library(
    name = "hugepages",
    srcs = [
        "hugepages.cc",
    ] + select({
        "@platforms//os:windows": [
            "hugepages_win.cc",
        ],
        "@platforms//os:linux": [
            "hugepages_linux.cc",
        ],
        "//conditions:default": [
            "hugepages_unsupported.cc",
        ],
    }),
    hdrs = ["hugepages.h"],
    deps = [
        ":error_code",
        ":file_util",
        ":memory_region",
        ":potentially_blocking_region",
        "//tensorstore/internal:env",
        "//tensorstore/internal/log:verbose_flag",
        "//tensorstore/internal/tracing",
        "//tensorstore/util:division",
        "//tensorstore/util:result",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
    ],
)

tensorstore_cc_test(
    name = "hugepages_test",
    srcs = ["hugepages_test.cc"],
    deps = [
        ":hugepages",
        "//tensorstore/util:status_testutil",
        "@abseil-cpp//absl/flags:reflection",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:cord",
        "@googletest//:gtest_main",
    ],
)
