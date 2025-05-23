load(
    "//bazel:tensorstore.bzl",
    "tensorstore_cc_binary",
    "tensorstore_cc_library",
)

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

tensorstore_cc_binary(
    name = "tscli",
    srcs = [
        "main.cc",
    ],
    deps = [
        ":command",
        ":tscli_commands",
        "//tensorstore:context",
        "//tensorstore/driver:all_drivers",
        "//tensorstore/internal/metrics:collect",
        "//tensorstore/internal/metrics:registry",
        "//tensorstore/kvstore:all_drivers",
        "//tensorstore/util:json_absl_flag",
        "//tensorstore/util:span",
        "@abseil-cpp//absl/base:log_severity",
        "@abseil-cpp//absl/base:no_destructor",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log:globals",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/status",
    ],
)

tensorstore_cc_library(
    name = "tscli_commands",
    srcs = [
        "copy_command.cc",
        "list_command.cc",
        "ocdbt_dump_command.cc",
        "print_spec_command.cc",
        "print_stats_command.cc",
        "search_command.cc",
    ],
    hdrs = [
        "copy_command.h",
        "list_command.h",
        "ocdbt_dump_command.h",
        "print_spec_command.h",
        "print_stats_command.h",
        "search_command.h",
    ],
    deps = [
        ":command",
        "//tensorstore:box",
        "//tensorstore:context",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:spec",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore",
        "//tensorstore/tscli/lib:kvstore_copy",
        "//tensorstore/tscli/lib:kvstore_list",
        "//tensorstore/tscli/lib:ocdbt_dump",
        "//tensorstore/tscli/lib:ts_print_spec",
        "//tensorstore/tscli/lib:ts_print_stats",
        "//tensorstore/tscli/lib:ts_search",
        "//tensorstore/util:json_absl_flag",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
    ],
)

tensorstore_cc_library(
    name = "command",
    srcs = [
        "command.cc",
        "command_parser.cc",
    ],
    hdrs = [
        "command.h",
        "command_parser.h",
    ],
    deps = [
        "//tensorstore:context",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:string_view",
    ],
)
