# Placeholder: load py_binary

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "h2_server",
    testonly = 1,
    srcs = ["h2_server.py"],
    data = [
        "test.crt",
        "test.key",
    ],
    tags = [
        "manual",
        "notap",
        "skip-cmake",
    ],
    deps = ["@pypa_h2//:h2"],
)
