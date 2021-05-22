load("@build_bazel_rules_nodejs//:index.bzl", _nodejs_binary = "nodejs_binary")

def nodejs_binary(name, srcs = [], deps = [], data = [], templated_args = [], **kwargs):
    _nodejs_binary(
        name = name,
        data = srcs + deps + data,
        **kwargs
    )
