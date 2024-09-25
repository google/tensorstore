workspace(
    name = "tensorstore",
)

load("//:external.bzl", "tensorstore_dependencies")

tensorstore_dependencies()

load("@bazel_features//:deps.bzl", "bazel_features_deps")

bazel_features_deps()

register_toolchains("@local_config_python//:py_toolchain")

# Register proto toolchains.
load("@rules_proto//proto:toolchains.bzl", "rules_proto_toolchains")

rules_proto_toolchains()
