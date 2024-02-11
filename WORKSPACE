workspace(
    name = "tensorstore",
)

load("//:external.bzl", "tensorstore_dependencies")

tensorstore_dependencies()

register_toolchains("@local_config_python//:py_toolchain")

# Register proto toolchains.
load("@rules_proto//proto:repositories.bzl", "rules_proto_toolchains")

rules_proto_toolchains()
