# Copyright 2025 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Version of run_binary from bazel-skylib that may avoid a separate exec build."""

# The `value` is a list of Label objects indicating which platform constraints are satisfied.
PlatformConstraintsInfo = provider(
    "List of satisfied platform constraints.",
    fields = ["value"],
)

# This should include all constraints that are needed to identify cases where the target and exec
# platforms are incompatible.
PLATFORM_CONSTRAINTS = [
    "@platforms//cpu:x86_64",
    "@platforms//cpu:arm64",
    "@platforms//cpu:ppc64le",
    "@platforms//os:windows",
    "@platforms//os:linux",
    "@platforms//os:macos",
    "@platforms//os:ios",
    "@platforms//os:freebsd",
    "@platforms//os:android",
]

def _platform_constraints_impl(ctx):
    return PlatformConstraintsInfo(value = [
        constraint.label
        for constraint in ctx.attr.constraints
        if ctx.target_platform_has_constraint(constraint[platform_common.ConstraintValueInfo])
    ])

# The platform_constraints rule serves to populate a `PlatformConstraintsInfo` object for either the
# exec or target platform.
_platform_constraints = rule(
    implementation = _platform_constraints_impl,
    attrs = {
        "constraints": attr.label_list(
            mandatory = True,
            providers = [platform_common.ConstraintValueInfo],
        ),
    },
)

def _run_binary_impl(ctx):
    exec_constraints = ctx.attr.exec_platform_constraints[PlatformConstraintsInfo].value
    target_constraints = ctx.attr.target_platform_constraints[PlatformConstraintsInfo].value
    tool_cfg = "target" if exec_constraints == target_constraints else "exec"
    tool_attr_name = "tool_" + tool_cfg
    tool_as_list = [getattr(ctx.attr, tool_attr_name)]

    # The implementation below is derived from bazel-skylib.
    args = [
        ctx.expand_location(a, tool_as_list)
        for a in ctx.attr.args
    ]
    envs = {
        k: ctx.expand_location(v, tool_as_list)
        for k, v in ctx.attr.env.items()
    }
    ctx.actions.run(
        outputs = ctx.outputs.outs,
        inputs = ctx.files.srcs,
        tools = [getattr(ctx.executable, tool_attr_name)],
        executable = getattr(ctx.executable, tool_attr_name),
        arguments = args,
        mnemonic = "RunBinary",
        use_default_shell_env = False,
        env = ctx.configuration.default_shell_env | envs,
    )
    return DefaultInfo(
        files = depset(ctx.outputs.outs),
        runfiles = ctx.runfiles(files = ctx.outputs.outs),
    )

_run_binary = rule(
    implementation = _run_binary_impl,
    attrs = {
        "tool_exec": attr.label(
            executable = True,
            allow_files = True,
            mandatory = True,
            cfg = "exec",
        ),
        "tool_target": attr.label(
            executable = True,
            allow_files = True,
            mandatory = True,
            cfg = "target",
        ),
        "env": attr.string_dict(),
        "srcs": attr.label_list(
            allow_files = True,
        ),
        "outs": attr.output_list(
            mandatory = True,
        ),
        "args": attr.string_list(),
        "exec_platform_constraints": attr.label(
            mandatory = True,
            cfg = "exec",
            providers = [PlatformConstraintsInfo],
        ),
        "target_platform_constraints": attr.label(
            mandatory = True,
            cfg = "target",
            providers = [PlatformConstraintsInfo],
        ),
    },
)

def run_binary(
        name,
        tool,
        env = {},
        srcs = [],
        outs = [],
        args = [],
        platform_constraints = PLATFORM_CONSTRAINTS,
        **kwargs):
    """Runs a binary as a build step, avoiding separate exec build if possible.

    Equivalent to the `run_binary` rule from bazel-skylib, except that the
    `tool` is built in the *target* configuration rather than the *exec*
    configuration if the target platform is the same as the exec platform.

    In builds where many of the dependencies of `tool` are also needed in the
    `target` configuration, this avoids building those dependencies twice.

    Args:
      name: Rule name
      tool: Label specifying the tool binary to run.
      env: Additional environment variables to set.
      srcs: Source files required.
      outs: Outputs.
      args: Command-line arguments for the tool.
      platform_constraints: List of platform constraints to check compatibility
        between exec and target configuration.
      **kwargs: Additional attributes common to all rules.
    """
    constraints_rule_name = name + "_platform_constraints"
    _platform_constraints(
        name = constraints_rule_name,
        constraints = platform_constraints,
        visibility = ["//visibility:private"],
    )
    _run_binary(
        name = name,
        tool_exec = tool,
        tool_target = tool,
        exec_platform_constraints = constraints_rule_name,
        target_platform_constraints = constraints_rule_name,
        env = env,
        srcs = srcs,
        outs = outs,
        args = args,
        **kwargs
    )
