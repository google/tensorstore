# Copyright 2020 The TensorStore Authors
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

load("@bazel_skylib//lib:selects.bzl", "selects")

def _sorted_unique_strings(x):
    return sorted({k: True for k in x}.keys())

def _escape_target_name(name):
    name = name.replace("~", "~~")
    name = name.replace(":", "~c")
    name = name.replace("/", "~/")
    return name

# https://github.com/bazelbuild/proposals/blob/master/designs/2018-11-09-config-setting-chaining.md
def all_conditions(conditions):
    conditions = _sorted_unique_strings([x for x in conditions if x != "//conditions:default"])
    if len(conditions) == 0:
        return "//conditions:default"
    if len(conditions) == 1:
        return conditions[0]
    name = "all=" + ",".join([_escape_target_name(x) for x in conditions])
    if native.existing_rule(name) == None:
        selects.config_setting_group(
            name = name,
            match_all = conditions,
        )
    return name

# Used to emit combined settings for os / cpu / compiler.
def emit_os_cpu_compiler_group(os, cpu, compiler):
    n = [x for x in [os, cpu, compiler] if x != None]
    if len(n) < 2:
        return
    name = "_".join(n)

    if native.existing_rule(name) == None:
        m = [
            "@platforms//os:" + os if os != None else None,
            "@platforms//cpu:" + cpu if cpu != None else None,
            ":compiler_" + compiler if compiler != None else None,
        ]
        selects.config_setting_group(
            name = name,
            match_all = [x for x in m if x != None],
        )
