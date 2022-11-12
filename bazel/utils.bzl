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

def escape_target_name(name):
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
    name = "all=" + ",".join([escape_target_name(x) for x in conditions])
    if native.existing_rule(name) == None:
        selects.config_setting_group(
            name = name,
            match_all = conditions,
        )
    return name

def repository_source_root():
    return Label(native.repository_name() + "//:dummy").workspace_root

def package_source_root():
    root = repository_source_root()
    p = native.package_name()
    if root != "" and p != "":
        root += "/"
    p = root + p
    return p

def package_relative_path(path):
    root = package_source_root()
    if path == ".":
        path = root
    else:
        path = root + "/" + path
    if path == "":
        path = "."
    return path
