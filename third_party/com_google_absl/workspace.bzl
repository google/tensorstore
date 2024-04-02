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

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")

# REPO_BRANCH = master

def repo():
    maybe(
        third_party_http_archive,
        name = "com_google_absl",
        strip_prefix = "abseil-cpp-fb3621f4f897824c0dbe0615fa94543df6192f30",
        urls = [
            "https://storage.googleapis.com/tensorstore-bazel-mirror/github.com/abseil/abseil-cpp/archive/fb3621f4f897824c0dbe0615fa94543df6192f30.tar.gz",  # lts_2023_08_02(2023-09-19)
        ],
        sha256 = "0320586856674d16b0b7a4d4afb22151bdc798490bb7f295eddd8f6a62b46fea",
        patches = [
            # mingw build fix; https://github.com/abseil/abseil-cpp/commit/2f77684e8dc473a48dbc19167ffe69c40ce8ada4
            Label("//third_party:com_google_absl/patches/mingw.diff"),
        ],
        patch_args = ["-p1"],
        cmake_name = "absl",
        cmake_target_mapping = ABSL_CMAKE_MAPPING,
        cmake_settings = {
            "ABSL_PROPAGATE_CXX_STD": "ON",
            "ABSL_BUILD_TESTING": "OFF",
            "BUILD_TESTING": "OFF",
            "ABSL_BUILD_TEST_HELPERS": "ON",
            "ABSL_USE_EXTERNAL_GOOGLETEST": "ON",
            "ABSL_FIND_GOOGLETEST": "ON",
        },
    )

# Mapping from Bazel label to CMake target
ABSL_CMAKE_MAPPING = {
    "//absl/algorithm:algorithm": "absl::algorithm",
    "//absl/algorithm:container": "absl::algorithm_container",
    "//absl/base:base": "absl::base",
    "//absl/base:core_headers": "absl::core_headers",
    "//absl/base:dynamic_annotations": "absl::dynamic_annotations",
    "//absl/base:log_severity": "absl::log_severity",
    "//absl/base:prefetch": "absl::prefetch",
    "//absl/cleanup:cleanup": "absl::cleanup",
    "//absl/container:btree": "absl::btree",
    "//absl/container:fixed_array": "absl::fixed_array",
    "//absl/container:flat_hash_map": "absl::flat_hash_map",
    "//absl/container:flat_hash_set": "absl::flat_hash_set",
    "//absl/container:inlined_vector": "absl::inlined_vector",
    "//absl/container:node_hash_map": "absl::node_hash_map",
    "//absl/container:node_hash_set": "absl::node_hash_set",
    "//absl/crc:crc32c": "absl::crc32c",
    "//absl/debugging:debugging": "absl::debugging",
    "//absl/debugging:failure_signal_handler": "absl::failure_signal_handler",
    "//absl/debugging:leak_check": "absl::leak_check",
    "//absl/debugging:stacktrace": "absl::stacktrace",
    "//absl/debugging:symbolize": "absl::symbolize",
    "//absl/flags:commandlineflag": "absl::flags_commandlineflag",
    "//absl/flags:config": "absl::flags_config",
    "//absl/flags:flag": "absl::flags",
    "//absl/flags:flags_marshalling": "absl::flags_marshalling",
    "//absl/flags:marshalling": "absl::flags_marshalling",
    "//absl/flags:parse": "absl::flags_parse",
    "//absl/flags:reflection": "absl::flags_reflection",
    "//absl/flags:usage": "absl::flags_usage",
    "//absl/functional:any_invocable": "absl::any_invocable",
    "//absl/functional:bind_front": "absl::bind_front",
    "//absl/functional:function_ref": "absl::function_ref",
    "//absl/hash:hash": "absl::hash",
    "//absl/log:absl_check": "absl::absl_check",
    "//absl/log:absl_log": "absl::absl_log",
    "//absl/log:check": "absl::check",
    "//absl/log:die_if_null": "absl::die_if_null",
    "//absl/log:log_sink": "absl::log_sink",
    "//absl/log:log": "absl::log",
    "//absl/memory:memory": "absl::memory",
    "//absl/meta:meta": "absl::meta",
    "//absl/meta:type_traits": "absl::type_traits",
    "//absl/numeric:bits": "absl::bits",
    "//absl/numeric:int128": "absl::int128",
    "//absl/numeric:numeric_representation": "absl::numeric_representation",
    "//absl/numeric:numeric": "absl::numeric",
    "//absl/profiling:exponential_biased": "absl::exponential_biased",
    "//absl/profiling:periodic_sampler": "absl::periodic_sampler",
    "//absl/profiling:sample_recorder": "absl::sample_recorder",
    "//absl/random:bit_gen_ref": "absl::random_bit_gen_ref",
    "//absl/random:distributions": "absl::random_distributions",
    "//absl/random:mocking_bit_gen": "absl::random_mocking_bit_gen",
    "//absl/random:random": "absl::random_random",
    "//absl/random:seed_gen_exception": "absl::random_seed_gen_exception",
    "//absl/random:seed_sequences": "absl::random_seed_sequences",
    "//absl/status:status": "absl::status",
    "//absl/status:statusor": "absl::statusor",
    "//absl/strings:cord": "absl::cord",
    "//absl/strings:str_format": "absl::str_format",
    "//absl/strings:strings": "absl::strings",
    "//absl/synchronization:synchronization": "absl::synchronization",
    "//absl/time:civil_time": "absl::civil_time",
    "//absl/time:time_zone": "absl::time_zone",
    "//absl/time:time": "absl::time",
    "//absl/types:any": "absl::any",
    "//absl/types:bad_any_cast": "absl::bad_any_cast",
    "//absl/types:bad_optional_access": "absl::bad_optional_access",
    "//absl/types:bad_variant_access": "absl::bad_variant_access",
    "//absl/types:compare": "absl::compare",
    "//absl/types:optional": "absl::optional",
    "//absl/types:span": "absl::span",
    "//absl/types:variant": "absl::variant",
    "//absl/utility:utility": "absl::utility",
    # Internal targets mapping
    "//absl/base:endian": "absl::endian",
    "//absl/base:config": "absl::config",
    "//absl/container:layout": "absl::layout",
    "//absl/strings:internal": "absl::strings_internal",
    # Not available in abseil CMakeLists.txt
    "//absl/debugging:leak_check_disable": "",
    # Testonly targets
    "//absl/hash:hash_testing": "absl::hash_testing",
    "//absl/strings:cord_test_helpers": "absl::cord_test_helpers",
}
