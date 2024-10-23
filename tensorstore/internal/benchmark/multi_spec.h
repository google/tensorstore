// Copyright 2024 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORSTORE_INTERNAL_BENCHMARK_MULTI_SPEC_H_
#define TENSORSTORE_INTERNAL_BENCHMARK_MULTI_SPEC_H_

#include <string>
#include <vector>

#include "tensorstore/spec.h"

namespace tensorstore {
namespace internal_benchmark {

/// Reads a list of tensorstore::Spec from a text file.
///
/// The text file can contain either a list of tensorstore::Spec in JSON format,
/// or a list of tensorstore::Spec in a text format.
///
/// The text format is a list of lines, where each line contains a single
/// tensorstore::Spec in JSON format. Empty lines and lines starting with '#'
/// are ignored.
std::vector<tensorstore::Spec> ReadSpecsFromFile(
    const std::string& txt_file_path);

}  // namespace internal_benchmark
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_BENCHMARK_MULTI_SPEC_H_
