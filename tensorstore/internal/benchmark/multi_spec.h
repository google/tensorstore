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

#include <stddef.h>

#include <cstdint>
#include <string>
#include <vector>

#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/box.h"  // IWYU pragma: keep
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_array.h"  // IWYU pragma: keep

namespace tensorstore {
namespace internal_benchmark {

struct ShardVariable {
  std::string name;
  std::vector<Index> shape;
  std::vector<Index> chunks;
  std::string dtype;
  std::vector<Box<>> array_boxes;

  constexpr static auto default_json_binder =
      [](auto is_loading, const auto& options, auto* obj, auto* j) {
        namespace jb = tensorstore::internal_json_binding;
        using Self = ShardVariable;
        return jb::Object(
            jb::Member("name", jb::Projection<&Self::name>()),
            jb::Member("shape", jb::Projection<&Self::shape>()),
            jb::Member("chunks", jb::Projection<&Self::chunks>()),
            jb::Member("dtype", jb::Projection<&Self::dtype>()),
            jb::Member("array_boxes", jb::Projection<&Self::array_boxes>()),
            jb::DiscardExtraMembers /**/
            )(is_loading, options, obj, j);
      };
};

/// Read a list of ShardVariable from a file or a json object.
std::vector<ShardVariable> ReadFromFileOrFlag(std::string flag);

}  // namespace internal_benchmark
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_BENCHMARK_MULTI_SPEC_H_
