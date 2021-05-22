// Copyright 2020 The TensorStore Authors
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

#include "python/tensorstore/index.h"

#include <string>
#include <variant>
#include <vector>

#include "tensorstore/index.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_python {

IndexVectorOrScalarContainer ToIndexVectorOrScalarContainer(
    const OptionallyImplicitIndexVectorOrScalarContainer& x,
    Index implicit_value) {
  if (auto* index = std::get_if<OptionallyImplicitIndex>(&x)) {
    return index->value_or(implicit_value);
  }
  const auto& v = std::get<SequenceParameter<OptionallyImplicitIndex>>(x);
  std::vector<Index> out_v;
  out_v.reserve(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out_v.push_back(v[i].value_or(implicit_value));
  }
  return out_v;
}

internal_index_space::IndexVectorOrScalarView ToIndexVectorOrScalar(
    const IndexVectorOrScalarContainer& x) {
  constexpr static Index temp = 0;
  if (auto* index = std::get_if<Index>(&x)) {
    return *index;
  } else {
    const auto& v = std::get<std::vector<Index>>(x);
    if (v.empty()) {
      // Ensure we don't have a null pointer.
      return span(&temp, 0);
    }
    return span(v);
  }
}

std::string IndexVectorRepr(const IndexVectorOrScalarContainer& x,
                            bool implicit, bool subscript) {
  return internal::IndexVectorRepr(ToIndexVectorOrScalar(x), implicit,
                                   subscript);
}

}  // namespace internal_python
}  // namespace tensorstore
