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

#include "absl/strings/str_join.h"

namespace tensorstore {
namespace internal_python {

IndexVectorOrScalarContainer ToIndexVectorOrScalarContainer(
    const OptionallyImplicitIndexVectorOrScalarContainer& x,
    Index implicit_value) {
  if (auto* index = std::get_if<OptionallyImplicitIndex>(&x)) {
    return index->value_or(implicit_value);
  }
  const auto& v = std::get<std::vector<OptionallyImplicitIndex>>(x);
  std::vector<Index> out_v;
  out_v.reserve(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out_v.push_back(v[i].value_or(implicit_value));
  }
  return out_v;
}

std::string OptionallyImplicitIndexRepr(Index value) {
  if (value == kImplicit) return "None";
  return std::to_string(value);
}

std::string IndexVectorRepr(const IndexVectorOrScalarContainer& x,
                            bool implicit, bool subscript) {
  if (auto* index = std::get_if<Index>(&x)) {
    if (implicit) return OptionallyImplicitIndexRepr(*index);
    return StrCat(*index);
  }
  const auto& v = std::get<std::vector<Index>>(x);
  std::string out;
  if (!subscript) out += "[";
  for (size_t i = 0; i < v.size(); ++i) {
    if (i != 0) out += ',';
    if (implicit) {
      out += OptionallyImplicitIndexRepr(v[i]);
    } else {
      AppendToString(&out, v[i]);
    }
  }
  if (subscript) {
    if (v.size() == 1) {
      out += ",";
    } else if (v.empty()) {
      out += "()";
    }
  } else {
    out += "]";
  }
  return out;
}

internal_index_space::IndexVectorOrScalar ToIndexVectorOrScalar(
    const IndexVectorOrScalarContainer& x) {
  if (auto* index = std::get_if<Index>(&x)) return *index;
  return span(std::get<std::vector<Index>>(x));
}

}  // namespace internal_python
}  // namespace tensorstore
