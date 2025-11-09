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

#ifndef TENSORSTORE_INTERNAL_BENCHMARK_VECTOR_FLAG_H_
#define TENSORSTORE_INTERNAL_BENCHMARK_VECTOR_FLAG_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/marshalling.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"

namespace tensorstore {

// Provides a flag that can be used to specify a vector of values.
//
// The flag is specified as a comma-separated list of values.  Each value is
// parsed as a separate flag of the specified type.
//
// Example:
//   ABSL_FLAG(tensorstore::VectorFlag<int64_t>, chunk_shape, {},
//             "Read/write chunks of --chunk_shape dimensions.");

template <typename T>
struct VectorFlag {
  VectorFlag() = default;
  VectorFlag(std::vector<T> e) : elements(std::move(e)) {}
  VectorFlag(T x) : elements({std::move(x)}) {}

  std::vector<T> elements;

  friend std::string AbslUnparseFlag(const VectorFlag<T>& list) {
    auto unparse_element = [](std::string* const out, const T element) {
      absl::StrAppend(out, absl::UnparseFlag(element));
    };
    return absl::StrJoin(list.elements, ",", unparse_element);
  }

  friend bool AbslParseFlag(std::string_view text, VectorFlag<T>* list,
                            std::string* error) {
    list->elements.clear();
    for (const auto& part : absl::StrSplit(text, ',', absl::SkipWhitespace())) {
      T element;
      // Let flag module parse the element type for us.
      if (!absl::ParseFlag(part, &element, error)) {
        return false;
      }
      list->elements.push_back(element);
    }
    return true;
  }
};

}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_BENCHMARK_VECTOR_FLAG_H_
