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

#include "tensorstore/internal/json.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

absl::Status MaybeAnnotateMemberError(const absl::Status& status,
                                      std::string_view member_name) {
  if (status.ok()) return status;
  return MaybeAnnotateStatus(
      status, StrCat("Error parsing object member ", QuoteString(member_name)));
}

absl::Status MaybeAnnotateMemberConvertError(const absl::Status& status,
                                             std::string_view member_name) {
  if (status.ok()) return status;
  return MaybeAnnotateStatus(status, StrCat("Error converting object member ",
                                            QuoteString(member_name)));
}

absl::Status MaybeAnnotateArrayElementError(const absl::Status& status,
                                            std::size_t i, bool is_loading) {
  return MaybeAnnotateStatus(
      status,
      tensorstore::StrCat("Error ", is_loading ? "parsing" : "converting",
                          " value at position ", i));
}

}  // namespace internal_json_binding
}  // namespace tensorstore
