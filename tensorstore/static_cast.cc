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

#include "tensorstore/static_cast.h"

#include "absl/status/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_cast {
absl::Status CastError(std::string_view source_description,
                       std::string_view target_description) {
  return absl::InvalidArgumentError(tensorstore::StrCat(
      "Cannot cast ", source_description, " to ", target_description));
}

}  // namespace internal_cast
}  // namespace tensorstore
