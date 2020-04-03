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

#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal_cast {
Status CastError(absl::string_view source_description,
                 absl::string_view target_description) {
  return absl::InvalidArgumentError(
      StrCat("Cannot cast ", source_description, " to ", target_description));
}

}  // namespace internal_cast
}  // namespace tensorstore
