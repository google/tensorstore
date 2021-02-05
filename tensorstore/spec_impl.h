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

#ifndef TENSORSTORE_SPEC_IMPL_H_
#define TENSORSTORE_SPEC_IMPL_H_

/// \file
/// Implementation details of `Spec`.

#include <string>

#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/rank.h"

namespace tensorstore {
namespace internal_spec {

class SpecAccess {
 public:
  template <typename T>
  static decltype(auto) impl(T&& spec) {
    return (spec.impl_);
  }
};

}  // namespace internal_spec
}  // namespace tensorstore

#endif  // TENSORSTORE_SPEC_IMPL_H_
