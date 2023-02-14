// Copyright 2021 The TensorStore Authors
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

#ifndef THIRD_PARTY_PY_TENSORSTORE_CHUNK_LAYOUT_H_
#define THIRD_PARTY_PY_TENSORSTORE_CHUNK_LAYOUT_H_

/// \file
///
/// Defines `tensorstore.ChunkLayout`.

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "tensorstore/chunk_layout.h"

namespace pybind11 {
namespace detail {

/// Conversion between `ChunkLayout::Usage` and `str`/`None`.
template <>
struct type_caster<tensorstore::ChunkLayout::Usage> {
  using T = tensorstore::ChunkLayout::Usage;
  PYBIND11_TYPE_CASTER(T, _("tensorstore.ChunkLayout.Usage"));
  bool load(handle src, bool convert);
  static handle cast(tensorstore::ChunkLayout::Usage usage,
                     return_value_policy /* policy */, handle /* parent */);
};

}  // namespace detail
}  // namespace pybind11

#endif  // THIRD_PARTY_PY_TENSORSTORE_CHUNK_CHUNK_LAYOUT_H_
