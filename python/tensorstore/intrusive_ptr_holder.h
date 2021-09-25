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

#ifndef THIRD_PARTY_PY_TENSORSTORE_INTRUSIVE_PTR_HOLDER_H_
#define THIRD_PARTY_PY_TENSORSTORE_INTRUSIVE_PTR_HOLDER_H_

#include <pybind11/pybind11.h>
// Other headers must be included after pybind11 to ensure header-order
// inclusion constraints are satisfied.

#include "tensorstore/internal/intrusive_ptr.h"

// Declare `tensorstore::internal::IntrusivePtr<T>` a pybind11-compatible smart
// pointer type.
//
// IntrusivePtr types that use a custom traits type must be declared separately.
//
//  When registering a class via `pybind1::class_` a registered smart pointer
// type may be specified as an additional template parameter, in which case the
// specified smart pointer type is used in place of `std::unique_ptr`.
PYBIND11_DECLARE_HOLDER_TYPE(T, ::tensorstore::internal::IntrusivePtr<T>,
                             /*always_construct_holder=*/true)

#endif  // THIRD_PARTY_PY_TENSORSTORE_INTRUSIVE_PTR_HOLDER_H_
