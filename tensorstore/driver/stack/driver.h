// Copyright 2023 The TensorStore Authors
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

#ifndef TENSORSTORE_DRIVER_STACK_DRIVER_H_
#define TENSORSTORE_DRIVER_STACK_DRIVER_H_

#include <cassert>
#include <variant>

#include "tensorstore/driver/driver_handle.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_stack {

using StackOpenOptions = TransactionalOpenOptions;

// Specifies a stack layer as either a `TransformedDriverSpec` or a
// `DriverHandle`.
struct StackLayerSpec {
  StackLayerSpec() = default;
  explicit StackLayerSpec(const internal::TransformedDriverSpec& spec)
      : transform(spec.transform), driver_spec(spec.driver_spec) {}
  explicit StackLayerSpec(const internal::DriverHandle& handle)
      : transform(handle.transform),
        driver(handle.driver),
        transaction(handle.transaction) {}
  template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
  explicit StackLayerSpec(const TensorStore<Element, Rank, Mode>& store)
      : StackLayerSpec(internal::TensorStoreAccess::handle(store)) {}
  explicit StackLayerSpec(const Spec& spec)
      : StackLayerSpec(internal_spec::SpecAccess::impl(spec)) {}
  template <typename... T>
  explicit StackLayerSpec(const std::variant<T...>& layer_spec) {
    std::visit([&](auto& obj) { *this = StackLayerSpec(obj); }, layer_spec);
  }

  bool is_open() const { return static_cast<bool>(driver); }

  internal::TransformedDriverSpec GetTransformedDriverSpec() const {
    assert(!is_open());
    return internal::TransformedDriverSpec{driver_spec, transform};
  }

  // Index transform that applies to either `driver_spec` or `driver`.
  IndexTransform<> transform;

  // Indicates a layer backed by a driver spec that will be opened on demand.
  internal::DriverSpecPtr driver_spec;

  // Indicates a layer backed by an open driver.
  internal::ReadWritePtr<internal::Driver> driver;

  // May be specified only if `driver` is non-null.
  Transaction transaction{no_transaction};
};

Result<internal::DriverHandle> Overlay(span<const StackLayerSpec> layer_specs,
                                       StackOpenOptions&& options);
Result<internal::DriverHandle> Stack(span<const StackLayerSpec> layer_specs,
                                     DimensionIndex stack_dimension,
                                     StackOpenOptions&& options);
Result<internal::DriverHandle> Concat(span<const StackLayerSpec> layer_specs,
                                      DimensionIdentifier concat_dimension,
                                      StackOpenOptions&& options);

template <typename T>
struct StackResult {
  using type = TensorStore<>;
};
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct StackResult<TensorStore<Element, Rank, Mode>> {
  using type = TensorStore<Element, RankConstraint::Add(Rank, 1).rank, Mode>;
};

template <typename T>
struct OverlayResult {
  using type = TensorStore<>;
};
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
struct OverlayResult<TensorStore<Element, Rank, Mode>> {
  using type = TensorStore<Element, Rank, Mode>;
};
}  // namespace internal_stack
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_STACK_DRIVER_H_
