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

#ifndef TENSORSTORE_STACK_H_
#define TENSORSTORE_STACK_H_

#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "tensorstore/driver/stack/driver.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_identifier.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/open_options.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/option.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {

using StackOpenOptions = TransactionalOpenOptions;

/// Virtually overlays a sequence of `TensorStore` or `Spec` layers within a
/// common domain.
///
/// \param layers Sequence of layers to overlay.  Later layers take precedence.
template <typename Layers = std::vector<std::variant<Spec, TensorStore<>>>>
Result<typename internal_stack::OverlayResult<
    internal::remove_cvref_t<typename Layers::value_type>>::type>
Overlay(const Layers& layers, StackOpenOptions&& options) {
  std::vector<internal_stack::StackLayerSpec> layers_internal(
      std::begin(layers), std::end(layers));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle,
      internal_stack::Overlay(layers_internal, std::move(options)));
  return {std::in_place,
          internal::TensorStoreAccess::Construct<
              typename internal_stack::OverlayResult<
                  internal::remove_cvref_t<typename Layers::value_type>>::type>(
              std::move(handle))};
}
template <typename Layers = std::vector<std::variant<Spec, TensorStore<>>>,
          typename... Option>
std::enable_if_t<
    IsCompatibleOptionSequence<StackOpenOptions, Option...>,
    Result<typename internal_stack::OverlayResult<
        internal::remove_cvref_t<typename Layers::value_type>>::type>>
Overlay(const Layers& layers, Option&&... option) {
  TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(StackOpenOptions, options,
                                                option)
  return Overlay(layers, std::move(options));
}

/// Virtually stacks a sequence of `TensorStore` or `Spec` layers along a new
/// dimension.
///
/// \param layers Sequence of layers to stack.
template <typename Layers = std::vector<std::variant<Spec, TensorStore<>>>>
Result<typename internal_stack::StackResult<
    internal::remove_cvref_t<typename Layers::value_type>>::type>
Stack(const Layers& layers, DimensionIndex stack_dimension,
      StackOpenOptions&& options) {
  std::vector<internal_stack::StackLayerSpec> layers_internal(
      std::begin(layers), std::end(layers));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle, internal_stack::Stack(layers_internal, stack_dimension,
                                         std::move(options)));
  return {std::in_place,
          internal::TensorStoreAccess::Construct<
              typename internal_stack::StackResult<
                  internal::remove_cvref_t<typename Layers::value_type>>::type>(
              std::move(handle))};
}
template <typename Layers = std::vector<std::variant<Spec, TensorStore<>>>,
          typename... Option>
std::enable_if_t<
    IsCompatibleOptionSequence<StackOpenOptions, Option...>,
    Result<typename internal_stack::StackResult<
        internal::remove_cvref_t<typename Layers::value_type>>::type>>
Stack(const Layers& layers, DimensionIndex stack_dimension,
      Option&&... option) {
  TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(StackOpenOptions, options,
                                                option)
  return Stack(layers, stack_dimension, std::move(options));
}

/// Virtually concatenates a sequence of `TensorStore` or `Spec` layers along an
/// existing dimension.
///
/// \param layers Sequence of layers to concatenate.
template <typename Layers = std::vector<std::variant<Spec, TensorStore<>>>>
Result<typename internal_stack::OverlayResult<
    internal::remove_cvref_t<typename Layers::value_type>>::type>
Concat(const Layers& layers, DimensionIdentifier concat_dimension,
       StackOpenOptions&& options) {
  std::vector<internal_stack::StackLayerSpec> layers_internal(
      std::begin(layers), std::end(layers));
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto handle, internal_stack::Concat(layers_internal, concat_dimension,
                                          std::move(options)));
  return {std::in_place,
          internal::TensorStoreAccess::Construct<
              typename internal_stack::OverlayResult<
                  internal::remove_cvref_t<typename Layers::value_type>>::type>(
              std::move(handle))};
}
template <typename Layers = std::vector<std::variant<Spec, TensorStore<>>>,
          typename... Option>
std::enable_if_t<
    IsCompatibleOptionSequence<StackOpenOptions, Option...>,
    Result<typename internal_stack::OverlayResult<
        internal::remove_cvref_t<typename Layers::value_type>>::type>>
Concat(const Layers& layers, DimensionIdentifier concat_dimension,
       Option&&... option) {
  TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(StackOpenOptions, options,
                                                option)
  return Concat(layers, concat_dimension, std::move(options));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_STACK_H_
