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

#ifndef TENSORSTORE_OPEN_H_
#define TENSORSTORE_OPEN_H_

#include <type_traits>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

/// Opens a TensorStore from a Spec.
///
/// Options are specified in any order after `spec`.  The meaning of the option
/// is determined by its type.
///
/// Supported option types include:
///
/// - Context: Shared resource context to use.  Defaults to
///   `Context::Default()`.
///
/// - Transaction: Transaction to use for opening.  Defaults to
///   `no_transaction`.
///
/// - ReadWriteMode: specifies whether reading and/or writing is supported.
///   Defaults to `Mode`.  Specifying multiple modes as separate options is
///   equivalent to ORing them together.
///
/// - OpenMode: specifies the open mode (overriding any open mode set on
///   `spec`).  Specifying multiple modes as separate options is equivalent to
///   ORing them together.
///
/// - RecheckCached, RecheckCachedData, RecheckCachedMetadata: specifies cache
///   staleness bounds, overriding the corresponding bound or default from
///   `spec`.
///
/// - kvstore::Spec: specifies the underlying storage, if applicable.
///
/// For option types other than `ReadWriteMode` and `OpenMode` (for which
/// multiple modes are ORed together), if the same option type is specified more
/// than once, the later value takes precedence; however, for the sake of
/// readability, it is not recommended to rely on this override behavior.
///
/// Example usage::
///
///     tensorstore::Context context = ...;
///     TENSORSTORE_ASSIGN_OR_RETURN(auto store,
///         tensorstore::Open({{"driver", "zarr"},
///                            {"kvstore", {{"driver", "file"},
///                                         {"path", "/tmp/data"}}}},
///                           context,
///                           tensorstore::OpenMode::open,
///                           tensorstore::RecheckCached{false},
///                           tensorstore::ReadWriteMode::read).result());
///
/// \tparam Element Constrains data type at compile time, defaults to `void` (no
///     constraint).
/// \tparam Rank Constrains rank at compile time, defaults to `dynamic_rank`.
/// \tparam Mode Constrains read-write mode at compile-time, defaults to
///     `ReadWriteMode::dynamic`.
/// \param spec The Spec to open.
/// \param option Any option compatible with `TransactionalOpenOptions`.
/// \relates TensorStore
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic>
Future<TensorStore<Element, Rank, Mode>> Open(
    Spec spec, TransactionalOpenOptions&& options) {
  if constexpr (Mode != ReadWriteMode::dynamic) {
    if (options.read_write_mode == ReadWriteMode::dynamic) {
      options.read_write_mode = Mode;
    } else if (!internal::IsModePossible(options.read_write_mode, Mode)) {
      return internal::InvalidModeError(options.read_write_mode, Mode);
    }
  }
  return internal::ConvertTensorStoreFuture<Element, Rank, Mode>(
      internal::OpenDriver(std::move(internal_spec::SpecAccess::impl(spec)),
                           std::move(options)));
}
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic, typename... Option>
std::enable_if_t<
    IsCompatibleOptionSequence<TransactionalOpenOptions, Option...>,
    Future<TensorStore<Element, Rank, Mode>>>
Open(Spec spec, Option&&... option) {
  TENSORSTORE_INTERNAL_ASSIGN_OPTIONS_OR_RETURN(TransactionalOpenOptions,
                                                options, option)
  return tensorstore::Open<Element, Rank, Mode>(std::move(spec),
                                                std::move(options));
}
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic,
          typename J = ::nlohmann::json, typename... Option>
std::enable_if_t<
    (IsCompatibleOptionSequence<TransactionalOpenOptions, Option...> &&
     std::is_same_v<J, ::nlohmann::json>),
    Future<TensorStore<Element, Rank, Mode>>>
Open(J json_spec, Option&&... option) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec, Spec::FromJson(json_spec));
  return tensorstore::Open<Element, Rank, Mode>(
      std::move(spec), std::forward<Option>(option)...);
}
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic,
          typename J = ::nlohmann::json>
std::enable_if_t<std::is_same_v<J, ::nlohmann::json>,
                 Future<TensorStore<Element, Rank, Mode>>>
Open(J json_spec, TransactionalOpenOptions&& options) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec, Spec::FromJson(json_spec));
  return tensorstore::Open<Element, Rank, Mode>(std::move(spec),
                                                std::move(options));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_OPEN_H_
