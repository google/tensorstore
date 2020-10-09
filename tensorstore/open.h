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

#include "tensorstore/context.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"
#include "tensorstore/util/status.h"

namespace tensorstore {

namespace internal_open {
Status InvalidModeError(ReadWriteMode mode, ReadWriteMode static_mode);
Status ValidateDataTypeAndRank(internal::DriverConstraints expected,
                               internal::DriverConstraints actual);
}  // namespace internal_open

/// Opens a TensorStore from a Spec.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic>
Future<TensorStore<Element, Rank, Mode>> Open(Context context,
                                              Transaction transaction,
                                              Spec spec,
                                              OpenOptions options = {}) {
  if constexpr (Mode != ReadWriteMode::dynamic) {
    if (options.read_write_mode == ReadWriteMode::dynamic) {
      options.read_write_mode = Mode;
    }
    if (!internal::IsModePossible(options.read_write_mode, Mode)) {
      return internal_open::InvalidModeError(options.read_write_mode, Mode);
    }
  }
  return MapFutureValue(
      InlineExecutor{},
      [](internal::DriverReadWriteHandle handle)
          -> Result<TensorStore<Element, Rank, Mode>> {
        TENSORSTORE_RETURN_IF_ERROR(internal_open::ValidateDataTypeAndRank(
            {StaticOrDynamicDataTypeOf<Element>(), Rank},
            {handle.driver->data_type(), handle.transform.input_rank()}));
        return internal::TensorStoreAccess::Construct<
            TensorStore<Element, Rank, Mode>>(std::move(handle));
      },
      internal::OpenDriver(std::move(context), std::move(transaction),
                           internal_spec::SpecAccess::impl(spec), options));
}

template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic>
Future<TensorStore<Element, Rank, Mode>> Open(Context context, Spec spec,
                                              OpenOptions options = {}) {
  return tensorstore::Open<Element, Rank, Mode>(
      std::move(context), no_transaction, std::move(spec), std::move(options));
}

/// Opens a TensorStore from a JSON specification.
template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic, typename J>
absl::enable_if_t<std::is_same<J, ::nlohmann::json>::value,
                  Future<TensorStore<Element, Rank, Mode>>>
Open(Context context, Transaction transaction, const J& json_spec,
     OpenOptions options = {}) {
  TENSORSTORE_ASSIGN_OR_RETURN(auto spec, Spec::FromJson(json_spec));
  return tensorstore::Open<Element, Rank, Mode>(
      std::move(context), std::move(transaction), std::move(spec),
      std::move(options));
}

template <typename Element = void, DimensionIndex Rank = dynamic_rank,
          ReadWriteMode Mode = ReadWriteMode::dynamic, typename J>
absl::enable_if_t<std::is_same<J, ::nlohmann::json>::value,
                  Future<TensorStore<Element, Rank, Mode>>>
Open(Context context, const J& json_spec, OpenOptions options = {}) {
  return tensorstore::Open<Element, Rank, Mode>(
      std::move(context), no_transaction, std::move(json_spec),
      std::move(options));
}

}  // namespace tensorstore

#endif  // TENSORSTORE_OPEN_H_
