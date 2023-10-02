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

#ifndef TENSORSTORE_DRIVER_ZARR3_CODEC_REGISTRY_H_
#define TENSORSTORE_DRIVER_ZARR3_CODEC_REGISTRY_H_

#include <functional>
#include <optional>
#include <string_view>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/driver/zarr3/codec/codec_spec.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_registry.h"
#include "tensorstore/util/quote_string.h"

namespace tensorstore {
namespace internal_zarr3 {

template <
    typename ValueBinder = decltype(internal_json_binding::DefaultBinder<>)>
constexpr inline auto OptionalIfConstraintsBinder(
    ValueBinder value_binder = internal_json_binding::DefaultBinder<>) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json* j) {
    if constexpr (is_loading) {
      if (options.constraints && j->is_discarded()) {
        obj->reset();
        return absl::OkStatus();
      }
      return value_binder(is_loading, options, &obj->emplace(), j);
    } else {
      if (!obj->has_value()) {
        *j = ::nlohmann::json::value_t::discarded;
        return absl::OkStatus();
      }
      return value_binder(is_loading, options, &**obj, j);
    }
  };
}

template <typename T, typename Binder, typename TryMerge = std::equal_to<void>>
absl::Status MergeConstraint(std::string_view member, T& a, const T& b,
                             Binder binder, TryMerge try_merge = {}) {
  if (!try_merge(a, b)) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Incompatible %s: %s vs %s", tensorstore::QuoteString(member),
        internal_json_binding::ToJson(a, binder).value().dump(),
        internal_json_binding::ToJson(b, binder).value().dump()));
  }
  return absl::OkStatus();
}

template <typename T, typename Binder, typename TryMerge = std::equal_to<void>>
absl::Status MergeConstraint(std::string_view member, std::optional<T>& a,
                             const std::optional<T>& b, Binder binder,
                             TryMerge try_merge = {}) {
  if (!a && b) {
    a = *b;
    return absl::OkStatus();
  }
  if (a && b) {
    return MergeConstraint(member, *a, *b, binder, try_merge);
  }
  return absl::OkStatus();
}

template <auto Member, typename T,
          typename Binder = decltype(internal_json_binding::DefaultBinder<>),
          typename TryMerge = std::equal_to<void>>
absl::Status MergeConstraint(
    std::string_view member, T& a, const T& b,
    Binder binder = internal_json_binding::DefaultBinder<>,
    TryMerge try_merge = {}) {
  auto& a_member = a.*Member;
  auto& b_member = b.*Member;
  return MergeConstraint(member, a_member, b_member, binder, try_merge);
}

using CodecRegistry =
    internal::JsonRegistry<ZarrCodecSpec, ZarrCodecSpec::FromJsonOptions,
                           ZarrCodecSpec::ToJsonOptions, ZarrCodecSpec::Ptr>;

CodecRegistry& GetCodecRegistry();

template <typename T, typename Binder>
void RegisterCodec(std::string_view id, Binder binder) {
  GetCodecRegistry().Register<T>(id, std::move(binder));
}

}  // namespace internal_zarr3
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR3_CODEC_REGISTRY_H_
