// Copyright 2024 The TensorStore Authors
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

#ifndef TENSORSTORE_INTERNAL_JSON_BINDING_BOX_H_
#define TENSORSTORE_INTERNAL_JSON_BINDING_BOX_H_

#include <type_traits>

#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/box.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_json_binding {

/// Implementation details for jb::Box and similar.
template <bool kDiscardEmpty>
struct BoxBinderImpl {
  template <typename Options, DimensionIndex Rank>
  absl::Status operator()(std::true_type, const Options& options,
                          BoxView<Rank, true>* obj, ::nlohmann::json* j) const {
    Box<Rank> box;
    TENSORSTORE_RETURN_IF_ERROR(LoadImpl(options, &box, j));
    obj->DeepAssign(box);
    return absl::OkStatus();
  }

  template <typename Options, DimensionIndex Rank>
  absl::Status operator()(std::true_type, const Options& options,
                          Box<Rank>* obj, ::nlohmann::json* j) const {
    return LoadImpl(options, obj, j);
  }

  template <typename Options, DimensionIndex Rank, bool Mutable>
  absl::Status operator()(std::false_type, const Options& options,
                          const BoxView<Rank, Mutable>* obj,
                          ::nlohmann::json* j) const {
    return SaveImpl(options, obj, j);
  }

  template <typename Options, DimensionIndex Rank>
  absl::Status operator()(std::false_type, const Options& options,
                          const Box<Rank>* obj, ::nlohmann::json* j) const {
    return SaveImpl(options, obj, j);
  }

 private:
  template <typename Options, typename Obj>
  absl::Status SaveImpl(const Options& options, const Obj* obj,
                        ::nlohmann::json* j) const {
    if constexpr (kDiscardEmpty) {
      if (obj->empty()) {
        *j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
        return absl::OkStatus();
      }
    }
    *j = ::nlohmann::json::object_t();
    auto* j_obj = j->template get_ptr<::nlohmann::json::object_t*>();
    (*j_obj)["shape"] =
        ::nlohmann::json::array_t(obj->shape().cbegin(), obj->shape().cend());
    (*j_obj)["origin"] =
        ::nlohmann::json::array_t(obj->origin().cbegin(), obj->origin().cend());
    return absl::OkStatus();
  }

  template <typename Options, DimensionIndex Rank>
  absl::Status LoadImpl(const Options& options, Box<Rank>* obj,
                        ::nlohmann::json* j) const {
    if constexpr (kDiscardEmpty) {
      if (j->is_discarded()) return absl::OkStatus();
    }
    auto* j_obj = j->template get_ptr<::nlohmann::json::object_t*>();
    if (!j_obj) {
      return internal_json::ExpectedError(*j, "object");
    }
    // TODO: Allow 0-origin to be omitted.
    auto shape_it = j_obj->find("shape");
    auto origin_it = j_obj->find("origin");
    if (origin_it == j_obj->end() || shape_it == j_obj->end() ||
        !shape_it->second.template get_ptr<::nlohmann::json::array_t*>() ||
        !origin_it->second.template get_ptr<::nlohmann::json::array_t*>()) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Expected \"shape\" and \"origin\" as members of box: ", j->dump()));
    }
    if (shape_it->second.template get_ptr<::nlohmann::json::array_t*>()
            ->size() !=
        origin_it->second.template get_ptr<::nlohmann::json::array_t*>()
            ->size()) {
      return absl::InvalidArgumentError(tensorstore::StrCat(
          "Expected \"shape\" and \"origin\" have the same rank: ", j->dump()));
    }
    if constexpr (Rank != dynamic_rank) {
      if (shape_it->second.template get_ptr<::nlohmann::json::array_t*>()
                  ->size() != Rank ||
          origin_it->second.template get_ptr<::nlohmann::json::array_t*>()
                  ->size() != Rank) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Expected \"shape\" and \"origin\" have rank ",
                                Rank, ", got: ", j->dump()));
      }
    }

    *obj = Box<>(
        tensorstore::span(
            *origin_it->second.template get_ptr<::nlohmann::json::array_t*>()),
        tensorstore::span(
            *shape_it->second.template get_ptr<::nlohmann::json::array_t*>()));
    return absl::OkStatus();
  }
};

template <DimensionIndex Rank>
constexpr inline auto DefaultBinder<Box<Rank>> = BoxBinderImpl<false>();

template <DimensionIndex Rank>
constexpr inline auto DefaultBinder<BoxView<Rank, true>> =
    BoxBinderImpl<false>();

template <bool kDiscardEmpty>
constexpr auto BoxBinder() {
  return BoxBinderImpl<kDiscardEmpty>();
};

}  // namespace internal_json_binding
}  // namespace tensorstore

#endif  // TENSORSTORE_INTERNAL_JSON_BINDING_BOX_H_
