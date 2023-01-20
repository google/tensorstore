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

#include "tensorstore/index_space/json.h"

#include "absl/container/fixed_array.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/json/array.h"
#include "tensorstore/internal/json_binding/array.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace {

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_index_space::BuilderFlags;
using ::tensorstore::internal_index_space::OutputIndexMapInitializer;
using ::tensorstore::internal_index_space::TransformRep;

struct DomainJsonKeys {
  const char* rank;
  const char* inclusive_min;
  const char* inclusive_max;
  const char* shape;
  const char* exclusive_max;
  const char* labels;
};

constexpr DomainJsonKeys kIndexDomainJsonKeys = {
    "rank",  "inclusive_min", "inclusive_max",
    "shape", "exclusive_max", "labels",
};

constexpr DomainJsonKeys kIndexTransformJsonKeys = {
    "input_rank",  "input_inclusive_min", "input_inclusive_max",
    "input_shape", "input_exclusive_max", "input_labels",
};

// ImplicitPairBinder expects Obj to be a std::pair<value, bool-like>*,
// and expresses the implicit value as being wrapped in an array of length 1.
template <typename ElementBinder>
struct ImplicitPairBinder {
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS ElementBinder element_binder;

  template <typename Options, typename Obj>
  absl::Status operator()(std::true_type is_loading, const Options& options,
                          Obj* obj, ::nlohmann::json* j) const {
    auto&& [element, is_implicit] = *obj;
    if (const auto* k = j->get_ptr<const ::nlohmann::json::array_t*>()) {
      if (k->size() != 1) {
        return internal_json::ExpectedError(
            *k, "array of size 1 indicating an implicit value");
      }
      is_implicit = true;
      return element_binder(is_loading, options, &element, &(*k)[0]);
    } else {
      is_implicit = false;
      return element_binder(is_loading, options, &element, j);
    }
  }

  template <typename Options, typename Obj>
  absl::Status operator()(std::false_type is_loading, const Options& options,
                          const Obj* obj, ::nlohmann::json* j) const {
    auto&& [element, is_implicit] = *obj;
    if (is_implicit) {
      ::nlohmann::json::array_t k(1);
      TENSORSTORE_RETURN_IF_ERROR(
          element_binder(is_loading, options, &element, &k[0]));
      *j = std::move(k);
    } else {
      return element_binder(is_loading, options, &element, j);
    }
    return absl::OkStatus();
  }
};

/// Similar to `DimensionIndexedVector`, adapted to load implicit
/// values described by ImplicitVector, above.
template <typename RankProjection, typename ValuesProjection,
          typename ImplicitProjection, typename ElementBinder>
struct ImplicitArrayBinderImpl {
  RankProjection rank_ptr;
  ValuesProjection values_ptr;
  ImplicitProjection implicit_ptr;
  TENSORSTORE_ATTRIBUTE_NO_UNIQUE_ADDRESS ElementBinder element_binder;

  template <typename Loading, typename Options, typename Obj>
  absl::Status operator()(Loading is_loading, const Options& options, Obj* obj,
                          ::nlohmann::json* j) const {
    return jb::OptionalArray(
        [this](const auto& obj) { return std::invoke(values_ptr, obj).size(); },
        [this](auto& obj, size_t size) {
          TENSORSTORE_RETURN_IF_ERROR(ValidateRank(size));
          auto&& rank = std::invoke(rank_ptr, obj);
          if (rank == dynamic_rank) {
            rank = size;
          } else if (rank != static_cast<DimensionIndex>(size)) {
            return internal_json::JsonValidateArrayLength(size, rank);
          }
          std::invoke(values_ptr, obj).resize(size);
          return absl::OkStatus();
        },
        [this](auto& obj, size_t i) {
          auto& value = std::invoke(values_ptr, obj)[i];
          auto implicit_value = std::invoke(implicit_ptr, obj)[i];
          return std::pair<decltype(value), decltype(implicit_value)>(
              value, implicit_value);
        },
        element_binder)(is_loading, options, obj, j);
  }
};

template <typename T>
using InlinedVector = absl::InlinedVector<T, internal::kNumInlinedDims>;

struct TransformParserOutput {
  Index offset = 0;
  Index stride = 1;
  std::optional<DimensionIndex> input_dimension;
  IndexInterval index_array_bounds;
  SharedArray<const Index, dynamic_rank> index_array;
};

struct TransformParserData {
  IntervalForm interval_form = IntervalForm::half_open;
  BuilderFlags flags{0};
  DimensionIndex rank = dynamic_rank;
  InlinedVector<Index> lower_bounds;
  InlinedVector<Index> upper_bounds;
  DimensionSet implicit_lower_bounds;
  DimensionSet implicit_upper_bounds;
  InlinedVector<std::string> labels;

  // outputs
  std::optional<InlinedVector<TransformParserOutput>> output;

  // Convert to a transform.
  Result<TransformRep::Ptr<>> Finalize();
};

constexpr auto TransformParserOutputBinder = jb::Object(
    jb::Member("offset",
               jb::Projection(&TransformParserOutput::offset,
                              jb::DefaultValue([](Index* o) { *o = 0; }))),
    jb::AtMostOne("input_dimension", "index_array"),
    jb::Member("input_dimension",
               jb::Projection(&TransformParserOutput::input_dimension,
                              jb::Optional())),
    jb::OptionalMember(
        "index_array",
        jb::Projection(&TransformParserOutput::index_array, jb::NestedArray())),
    jb::OptionalMember(
        "index_array_bounds",
        jb::Sequence(jb::Initialize([](auto* obj) {
                       if (!obj->index_array.data()) {
                         return absl::InvalidArgumentError(
                             "\"index_array_bounds\" is only valid with "
                             "\"index_array\"");
                       }
                       return absl::OkStatus();
                     }),
                     jb::Projection(&TransformParserOutput::index_array_bounds,
                                    jb::DefaultValue(
                                        [](auto* obj) {
                                          *obj = IndexInterval::Infinite();
                                        },
                                        jb::IndexIntervalBinder)))),
    jb::OptionalMember(
        "stride",
        jb::Sequence(
            jb::Initialize([](auto* obj) {
              if (!obj->input_dimension && !obj->index_array.data()) {
                return absl::InvalidArgumentError(
                    "Either \"input_dimension\" or \"index_array\" must be "
                    "specified in "
                    "conjunction with \"stride\"");
              }
              return absl::OkStatus();
            }),

            jb::Projection(&TransformParserOutput::stride,
                           jb::DefaultValue([](Index* s) { *s = 1; }))))
    /**/);

template <typename T, typename ElementBinder>
constexpr auto LowerBoundsBinder(ElementBinder element_binder) {
  using Binder = ImplicitPairBinder<internal::remove_cvref_t<ElementBinder>>;
  auto rank_ptr = &T::rank;
  auto value_ptr = &T::lower_bounds;
  auto implicit_ptr = &T::implicit_lower_bounds;
  return ImplicitArrayBinderImpl<decltype(rank_ptr), decltype(value_ptr),
                                 decltype(implicit_ptr), Binder>{
      std::move(rank_ptr), std::move(value_ptr), std::move(implicit_ptr),
      Binder{std::move(element_binder)}};
}

template <typename T, typename ElementBinder>
constexpr auto UpperBoundsBinder(ElementBinder element_binder) {
  using Binder = ImplicitPairBinder<internal::remove_cvref_t<ElementBinder>>;
  auto rank_ptr = &T::rank;
  auto value_ptr = &T::upper_bounds;
  auto implicit_ptr = &T::implicit_upper_bounds;
  return ImplicitArrayBinderImpl<decltype(rank_ptr), decltype(value_ptr),
                                 decltype(implicit_ptr), Binder>{
      std::move(rank_ptr), std::move(value_ptr), std::move(implicit_ptr),
      Binder{std::move(element_binder)}};
}

constexpr auto IndexTransformParser(
    bool is_transform, DimensionIndex input_rank_constraint = dynamic_rank) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json::object_t* j) -> absl::Status {
    using T = TransformParserData;

    auto* keys =
        is_transform ? &kIndexTransformJsonKeys : &kIndexDomainJsonKeys;
    DimensionIndex* rank = is_loading ? &obj->rank : nullptr;

    return jb::Sequence(
        jb::AtLeastOne(keys->rank, keys->inclusive_min, keys->shape,
                       keys->inclusive_max, keys->exclusive_max, keys->labels),
        // "rank" / "input_rank" are only emitted when none of the implicit
        // fields are in the json object.
        [=](auto is_loading, const auto& options, auto* obj,
            ::nlohmann::json::object_t* j) -> absl::Status {
          if constexpr (!is_loading) {
            if (j->count(keys->inclusive_min) ||
                j->count(keys->exclusive_max) || j->count(keys->labels)) {
              return absl::OkStatus();
            }
          }
          return jb::Member(
              keys->rank,
              jb::Projection(&T::rank,
                             jb::DefaultValue(
                                 [](DimensionIndex* o) { *o = dynamic_rank; },
                                 jb::Integer<DimensionIndex>(0, kMaxRank))) /**/
              )(is_loading, options, obj, j);
        },
        jb::OptionalMember(keys->inclusive_min,
                           jb::Sequence(LowerBoundsBinder<T>(
                                            jb::BoundsBinder<-kInfIndex, 0>()),
                                        jb::Initialize([](auto* obj) {
                                          obj->flags |=
                                              (BuilderFlags::kSetLower |
                                               BuilderFlags::kSetImplicitLower);
                                        }))),
        /// Upper bounds can be specified in several ways.
        jb::AtMostOne(keys->shape, keys->inclusive_max, keys->exclusive_max),
        jb::OptionalMember(
            keys->shape,
            jb::LoadSave(jb::Sequence(
                UpperBoundsBinder<T>(jb::BoundsBinder<0, +kInfSize>()),
                jb::Initialize([](auto* obj) {
                  obj->interval_form = IntervalForm::sized;
                  obj->flags |= (BuilderFlags::kSetUpper |
                                 BuilderFlags::kSetImplicitUpper);
                })))),
        jb::OptionalMember(
            keys->inclusive_max,
            jb::LoadSave(jb::Sequence(
                UpperBoundsBinder<T>(jb::BoundsBinder<0, +kInfIndex>()),
                jb::Initialize([](auto* obj) {
                  obj->interval_form = IntervalForm::closed;
                  obj->flags |= (BuilderFlags::kSetUpper |
                                 BuilderFlags::kSetImplicitUpper);
                })))),
        jb::OptionalMember(
            keys->exclusive_max,
            jb::Sequence(
                UpperBoundsBinder<T>(jb::BoundsBinder<0, +kInfIndex + 1>()),
                jb::Initialize([](auto* obj) {
                  obj->interval_form = IntervalForm::half_open;
                  obj->flags |= (BuilderFlags::kSetUpper |
                                 BuilderFlags::kSetImplicitUpper);
                }))),
        jb::OptionalMember(
            keys->labels,
            jb::Projection(&T::labels, jb::DimensionLabelVector(rank))),
        jb::Initialize([=](auto* obj) {
          if (!RankConstraint::EqualOrUnspecified(input_rank_constraint,
                                                  obj->rank)) {
            return absl::InvalidArgumentError(tensorstore::StrCat(
                "Expected ", keys->rank, " to be ", input_rank_constraint,
                ", but is: ", obj->rank));
          }
          return absl::OkStatus();
        })
        /**/)(is_loading, options, obj, j);
  };
}

constexpr auto IndexTransformOutputParser(
    DimensionIndex output_rank_constraint = dynamic_rank) {
  return [=](auto is_loading, const auto& options, auto* obj,
             ::nlohmann::json::object_t* j) -> absl::Status {
    return jb::Sequence(
        jb::Member("output", jb::Projection(&TransformParserData::output,
                                            jb::Optional(jb::Array(
                                                TransformParserOutputBinder)))),
        jb::Initialize([=](auto* obj) {
          // output rank was constrained,
          if (obj->output) {
            if (output_rank_constraint != dynamic_rank &&
                obj->output->size() != output_rank_constraint) {
              return absl::InvalidArgumentError(tensorstore::StrCat(
                  "Expected output rank to be ", output_rank_constraint,
                  ", but is: ", obj->output->size()));
            }
            return absl::OkStatus();
          }
          const DimensionIndex rank = obj->rank;
          if (output_rank_constraint != dynamic_rank &&
              output_rank_constraint != rank) {
            // The constraint is something other than the input rank.
            return absl::InvalidArgumentError("Missing \"output\" member");
          }
          return absl::OkStatus();
        }) /**/)(is_loading, options, obj, j);
  };
}

Result<TransformRep::Ptr<>> TransformParserData::Finalize() {
  if (!output) {
    // No outputs specified, so initialize them from the input rank.
    output.emplace(rank);
    for (DimensionIndex i = 0; i < rank; ++i) {
      (*output)[i].input_dimension = i;
    }
  }

  const DimensionIndex output_rank = output->size();

  auto transform = TransformRep::Allocate(rank, output_rank);
  transform->input_rank = rank;
  transform->output_rank = output_rank;
  if ((flags & BuilderFlags::kSetLower) != BuilderFlags::kDefault) {
    std::copy(lower_bounds.begin(), lower_bounds.end(),
              transform->input_origin().begin());
    transform->implicit_lower_bounds = implicit_lower_bounds;
  }
  if ((flags & BuilderFlags::kSetUpper) != BuilderFlags::kDefault) {
    std::copy(upper_bounds.begin(), upper_bounds.end(),
              transform->input_shape().begin());
    transform->implicit_upper_bounds = implicit_upper_bounds;
  }
  if (!labels.empty()) {
    std::copy(labels.begin(), labels.end(), transform->input_labels().begin());
  }
  InlinedVector<OutputIndexMapInitializer> output_maps;
  output_maps.reserve(output_rank);
  auto maps = transform->output_index_maps();
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& out = (*output)[output_dim];
    auto& map = maps[output_dim];
    map.offset() = out.offset;
    map.stride() = out.stride;
    output_maps.emplace_back(
        out.input_dimension
            ? OutputIndexMapInitializer(out.input_dimension.value())
            : OutputIndexMapInitializer(out.index_array,
                                        out.index_array_bounds));
  }
  TENSORSTORE_RETURN_IF_ERROR(SetOutputIndexMapsAndValidateTransformRep(
      transform.get(), output_maps, interval_form, flags));
  return transform;
}

TransformParserData MakeIndexDomainViewDataForSaving(IndexDomainView<> domain) {
  const DimensionIndex rank = domain.rank();

  TransformParserData tmp;
  tmp.rank = rank;
  tmp.lower_bounds.resize(rank);
  tmp.upper_bounds.resize(rank);
  tmp.labels.assign(domain.labels().begin(), domain.labels().end());
  tmp.implicit_lower_bounds = domain.implicit_lower_bounds();
  tmp.implicit_upper_bounds = domain.implicit_upper_bounds();

  // Compute the `inclusive_min` and `exclusive_max` members.
  bool all_implicit_lower = true;
  bool all_implicit_upper = true;
  for (DimensionIndex i = 0; i < rank; ++i) {
    tmp.lower_bounds[i] = domain[i].inclusive_min();
    tmp.upper_bounds[i] = domain[i].exclusive_max();
    all_implicit_lower = all_implicit_lower && tmp.implicit_lower_bounds[i] &&
                         (tmp.lower_bounds[i] == -kInfIndex);
    all_implicit_upper = all_implicit_upper && tmp.implicit_upper_bounds[i] &&
                         (tmp.upper_bounds[i] == (+kInfIndex + 1));
  }

  // Avoid outputting implicit bounds.
  // NOTE: Move this logic to the binder.
  if (all_implicit_lower) {
    tmp.lower_bounds.resize(0);
  }
  if (all_implicit_upper) {
    tmp.upper_bounds.resize(0);
  }
  return tmp;
}

TransformParserData MakeIndexTransformViewDataForSaving(
    IndexTransformView<> transform) {
  auto input_domain = transform.input_domain();

  TransformParserData tmp = MakeIndexDomainViewDataForSaving(input_domain);
  const DimensionIndex input_rank = transform.input_rank();
  const DimensionIndex output_rank = transform.output_rank();
  bool all_identity = (output_rank == input_rank);
  tmp.output.emplace(output_rank);

  auto maps = transform.output_index_maps();
  for (DimensionIndex i = 0; i < output_rank; ++i) {
    auto& output = (*tmp.output)[i];
    const auto map = maps[i];
    if (map.offset() != 0) {
      output.offset = map.offset();
      all_identity = false;
    }
    if (map.method() != OutputIndexMethod::constant && map.stride() != 1) {
      output.stride = map.stride();
      all_identity = false;
    }
    switch (map.method()) {
      case OutputIndexMethod::constant:
        all_identity = false;
        break;
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        output.input_dimension = input_dim;
        if (input_dim != i) all_identity = false;
        break;
      }
      case OutputIndexMethod::array: {
        all_identity = false;
        const auto index_array_data = map.index_array();
        output.index_array = UnbroadcastArrayPreserveRank(
            UnownedToShared(index_array_data.array_ref()));
        // If `index_array` contains values outside `index_range`, encode
        // `index_range` as well to avoid expanding the range.
        IndexInterval index_range = index_array_data.index_range();
        if (index_range != IndexInterval::Infinite() &&
            !ValidateIndexArrayBounds(index_range, output.index_array).ok()) {
          output.index_array_bounds = index_range;
        }
        break;
      }
    }
  }
  if (all_identity) {
    tmp.output = std::nullopt;
  }

  return tmp;
}

}  // namespace

void to_json(::nlohmann::json& j,  // NOLINT
             IndexTransformView<> transform) {
  if (!transform.valid()) {
    j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
    return;
  }
  auto binder = jb::Object(IndexTransformParser(/*is_transform=*/true),
                           IndexTransformOutputParser());

  auto tmp = MakeIndexTransformViewDataForSaving(transform);

  ::nlohmann::json::object_t obj;
  auto status = binder(std::false_type{}, IncludeDefaults{false}, &tmp, &obj);
  status.IgnoreError();
  assert(status.ok());

  j = std::move(obj);
}

void to_json(::nlohmann::json& j,  // NOLINT
             IndexDomainView<> domain) {
  if (!domain.valid()) {
    j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
    return;
  }
  auto binder = jb::Object(IndexTransformParser(/*is_transform=*/false));

  auto tmp = MakeIndexDomainViewDataForSaving(domain);

  ::nlohmann::json::object_t obj;
  auto status = binder(std::false_type{}, IncludeDefaults{false}, &tmp, &obj);
  status.IgnoreError();
  assert(status.ok());

  j = std::move(obj);
}

void to_json(::nlohmann::json& j,  // NOLINT
             IndexInterval interval) {
  auto status = jb::IndexIntervalBinder(std::false_type{},
                                        IncludeDefaults{false}, &interval, &j);
  status.IgnoreError();
  assert(status.ok());
}

namespace internal_index_space {

Result<TransformRep::Ptr<>> ParseIndexTransformFromJson(
    const ::nlohmann::json& j, DimensionIndex input_rank_constraint,
    DimensionIndex output_rank_constraint) {
  if (j.is_discarded()) return TransformRep::Ptr<>(nullptr);
  auto result = [&]() -> Result<TransformRep::Ptr<>> {
    auto binder = jb::Object(IndexTransformParser(true, input_rank_constraint),
                             IndexTransformOutputParser(output_rank_constraint)
                             /**/);
    TENSORSTORE_ASSIGN_OR_RETURN(auto parser_data,
                                 jb::FromJson<TransformParserData>(j, binder));
    return parser_data.Finalize();
  }();

  if (result) return result;
  return MaybeAnnotateStatus(result.status(),
                             "Error parsing index transform from JSON");
}

Result<TransformRep::Ptr<>> ParseIndexDomainFromJson(
    const ::nlohmann::json& j, DimensionIndex rank_constraint) {
  if (j.is_discarded()) return TransformRep::Ptr<>(nullptr);
  auto result = [&]() -> Result<TransformRep::Ptr<>> {
    auto binder = jb::Object(IndexTransformParser(false, rank_constraint));
    TENSORSTORE_ASSIGN_OR_RETURN(auto parser_data,
                                 jb::FromJson<TransformParserData>(j, binder))
    return parser_data.Finalize();
  }();

  if (result) return result;
  return MaybeAnnotateStatus(result.status(),
                             "Error parsing index domain from JSON");
}

}  // namespace internal_index_space

namespace internal_json_binding {

TENSORSTORE_DEFINE_JSON_BINDER(
    ConstrainedRankJsonBinder,
    [](auto is_loading, const auto& options, auto* obj, auto* j) {
      if constexpr (is_loading) {
        if (j->is_discarded()) {
          *obj = options.rank().rank;
          return absl::OkStatus();
        }
        TENSORSTORE_RETURN_IF_ERROR(
            Integer<DimensionIndex>(0, kMaxRank)(is_loading, options, obj, j));
      } else {
        if ((!IncludeDefaults(options).include_defaults() &&
             options.rank().rank != dynamic_rank) ||
            *obj == dynamic_rank) {
          *j = ::nlohmann::json::value_t::discarded;
        } else {
          *j = *obj;
        }
      }
      if (!RankConstraint::EqualOrUnspecified(options.rank().rank, *obj)) {
        return absl::InvalidArgumentError(tensorstore::StrCat(
            "Expected ", options.rank().rank, ", but received: ", *obj));
      }
      return absl::OkStatus();
    })

}  // namespace internal_json_binding
}  // namespace tensorstore
