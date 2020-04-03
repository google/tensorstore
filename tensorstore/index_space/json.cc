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
#include "absl/container/inlined_vector.h"
#include <nlohmann/json.hpp>
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/json.h"
#include "tensorstore/internal/json_array.h"

namespace tensorstore {
namespace {
::nlohmann::json EncodeImplicit(::nlohmann::json v, bool implicit) {
  if (!implicit) return v;
  ::nlohmann::json::array_t j;
  j.push_back(std::move(v));
  return j;
}
}  // namespace

void to_json(::nlohmann::json& j,  // NOLINT
             IndexTransformView<> transform) {
  if (!transform.valid()) {
    j = ::nlohmann::json(::nlohmann::json::value_t::discarded);
    return;
  }
  ::nlohmann::json::object_t obj;
  const DimensionIndex input_rank = transform.input_rank();
  auto input_domain = transform.input_domain();

  // Compute the `"input_inclusive_min"` and `"input_exclusive_max"` members.
  {
    auto implicit_lower_bounds = transform.implicit_lower_bounds();
    auto implicit_upper_bounds = transform.implicit_upper_bounds();
    ::nlohmann::json::array_t j_inclusive_min, j_exclusive_max;
    j_inclusive_min.reserve(input_rank);
    j_exclusive_max.reserve(input_rank);
    for (DimensionIndex i = 0; i < input_rank; ++i) {
      const auto d = input_domain[i];
      const bool implicit_lower = implicit_lower_bounds[i];
      const bool implicit_upper = implicit_upper_bounds[i];
      j_inclusive_min.push_back(
          d.inclusive_min() == -kInfIndex
              ? EncodeImplicit("-inf", implicit_lower)
              : EncodeImplicit(d.inclusive_min(), implicit_lower));
      j_exclusive_max.push_back(
          d.inclusive_max() == kInfIndex
              ? EncodeImplicit("+inf", implicit_upper)
              : EncodeImplicit(d.exclusive_max(), implicit_upper));
    }
    obj.emplace("input_inclusive_min", std::move(j_inclusive_min));
    obj.emplace("input_exclusive_max", std::move(j_exclusive_max));
  }

  // Compute the `"input_labels"` member.
  {
    ::nlohmann::json::array_t j_labels;
    j_labels.reserve(input_rank);
    auto input_labels = transform.input_labels();
    bool encode_labels = false;
    for (DimensionIndex i = 0; i < input_rank; ++i) {
      auto const& label = input_labels[i];
      if (!label.empty()) {
        encode_labels = true;
      }
      j_labels.push_back(label);
    }
    // If all labels are empty, skip the `"input_labels"` member to produce a
    // shorter representation.
    if (encode_labels) {
      obj.emplace("input_labels", std::move(j_labels));
    }
  }

  // Compute the `"output"` member, which encodes the output index maps in
  // output dimension index order.
  {
    const DimensionIndex output_rank = transform.output_rank();
    ::nlohmann::json::array_t j_outputs;
    j_outputs.reserve(output_rank);
    auto maps = transform.output_index_maps();
    bool all_identity = output_rank == input_rank;
    absl::FixedArray<Index, internal::kNumInlinedDims> index_array_shape(
        input_rank);
    for (DimensionIndex i = 0; i < output_rank; ++i) {
      ::nlohmann::json::object_t j_output;
      const auto map = maps[i];
      if (map.offset() != 0) {
        j_output.emplace("offset", map.offset());
        all_identity = false;
      }
      if (map.method() != OutputIndexMethod::constant && map.stride() != 1) {
        j_output.emplace("stride", map.stride());
        all_identity = false;
      }
      switch (map.method()) {
        case OutputIndexMethod::constant:
          all_identity = false;
          break;
        case OutputIndexMethod::single_input_dimension: {
          const DimensionIndex input_dim = map.input_dimension();
          j_output.emplace("input_dimension", input_dim);
          if (input_dim != i) all_identity = false;
          break;
        }
        case OutputIndexMethod::array: {
          all_identity = false;
          const auto index_array_data = map.index_array();
          for (DimensionIndex input_dim = 0; input_dim < input_rank;
               ++input_dim) {
            index_array_shape[input_dim] =
                index_array_data.byte_strides()[input_dim] == 0
                    ? 1
                    : input_domain.shape()[input_dim];
          }
          ArrayView<const Index, dynamic_rank> index_array(
              AddByteOffset(
                  ElementPointer<const Index>(
                      index_array_data.element_pointer()),
                  IndexInnerProduct(input_rank, input_domain.origin().data(),
                                    index_array_data.byte_strides().data())),
              StridedLayoutView<>(input_rank, index_array_shape.data(),
                                  index_array_data.byte_strides().data()));
          // TODO(jbms): check bounds
          j_output.emplace("index_array",
                           internal::JsonEncodeNestedArray(
                               index_array, [](const Index* x) { return *x; }));
          break;
        }
      }
      j_outputs.emplace_back(std::move(j_output));
    }
    // If the transform is actually an identity transform, skip the `"output"`
    // member to produce a shorter representation.
    if (!all_identity) {
      obj.emplace("output", std::move(j_outputs));
    }
  }
  j = std::move(obj);
}

namespace internal_index_space {
namespace {

template <typename T>
using InlinedVector = absl::InlinedVector<T, internal::kNumInlinedDims>;

/// Parses `j` as a 64-bit signed integer, except that `"-inf"` is optionally
/// converted to `neg_infinity` and `"+inf"` is optionally converted to
/// `pos_infinity`.  The number may be specified as a JSON number or as a string
/// containing a base 10 representation.
///
/// \param j The JSON value to parse.
/// \param neg_infinity If non-zero, `"-inf"` is accepted and converted to the
///     specified value.
/// \param pos_infinity If non-zero, `"+inf"` is accepted and converted to the
///     specified value.
/// \error `absl::StatusCode::kInvalidArgument` if `j` is invalid.
Result<Index> ParseBound(const ::nlohmann::json& j, Index neg_infinity,
                         Index pos_infinity) {
  if (const auto* j_string = j.get_ptr<const ::nlohmann::json::string_t*>()) {
    if (neg_infinity != 0 && *j_string == "-inf") return neg_infinity;
    if (pos_infinity != 0 && *j_string == "+inf") return pos_infinity;
  }
  if (auto value = internal::JsonValueAs<Index>(j)) return *value;
  // Uses the same format as internal_json::ExpectedError
  return absl::InvalidArgumentError(StrCat(
      "Expected 64-bit signed integer", neg_infinity != 0 ? " or \"-inf\"" : "",
      pos_infinity != 0 ? " or \"+inf\"" : "", ", but received: ", j.dump()));
}

template <typename SubParser>
Status ParseImplicit(const ::nlohmann::json& j, bool* implicit,
                     SubParser sub_parser) {
  if (const auto* j_array = j.get_ptr<const ::nlohmann::json::array_t*>()) {
    if (j_array->size() != 1) {
      return internal_json::ExpectedError(
          j, "array of size 1 indicating an implicit value");
    }
    *implicit = true;
    return sub_parser((*j_array)[0]);
  }
  *implicit = false;
  return sub_parser(j);
}

template <typename T, typename SubParser>
Status ParseInputDimsData(const ::nlohmann::json& j_bounds,
                          absl::optional<DimensionIndex>* input_rank,
                          InlinedVector<T>* values,
                          InlinedVector<bool>* implicit, SubParser sub_parser) {
  return internal::JsonParseArray(
      j_bounds,
      [&](DimensionIndex rank) {
        if (*input_rank) {
          TENSORSTORE_RETURN_IF_ERROR(
              internal::JsonValidateArrayLength(rank, **input_rank));
        } else {
          *input_rank = rank;
        }
        values->resize(rank);
        implicit->resize(rank);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& j_bound, DimensionIndex i) {
        return ParseImplicit(
            j_bound, &(*implicit)[i], [&](const ::nlohmann::json& v) {
              TENSORSTORE_ASSIGN_OR_RETURN((*values)[i], sub_parser(v));
              return absl::OkStatus();
            });
      });
}

Status ParseInputBounds(const ::nlohmann::json& j_bounds,
                        absl::optional<DimensionIndex>* input_rank,
                        InlinedVector<Index>* bounds,
                        InlinedVector<bool>* implicit, Index neg_infinity,
                        Index pos_infinity) {
  return ParseInputDimsData(j_bounds, input_rank, bounds, implicit,
                            [&](const ::nlohmann::json& v) {
                              return ParseBound(v, neg_infinity, pos_infinity);
                            });
}

Status ParseInputLabels(const ::nlohmann::json& j_bounds,
                        absl::optional<DimensionIndex>* input_rank,
                        InlinedVector<std::string>* labels) {
  return internal::JsonParseArray(
      j_bounds,
      [&](DimensionIndex rank) {
        if (*input_rank) {
          TENSORSTORE_RETURN_IF_ERROR(
              internal::JsonValidateArrayLength(rank, **input_rank));
        } else {
          *input_rank = rank;
        }
        labels->resize(rank);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& v, DimensionIndex i) {
        if (!v.is_string()) {
          return internal_json::ExpectedError(v, "string");
        }
        (*labels)[i] = v.get<std::string>();
        return absl::OkStatus();
      });
}

struct OutputOffsetAndStride {
  Index offset = 0;
  Index stride = 1;
};

Result<std::int64_t> ParseInt64(const ::nlohmann::json& j) {
  if (auto x = internal::JsonValueAs<std::int64_t>(j)) {
    return *x;
  }
  return internal_json::ExpectedError(j, "64-bit signed integer");
}

Status ParseOutput(const ::nlohmann::json& j,
                   OutputOffsetAndStride* offset_and_stride,
                   OutputIndexMapInitializer* output_map) {
  TENSORSTORE_RETURN_IF_ERROR(internal::JsonValidateObjectMembers(
      j, {"offset", "input_dimension", "index_array", "stride"}));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
      j, "offset", [&](const ::nlohmann::json& value) {
        return internal::JsonRequireValueAs(value, &offset_and_stride->offset);
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
      j, "input_dimension", [&](const ::nlohmann::json& value) {
        int64_t input_dimension;
        auto status = internal::JsonRequireValueAs(value, &input_dimension);
        if (status.ok()) {
          output_map->input_dimension = input_dimension;
        }
        return status;
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
      j, "index_array", [&](const ::nlohmann::json& value) {
        if (output_map->input_dimension) {
          return absl::InvalidArgumentError(
              "At most one of \"input_dimension\" and "
              "\"index_array\" must be specified");
        }
        // There's a bit of a mismatch between JsonParseNestedarray and
        // JsonRequire...
        TENSORSTORE_ASSIGN_OR_RETURN(
            output_map->index_array,
            internal::JsonParseNestedArray(value, &ParseInt64));
        return absl::OkStatus();
      }));

  TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
      j, "stride", [&](const ::nlohmann::json& value) {
        if (!output_map->input_dimension && !output_map->index_array.data()) {
          return absl::InvalidArgumentError(
              "Either \"input_dimension\" or \"index_array\" must be "
              "specified in "
              "conjunction with \"stride\"");
        }
        return internal::JsonRequireValueAs(value, &offset_and_stride->stride);
      }));

  return absl::OkStatus();
}

Status ParseOutputs(
    const ::nlohmann::json& j_outputs,
    absl::optional<DimensionIndex>* output_rank,
    InlinedVector<OutputOffsetAndStride>* output_offsets_and_strides,
    InlinedVector<OutputIndexMapInitializer>* output_maps) {
  return internal::JsonParseArray(
      j_outputs,
      [&](DimensionIndex rank) {
        *output_rank = rank;
        output_offsets_and_strides->resize(rank);
        output_maps->resize(rank);
        return absl::OkStatus();
      },
      [&](const ::nlohmann::json& j_output, DimensionIndex i) {
        return ParseOutput(j_output, &(*output_offsets_and_strides)[i],
                           &(*output_maps)[i]);
      });
}

}  // namespace

Result<TransformRep::Ptr<>> ParseIndexTransformAsJson(
    const ::nlohmann::json& j, DimensionIndex input_rank_constraint,
    DimensionIndex output_rank_constraint) {
  if (j.is_discarded()) return TransformRep::Ptr<>(nullptr);
  auto result = [&]() -> Result<TransformRep::Ptr<>> {
    absl::optional<DimensionIndex> input_rank, output_rank;
    IntervalForm interval_form = IntervalForm::half_open;
    InlinedVector<Index> input_lower, input_upper;
    InlinedVector<std::string> input_labels;
    InlinedVector<OutputOffsetAndStride> output_offsets_and_strides;
    InlinedVector<OutputIndexMapInitializer> output_maps;
    InlinedVector<bool> implicit_lower_bounds, implicit_upper_bounds;
    BuilderFlags flags = 0;
    bool has_output = false;

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonValidateObjectMembers(
        j, {"input_rank", "input_inclusive_min", "input_shape",
            "input_inclusive_max", "input_exclusive_max", "input_labels",
            "output"}));

    const auto upper_bound_error = [] {
      return absl::InvalidArgumentError(
          "At most one of \"input_shape\", \"input_inclusive_max\", and "
          "\"input_exclusive_max\" members must be specified");
    };

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
        j, "input_rank", [&](const ::nlohmann::json& value) {
          DimensionIndex rank;
          TENSORSTORE_RETURN_IF_ERROR(
              internal::JsonRequireInteger(value, &rank,
                                           /*strict=*/true,
                                           /*min_value=*/0));
          input_rank = rank;
          return absl::OkStatus();
        }));

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
        j, "input_inclusive_min", [&](const ::nlohmann::json& value) {
          flags |= (kSetLower | kSetImplicitLower);
          return ParseInputBounds(value, &input_rank, &input_lower,
                                  &implicit_lower_bounds, -kInfIndex, 0);
        }));

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
        j, "input_shape", [&](const ::nlohmann::json& value) {
          interval_form = IntervalForm::sized;
          flags |= (kSetUpper | kSetImplicitUpper);
          return ParseInputBounds(value, &input_rank, &input_upper,
                                  &implicit_upper_bounds, 0, +kInfSize);
        }));

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
        j, "input_inclusive_max", [&](const ::nlohmann::json& value) {
          if (flags & kSetUpper) return upper_bound_error();
          flags |= (kSetUpper | kSetImplicitUpper);
          interval_form = IntervalForm::closed;
          return ParseInputBounds(value, &input_rank, &input_upper,
                                  &implicit_upper_bounds, 0, +kInfIndex);
        }));

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
        j, "input_exclusive_max", [&](const ::nlohmann::json& value) {
          if (flags & kSetUpper) return upper_bound_error();
          flags |= (kSetUpper | kSetImplicitUpper);
          interval_form = IntervalForm::half_open;
          return ParseInputBounds(value, &input_rank, &input_upper,
                                  &implicit_upper_bounds, 0, +kInfIndex + 1);
        }));

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
        j, "input_labels", [&](const ::nlohmann::json& value) {
          return ParseInputLabels(value, &input_rank, &input_labels);
        }));

    TENSORSTORE_RETURN_IF_ERROR(internal::JsonHandleObjectMember(  //
        j, "output", [&](const ::nlohmann::json& value) {
          has_output = true;
          return ParseOutputs(value, &output_rank, &output_offsets_and_strides,
                              &output_maps);
        }));

    if (!input_rank) {
      return absl::InvalidArgumentError(
          "At least one of \"input_rank\", \"input_inclusive_min\", "
          "\"input_shape\", \"input_inclusive_max\", \"input_exclusive_max\", "
          "or \"input_labels\" must be specified");
    }

    if (input_rank_constraint != dynamic_rank &&
        *input_rank != input_rank_constraint) {
      return absl::InvalidArgumentError(StrCat("Expected input rank to be ",
                                               input_rank_constraint,
                                               ", but is: ", *input_rank));
    }

    if (output_rank_constraint != dynamic_rank) {
      if (!output_rank) {
        output_rank = output_rank_constraint;
      } else if (*output_rank != output_rank_constraint) {
        return absl::InvalidArgumentError(StrCat("Expected output rank to be ",
                                                 output_rank_constraint,
                                                 ", but is: ", *output_rank));
      }
    }

    if (!has_output) {
      if (output_rank && *output_rank != *input_rank) {
        return absl::InvalidArgumentError("Missing \"output\" member");
      }
      output_rank = *input_rank;
      output_maps.resize(*input_rank);
      output_offsets_and_strides.resize(*input_rank);
      for (DimensionIndex i = 0; i < *input_rank; ++i) {
        output_maps[i].input_dimension = i;
      }
    }
    auto transform = TransformRep::Allocate(*input_rank, *output_rank);
    transform->input_rank = *input_rank;
    transform->output_rank = *output_rank;
    if (flags & kSetLower) {
      std::copy(input_lower.begin(), input_lower.end(),
                transform->input_origin().begin());
      std::copy(implicit_lower_bounds.begin(), implicit_lower_bounds.end(),
                transform->implicit_lower_bounds(*input_rank).begin());
    }
    if (flags & kSetUpper) {
      std::copy(input_upper.begin(), input_upper.end(),
                transform->input_shape().begin());
      std::copy(implicit_upper_bounds.begin(), implicit_upper_bounds.end(),
                transform->implicit_upper_bounds(*input_rank).begin());
    }
    if (!input_labels.empty()) {
      std::copy(input_labels.begin(), input_labels.end(),
                transform->input_labels().begin());
    }
    auto maps = transform->output_index_maps();
    for (DimensionIndex output_dim = 0; output_dim < *output_rank;
         ++output_dim) {
      auto& map = maps[output_dim];
      map.offset() = output_offsets_and_strides[output_dim].offset;
      map.stride() = output_offsets_and_strides[output_dim].stride;
    }
    TENSORSTORE_RETURN_IF_ERROR(SetOutputIndexMapsAndValidateTransformRep(
        transform.get(), output_maps, interval_form, flags));
    return transform;
  }();

  if (result) return result;
  return MaybeAnnotateStatus(result.status(),
                             "Error parsing index transform from JSON");
}

}  // namespace internal_index_space

namespace jb = tensorstore::internal::json_binding;

TENSORSTORE_DEFINE_JSON_BINDER(
    IndexTransformSpecBinder,
    jb::Validate(
        [](const auto& options, auto* obj) {
          TENSORSTORE_ASSIGN_OR_RETURN(
              *obj, tensorstore::ComposeIndexTransformSpecs(
                        std::move(*obj), IndexTransformSpec{options.rank}));
          return absl::OkStatus();
        },
        jb::Sequence(
            jb::Member("rank",
                       jb::GetterSetter(
                           [](const IndexTransformSpec& s) -> DimensionIndex {
                             return s.transform().valid() ? dynamic_rank
                                                          : s.input_rank();
                           },
                           [](IndexTransformSpec& s, DimensionIndex rank) {
                             s = rank;
                           },
                           jb::DefaultValue(
                               [options](DimensionIndex* r) {
                                 *r = options.rank;
                               },
                               jb::DefaultValue</*NeverIncludeDefaults=*/true>(
                                   [](DimensionIndex* r) { *r = dynamic_rank; },
                                   jb::Integer<DimensionIndex>(0))))),
            jb::Member(
                "transform",
                jb::GetterSetter<IndexTransform<>>(
                    [](const IndexTransformSpec& s) -> IndexTransformView<> {
                      return s.transform();
                    },
                    [](IndexTransformSpec& s, IndexTransform<> transform) {
                      TENSORSTORE_ASSIGN_OR_RETURN(
                          s, tensorstore::ComposeIndexTransformSpecs(
                                 IndexTransformSpec{std::move(transform)},
                                 std::move(s)));
                      return absl::OkStatus();
                    })))))

}  // namespace tensorstore
