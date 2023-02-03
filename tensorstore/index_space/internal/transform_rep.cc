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

#include "tensorstore/index_space/internal/transform_rep.h"

#include <memory>
#include <new>
#include <utility>

#include "absl/base/optimization.h"
#include "absl/container/fixed_array.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/internal/transform_rep_impl.h"
#include "tensorstore/internal/dimension_labels.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/util/division.h"
#include "tensorstore/util/quote_string.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_index_space {

namespace {
void FreeIndexArrayData(IndexArrayData* data) {
  std::destroy_at(data);
  std::free(data);
}

void CopyTrivialFields(TransformRep* source, TransformRep* dest) {
  assert(dest->input_rank_capacity >= source->input_rank &&
         dest->output_rank_capacity >= source->output_rank);
  const DimensionIndex input_rank = dest->input_rank = source->input_rank;
  dest->output_rank = source->output_rank;
  std::copy_n(source->input_origin().begin(), input_rank,
              dest->input_origin().begin());
  std::copy_n(source->input_shape().begin(), input_rank,
              dest->input_shape().begin());
  dest->implicit_lower_bounds = source->implicit_lower_bounds;
  dest->implicit_upper_bounds = source->implicit_upper_bounds;
}

}  // namespace

void CopyInputLabels(TransformRep* source, TransformRep* dest, bool can_move) {
  assert(dest->input_rank_capacity >= source->input_rank);
  const DimensionIndex input_rank = source->input_rank;
  if (can_move) {
    std::copy_n(std::make_move_iterator(source->input_labels().begin()),
                input_rank, dest->input_labels().begin());
  } else {
    std::copy_n(source->input_labels().begin(), input_rank,
                dest->input_labels().begin());
  }
}

void OutputIndexMap::SetConstant() {
  if (method() == OutputIndexMethod::array) {
    FreeIndexArrayData(&index_array_data());
  }
  value_ = 0;
}

void OutputIndexMap::SetSingleInputDimension(DimensionIndex input_dim) {
  if (method() == OutputIndexMethod::array) {
    FreeIndexArrayData(&index_array_data());
  }
  value_ = (input_dim << 1) | 1;
}

IndexArrayData& OutputIndexMap::SetArrayIndexing(DimensionIndex rank) {
  IndexArrayData* data;
  if (method() == OutputIndexMethod::array) {
    // An IndexArrayData has already been allocated.
    data = &index_array_data();
    if (data->rank_capacity >= rank) return *data;
    // Resize the IndexArrayData to have sufficient capacity in the trailing
    // byte_strides array.
    SharedElementPointer<const Index> element_pointer =
        std::move(data->element_pointer);
    auto bounds = data->index_range;
    std::destroy_at(data);
    IndexArrayData* new_data = static_cast<IndexArrayData*>(
        std::realloc(static_cast<void*>(data),
                     sizeof(IndexArrayData) + sizeof(Index) * rank));
    if (new_data) data = new_data;
    new (data) IndexArrayData;
    data->element_pointer = std::move(element_pointer);
    data->index_range = bounds;
    if (!new_data) TENSORSTORE_THROW_BAD_ALLOC;
    data->rank_capacity = rank;
  } else {
    // No existing IndexArrayData has been allocated.  Allocate it with
    // sufficient capacity in the trailing byte_strides array.
    data = static_cast<IndexArrayData*>(
        std::malloc(sizeof(IndexArrayData) + sizeof(Index) * rank));
    if (!data) {
      TENSORSTORE_THROW_BAD_ALLOC;
    }
    new (data) IndexArrayData;
    data->rank_capacity = rank;
  }
  value_ = reinterpret_cast<std::uintptr_t>(data);
  return *data;
}

IndexArrayData& OutputIndexMap::SetArrayIndexing(DimensionIndex rank,
                                                 const IndexArrayData& other) {
  assert(other.rank_capacity >= rank);
  auto& data = SetArrayIndexing(rank);
  data.element_pointer = other.element_pointer;
  data.index_range = other.index_range;
  std::memcpy(data.byte_strides, other.byte_strides, sizeof(Index) * rank);
  return data;
}

void OutputIndexMap::Assign(DimensionIndex rank, const OutputIndexMap& other) {
  if (other.method() == OutputIndexMethod::array) {
    SetArrayIndexing(rank, other.index_array_data());
  } else {
    value_ = other.value_;
  }
  offset_ = other.offset_;
  stride_ = other.stride_;
}

TransformRep::Ptr<> TransformRep::Allocate(
    DimensionIndex input_rank_capacity, DimensionIndex output_rank_capacity) {
  ABSL_CHECK(input_rank_capacity >= 0 && output_rank_capacity >= 0 &&
             input_rank_capacity <= kMaxRank &&
             output_rank_capacity <= kMaxRank);
  const size_t total_size =
      // header size
      sizeof(TransformRep) +
      // size of OutputIndexMap array
      sizeof(OutputIndexMap) * output_rank_capacity +
      // size of input_origin, input_shape, and input_labels arrays
      input_rank_capacity * (sizeof(Index) * 2 + sizeof(std::string));
  char* base_ptr = static_cast<char*>(::operator new(total_size));
  TransformRep* ptr =  // NOLINT
      new (base_ptr + sizeof(OutputIndexMap) * output_rank_capacity)
          TransformRep;
  ptr->reference_count.store(1, std::memory_order_relaxed);
  ptr->input_rank_capacity = input_rank_capacity;
  ptr->output_rank_capacity = output_rank_capacity;
  std::uninitialized_default_construct_n(ptr->output_index_maps().begin(),
                                         output_rank_capacity);
  std::uninitialized_default_construct_n(ptr->input_labels().begin(),
                                         input_rank_capacity);
  return TransformRep::Ptr<>(ptr, internal::adopt_object_ref);
}

void DestroyLabelFields(TransformRep* ptr) {
  std::destroy_n(ptr->input_labels().begin(), ptr->input_rank_capacity);
}

void TransformRep::Free(TransformRep* ptr) {
  assert(ptr->reference_count == 0);
  DestroyLabelFields(ptr);
  std::destroy_n(ptr->output_index_maps().begin(), ptr->output_rank_capacity);
  ::operator delete(static_cast<void*>(ptr->output_index_maps().data()));
}

void CopyTransformRep(TransformRep* source, TransformRep* dest) {
  assert(source != nullptr);
  assert(dest != nullptr);
  assert(dest->output_rank_capacity >= source->output_rank);
  CopyTransformRepDomain(source, dest);
  const DimensionIndex input_rank = source->input_rank;
  const DimensionIndex output_rank = dest->output_rank = source->output_rank;
  span<const OutputIndexMap> source_maps =
      source->output_index_maps().first(output_rank);
  span<OutputIndexMap> dest_maps = dest->output_index_maps().first(output_rank);
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    dest_maps[output_dim].Assign(input_rank, source_maps[output_dim]);
  }
}

void CopyTransformRepDomain(TransformRep* source, TransformRep* dest) {
  assert(source != nullptr);
  assert(dest != nullptr);
  assert(dest->input_rank_capacity >= source->input_rank);
  const DimensionIndex input_rank = dest->input_rank = source->input_rank;
  std::copy_n(source->input_origin().begin(), input_rank,
              dest->input_origin().begin());
  std::copy_n(source->input_shape().begin(), input_rank,
              dest->input_shape().begin());
  dest->implicit_lower_bounds = source->implicit_lower_bounds;
  dest->implicit_upper_bounds = source->implicit_upper_bounds;
  std::copy_n(source->input_labels().begin(), input_rank,
              dest->input_labels().begin());
}

void MoveTransformRep(TransformRep* source, TransformRep* dest) {
  CopyTrivialFields(source, dest);
  std::copy_n(std::make_move_iterator(source->output_index_maps().begin()),
              source->output_rank, dest->output_index_maps().begin());
  CopyInputLabels(source, dest, /*can_move=*/true);
}

void ResetOutputIndexMaps(TransformRep* ptr) {
  auto output_index_maps = ptr->output_index_maps();
  for (DimensionIndex output_dim = 0, output_rank = ptr->output_rank;
       output_dim < output_rank; ++output_dim) {
    output_index_maps[output_dim].SetConstant();
  }
  ptr->output_rank = 0;
}

TransformRep::Ptr<> MutableRep(TransformRep::Ptr<> ptr, bool domain_only) {
  if (!ptr) return ptr;
  if (ptr->is_unique()) {
    if (domain_only) {
      ResetOutputIndexMaps(ptr.get());
      ptr->output_rank = 0;
    }
    return ptr;
  }
  if (domain_only) {
    auto new_rep = TransformRep::Allocate(ptr->input_rank, 0);
    CopyTransformRepDomain(ptr.get(), new_rep.get());
    new_rep->output_rank = 0;
    internal_index_space::DebugCheckInvariants(new_rep.get());
    return new_rep;
  } else {
    auto new_rep = TransformRep::Allocate(ptr->input_rank, ptr->output_rank);
    CopyTransformRep(ptr.get(), new_rep.get());
    internal_index_space::DebugCheckInvariants(new_rep.get());
    return new_rep;
  }
}

TransformRep::Ptr<> NewOrMutableRep(TransformRep* ptr,
                                    DimensionIndex input_rank_capacity,
                                    DimensionIndex output_rank_capacity,
                                    bool domain_only) {
  assert(ptr);
  if (ptr->input_rank_capacity >= input_rank_capacity &&
      ptr->output_rank_capacity >= output_rank_capacity && ptr->is_unique()) {
    if (domain_only) {
      ResetOutputIndexMaps(ptr);
    }
    return TransformRep::Ptr<>(ptr);
  } else {
    return TransformRep::Allocate(input_rank_capacity,
                                  domain_only ? 0 : output_rank_capacity);
  }
}

bool IsDomainExplicitlyEmpty(TransformRep* ptr) {
  DimensionSet implicit_dims = ptr->implicit_dimensions();

  const Index* input_shape = ptr->input_shape().data();
  for (DimensionIndex input_dim = 0, input_rank = ptr->input_rank;
       input_dim < input_rank; ++input_dim) {
    if (!implicit_dims[input_dim] && input_shape[input_dim] == 0) {
      return true;
    }
  }
  return false;
}

void ReplaceAllIndexArrayMapsWithConstantMaps(TransformRep* ptr) {
  for (DimensionIndex output_dim = 0, output_rank = ptr->output_rank;
       output_dim < output_rank; ++output_dim) {
    auto& map = ptr->output_index_maps()[output_dim];
    if (map.method() != OutputIndexMethod::array) continue;
    map.SetConstant();
    map.offset() = 0;
    map.stride() = 0;
  }
}

bool AreIndexMapsEqual(const OutputIndexMap& a, const OutputIndexMap& b,
                       BoxView<> input_domain) {
  const auto method = a.method();
  if (method != b.method() || a.offset() != b.offset()) return false;
  switch (method) {
    case OutputIndexMethod::constant:
      return true;
    case OutputIndexMethod::single_input_dimension:
      return a.input_dimension() == b.input_dimension() &&
             a.stride() == b.stride();
    case OutputIndexMethod::array: {
      const auto& index_array_data_a = a.index_array_data();
      const auto& index_array_data_b = b.index_array_data();
      if (a.stride() != b.stride()) return false;
      if (index_array_data_a.index_range != index_array_data_b.index_range) {
        return false;
      }
      return ArrayView<const Index, dynamic_rank, offset_origin>(
                 index_array_data_a.element_pointer,
                 StridedLayoutView<dynamic_rank, offset_origin>(
                     input_domain.rank(), input_domain.origin().data(),
                     input_domain.shape().data(),
                     index_array_data_a.byte_strides)) ==
             ArrayView<const Index, dynamic_rank, offset_origin>(
                 index_array_data_b.element_pointer,
                 StridedLayoutView<dynamic_rank, offset_origin>(
                     input_domain.rank(), input_domain.origin().data(),
                     input_domain.shape().data(),
                     index_array_data_b.byte_strides));
    }
  }
  ABSL_UNREACHABLE();  // COV_NF_LINE
}

bool AreDomainsEqual(TransformRep* a, TransformRep* b) {
  if (!a != !b) return false;
  if (!a) return true;
  if (a->input_rank != b->input_rank) return false;
  const DimensionIndex input_rank = a->input_rank;
  const BoxView<> input_domain_a = a->input_domain(input_rank);
  if (input_domain_a != b->input_domain(input_rank)) return false;
  if (a->implicit_lower_bounds != b->implicit_lower_bounds ||
      a->implicit_upper_bounds != b->implicit_upper_bounds) {
    return false;
  }
  span<const std::string> input_labels_a = a->input_labels().first(input_rank);
  if (!std::equal(input_labels_a.begin(), input_labels_a.end(),
                  b->input_labels().begin())) {
    return false;
  }

  return true;
}

bool AreEqual(TransformRep* a, TransformRep* b) {
  if (!AreDomainsEqual(a, b)) return false;
  if (!a) return true;
  if (a->output_rank != b->output_rank) return false;
  const DimensionIndex input_rank = a->input_rank;
  const DimensionIndex output_rank = a->output_rank;
  const BoxView<> input_domain_a = a->input_domain(input_rank);
  span<const OutputIndexMap> a_maps = a->output_index_maps().first(output_rank);
  span<const OutputIndexMap> b_maps = b->output_index_maps().first(output_rank);

  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    if (!AreIndexMapsEqual(a_maps[output_dim], b_maps[output_dim],
                           input_domain_a)) {
      return false;
    }
  }

  return true;
}

void PrintToOstream(std::ostream& os, TransformRep* transform) {
  if (!transform) {
    os << "<Invalid index space transform>";
    return;
  }
  const DimensionIndex input_rank = transform->input_rank;
  const DimensionIndex output_rank = transform->output_rank;
  os << "Rank " << transform->input_rank << " -> " << transform->output_rank
     << " index space transform:\n";
  os << "  Input domain:\n";
  const BoxView<> input_domain = transform->input_domain(input_rank);
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    const auto d = transform->input_dimension(input_dim);
    os << "    " << input_dim << ": " << d.optionally_implicit_domain();
    if (!d.label().empty()) {
      os << " " << QuoteString(d.label());
    }
    os << '\n';
  }
  span<const OutputIndexMap> maps =
      transform->output_index_maps().first(output_rank);
  absl::FixedArray<Index, internal::kNumInlinedDims> index_array_shape(
      input_rank);
  os << "  Output index maps:\n";
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    const auto& map = maps[output_dim];
    os << "    out[" << output_dim << "] = " << map.offset();
    if (map.method() != OutputIndexMethod::constant) {
      os << " + " << map.stride() << " * ";
    }
    switch (map.method()) {
      case OutputIndexMethod::constant:
        break;
      case OutputIndexMethod::single_input_dimension:
        os << "in[" << map.input_dimension() << "]";
        break;
      case OutputIndexMethod::array: {
        const auto& index_array_data = map.index_array_data();
        for (DimensionIndex input_dim = 0; input_dim < input_rank;
             ++input_dim) {
          index_array_shape[input_dim] =
              index_array_data.byte_strides[input_dim] == 0
                  ? 1
                  : input_domain.shape()[input_dim];
        }

        ArrayView<const Index, dynamic_rank> index_array(
            AddByteOffset(
                ElementPointer<const Index>(index_array_data.element_pointer),
                IndexInnerProduct(input_rank, input_domain.origin().data(),
                                  index_array_data.byte_strides)),
            StridedLayoutView<>(input_rank, index_array_shape.data(),
                                index_array_data.byte_strides));
        os << "bounded(" << index_array_data.index_range
           << ", array(in)), where array =\n";
        os << "      " << index_array;
        break;
      }
    }
    os << '\n';
  }
}

void PrintDomainToOstream(std::ostream& os, TransformRep* transform) {
  if (!transform) {
    os << "<invalid index domain>";
    return;
  }
  os << "{ ";
  for (DimensionIndex i = 0, rank = transform->input_rank; i < rank; ++i) {
    if (i != 0) os << ", ";
    const InputDimensionRef dim_ref = transform->input_dimension(i);
    const IndexDomainDimension<view> d{dim_ref.optionally_implicit_domain(),
                                       dim_ref.label()};
    os << d;
  }
  os << " }";
}

Result<Index> OutputIndexMap::operator()(
    span<const Index> input_indices) const {
  Index base_output_index;
  switch (method()) {
    case OutputIndexMethod::constant:
      base_output_index = 0;
      break;
    case OutputIndexMethod::single_input_dimension: {
      const DimensionIndex input_dim = input_dimension();
      assert(input_dim >= 0 && input_dim < input_indices.size());
      base_output_index = input_indices[input_dim];
      break;
    }
    case OutputIndexMethod::array: {
      const IndexArrayData& data = index_array_data();
      assert(data.element_pointer &&
             input_indices.size() <= data.rank_capacity);
      base_output_index =
          data.element_pointer.byte_strided_pointer()[IndexInnerProduct(
              input_indices.size(), input_indices.data(), data.byte_strides)];
      TENSORSTORE_RETURN_IF_ERROR(
          CheckContains(data.index_range, base_output_index),
          MaybeAnnotateStatus(
              _, "Checking result of index array output index map"));
      break;
    }
  }
  return base_output_index * stride() + offset();
}

absl::Status TransformIndices(TransformRep* data,
                              span<const Index> input_indices,
                              span<Index> output_indices) {
  assert(data && data->input_rank == input_indices.size() &&
         data->output_rank == output_indices.size());
  const DimensionIndex output_rank = data->output_rank;
  const DimensionIndex input_rank = data->input_rank;
  span<const OutputIndexMap> output_index_maps =
      data->output_index_maps().first(output_rank);
  for (DimensionIndex i = 0; i < input_rank; ++i) {
    auto oi_interval = data->input_dimension(i).optionally_implicit_domain();
    if (!Contains(oi_interval.effective_interval(), input_indices[i])) {
      return absl::OutOfRangeError(tensorstore::StrCat(
          "Index ", input_indices[i], " is not contained in the domain ",
          oi_interval, " for input dimension ", i));
    }
  }
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    TENSORSTORE_ASSIGN_OR_RETURN(
        output_indices[output_dim],
        output_index_maps[output_dim](input_indices),
        MaybeAnnotateStatus(
            _, tensorstore::StrCat("Computing index for output dimension ",
                                   output_dim)));
  }
  return absl::OkStatus();
}

absl::Status ReplaceZeroRankIndexArrayIndexMap(Index index,
                                               IndexInterval bounds,
                                               Index* output_offset,
                                               Index* output_stride) {
  TENSORSTORE_RETURN_IF_ERROR(CheckContains(bounds, index));
  Index new_offset;
  if (internal::MulOverflow(index, *output_stride, &new_offset) ||
      internal::AddOverflow(new_offset, *output_offset, output_offset)) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        "Integer overflow computing offset for output dimension."));
  }
  *output_stride = 0;
  return absl::OkStatus();
}

TransformRep::Ptr<> GetSubDomain(TransformRep* rep,
                                 span<const DimensionIndex> dims) {
  assert(rep);
  [[maybe_unused]] const DimensionIndex old_rank = rep->input_rank;
  const DimensionIndex new_rank = dims.size();
  auto new_rep = TransformRep::Allocate(new_rank, 0);
  new_rep->output_rank = 0;
  new_rep->input_rank = new_rank;
#ifndef NDEBUG
  absl::FixedArray<bool, internal::kNumInlinedDims> seen_dims(old_rank, false);
#endif
  for (DimensionIndex new_dim = 0; new_dim < dims.size(); ++new_dim) {
    const DimensionIndex old_dim = dims[new_dim];
    assert(old_dim >= 0 && old_dim < old_rank);
#ifndef NDEBUG
    assert(!seen_dims[old_dim]);
    seen_dims[old_dim] = true;
#endif
    new_rep->input_dimension(new_dim) = rep->input_dimension(old_dim);
  }
  return new_rep;
}

bool IsUnlabeled(span<const std::string> labels) {
  return std::all_of(labels.begin(), labels.end(),
                     [](std::string_view s) { return s.empty(); });
}

DimensionSet GetIndexArrayInputDimensions(TransformRep* transform) {
  DimensionSet set;
  const DimensionIndex output_rank = transform->output_rank;
  const DimensionIndex input_rank = transform->input_rank;
  auto output_maps = transform->output_index_maps();
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& map = output_maps[output_dim];
    if (map.method() != OutputIndexMethod::array) continue;
    const auto& index_array_data = map.index_array_data();
    for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
      if (index_array_data.byte_strides[input_dim] != 0) {
        set[input_dim] = true;
      }
    }
  }
  return set;
}

TransformRep::Ptr<> WithImplicitDimensions(TransformRep::Ptr<> transform,
                                           DimensionSet implicit_lower_bounds,
                                           DimensionSet implicit_upper_bounds,
                                           bool domain_only) {
  transform = MutableRep(std::move(transform), domain_only);
  if (!domain_only && (implicit_lower_bounds || implicit_upper_bounds)) {
    // Ensure all dimensions on which an index array depends remain explicit.
    auto index_array_dims =
        internal_index_space::GetIndexArrayInputDimensions(transform.get());
    implicit_lower_bounds &= ~index_array_dims;
    implicit_upper_bounds &= ~index_array_dims;
  }
  const auto mask = DimensionSet::UpTo(transform->input_rank);
  transform->implicit_lower_bounds = implicit_lower_bounds & mask;
  transform->implicit_upper_bounds = implicit_upper_bounds & mask;
  return transform;
}

#ifndef NDEBUG
void DebugCheckInvariants(TransformRep* rep) {
  assert(rep);
  assert(rep->reference_count > 0);
  const DimensionIndex input_rank = rep->input_rank,
                       output_rank = rep->output_rank;
  assert(rep->input_rank_capacity <= kMaxRank);
  assert(rep->output_rank_capacity <= kMaxRank);
  assert(input_rank <= rep->input_rank_capacity);
  assert(output_rank <= rep->output_rank_capacity);
  assert(input_rank >= 0);
  assert(output_rank >= 0);
  const auto mask = DimensionSet::UpTo(rep->input_rank);
  assert((rep->implicit_lower_bounds & mask) == rep->implicit_lower_bounds);
  assert((rep->implicit_upper_bounds & mask) == rep->implicit_upper_bounds);
  TENSORSTORE_CHECK_OK(internal::ValidateDimensionLabelsAreUnique(
      rep->input_labels().first(input_rank)));
  auto input_origin = rep->input_origin().data();
  auto input_shape = rep->input_shape().data();
  for (DimensionIndex input_dim = 0; input_dim < input_rank; ++input_dim) {
    TENSORSTORE_CHECK_OK(
        IndexInterval::Sized(input_origin[input_dim], input_shape[input_dim]));
  }
  const bool is_domain_explicitly_empty = IsDomainExplicitlyEmpty(rep);
  const auto implicit_dims = rep->implicit_dimensions();
  for (DimensionIndex output_dim = 0; output_dim < output_rank; ++output_dim) {
    auto& map = rep->output_index_maps()[output_dim];
    switch (map.method()) {
      case OutputIndexMethod::constant: {
        assert(map.stride() == 0);
        break;
      }
      case OutputIndexMethod::single_input_dimension: {
        const DimensionIndex input_dim = map.input_dimension();
        assert(input_dim >= 0 && input_dim < input_rank);
        assert(map.stride() != 0);
        break;
      }
      case OutputIndexMethod::array: {
        assert(map.stride() != 0);
        const auto& index_array_data = map.index_array_data();
        assert(index_array_data.rank_capacity >= input_rank);
        assert(index_array_data.rank_capacity <= kMaxRank);
        assert(!is_domain_explicitly_empty);
        for (DimensionIndex input_dim = 0; input_dim < input_rank;
             ++input_dim) {
          const Index byte_stride = index_array_data.byte_strides[input_dim];
          if (byte_stride == 0) continue;
          const auto bounds = IndexInterval::UncheckedSized(
              input_origin[input_dim], input_shape[input_dim]);
          assert(IsFinite(bounds));
          assert(!implicit_dims[input_dim]);
        }
        break;
      }
    }
  }

  // Validate that no index arrays remain allocated
  for (DimensionIndex output_dim = output_rank,
                      output_rank_capacity = rep->output_rank_capacity;
       output_dim < output_rank_capacity; ++output_dim) {
    assert(rep->output_index_maps()[output_dim].method() !=
           OutputIndexMethod::array);
  }
}
#endif

}  // namespace internal_index_space
}  // namespace tensorstore
