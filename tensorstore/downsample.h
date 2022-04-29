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

#ifndef TENSORSTORE_DOWNSAMPLE_H_
#define TENSORSTORE_DOWNSAMPLE_H_

/// \file
/// Downsampling adapter for TensorStore objects.

#include "tensorstore/downsample_method.h"
#include "tensorstore/driver/downsample/downsample.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/spec.h"
#include "tensorstore/tensorstore.h"

namespace tensorstore {

/// Returns a downsampled view of a `TensorStore`.
///
/// \param store Base view to downsample, must support reading.
/// \param downsample_factors Factors by which to downsample each dimension of
///     `store`.  Must have length equal to `store.rank()` and all factors must
///     be positive.  Specifying a factor of 1 indicates not to downsample a
///     given dimension.  May be specified as a braced list,
///     e.g. `Downsample(store, {2, 3}, DownsampleMethod::kMean)`.
/// \param downsample_method The downsampling method.
/// \returns The downsampled view, with the same rank as `store` but downsampled
///     domain.
/// \error `absl::StatusCode::kInvalidArgument` if `downsample_factors` is
///     invalid, or `downsample_method` is not supported for
///     `store.dtype()`.
/// \ingroup downsample
/// \id TensorStore
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode>
Result<TensorStore<Element, Rank, ReadWriteMode::read>> Downsample(
    TensorStore<Element, Rank, Mode> store,
    internal::type_identity_t<span<const Index, Rank>> downsample_factors,
    DownsampleMethod downsample_method) {
  static_assert(Mode != ReadWriteMode::write,
                "Cannot downsample write-only TensorStore");
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto transformed_driver,
      internal::MakeDownsampleDriver(
          std::move(internal::TensorStoreAccess::handle(store)),
          downsample_factors, downsample_method));
  return internal::TensorStoreAccess::Construct<
      TensorStore<Element, Rank, ReadWriteMode::read>>(
      std::move(transformed_driver));
}

// Overload that allows `downsample_factors` to be specified as a braced list,
// e.g. `Downsample(store, {2, 3}, DownsampleMethod::kMean)`.
template <typename Element, DimensionIndex Rank, ReadWriteMode Mode,
          DimensionIndex FactorsRank>
std::enable_if_t<RankConstraint::Implies(FactorsRank, Rank),
                 Result<TensorStore<Element, Rank, ReadWriteMode::read>>>
Downsample(TensorStore<Element, Rank, Mode> store,
           const Index (&downsample_factors)[FactorsRank],
           DownsampleMethod downsample_method) {
  return Downsample(std::move(store), span(downsample_factors),
                    downsample_method);
}

/// Returns a downsampled view of a `Spec`.
///
/// \param base_spec Base spec to downsample.
/// \param downsample_factors Factors by which to downsample each dimension.
///     Must have length compatible with `base_spec.rank()` and all factors must
///     be positive.  May be specified as a braced list,
///     e.g. `Downsample(base_spec, {2, 3}, DownsampleMethod::kMean)`.
/// \param downsample_method The downsampling method.
/// \returns The downsampled view.
/// \ingroup downsample
/// \id Spec
Result<Spec> Downsample(const Spec& base_spec,
                        span<const Index> downsample_factors,
                        DownsampleMethod downsample_method);

// Overload that allows `downsample_factors` to be specified as a braced
// list, e.g. `Downsample(spec, {2, 3}, DownsampleMethod::kMean)`.
template <DimensionIndex FactorsRank>
Result<Spec> Downsample(const Spec& base_spec,
                        const Index (&downsample_factors)[FactorsRank],
                        DownsampleMethod downsample_method) {
  return tensorstore::Downsample(
      base_spec, span<const Index>(downsample_factors), downsample_method);
}

}  // namespace tensorstore

#endif  // TENSORSTORE_DOWNSAMPLE_H_
