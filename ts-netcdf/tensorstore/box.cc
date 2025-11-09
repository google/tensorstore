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

#include "tensorstore/box.h"

#include <algorithm>
#include <ostream>

#include "absl/status/status.h"
#include "tensorstore/serialization/serialization.h"
#include "tensorstore/serialization/span.h"  // IWYU pragma: keep
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_box {

std::string DescribeForCast(DimensionIndex rank) {
  return tensorstore::StrCat("box with ",
                             StaticCastTraits<DimensionIndex>::Describe(rank));
}

std::ostream& PrintToOstream(std::ostream& os, const BoxView<>& view) {
  return os << "{origin=" << view.origin() << ", shape=" << view.shape() << "}";
}

bool AreEqual(const BoxView<>& box_a, const BoxView<>& box_b) {
  return box_a.rank() == box_b.rank() &&
         std::equal(box_a.shape().begin(), box_a.shape().end(),
                    box_b.shape().begin()) &&
         std::equal(box_a.origin().begin(), box_a.origin().end(),
                    box_b.origin().begin());
}

bool IsFinite(BoxView<> box) {
  for (DimensionIndex i = 0; i < box.rank(); ++i) {
    if (!IsFinite(box[i])) return false;
  }
  return true;
}

}  // namespace internal_box

namespace serialization {

namespace internal_serialization {
bool EncodeBoxView(EncodeSink& sink, BoxView<> box) {
  return serialization::EncodeTuple(sink, box.origin(), box.shape());
}

bool DecodeBoxView(DecodeSource& source, MutableBoxView<> box) {
  return serialization::DecodeTuple(source, box.origin(), box.shape());
}
}  // namespace internal_serialization

bool RankSerializer::Encode(EncodeSink& sink, DimensionIndex rank) {
  assert(IsValidRank(rank));
  return sink.writer().WriteByte(static_cast<uint8_t>(rank));
}

bool RankSerializer::Decode(DecodeSource& source, DimensionIndex& rank) {
  uint8_t v;
  if (!source.reader().ReadByte(v)) return false;
  if (v > kMaxRank) {
    source.Fail(DecodeError(
        tensorstore::StrCat("Invalid rank value: ", static_cast<size_t>(v))));
  }
  rank = static_cast<DimensionIndex>(v);
  return true;
}

}  // namespace serialization

}  // namespace tensorstore
