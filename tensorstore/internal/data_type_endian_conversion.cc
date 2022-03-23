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

#include "tensorstore/internal/data_type_endian_conversion.h"

#include <cassert>
#include <complex>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "tensorstore/array.h"
#include "tensorstore/data_type.h"
#include "tensorstore/internal/elementwise_function.h"
#include "tensorstore/internal/unaligned_data_type_functions.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/iterate.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"

namespace tensorstore {
namespace internal {

void EncodeArray(ArrayView<const void> source, ArrayView<void> target,
                 endian target_endian) {
  const DataType dtype = source.dtype();
  assert(absl::c_equal(source.shape(), target.shape()));
  assert(dtype == target.dtype());
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(dtype.id())];
  assert(functions.copy != nullptr);  // fail on non-trivial types
  internal::IterateOverStridedLayouts<2>(
      {/*function=*/(target_endian == endian::native) ? functions.copy
                                                      : functions.swap_endian,
       /*context=*/nullptr},
      /*status=*/nullptr, source.shape(),
      {{const_cast<void*>(source.data()), target.data()}},
      {{source.byte_strides().data(), target.byte_strides().data()}},
      /*constraints=*/skip_repeated_elements, {{dtype.size(), dtype.size()}});
}

namespace {
static_assert(sizeof(bool) == 1);
struct DecodeBoolArray {
  void operator()(unsigned char* source, bool* output, absl::Status*) const {
    *output = static_cast<bool>(*source);
  }
};

struct DecodeBoolArrayInplace {
  void operator()(unsigned char* source, absl::Status*) const {
    *source = static_cast<bool>(*source);
  }
};
}  // namespace

void DecodeArray(ArrayView<const void> source, endian source_endian,
                 ArrayView<void> target) {
  const DataType dtype = source.dtype();
  assert(absl::c_equal(source.shape(), target.shape()));
  assert(dtype == target.dtype());
  if (dtype.id() != DataTypeId::bool_t) {
    EncodeArray(source, target, source_endian);
    return;
  }
  // `bool` requires special decoding to ensure the decoded result only contains
  // 0 or 1.
  internal::IterateOverStridedLayouts<2>(
      {/*function=*/SimpleElementwiseFunction<
           DecodeBoolArray(unsigned char, bool), absl::Status*>(),
       /*context=*/nullptr},
      /*status=*/nullptr, source.shape(),
      {{const_cast<void*>(source.data()), target.data()}},
      {{source.byte_strides().data(), target.byte_strides().data()}},
      /*constraints=*/skip_repeated_elements, {{1, 1}});
}

void DecodeArray(SharedArrayView<void>* source, endian source_endian,
                 StridedLayoutView<> decoded_layout) {
  assert(source != nullptr);
  assert(absl::c_equal(source->shape(), decoded_layout.shape()));
  const DataType dtype = source->dtype();
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(dtype.id())];
  assert(functions.copy != nullptr);  // fail on non-trivial types
  if ((reinterpret_cast<std::uintptr_t>(source->data()) % dtype->alignment) ==
          0 &&
      std::all_of(source->byte_strides().begin(), source->byte_strides().end(),
                  [&](Index byte_stride) {
                    return (byte_stride % dtype->alignment) == 0;
                  })) {
    // Source array is already suitably aligned.  Can decode in place.
    const ElementwiseFunction<1, absl::Status*>* convert_func = nullptr;
    if (dtype.id() == DataTypeId::bool_t) {
      convert_func =
          SimpleElementwiseFunction<DecodeBoolArrayInplace(unsigned char),
                                    absl::Status*>();
    } else if (source_endian != endian::native &&
               functions.swap_endian_inplace) {
      convert_func = functions.swap_endian_inplace;
    }
    if (convert_func) {
      internal::IterateOverStridedLayouts<1>(
          {/*function=*/convert_func,
           /*context=*/nullptr},
          /*status=*/nullptr, source->shape(), {{source->data()}},
          {{source->byte_strides().data()}},
          /*constraints=*/skip_repeated_elements, {{dtype.size()}});
    }
  } else {
    // Source array is not suitably aligned.  We could still decode in-place,
    // but the caller is expecting the decoded result to be in a properly
    // aligned array.  Therefore, we allocate a separate array and decode into
    // it.
    *source = CopyAndDecodeArray(*source, source_endian, decoded_layout);
  }
}

SharedArrayView<void> CopyAndDecodeArray(ArrayView<const void> source,
                                         endian source_endian,
                                         StridedLayoutView<> decoded_layout) {
  SharedArrayView<void> target(
      internal::AllocateAndConstructSharedElements(
          decoded_layout.num_elements(), default_init, source.dtype()),
      decoded_layout);
  DecodeArray(source, source_endian, target);
  return target;
}

SharedArrayView<const void> TryViewCordAsArray(const absl::Cord& source,
                                               Index offset, DataType dtype,
                                               endian source_endian,
                                               StridedLayoutView<> layout) {
  const auto& functions =
      kUnalignedDataTypeFunctions[static_cast<size_t>(dtype.id())];
  assert(functions.copy != nullptr);  // fail on non-trivial types
  if (source_endian != endian::native && functions.swap_endian_inplace) {
    // Source data requires endian conversion, and we can't modify the data in
    // an `absl::Cord`.
    return {};
  }
  auto maybe_flat = source.TryFlat();
  if (!maybe_flat) {
    // Source string is not flat.
    return {};
  }
  ByteStridedPointer<const void> ptr = maybe_flat->data();
  ptr += offset;

  if ((reinterpret_cast<std::uintptr_t>(ptr.get()) % dtype->alignment) != 0 ||
      !std::all_of(layout.byte_strides().begin(), layout.byte_strides().end(),
                   [&](Index byte_stride) {
                     return (byte_stride % dtype->alignment) == 0;
                   })) {
    // Source string is not suitably aligned.
    return {};
  }
  auto shared_cord = std::make_shared<absl::Cord>(source);
  // Verify that `shared_cord` has the same flat buffer (this will only fail in
  // the unusual case that `source` was using the inline representation).
  if (auto shared_flat = shared_cord->TryFlat();
      !shared_flat || shared_flat->data() != maybe_flat->data()) {
    return {};
  }

  return SharedArrayView<const void>(
      SharedElementPointer<const void>(
          std::shared_ptr<const void>(std::move(shared_cord), ptr.get()),
          dtype),
      layout);
}

}  // namespace internal
}  // namespace tensorstore
