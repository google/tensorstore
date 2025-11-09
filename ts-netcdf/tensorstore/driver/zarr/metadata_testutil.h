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

#ifndef TENSORSTORE_DRIVER_ZARR_METADATA_TESTUTIL_H_
#define TENSORSTORE_DRIVER_ZARR_METADATA_TESTUTIL_H_

#include <iosfwd>

#include "tensorstore/driver/zarr/metadata.h"

namespace tensorstore {
namespace internal_zarr {

bool operator==(const ZarrDType::BaseDType& a, const ZarrDType::BaseDType& b);
inline bool operator!=(const ZarrDType::BaseDType& a,
                       const ZarrDType::BaseDType& b) {
  return !(a == b);
}

void PrintTo(const ZarrDType::BaseDType& x, std::ostream* os);

bool operator==(const ZarrDType::Field& a, const ZarrDType::Field& b);
inline bool operator!=(const ZarrDType::Field& a, const ZarrDType::Field& b) {
  return !(a == b);
}
void PrintTo(const ZarrDType::Field& x, std::ostream* os);

bool operator==(const ZarrDType& a, const ZarrDType& b);
inline bool operator!=(const ZarrDType& a, const ZarrDType& b) {
  return !(a == b);
}
void PrintTo(const ZarrDType& x, std::ostream* os);

}  // namespace internal_zarr
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_ZARR_METADATA_TESTUTIL_H_
